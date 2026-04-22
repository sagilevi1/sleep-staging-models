"""
Training Script for DREAMT Triple-Stream Sleep Staging Model

Features:
- Subject-level train/val/test split (no data leakage)
- 6-class classification (P, W, N1, N2, N3, REM)
- Mixed precision training (AMP)
- Class-weighted loss for imbalanced data
- Metrics: Loss, Accuracy, Cohen's Kappa, Per-class F1
- TensorBoard logging
- Early stopping
- Checkpoint saving
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from dreamt_triple_dataset import (
    DatasetConfig,
    DreamtTripleStreamDataset,
    get_dataloaders,
    STAGE_NAMES,
    NUM_CLASSES,
)
from triple_stream_model import TripleStreamSleepNet, create_triple_stream_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Trainer Class
# =============================================================================

class TripleStreamTrainer:
    """Trainer for Triple-Stream Sleep Staging Model."""
    
    def __init__(self, config: dict, run_id: Optional[int] = None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            run_id: Optional run ID for multiple runs
        """
        self.config = config
        self.run_id = run_id
        
        # Setup device
        device_name = config.get("hardware", {}).get("device", "cuda")
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Setup directories
        self._setup_directories()
        
        # Save config
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # TensorBoard (optional)
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
        
        # Mixed precision
        self.use_amp = config.get("hardware", {}).get("use_amp", True) and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using mixed precision training (AMP)")
        
        # Gradient accumulation
        self.grad_accum_steps = config.get("hardware", {}).get("gradient_accumulation_steps", 1)
    
    def _setup_directories(self):
        """Create output directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"triple_stream_{timestamp}"
        if self.run_id is not None:
            model_name += f"_run{self.run_id}"
        
        base_dir = Path(self.config.get("output", {}).get("save_dir", "./outputs/triple_stream"))
        self.output_dir = base_dir / model_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.results_dir = self.output_dir / "results"
        
        for d in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders."""
        data_config = self.config.get("data", {})
        
        dataset_config = DatasetConfig(
            data_dir=data_config.get("data_dir", "C:/Users/SagiLevi/data_64Hz"),
            fs=data_config.get("fs", 64.0),
            window_sec=data_config.get("window_sec", 30.0),
            train_ratio=data_config.get("train_ratio", 0.70),
            val_ratio=data_config.get("val_ratio", 0.15),
            test_ratio=data_config.get("test_ratio", 0.15),
            seed=data_config.get("seed", 42),
            min_windows_per_subject=data_config.get("min_windows_per_subject", 10),
        )
        
        train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
            dataset_config,
            batch_size=self.config.get("training", {}).get("batch_size", 16),
            num_workers=data_config.get("num_workers", 4),
            pin_memory=True,
        )
        
        # Store class weights for loss function
        self.class_weights = train_ds.get_class_weights().to(self.device)
        logger.info(f"Class weights: {self.class_weights.tolist()}")
        
        return train_loader, val_loader, test_loader
    
    def _create_model(self) -> TripleStreamSleepNet:
        """Create model."""
        model_config = self.config.get("model", {})
        
        model = create_triple_stream_model(
            n_classes=model_config.get("n_classes", NUM_CLASSES),
            d_model=model_config.get("d_model", 256),
            n_heads=model_config.get("n_heads", 8),
            n_fusion_blocks=model_config.get("n_fusion_blocks", 3),
            dropout=model_config.get("dropout", 0.2),
        )
        
        model = model.to(self.device)
        
        logger.info(f"Model parameters: {model.get_num_parameters():,}")
        logger.info(f"Trainable parameters: {model.get_num_trainable_parameters():,}")
        
        return model
    
    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            epoch_loss, epoch_accuracy
        """
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        log_freq = self.config.get("output", {}).get("log_frequency", 10)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        
        optimizer.zero_grad()
        
        for batch_idx, (bvp, acc, ibi_feat, labels) in enumerate(pbar):
            bvp = bvp.to(self.device)
            acc = acc.to(self.device)
            ibi_feat = ibi_feat.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    logits = model(bvp, acc, ibi_feat)
                    loss = criterion(logits, labels)
                    loss = loss / self.grad_accum_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(bvp, acc, ibi_feat)
                loss = criterion(logits, labels)
                loss = loss / self.grad_accum_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item() * self.grad_accum_steps * labels.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            if batch_idx % log_freq == 0:
                pbar.set_postfix({
                    "loss": running_loss / total,
                    "acc": correct / total,
                })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> Dict:
        """
        Validate model.
        
        Returns:
            Dictionary with metrics
        """
        model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for bvp, acc, ibi_feat, labels in tqdm(dataloader, desc="Validation"):
            bvp = bvp.to(self.device)
            acc = acc.to(self.device)
            ibi_feat = ibi_feat.to(self.device)
            labels = labels.to(self.device)
            
            if self.use_amp:
                with autocast():
                    logits = model(bvp, acc, ibi_feat)
                    loss = criterion(logits, labels)
            else:
                logits = model(bvp, acc, ibi_feat)
                loss = criterion(logits, labels)
            
            running_loss += loss.item() * labels.size(0)
            
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        epoch_loss = running_loss / len(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        
        # Per-class metrics
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
        
        return {
            "loss": epoch_loss,
            "accuracy": accuracy,
            "kappa": kappa,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "confusion_matrix": cm,
            "all_preds": all_preds,
            "all_labels": all_labels,
        }
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict,
        is_best: bool = False,
    ):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {k: v for k, v in metrics.items() if k not in ["all_preds", "all_labels", "confusion_matrix"]},
            "config": self.config,
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, path)
            logger.info(f"Saved best model (kappa={metrics['kappa']:.4f})")
        
        # Save periodic checkpoint
        save_freq = self.config.get("output", {}).get("save_frequency", 5)
        if epoch % save_freq == 0:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, path)
    
    def _plot_confusion_matrix(self, cm: np.ndarray, epoch: int, prefix: str = ""):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Normalize
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        
        # Create annotations
        annot = np.empty_like(cm, dtype=str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
        
        labels = [STAGE_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)]
        
        sns.heatmap(
            cm_norm * 100,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        
        plt.title(f"Confusion Matrix - {prefix}Epoch {epoch}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        save_path = self.results_dir / f"confusion_matrix_{prefix}epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def train(self) -> Dict:
        """
        Main training loop.
        
        Returns:
            Dictionary with final test results
        """
        logger.info("=" * 70)
        logger.info("Starting DREAMT Triple-Stream Training")
        logger.info("=" * 70)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self._create_dataloaders()
        
        # Create model
        model = self._create_model()
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Optimizer
        training_config = self.config.get("training", {})
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 1e-5),
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
        
        # Training loop
        num_epochs = training_config.get("num_epochs", 50)
        patience = training_config.get("patience", 15)
        
        best_kappa = 0.0
        best_epoch = 0
        patience_counter = 0
        
        history = defaultdict(list)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Train
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            val_metrics = self._validate(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_metrics["kappa"])
            
            # Log metrics
            logger.info(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val Kappa: {val_metrics['kappa']:.4f}, Val F1: {val_metrics['f1_weighted']:.4f}")
            
            # TensorBoard logging (if available)
            if self.writer is not None:
                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
                self.writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("Val/Accuracy", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("Val/Kappa", val_metrics["kappa"], epoch)
                self.writer.add_scalar("Val/F1_Weighted", val_metrics["f1_weighted"], epoch)
                self.writer.add_scalar("Val/F1_Macro", val_metrics["f1_macro"], epoch)
                self.writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            
            # Save history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_kappa"].append(val_metrics["kappa"])
            history["val_f1"].append(val_metrics["f1_weighted"])
            
            # Check for improvement
            if val_metrics["kappa"] > best_kappa:
                best_kappa = val_metrics["kappa"]
                best_epoch = epoch
                patience_counter = 0
                
                self._save_checkpoint(model, optimizer, epoch, val_metrics, is_best=True)
                self._plot_confusion_matrix(val_metrics["confusion_matrix"], epoch, prefix="best_")
            else:
                patience_counter += 1
            
            # Periodic save
            self._save_checkpoint(model, optimizer, epoch, val_metrics, is_best=False)
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break
            
            # Memory cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"\nBest validation kappa: {best_kappa:.4f} at epoch {best_epoch}")
        
        # =====================================================================
        # Test on best model
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("Evaluating best model on test set")
        logger.info("=" * 70)
        
        # Load best model
        checkpoint = torch.load(self.checkpoint_dir / "best_model.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Test
        test_metrics = self._validate(model, test_loader, criterion)
        
        logger.info(f"\nTest Results:")
        logger.info(f"  Loss: {test_metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Kappa: {test_metrics['kappa']:.4f}")
        logger.info(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        logger.info(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        
        # Classification report
        report = classification_report(
            test_metrics["all_labels"],
            test_metrics["all_preds"],
            target_names=[STAGE_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)],
            output_dict=True,
        )
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(
            test_metrics["all_labels"],
            test_metrics["all_preds"],
            target_names=[STAGE_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)],
        ))
        
        # Save confusion matrix
        self._plot_confusion_matrix(test_metrics["confusion_matrix"], best_epoch, prefix="test_")
        
        # Save results
        results = {
            "best_epoch": best_epoch,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_kappa": test_metrics["kappa"],
            "test_f1_weighted": test_metrics["f1_weighted"],
            "test_f1_macro": test_metrics["f1_macro"],
            "classification_report": report,
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            "config": self.config,
        }
        
        results_path = self.results_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to: {results_path}")
        
        # Plot training curves
        self._plot_training_curves(history)
        
        if self.writer is not None:
            self.writer.close()
        
        return results
    
    def _plot_training_curves(self, history: Dict):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(history["train_loss"]) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train")
        axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, history["train_acc"], "b-", label="Train")
        axes[0, 1].plot(epochs, history["val_acc"], "r-", label="Val")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Kappa
        axes[1, 0].plot(epochs, history["val_kappa"], "g-", label="Val Kappa")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Kappa")
        axes[1, 0].set_title("Cohen's Kappa")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1
        axes[1, 1].plot(epochs, history["val_f1"], "m-", label="Val F1")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].set_title("F1 Score (Weighted)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "training_curves.png", dpi=150)
        plt.close()


# =============================================================================
# Main
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DREAMT Triple-Stream Sleep Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config_triple_stream.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of training runs",
    )
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}
    
    # Multiple runs
    all_results = []
    
    for run in range(1, args.runs + 1):
        if args.runs > 1:
            logger.info(f"\n{'#' * 70}")
            logger.info(f"RUN {run}/{args.runs}")
            logger.info("#" * 70)
        
        # Set random seeds
        seed = config.get("data", {}).get("seed", 42) + run - 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Train
        trainer = TripleStreamTrainer(config, run_id=run if args.runs > 1 else None)
        results = trainer.train()
        all_results.append(results)
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Summary for multiple runs
    if args.runs > 1:
        logger.info("\n" + "=" * 70)
        logger.info(f"SUMMARY ({args.runs} runs)")
        logger.info("=" * 70)
        
        kappas = [r["test_kappa"] for r in all_results]
        accuracies = [r["test_accuracy"] for r in all_results]
        f1s = [r["test_f1_weighted"] for r in all_results]
        
        logger.info(f"Test Kappa: {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")
        logger.info(f"Test Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
        logger.info(f"Test F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        logger.info(f"All kappas: {[f'{k:.4f}' for k in kappas]}")
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

