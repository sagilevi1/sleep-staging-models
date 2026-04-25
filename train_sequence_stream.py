"""
Train Sequence Sleep Staging Model
===================================
Same engineering style as train_triple_stream.py, but operates on
sequences of L windows and predicts a label per window position.

Loss is computed across all L positions per sample (mean reduction).
Validation/Test metrics are computed by flattening predictions over
(B, L) and comparing against true per-window labels — directly
comparable to the per-window kappa from the baseline trainer.
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
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from dreamt_numpy_dataset import STAGE_NAMES, NUM_CLASSES
from dreamt_sequence_dataset import (
    SequenceDatasetConfig,
    get_sequence_dataloaders,
)
from sequence_model import create_sequence_model
from augmentations import AugmentationConfig
from losses import build_loss
from experiment_tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SequenceTrainer:
    def __init__(
        self,
        config: dict,
        run_id: Optional[int] = None,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.config = config
        self.run_id = run_id
        self.tracker = tracker

        device_name = config.get("hardware", {}).get("device", "cuda")
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self._setup_directories()
        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        self.writer = SummaryWriter(self.log_dir) if HAS_TENSORBOARD else None
        if self.writer is None:
            logger.warning("TensorBoard not available")

        self.use_amp = config.get("hardware", {}).get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        self.grad_accum_steps = config.get("hardware", {}).get("gradient_accumulation_steps", 1)

    def _setup_directories(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"sequence_stream_{timestamp}"
        if self.run_id is not None:
            model_name += f"_run{self.run_id}"

        base_dir = Path(self.config.get("output", {}).get("save_dir", "./outputs/sequence_stream"))
        self.output_dir = base_dir / model_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.results_dir = self.output_dir / "results"
        for d in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    # ── Dataloaders ─────────────────────────────────────────────────────────

    def _create_dataloaders(self):
        data_cfg = self.config.get("data", {})
        seq_cfg = self.config.get("sequence", {})
        batch_size = self.config.get("training", {}).get("batch_size", 8)
        num_workers = data_cfg.get("num_workers", 0)
        pin_memory = self.device.type == "cuda"

        preprocessed_dir = data_cfg.get("preprocessed_dir", "").strip()
        if not preprocessed_dir:
            raise ValueError("Sequence trainer requires data.preprocessed_dir to be set.")

        ds_cfg = SequenceDatasetConfig(
            preprocessed_dir=preprocessed_dir,
            sequence_length=int(seq_cfg.get("length", 20)),
            sequence_stride=int(seq_cfg.get("stride", 5)),
            fs=data_cfg.get("fs", 64.0),
            window_sec=data_cfg.get("window_sec", 30.0),
            seed=data_cfg.get("seed", 42),
        )
        aug_cfg = AugmentationConfig.from_config(self.config.get("training", {}))

        train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = \
            get_sequence_dataloaders(
                ds_cfg, batch_size=batch_size, num_workers=num_workers,
                pin_memory=pin_memory, augment_config=aug_cfg,
            )

        self.class_weights = train_ds.get_class_weights().to(self.device)
        logger.info(f"Class weights: {self.class_weights.tolist()}")

        if self.tracker is not None:
            try:
                self.tracker.log_dataset_info(train_ds, val_ds, test_ds)
                self.tracker.log_preprocessing_info({
                    "bvp_filter": "none",
                    "bvp_norm": "per-window-zscore",
                    "acc_norm": "per-window-per-channel-zscore",
                    "ibi_norm": "global-zscore-train-stats",
                    "modalities": ["bvp", "acc", "ibi"],
                    "ibi_n_features": self.config.get("model", {}).get("ibi_n_features", 5),
                    "sequence_length": ds_cfg.sequence_length,
                    "sequence_stride": ds_cfg.sequence_stride,
                })
            except Exception as e:
                logger.warning(f"Tracker.log_dataset_info failed: {e}")

        return train_loader, val_loader, test_loader

    # ── Model ───────────────────────────────────────────────────────────────

    def _create_model(self):
        m = self.config.get("model", {})
        s = self.config.get("sequence", {})
        model = create_sequence_model(
            n_classes=m.get("n_classes", NUM_CLASSES),
            d_model=m.get("d_model", 256),
            n_heads=m.get("n_heads", 8),
            n_fusion_blocks=m.get("n_fusion_blocks", 3),
            dropout=m.get("dropout", 0.4),
            ibi_n_features=m.get("ibi_n_features", 5),
            seq_model=s.get("model", "bilstm"),
            seq_hidden=s.get("hidden_size", 256),
            seq_layers=s.get("num_layers", 1),
            seq_bidirectional=s.get("bidirectional", True),
            seq_dropout=s.get("dropout", None),
        ).to(self.device)
        logger.info(f"Model parameters: {model.get_num_parameters():,}")
        logger.info(f"Trainable parameters: {model.get_num_trainable_parameters():,}")
        return model

    # ── Training step ───────────────────────────────────────────────────────

    def _train_epoch(self, model, loader, optimizer, criterion, epoch):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        log_freq = self.config.get("output", {}).get("log_frequency", 10)
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
        optimizer.zero_grad()

        for batch_idx, (bvp, acc, ibi, labels, _, _) in enumerate(pbar):
            bvp = bvp.to(self.device, non_blocking=True)
            acc = acc.to(self.device, non_blocking=True)
            ibi = ibi.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            B, L = labels.shape

            if self.use_amp:
                with autocast():
                    logits = model(bvp, acc, ibi)            # (B, L, C)
                    loss = criterion(
                        logits.reshape(B * L, -1),
                        labels.reshape(B * L),
                    ) / self.grad_accum_steps
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(bvp, acc, ibi)
                loss = criterion(
                    logits.reshape(B * L, -1),
                    labels.reshape(B * L),
                ) / self.grad_accum_steps
                loss.backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                running_loss += loss.item() * self.grad_accum_steps * B * L
                correct += preds.eq(labels).sum().item()
                total += B * L

            if batch_idx % log_freq == 0:
                pbar.set_postfix({"loss": running_loss / max(total, 1), "acc": correct / max(total, 1)})

        return running_loss / max(total, 1), correct / max(total, 1)

    # ── Validation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, model, loader, criterion):
        model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for bvp, acc, ibi, labels, _, _ in tqdm(loader, desc="Validation"):
            bvp = bvp.to(self.device); acc = acc.to(self.device)
            ibi = ibi.to(self.device); labels = labels.to(self.device)
            B, L = labels.shape

            if self.use_amp:
                with autocast():
                    logits = model(bvp, acc, ibi)
                    loss = criterion(logits.reshape(B * L, -1), labels.reshape(B * L))
            else:
                logits = model(bvp, acc, ibi)
                loss = criterion(logits.reshape(B * L, -1), labels.reshape(B * L))

            running_loss += loss.item() * B * L
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_labels.append(labels.cpu().numpy().reshape(-1))

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        n = len(all_labels)

        return {
            "loss": running_loss / max(n, 1),
            "accuracy": accuracy_score(all_labels, all_preds),
            "kappa": cohen_kappa_score(all_labels, all_preds),
            "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "confusion_matrix": confusion_matrix(
                all_labels, all_preds, labels=list(range(NUM_CLASSES))
            ),
            "all_preds": all_preds,
            "all_labels": all_labels,
        }

    # ── Save / plot ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {k: v for k, v in metrics.items()
                        if k not in ("all_preds", "all_labels", "confusion_matrix")},
            "config": self.config,
        }
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "best_model.pth")
            logger.info(f"Saved best model (kappa={metrics['kappa']:.4f})")
        save_freq = self.config.get("output", {}).get("save_frequency", 5)
        if epoch % save_freq == 0:
            torch.save(ckpt, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

    def _plot_confusion_matrix(self, cm, epoch, prefix=""):
        plt.figure(figsize=(10, 8))
        cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        annot = np.empty_like(cm, dtype=str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
        labels = [STAGE_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)]
        sns.heatmap(cm_norm * 100, annot=annot, fmt="", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix - {prefix}Epoch {epoch}")
        plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
        plt.savefig(self.results_dir / f"confusion_matrix_{prefix}epoch_{epoch}.png", dpi=150)
        plt.close()

    def _plot_training_curves(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train")
        axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val")
        axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(epochs, history["train_acc"], "b-", label="Train")
        axes[0, 1].plot(epochs, history["val_acc"], "r-", label="Val")
        axes[0, 1].set_title("Accuracy"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(epochs, history["val_kappa"], "g-", label="Val Kappa")
        axes[1, 0].set_title("Cohen's Kappa"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(epochs, history["val_f1"], "m-", label="Val F1")
        axes[1, 1].set_title("F1 Weighted"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "training_curves.png", dpi=150)
        plt.close()

    # ── Main training loop ──────────────────────────────────────────────────

    def train(self):
        logger.info("=" * 70)
        logger.info("DREAMT Sequence Sleep Staging Training")
        logger.info("=" * 70)

        train_loader, val_loader, test_loader = self._create_dataloaders()
        model = self._create_model()
        if self.tracker is not None:
            try:
                self.tracker.log_model(model)
            except Exception as e:
                logger.warning(f"Tracker.log_model failed: {e}")

        tcfg = self.config.get("training", {})
        criterion = build_loss(
            loss_type=tcfg.get("loss", "focal"),
            class_weights=self.class_weights,
            gamma=tcfg.get("focal_gamma", 2.0),
            label_smoothing=tcfg.get("label_smoothing", 0.0),
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=tcfg.get("learning_rate", 1e-4),
            weight_decay=tcfg.get("weight_decay", 1e-4),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=tcfg.get("scheduler_factor", 0.5),
            patience=tcfg.get("scheduler_patience", 3),
        )

        num_epochs = tcfg.get("num_epochs", 50)
        patience = tcfg.get("patience", 15)
        best_kappa, best_epoch, patience_counter = 0.0, 0, 0
        history = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            logger.info("\n" + "=" * 50)
            logger.info(f"Epoch {epoch}/{num_epochs}  LR={optimizer.param_groups[0]['lr']:.2e}")

            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion, epoch)
            val_metrics = self._validate(model, val_loader, criterion)
            scheduler.step(val_metrics["kappa"])

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val Kappa: {val_metrics['kappa']:.4f}, Val F1: {val_metrics['f1_weighted']:.4f}")

            if self.writer is not None:
                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
                self.writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("Val/Accuracy", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("Val/Kappa", val_metrics["kappa"], epoch)
                self.writer.add_scalar("Val/F1_Weighted", val_metrics["f1_weighted"], epoch)
                self.writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_kappa"].append(val_metrics["kappa"])
            history["val_f1"].append(val_metrics["f1_weighted"])

            if self.tracker is not None:
                try:
                    self.tracker.log_epoch(
                        epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                        val_metrics=val_metrics, lr=optimizer.param_groups[0]["lr"],
                    )
                except Exception as e:
                    logger.warning(f"Tracker.log_epoch failed: {e}")

            if val_metrics["kappa"] > best_kappa:
                best_kappa = val_metrics["kappa"]
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(model, optimizer, epoch, val_metrics, is_best=True)
                self._plot_confusion_matrix(val_metrics["confusion_matrix"], epoch, prefix="best_")
            else:
                patience_counter += 1

            self._save_checkpoint(model, optimizer, epoch, val_metrics, is_best=False)
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"\nBest validation kappa: {best_kappa:.4f} at epoch {best_epoch}")

        # Test
        best_path = self.checkpoint_dir / "best_model.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded best model from epoch {ckpt['epoch']}")
        test_metrics = self._validate(model, test_loader, criterion)
        logger.info(f"Test Kappa: {test_metrics['kappa']:.4f}")
        logger.info(f"Test F1 (w):  {test_metrics['f1_weighted']:.4f}")

        report = classification_report(
            test_metrics["all_labels"], test_metrics["all_preds"],
            target_names=[STAGE_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)],
            output_dict=True,
        )
        self._plot_confusion_matrix(test_metrics["confusion_matrix"], best_epoch, prefix="test_")

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
        with open(self.results_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        self._plot_training_curves(history)

        if self.tracker is not None:
            try:
                self.tracker.log_test(test_metrics, classification_report=report)
                self.tracker.copy_best_checkpoint(self.checkpoint_dir / "best_model.pth")
                self.tracker.copy_confusion_matrix(
                    self.results_dir / f"confusion_matrix_test_epoch_{best_epoch}.png"
                )
                self.tracker.finalize(best_epoch=best_epoch, best_val_kappa=best_kappa)
            except Exception as e:
                logger.warning(f"Tracker finalize failed: {e}")

        if self.writer is not None:
            self.writer.close()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DREAMT Sequence Sleep Model")
    parser.add_argument("--config", type=str, default="config_sequence.yaml")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run-name", type=str, default="sequence_stream")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--main-changes", type=str, default="")
    parser.add_argument("--no-track", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}; using defaults")
        config = {}

    all_results = []
    for run in range(1, args.runs + 1):
        if args.runs > 1:
            logger.info(f"\n{'#' * 70}\nRUN {run}/{args.runs}\n{'#' * 70}")

        seed = config.get("data", {}).get("seed", 42) + run - 1
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        tracker = None
        if not args.no_track:
            run_name = args.run_name + (f"_run{run}" if args.runs > 1 else "")
            tracker = ExperimentTracker(
                config=config, run_name=run_name, notes=args.notes,
                extra_metadata={
                    "config_path": args.config,
                    "main_changes": args.main_changes,
                    "seed": seed,
                },
            ).start()

        trainer = SequenceTrainer(
            config, run_id=run if args.runs > 1 else None, tracker=tracker,
        )
        results = trainer.train()
        all_results.append(results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if args.runs > 1:
        kappas = [r["test_kappa"] for r in all_results]
        accs = [r["test_accuracy"] for r in all_results]
        f1s = [r["test_f1_weighted"] for r in all_results]
        logger.info("\n" + "=" * 70)
        logger.info(f"SUMMARY ({args.runs} runs)")
        logger.info(f"Test Kappa: {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")
        logger.info(f"Test Acc:   {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        logger.info(f"Test F1:    {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    logger.info("\nTRAINING COMPLETE!")


if __name__ == "__main__":
    main()
