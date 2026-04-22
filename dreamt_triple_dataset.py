"""
DREAMT Triple-Stream Dataset for Sleep Staging

Loads PPG (BVP), ACC (3-channel), and IBI features from DREAMT 64Hz CSV files.
Implements strict subject-level train/val/test split to prevent data leakage.

Dataset: DREAMT wearable data
- ~100 subjects (CSV files)
- Columns: TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, IBI, Sleep_Stage
- Sampling rate: 64 Hz
- Labels: P, W, N1, N2, N3, REM (6 classes)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Stage mapping (6 classes, NO merging)
STAGE_MAP = {
    "P": 0, "PRE": 0, "PREPARATION": 0,
    "W": 1, "WAKE": 1, "AWAKE": 1,
    "N1": 2, "NREM1": 2,
    "N2": 3, "NREM2": 3,
    "N3": 4, "NREM3": 4, "SWS": 4,
    "R": 5, "REM": 5,
}

STAGE_NAMES = {0: "P", 1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}
NUM_CLASSES = 6

# Column name candidates for flexibility
COLUMN_CANDIDATES = {
    "timestamp": ["TIMESTAMP", "Time", "time", "ts"],
    "bvp": ["BVP", "PPG", "bvp", "ppg"],
    "acc_x": ["ACC_X", "AccX", "acc_x"],
    "acc_y": ["ACC_Y", "AccY", "acc_y"],
    "acc_z": ["ACC_Z", "AccZ", "acc_z"],
    "ibi": ["IBI", "ibi"],
    "stage": ["Sleep_Stage", "SleepStage", "stage", "Stage"],
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for DREAMT dataset."""
    data_dir: str
    fs: float = 64.0                    # Sampling frequency (Hz)
    window_sec: float = 30.0            # Window duration (seconds)
    train_ratio: float = 0.70           # Train split ratio
    val_ratio: float = 0.15             # Validation split ratio
    test_ratio: float = 0.15            # Test split ratio
    seed: int = 42                      # Random seed for reproducibility
    min_windows_per_subject: int = 10   # Minimum windows to include subject


# =============================================================================
# Utility Functions
# =============================================================================

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names to canonical names."""
    rename_map = {}
    existing = {c: c for c in df.columns}
    
    for logical, candidates in COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in existing:
                rename_map[existing[cand]] = logical
                break
    
    return df.rename(columns=rename_map)


def map_stage_to_id(stage_value) -> int:
    """Map stage string/value to numeric ID (0-5)."""
    if pd.isna(stage_value):
        return -1  # Invalid
    
    s = str(stage_value).strip().upper()
    if s in STAGE_MAP:
        return STAGE_MAP[s]
    
    # Try numeric
    try:
        v = int(float(s))
        if 0 <= v <= 5:
            return v
    except (ValueError, TypeError):
        pass
    
    return -1  # Unknown stage


def compute_ibi_features(ibi_values: np.ndarray) -> np.ndarray:
    """
    Compute HRV feature vector from IBI values within a window.
    
    Features (5 total):
    - mean_ibi: Mean inter-beat interval (ms)
    - std_ibi: Standard deviation of IBI
    - rmssd: Root mean square of successive differences
    - hr_mean: Mean heart rate (60000 / mean_ibi)
    - n_beats: Number of valid IBI values
    
    Args:
        ibi_values: Array of IBI values (may contain NaN)
    
    Returns:
        Feature vector of shape (5,)
    """
    # Remove NaN values
    ibi_clean = ibi_values[~np.isnan(ibi_values)]
    
    if len(ibi_clean) < 2:
        # Not enough data - return zeros
        return np.zeros(5, dtype=np.float32)
    
    mean_ibi = np.mean(ibi_clean)
    std_ibi = np.std(ibi_clean)
    
    # RMSSD: Root mean square of successive differences
    if len(ibi_clean) > 1:
        successive_diff = np.diff(ibi_clean)
        rmssd = np.sqrt(np.mean(successive_diff ** 2))
    else:
        rmssd = 0.0
    
    # Heart rate (assuming IBI is in milliseconds)
    # If IBI appears to be in seconds (< 10), convert
    if mean_ibi < 10:
        # IBI is likely in seconds, convert to ms for HR calculation
        hr_mean = 60.0 / mean_ibi if mean_ibi > 0 else 0.0
    else:
        # IBI is in milliseconds
        hr_mean = 60000.0 / mean_ibi if mean_ibi > 0 else 0.0
    
    n_beats = float(len(ibi_clean))
    
    return np.array([mean_ibi, std_ibi, rmssd, hr_mean, n_beats], dtype=np.float32)


def discover_subject_files(data_dir: Path) -> List[Path]:
    """
    Discover all subject CSV files in the data directory.
    Each CSV file = one subject.
    """
    files = []
    for p in data_dir.glob("*.csv"):
        # Skip processed/intermediate files
        if "processed" in str(p).lower():
            continue
        files.append(p)
    
    files = sorted(files)
    logger.info(f"Discovered {len(files)} subject CSV files in {data_dir}")
    return files


def infer_subject_id(file_path: Path) -> str:
    """Extract subject ID from filename (e.g., S002_whole_df.csv -> S002)."""
    name = file_path.stem
    # Take first part before underscore
    parts = name.split("_")
    return parts[0] if parts else name


# =============================================================================
# Dataset Class
# =============================================================================

class DreamtTripleStreamDataset(Dataset):
    """
    DREAMT Triple-Stream Dataset for Sleep Staging.
    
    Returns per 30-second window:
    - bvp: (1, window_samples) - PPG signal
    - acc: (3, window_samples) - Accelerometer (X, Y, Z)
    - ibi_features: (5,) - HRV feature vector [mean_ibi, std_ibi, rmssd, hr_mean, n_beats]
    - label: int (0-5) - Sleep stage
    
    CRITICAL: Subject-level split is enforced. Windows from the same subject
    are NEVER split across train/val/test sets.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        subject_ids: Optional[List[str]] = None,
        transform=None,
        verbose: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            config: Dataset configuration
            split: One of "train", "val", "test"
            subject_ids: Optional explicit list of subject IDs for this split.
                        If None, will perform automatic subject-level split.
            transform: Optional transform to apply to signals
            verbose: Whether to print progress
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.verbose = verbose
        
        self.fs = config.fs
        self.window_sec = config.window_sec
        self.window_samples = int(config.fs * config.window_sec)  # 64 * 30 = 1920
        
        self.data_dir = Path(config.data_dir)
        
        # Storage for windows
        self.windows: List[Dict] = []  # Each: {subject_id, file_path, t0_idx, t1_idx, label}
        
        # Data cache (loaded on demand)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Get subject files
        all_files = discover_subject_files(self.data_dir)
        
        if subject_ids is not None:
            # Use provided subject IDs
            self.subject_files = [f for f in all_files if infer_subject_id(f) in subject_ids]
        else:
            # Perform subject-level split
            self.subject_files = self._split_subjects(all_files, split)
        
        # Build window index
        self._build_window_index()
        
        if self.verbose:
            logger.info(f"[{split.upper()}] {len(self.subject_files)} subjects, {len(self.windows)} windows")
    
    def _split_subjects(self, all_files: List[Path], split: str) -> List[Path]:
        """
        Split subjects into train/val/test sets.
        
        CRITICAL: This is done at the SUBJECT level, not window level.
        """
        # Extract subject IDs
        subject_ids = [infer_subject_id(f) for f in all_files]
        file_map = {infer_subject_id(f): f for f in all_files}
        
        unique_subjects = sorted(set(subject_ids))
        n_subjects = len(unique_subjects)
        
        logger.info(f"Total unique subjects: {n_subjects}")
        
        # First split: train vs (val + test)
        val_test_ratio = self.config.val_ratio + self.config.test_ratio
        train_subjects, val_test_subjects = train_test_split(
            unique_subjects,
            test_size=val_test_ratio,
            random_state=self.config.seed
        )
        
        # Second split: val vs test
        val_ratio_adjusted = self.config.val_ratio / val_test_ratio
        val_subjects, test_subjects = train_test_split(
            val_test_subjects,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.config.seed
        )
        
        logger.info(f"Subject split: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
        
        # Select appropriate split
        if split == "train":
            selected = train_subjects
        elif split == "val":
            selected = val_subjects
        elif split == "test":
            selected = test_subjects
        else:
            raise ValueError(f"Unknown split: {split}")
        
        return [file_map[s] for s in selected if s in file_map]
    
    def _load_subject_data(self, file_path: Path) -> pd.DataFrame:
        """Load and preprocess a single subject's CSV file."""
        subject_id = infer_subject_id(file_path)
        
        if subject_id in self._data_cache:
            return self._data_cache[subject_id]
        
        # Load CSV
        df = pd.read_csv(file_path)
        df = canonicalize_columns(df)
        
        # Verify required columns
        required = ["timestamp", "bvp", "acc_x", "acc_y", "acc_z", "stage"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{file_path.name}: Missing required columns: {missing}")
        
        # Map stages to numeric
        df["stage_id"] = df["stage"].apply(map_stage_to_id)
        
        # Handle IBI column (may not exist or have many NaNs)
        if "ibi" not in df.columns:
            df["ibi"] = np.nan
        
        # Convert numeric columns
        for col in ["timestamp", "bvp", "acc_x", "acc_y", "acc_z", "ibi"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Cache
        self._data_cache[subject_id] = df
        
        return df
    
    def _build_window_index(self):
        """
        Build index of all valid windows from the assigned subjects.
        
        For each subject:
        1. Load data
        2. Segment into non-overlapping 30-second windows
        3. Assign majority label to each window
        """
        self.windows = []
        
        for file_path in self.subject_files:
            subject_id = infer_subject_id(file_path)
            
            try:
                df = self._load_subject_data(file_path)
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
                continue
            
            n_samples = len(df)
            n_windows = n_samples // self.window_samples
            
            if n_windows < self.config.min_windows_per_subject:
                logger.warning(f"{subject_id}: Only {n_windows} windows, skipping (min={self.config.min_windows_per_subject})")
                continue
            
            for win_idx in range(n_windows):
                t0_idx = win_idx * self.window_samples
                t1_idx = t0_idx + self.window_samples
                
                # Get majority label for this window
                window_stages = df["stage_id"].iloc[t0_idx:t1_idx].values
                valid_stages = window_stages[window_stages >= 0]
                
                if len(valid_stages) == 0:
                    continue  # Skip windows with no valid labels
                
                # Majority vote
                unique, counts = np.unique(valid_stages, return_counts=True)
                label = int(unique[np.argmax(counts)])
                
                self.windows.append({
                    "subject_id": subject_id,
                    "file_path": file_path,
                    "t0_idx": t0_idx,
                    "t1_idx": t1_idx,
                    "label": label
                })
        
        # Clear data cache to save memory (will reload on __getitem__)
        self._data_cache.clear()
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a single window.
        
        Returns:
            bvp: (1, window_samples) - PPG signal
            acc: (3, window_samples) - Accelerometer (X, Y, Z)
            ibi_features: (5,) - HRV feature vector
            label: int (0-5) - Sleep stage
        """
        window_info = self.windows[idx]
        
        # Load subject data
        df = self._load_subject_data(window_info["file_path"])
        
        t0 = window_info["t0_idx"]
        t1 = window_info["t1_idx"]
        
        # Extract BVP
        bvp = df["bvp"].iloc[t0:t1].values.astype(np.float32)
        
        # Extract ACC (3 channels)
        acc_x = df["acc_x"].iloc[t0:t1].values.astype(np.float32)
        acc_y = df["acc_y"].iloc[t0:t1].values.astype(np.float32)
        acc_z = df["acc_z"].iloc[t0:t1].values.astype(np.float32)
        acc = np.stack([acc_x, acc_y, acc_z], axis=0)  # (3, window_samples)
        
        # Extract IBI values and compute features
        ibi_values = df["ibi"].iloc[t0:t1].values.astype(np.float32)
        ibi_features = compute_ibi_features(ibi_values)
        
        # Handle NaN in signals
        bvp = np.nan_to_num(bvp, nan=0.0)
        acc = np.nan_to_num(acc, nan=0.0)
        
        # Add channel dimension to BVP
        bvp = bvp[np.newaxis, :]  # (1, window_samples)
        
        # Get label
        label = window_info["label"]
        
        # Apply transform if any
        if self.transform is not None:
            bvp, acc, ibi_features = self.transform(bvp, acc, ibi_features)
        
        # Convert to tensors
        bvp_tensor = torch.from_numpy(bvp)
        acc_tensor = torch.from_numpy(acc)
        ibi_tensor = torch.from_numpy(ibi_features)
        
        return bvp_tensor, acc_tensor, ibi_tensor, label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in this split."""
        counts = {}
        for w in self.windows:
            label = w["label"]
            counts[label] = counts.get(label, 0) + 1
        return counts
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data (inverse frequency)."""
        counts = self.get_class_distribution()
        total = sum(counts.values())
        
        weights = []
        for i in range(NUM_CLASSES):
            count = counts.get(i, 1)
            weight = total / (NUM_CLASSES * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_subjects(self) -> List[str]:
        """Get list of subject IDs in this split."""
        return [infer_subject_id(f) for f in self.subject_files]


# =============================================================================
# DataLoader Factory
# =============================================================================

def get_dataloaders(
    config: DatasetConfig,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, DreamtTripleStreamDataset, DreamtTripleStreamDataset, DreamtTripleStreamDataset]:
    """
    Create train, validation, and test dataloaders with subject-level split.
    
    Args:
        config: Dataset configuration
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    logger.info("=" * 60)
    logger.info("Creating DREAMT Triple-Stream DataLoaders")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Window: {config.window_sec}s @ {config.fs}Hz = {int(config.fs * config.window_sec)} samples")
    logger.info(f"Split ratios: train={config.train_ratio}, val={config.val_ratio}, test={config.test_ratio}")
    logger.info("=" * 60)
    
    # Create datasets with subject-level split
    train_dataset = DreamtTripleStreamDataset(config, split="train")
    val_dataset = DreamtTripleStreamDataset(config, split="val")
    test_dataset = DreamtTripleStreamDataset(config, split="test")
    
    # Verify no subject overlap
    train_subjects = set(train_dataset.get_subjects())
    val_subjects = set(val_dataset.get_subjects())
    test_subjects = set(test_dataset.get_subjects())
    
    assert train_subjects.isdisjoint(val_subjects), "Train/Val subjects overlap!"
    assert train_subjects.isdisjoint(test_subjects), "Train/Test subjects overlap!"
    assert val_subjects.isdisjoint(test_subjects), "Val/Test subjects overlap!"
    
    logger.info("[OK] Subject-level split verified: No overlap between splits")
    
    # Log class distributions
    for name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        dist = dataset.get_class_distribution()
        dist_str = ", ".join([f"{STAGE_NAMES.get(k, k)}:{v}" for k, v in sorted(dist.items())])
        logger.info(f"[{name}] Class distribution: {dist_str}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


# =============================================================================
# Sanity Check / Test
# =============================================================================

def sanity_check(data_dir: str, num_samples: int = 3):
    """
    Run sanity check on the dataset.
    
    Args:
        data_dir: Path to DREAMT data directory
        num_samples: Number of samples to inspect
    """
    print("\n" + "=" * 70)
    print("DREAMT Triple-Stream Dataset Sanity Check")
    print("=" * 70)
    
    config = DatasetConfig(
        data_dir=data_dir,
        fs=64.0,
        window_sec=30.0,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        config, batch_size=4, num_workers=0
    )
    
    print(f"\n--- Dataset Statistics ---")
    print(f"Train: {len(train_ds)} windows from {len(train_ds.get_subjects())} subjects")
    print(f"Val:   {len(val_ds)} windows from {len(val_ds.get_subjects())} subjects")
    print(f"Test:  {len(test_ds)} windows from {len(test_ds.get_subjects())} subjects")
    
    print(f"\n--- Sample Inspection ({num_samples} samples) ---")
    for i in range(min(num_samples, len(train_ds))):
        bvp, acc, ibi_feat, label = train_ds[i]
        print(f"\nSample {i}:")
        print(f"  BVP shape: {bvp.shape}, dtype: {bvp.dtype}")
        print(f"  BVP range: [{bvp.min():.3f}, {bvp.max():.3f}], mean: {bvp.mean():.3f}")
        print(f"  ACC shape: {acc.shape}, dtype: {acc.dtype}")
        print(f"  ACC range: [{acc.min():.3f}, {acc.max():.3f}]")
        print(f"  IBI features: {ibi_feat.numpy()}")
        print(f"  Label: {label} ({STAGE_NAMES.get(label, 'Unknown')})")
    
    print(f"\n--- Batch Test ---")
    batch = next(iter(train_loader))
    bvp_batch, acc_batch, ibi_batch, labels_batch = batch
    print(f"BVP batch: {bvp_batch.shape}")
    print(f"ACC batch: {acc_batch.shape}")
    print(f"IBI batch: {ibi_batch.shape}")
    print(f"Labels batch: {labels_batch.shape}, values: {labels_batch.tolist()}")
    
    print(f"\n--- Class Weights (for imbalanced training) ---")
    weights = train_ds.get_class_weights()
    for i, w in enumerate(weights):
        print(f"  {STAGE_NAMES.get(i, i)}: {w:.4f}")
    
    print("\n" + "=" * 70)
    print("[OK] Sanity check passed!")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # Default data directory
    data_dir = r"C:\Users\SagiLevi\data_64Hz"
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    sanity_check(data_dir)

