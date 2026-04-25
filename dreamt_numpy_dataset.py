"""
DREAMT Numpy Dataset — zero-CSV training backend
==================================================
Drop-in replacement for dreamt_triple_dataset.py that reads from
pre-extracted numpy arrays instead of raw CSV files.

RAM usage during training:
  - BVP and ACC loaded as memory-mapped files (OS pages on demand, ~0 overhead)
  - IBI, labels, subjects: fully in RAM (~20 MB total)
  - Per batch: only the 16-64 windows being processed are in Python heap

This is compatible with Colab Free (12 GB RAM) even for 100 subjects.

Normalization applied in __getitem__ (not stored in npy files):
  - BVP : per-window z-score  (zero-mean, unit-variance within window)
  - ACC : per-window per-channel z-score
  - IBI : global z-score using train-split statistics from ibi_stats.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from augmentations import AugmentationConfig, apply_window_augmentations

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (identical to dreamt_triple_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

STAGE_NAMES  = {0: "P", 1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}
NUM_CLASSES  = 6

# Re-exported so train_triple_stream.py can import them from here
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Mirrors the original DatasetConfig — unused fields are kept for compat."""
    data_dir: str = ""
    preprocessed_dir: str = ""
    fs: float = 64.0
    window_sec: float = 30.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    min_windows_per_subject: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DreamtNumpyDataset(Dataset):
    """
    Numpy-backed DREAMT Dataset for sleep staging.

    Reads from pre-extracted arrays created by preprocess_dreamt.py.
    BVP and ACC are memory-mapped so only accessed pages live in RAM.
    All normalization is applied on-the-fly in __getitem__.

    Returns per window:
        bvp          : (1, 1920)  float32  — z-scored within window
        acc          : (3, 1920)  float32  — z-scored per channel within window
        ibi_features : (5,)       float32  — globally z-scored (train stats)
        label        : int                 — sleep stage 0-5
    """

    def __init__(
        self,
        preprocessed_dir: str,
        split: str = "train",
        normalize: bool = True,
        verbose: bool = True,
        augment_config: Optional[AugmentationConfig] = None,
    ):
        self.split     = split
        self.normalize = normalize
        self.verbose   = verbose
        # Augmentations are only meaningful for training; trainer should pass
        # AugmentationConfig() (disabled) for val/test.
        self.augment_config = augment_config or AugmentationConfig()
        # Per-worker RNG (re-seeded from torch worker_init if multi-worker)
        self._rng = np.random.default_rng()

        base = Path(preprocessed_dir)
        self._validate_dir(base)

        # ── Load metadata ──────────────────────────────────────────────────
        with open(base / "split_boundaries.json") as f:
            bounds = json.load(f)
        with open(base / "ibi_stats.json") as f:
            ibi_stats = json.load(f)
        with open(base / "metadata.json") as f:
            meta = json.load(f)

        if split not in bounds:
            raise ValueError(f"Unknown split '{split}'. Expected one of {list(bounds)}")

        self.start_idx = bounds[split]["start"]
        self.end_idx   = bounds[split]["end"]
        self.n_windows = self.end_idx - self.start_idx

        self.window_samples = meta["window_samples"]   # 1920

        # ── IBI normalization constants ────────────────────────────────────
        self.ibi_mean = np.array(ibi_stats["mean"], dtype=np.float32)
        self.ibi_std  = np.array(ibi_stats["std"],  dtype=np.float32)

        # ── Memory-mapped signal arrays ────────────────────────────────────
        # mmap_mode='r' → OS pages data on demand; file never fully in RAM
        self._bvp = np.load(str(base / "bvp.npy"), mmap_mode="r")
        self._acc = np.load(str(base / "acc.npy"), mmap_mode="r")

        # Small arrays: fully in RAM (~20 MB total for 106k windows)
        self._ibi     = np.load(str(base / "ibi.npy"))
        self._labels  = np.load(str(base / "labels.npy"))
        self._sids    = np.load(str(base / "subjects.npy"))

        if self.verbose:
            logger.info(
                f"[{split.upper()}] {self.n_windows:,} windows  "
                f"(idx {self.start_idx}–{self.end_idx})  "
                f"subjects={len(set(self._sids[self.start_idx:self.end_idx]))}"
            )

    # ── Accessors ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        global_idx = self.start_idx + idx

        # Read from memmap (copies only this window into RAM)
        bvp = self._bvp[global_idx].copy().astype(np.float32)   # (1920,)
        acc = self._acc[global_idx].copy().astype(np.float32)   # (3, 1920)
        ibi = self._ibi[global_idx].copy().astype(np.float32)   # (5,)
        label = int(self._labels[global_idx])

        if self.normalize:
            # Per-window z-score for BVP
            bvp_std = bvp.std() + 1e-8
            bvp = (bvp - bvp.mean()) / bvp_std

            # Per-window per-channel z-score for ACC
            for ch in range(3):
                ch_std = acc[ch].std() + 1e-8
                acc[ch] = (acc[ch] - acc[ch].mean()) / ch_std

            # Global z-score for IBI (train-split statistics)
            ibi = (ibi - self.ibi_mean) / self.ibi_std

        # Augmentations (only active for train if configured)
        if self.augment_config.is_active():
            bvp, acc, ibi = apply_window_augmentations(
                bvp, acc, ibi, self.augment_config, rng=self._rng
            )

        # Add channel dim to BVP:  (1920,) → (1, 1920)
        bvp = bvp[np.newaxis, :]

        return (
            torch.from_numpy(bvp),
            torch.from_numpy(acc),
            torch.from_numpy(ibi),
            label,
        )

    # ── Utilities ──────────────────────────────────────────────────────────

    def get_subjects(self) -> List[str]:
        return list(set(self._sids[self.start_idx:self.end_idx].tolist()))

    def get_class_distribution(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for lbl in self._labels[self.start_idx:self.end_idx]:
            counts[int(lbl)] = counts.get(int(lbl), 0) + 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for CrossEntropyLoss."""
        counts = self.get_class_distribution()
        total  = sum(counts.values())
        weights = [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)]
        return torch.FloatTensor(weights)

    # ── Private ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_dir(base: Path):
        required = [
            "bvp.npy", "acc.npy", "ibi.npy", "labels.npy",
            "subjects.npy", "split_boundaries.json",
            "ibi_stats.json", "metadata.json",
        ]
        missing = [f for f in required if not (base / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Preprocessed directory '{base}' is missing: {missing}\n"
                "Run: python preprocess_dreamt.py --output_dir <dir>"
            )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory  (same signature as dreamt_triple_dataset.get_dataloaders)
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    config: DatasetConfig,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    augment_config: Optional[AugmentationConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader,
           DreamtNumpyDataset, DreamtNumpyDataset, DreamtNumpyDataset]:
    """
    Create train / val / test DataLoaders from pre-extracted numpy arrays.

    Args
    ----
    config        : DatasetConfig with preprocessed_dir set
    batch_size    : batch size
    num_workers   : 0 is recommended for Colab (no subprocess overhead)
    pin_memory    : True only when using CUDA

    Returns
    -------
    train_loader, val_loader, test_loader,
    train_dataset, val_dataset, test_dataset
    """
    preprocessed_dir = config.preprocessed_dir or config.data_dir
    if not preprocessed_dir:
        raise ValueError(
            "Set data.preprocessed_dir in your config to the folder produced by "
            "preprocess_dreamt.py"
        )

    logger.info("=" * 60)
    logger.info("Creating DREAMT Numpy DataLoaders")
    logger.info(f"Preprocessed dir: {preprocessed_dir}")
    logger.info("=" * 60)

    train_ds = DreamtNumpyDataset(
        preprocessed_dir, split="train", augment_config=augment_config,
    )
    val_ds   = DreamtNumpyDataset(preprocessed_dir, split="val")
    test_ds  = DreamtNumpyDataset(preprocessed_dir, split="test")
    if augment_config is not None and augment_config.is_active():
        logger.info(f"[AUG] Train augmentations enabled: {augment_config}")

    # Verify no subject overlap (subjects are stored per-window in subjects.npy)
    train_subjects = set(train_ds.get_subjects())
    val_subjects   = set(val_ds.get_subjects())
    test_subjects  = set(test_ds.get_subjects())

    assert train_subjects.isdisjoint(val_subjects),  "Train/Val subjects overlap!"
    assert train_subjects.isdisjoint(test_subjects), "Train/Test subjects overlap!"
    assert val_subjects.isdisjoint(test_subjects),   "Val/Test subjects overlap!"
    logger.info("[OK] Subject-level split verified: no overlap between splits")

    # Log class distributions
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        dist = ds.get_class_distribution()
        dist_str = ", ".join(
            f"{STAGE_NAMES.get(k, k)}:{v}" for k, v in sorted(dist.items())
        )
        logger.info(f"[{name}] Class distribution: {dist_str}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
