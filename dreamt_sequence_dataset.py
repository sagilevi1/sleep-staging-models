"""
DREAMT Sequence Dataset
========================
Returns sequences of L consecutive 30-second windows from the SAME subject
(no boundary crossing), so a sequence model (BiLSTM/Transformer) can learn
temporal context across an epoch.

Reads the same npy backend as DreamtNumpyDataset:
    bvp.npy, acc.npy, ibi.npy, labels.npy, subjects.npy,
    split_boundaries.json, ibi_stats.json, metadata.json

Each item:
    bvp_seq      : (L, 1, T)   float32   — z-scored per window
    acc_seq      : (L, 3, T)   float32   — z-scored per window per channel
    ibi_seq      : (L, F)      float32   — globally z-scored
    labels_seq   : (L,)        int64
    valid_mask   : (L,)        bool      — always all-True (we never cross subjects)
    subject_id   : str
    window_idx   : (L,)        int       — global window indices (for debugging)

Sequence enumeration:
    For each subject (within the split), generate sequences of length L
    starting at indices [0, stride, 2*stride, ...] such that start + L <= n_subj.
    Tail windows that don't fit are dropped (typical for L=20, stride=5).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from augmentations import AugmentationConfig, apply_window_augmentations
from dreamt_numpy_dataset import NUM_CLASSES, STAGE_NAMES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SequenceDatasetConfig:
    preprocessed_dir: str = ""
    sequence_length: int = 20
    sequence_stride: int = 5
    fs: float = 64.0
    window_sec: float = 30.0
    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DreamtSequenceDataset(Dataset):
    """Sequence-of-windows dataset, subject-aware (no boundary crossing)."""

    def __init__(
        self,
        preprocessed_dir: str,
        split: str = "train",
        sequence_length: int = 20,
        sequence_stride: int = 5,
        normalize: bool = True,
        verbose: bool = True,
        augment_config: Optional[AugmentationConfig] = None,
    ):
        self.split = split
        self.sequence_length = int(sequence_length)
        self.sequence_stride = int(sequence_stride)
        self.normalize = normalize
        self.verbose = verbose
        self.augment_config = augment_config or AugmentationConfig()
        self._rng = np.random.default_rng()

        base = Path(preprocessed_dir)
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

        with open(base / "split_boundaries.json") as f:
            bounds = json.load(f)
        with open(base / "ibi_stats.json") as f:
            ibi_stats = json.load(f)
        with open(base / "metadata.json") as f:
            meta = json.load(f)

        if split not in bounds:
            raise ValueError(f"Unknown split '{split}'. Got {list(bounds)}")

        self.start_idx = int(bounds[split]["start"])
        self.end_idx = int(bounds[split]["end"])
        self.window_samples = int(meta["window_samples"])

        self.ibi_mean = np.array(ibi_stats["mean"], dtype=np.float32)
        self.ibi_std = np.array(ibi_stats["std"], dtype=np.float32)

        self._bvp = np.load(str(base / "bvp.npy"), mmap_mode="r")
        self._acc = np.load(str(base / "acc.npy"), mmap_mode="r")
        self._ibi = np.load(str(base / "ibi.npy"))
        self._labels = np.load(str(base / "labels.npy"))
        self._sids = np.load(str(base / "subjects.npy"))

        # Build sequences: per-subject windows, then sliding windows of length L.
        self._sequences: List[Tuple[str, np.ndarray]] = []  # (subject, indices array)
        self._build_sequences()

        if self.verbose:
            n_subj = len(set(s for s, _ in self._sequences))
            logger.info(
                f"[{split.upper()} SEQ] {len(self._sequences):,} sequences  "
                f"(L={self.sequence_length}, stride={self.sequence_stride})  "
                f"subjects={n_subj}"
            )

    # ── Sequence enumeration ────────────────────────────────────────────────

    def _build_sequences(self) -> None:
        # Group consecutive same-subject indices in the split.
        sids_split = self._sids[self.start_idx:self.end_idx]
        if len(sids_split) == 0:
            return

        # Subjects appear in contiguous blocks (preprocess_dreamt.py guarantees
        # this; subjects.npy is built per-subject, then concatenated).
        # Find run boundaries:
        boundaries = [0]
        for i in range(1, len(sids_split)):
            if sids_split[i] != sids_split[i - 1]:
                boundaries.append(i)
        boundaries.append(len(sids_split))

        L = self.sequence_length
        stride = self.sequence_stride

        for b in range(len(boundaries) - 1):
            seg_start_local = boundaries[b]
            seg_end_local = boundaries[b + 1]
            seg_len = seg_end_local - seg_start_local
            if seg_len < L:
                continue
            sid = str(sids_split[seg_start_local])
            for s in range(0, seg_len - L + 1, stride):
                local_indices = np.arange(s, s + L)
                global_indices = self.start_idx + seg_start_local + local_indices
                self._sequences.append((sid, global_indices.astype(np.int64)))

    # ── PyTorch API ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        subject_id, gidx = self._sequences[idx]
        L = self.sequence_length
        T = self.window_samples

        bvp_seq = np.empty((L, 1, T), dtype=np.float32)
        acc_seq = np.empty((L, 3, T), dtype=np.float32)
        ibi_seq = np.empty((L, self.ibi_mean.shape[0]), dtype=np.float32)
        labels_seq = np.empty((L,), dtype=np.int64)

        for i, g in enumerate(gidx):
            bvp = self._bvp[g].copy().astype(np.float32)        # (T,)
            acc = self._acc[g].copy().astype(np.float32)        # (3, T)
            ibi = self._ibi[g].copy().astype(np.float32)        # (F,)
            label = int(self._labels[g])

            if self.normalize:
                bvp = (bvp - bvp.mean()) / (bvp.std() + 1e-8)
                for ch in range(3):
                    acc[ch] = (acc[ch] - acc[ch].mean()) / (acc[ch].std() + 1e-8)
                ibi = (ibi - self.ibi_mean) / self.ibi_std

            if self.augment_config.is_active():
                bvp, acc, ibi = apply_window_augmentations(
                    bvp, acc, ibi, self.augment_config, rng=self._rng
                )

            bvp_seq[i, 0] = bvp
            acc_seq[i] = acc
            ibi_seq[i] = ibi
            labels_seq[i] = label

        return (
            torch.from_numpy(bvp_seq),
            torch.from_numpy(acc_seq),
            torch.from_numpy(ibi_seq),
            torch.from_numpy(labels_seq),
            subject_id,
            torch.from_numpy(gidx),
        )

    # ── Utilities ───────────────────────────────────────────────────────────

    def get_subjects(self) -> List[str]:
        return list(set(s for s, _ in self._sequences))

    def get_class_distribution(self) -> Dict[int, int]:
        """Class distribution over the LABELS contained in the sequences.
        Each window appears once per sequence it belongs to (so windows in the
        overlap are double-counted — fine for class-weight estimation)."""
        counts: Dict[int, int] = {}
        for _, gidx in self._sequences:
            for g in gidx:
                lbl = int(self._labels[g])
                counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        counts = self.get_class_distribution()
        total = sum(counts.values()) or 1
        weights = [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)]
        return torch.FloatTensor(weights)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_sequence_dataloaders(
    config: SequenceDatasetConfig,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    augment_config: Optional[AugmentationConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader,
           DreamtSequenceDataset, DreamtSequenceDataset, DreamtSequenceDataset]:
    if not config.preprocessed_dir:
        raise ValueError("Set data.preprocessed_dir to use the sequence dataset.")

    logger.info("=" * 60)
    logger.info("Creating DREAMT Sequence DataLoaders")
    logger.info(
        f"  preprocessed_dir = {config.preprocessed_dir}\n"
        f"  L = {config.sequence_length}, stride = {config.sequence_stride}"
    )
    logger.info("=" * 60)

    train_ds = DreamtSequenceDataset(
        preprocessed_dir=config.preprocessed_dir,
        split="train",
        sequence_length=config.sequence_length,
        sequence_stride=config.sequence_stride,
        augment_config=augment_config,
    )
    val_ds = DreamtSequenceDataset(
        preprocessed_dir=config.preprocessed_dir,
        split="val",
        sequence_length=config.sequence_length,
        sequence_stride=config.sequence_stride,
    )
    test_ds = DreamtSequenceDataset(
        preprocessed_dir=config.preprocessed_dir,
        split="test",
        sequence_length=config.sequence_length,
        sequence_stride=config.sequence_stride,
    )

    train_subj = set(train_ds.get_subjects())
    val_subj = set(val_ds.get_subjects())
    test_subj = set(test_ds.get_subjects())
    assert train_subj.isdisjoint(val_subj), "Train/Val subjects overlap!"
    assert train_subj.isdisjoint(test_subj), "Train/Test subjects overlap!"
    assert val_subj.isdisjoint(test_subj), "Val/Test subjects overlap!"
    logger.info("[OK] Sequence dataset: subject-level split verified")

    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        dist = ds.get_class_distribution()
        dist_str = ", ".join(f"{STAGE_NAMES.get(k, k)}:{v}" for k, v in sorted(dist.items()))
        logger.info(f"[{name} SEQ] Window-class distribution: {dist_str}")

    def _collate(batch):
        bvps  = torch.stack([b[0] for b in batch], dim=0)
        accs  = torch.stack([b[1] for b in batch], dim=0)
        ibis  = torch.stack([b[2] for b in batch], dim=0)
        lbls  = torch.stack([b[3] for b in batch], dim=0)
        sids  = [b[4] for b in batch]
        widx  = torch.stack([b[5] for b in batch], dim=0)
        return bvps, accs, ibis, lbls, sids, widx

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=(num_workers > 0), collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), collate_fn=_collate,
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
