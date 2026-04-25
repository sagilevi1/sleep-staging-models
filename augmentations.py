"""
Signal augmentations for sleep staging.

All augmentations operate on numpy arrays in __getitem__ (cheap, per-sample).
They are gated by an `augment` flag so val/test can be deterministic.

Provided augmentations:
    - Gaussian noise on BVP and ACC (separate sigmas)
    - Time shift (circular roll within window)
    - Modality dropout (randomly zero one of {BVP, ACC, IBI})
    - Amplitude scaling (random per-channel gain)

Wire-up: see DreamtNumpyDataset for example usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AugmentationConfig:
    """All augmentations are off by default. Enable via config."""
    enabled: bool = False

    # Gaussian noise (post-z-score, so sigma is in standard deviations)
    bvp_noise_std: float = 0.0          # e.g. 0.05
    acc_noise_std: float = 0.0          # e.g. 0.05
    ibi_noise_std: float = 0.0          # e.g. 0.0  (IBI is short — be careful)

    # Time shift (fraction of window to roll, both directions)
    time_shift_max_frac: float = 0.0    # e.g. 0.1 → up to ±10% of window

    # Amplitude scaling (multiplicative; 1.0 ± scale_jitter)
    amplitude_jitter: float = 0.0       # e.g. 0.1 → uniform in [0.9, 1.1]

    # Modality dropout: probability of zeroing a single modality
    modality_dropout_prob: float = 0.0  # e.g. 0.1
    # Which modalities are eligible to be dropped (must keep at least one!)
    drop_bvp: bool = True
    drop_acc: bool = True
    drop_ibi: bool = True

    @classmethod
    def from_config(cls, training_cfg: Optional[dict]) -> "AugmentationConfig":
        if not training_cfg:
            return cls()
        aug = training_cfg.get("augmentation", {}) or {}
        # Backward-compat: top-level modality_dropout_prob still works.
        modality_dropout_prob = (
            aug.get("modality_dropout_prob",
                    training_cfg.get("modality_dropout_prob", 0.0))
        )
        return cls(
            enabled=bool(aug.get("enabled", False)),
            bvp_noise_std=float(aug.get("bvp_noise_std", 0.0)),
            acc_noise_std=float(aug.get("acc_noise_std", 0.0)),
            ibi_noise_std=float(aug.get("ibi_noise_std", 0.0)),
            time_shift_max_frac=float(aug.get("time_shift_max_frac", 0.0)),
            amplitude_jitter=float(aug.get("amplitude_jitter", 0.0)),
            modality_dropout_prob=float(modality_dropout_prob),
            drop_bvp=bool(aug.get("drop_bvp", True)),
            drop_acc=bool(aug.get("drop_acc", True)),
            drop_ibi=bool(aug.get("drop_ibi", True)),
        )

    def is_active(self) -> bool:
        return self.enabled and (
            self.bvp_noise_std > 0
            or self.acc_noise_std > 0
            or self.ibi_noise_std > 0
            or self.time_shift_max_frac > 0
            or self.amplitude_jitter > 0
            or self.modality_dropout_prob > 0
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample apply (numpy, called inside Dataset.__getitem__)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return x
    return x + rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(x.dtype)


def _time_shift(x: np.ndarray, max_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Circularly roll along the last axis."""
    if max_frac <= 0:
        return x
    n = x.shape[-1]
    max_shift = int(round(n * max_frac))
    if max_shift <= 0:
        return x
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return x
    return np.roll(x, shift, axis=-1)


def _amplitude_jitter(x: np.ndarray, jitter: float, rng: np.random.Generator) -> np.ndarray:
    if jitter <= 0:
        return x
    # Per-channel scaling (broadcast over time)
    if x.ndim == 1:
        gain = float(rng.uniform(1.0 - jitter, 1.0 + jitter))
        return (x * gain).astype(x.dtype)
    if x.ndim == 2:
        C = x.shape[0]
        gains = rng.uniform(1.0 - jitter, 1.0 + jitter, size=(C, 1)).astype(x.dtype)
        return x * gains
    return x


def apply_window_augmentations(
    bvp: np.ndarray,
    acc: np.ndarray,
    ibi: np.ndarray,
    cfg: AugmentationConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply all configured augmentations to a single window.

    Inputs (numpy):
        bvp : (T,)      already z-scored
        acc : (3, T)    already z-scored per channel
        ibi : (5,)      already z-scored
    Returns the same shapes/dtypes.
    """
    if not cfg.is_active():
        return bvp, acc, ibi

    rng = rng if rng is not None else np.random.default_rng()

    # Time shift (apply consistently to BVP and ACC so they stay aligned)
    if cfg.time_shift_max_frac > 0:
        n = bvp.shape[-1]
        max_shift = int(round(n * cfg.time_shift_max_frac))
        if max_shift > 0:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            if shift != 0:
                bvp = np.roll(bvp, shift, axis=-1)
                acc = np.roll(acc, shift, axis=-1)

    # Amplitude jitter (per-modality, per-channel)
    bvp = _amplitude_jitter(bvp, cfg.amplitude_jitter, rng)
    acc = _amplitude_jitter(acc, cfg.amplitude_jitter, rng)

    # Gaussian noise (modality-specific sigmas)
    bvp = _gaussian_noise(bvp, cfg.bvp_noise_std, rng)
    acc = _gaussian_noise(acc, cfg.acc_noise_std, rng)
    ibi = _gaussian_noise(ibi, cfg.ibi_noise_std, rng)

    # Modality dropout (zero one modality at most so we never lose all)
    if cfg.modality_dropout_prob > 0 and rng.random() < cfg.modality_dropout_prob:
        candidates = []
        if cfg.drop_bvp: candidates.append("bvp")
        if cfg.drop_acc: candidates.append("acc")
        if cfg.drop_ibi: candidates.append("ibi")
        if candidates:
            choice = rng.choice(candidates)
            if choice == "bvp":
                bvp = np.zeros_like(bvp)
            elif choice == "acc":
                acc = np.zeros_like(acc)
            elif choice == "ibi":
                ibi = np.zeros_like(ibi)

    return bvp, acc, ibi
