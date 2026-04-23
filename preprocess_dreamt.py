"""
DREAMT Dataset Preprocessing Script
=====================================
One-time script: reads all subject CSVs, extracts 30-second windows,
saves compact numpy arrays for zero-overhead training.

After this runs, training loads NO CSVs and uses NO pandas at all.

Usage (local):
    python preprocess_dreamt.py \
        --data_dir "C:/path/to/data_64Hz" \
        --output_dir "./dreamt_processed"

Usage (in Colab, if CSVs are on Drive):
    !python preprocess_dreamt.py \
        --data_dir "/content/drive/MyDrive/DREAMT/data_64Hz" \
        --output_dir "/content/drive/MyDrive/DREAMT/processed"

Output layout:
    dreamt_processed/
        bvp.npy              (N, 1920)    float32  PPG windows
        acc.npy              (N, 3, 1920) float32  ACC windows (X,Y,Z)
        ibi.npy              (N, 5)       float32  HRV features
        labels.npy           (N,)         int8     stage labels 0-5
        subjects.npy         (N,)         <U10     subject ID per window
        split_boundaries.json            train/val/test cut-points
        ibi_stats.json                   mean/std from train split only
        metadata.json                    full config for reproducibility

Windows are stored in split order: [train | val | test].
The DataLoader only needs the cut-point indices to select the right slice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (must stay in sync with dreamt_numpy_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

STAGE_MAP = {
    "P": 0, "PRE": 0, "PREPARATION": 0,
    "W": 1, "WAKE": 1, "AWAKE": 1,
    "N1": 2, "NREM1": 2,
    "N2": 3, "NREM2": 3,
    "N3": 4, "NREM3": 4, "SWS": 4,
    "R": 5, "REM": 5,
}

COLUMN_CANDIDATES = {
    "bvp":   ["BVP", "PPG", "bvp", "ppg"],
    "acc_x": ["ACC_X", "AccX", "acc_x"],
    "acc_y": ["ACC_Y", "AccY", "acc_y"],
    "acc_z": ["ACC_Z", "AccZ", "acc_z"],
    "ibi":   ["IBI", "ibi"],
    "stage": ["Sleep_Stage", "SleepStage", "stage", "Stage"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    existing = set(df.columns)
    for logical, candidates in COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in existing:
                rename_map[cand] = logical
                break
    return df.rename(columns=rename_map)


def map_stage(val) -> int:
    if pd.isna(val):
        return -1
    s = str(val).strip().upper()
    if s in STAGE_MAP:
        return STAGE_MAP[s]
    try:
        v = int(float(s))
        if 0 <= v <= 5:
            return v
    except (ValueError, TypeError):
        pass
    return -1


def compute_ibi_features(ibi_values: np.ndarray) -> np.ndarray:
    """
    5 HRV features from a 64Hz IBI signal window.

    In DREAMT, the IBI column is stored at 64Hz: each inter-beat interval
    value is repeated for every sample until the next heartbeat occurs.
    We must deduplicate consecutive identical values to recover the actual
    beat-to-beat sequence before computing HRV statistics.

    Features: [mean_ibi (s), std_ibi (s), rmssd (s), hr_mean (BPM), n_beats]
    """
    ibi_clean = ibi_values[~np.isnan(ibi_values)]
    if len(ibi_clean) < 2:
        return np.zeros(5, dtype=np.float32)

    # Deduplicate consecutive identical values → recover actual beat sequence
    change_mask = np.concatenate([[True], np.diff(ibi_clean) != 0])
    ibi_beats   = ibi_clean[change_mask]          # one value per heartbeat

    if len(ibi_beats) < 2:
        return np.zeros(5, dtype=np.float32)

    mean_ibi = float(np.mean(ibi_beats))
    std_ibi  = float(np.std(ibi_beats))
    rmssd    = float(np.sqrt(np.mean(np.diff(ibi_beats) ** 2)))
    hr_mean  = (60.0 / mean_ibi) if mean_ibi > 0 else 0.0   # IBI in seconds
    n_beats  = float(len(ibi_beats))
    return np.array([mean_ibi, std_ibi, rmssd, hr_mean, n_beats], dtype=np.float32)


def infer_subject_id(path: Path) -> str:
    return path.stem.split("_")[0]


def discover_files(data_dir: Path) -> List[Path]:
    files = sorted(
        f for f in data_dir.glob("*.csv")
        if "processed" not in f.stem.lower()
    )
    logger.info(f"Found {len(files)} subject CSV files in {data_dir}")
    return files


def subject_split(
    all_files: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Reproduce the exact same subject-level split as dreamt_triple_dataset.py."""
    subject_ids = sorted(set(infer_subject_id(f) for f in all_files))
    file_map = {infer_subject_id(f): f for f in all_files}

    val_test_ratio = 1.0 - train_ratio
    train_ids, val_test_ids = train_test_split(
        subject_ids, test_size=val_test_ratio, random_state=seed
    )
    val_ratio_adj = val_ratio / val_test_ratio
    val_ids, test_ids = train_test_split(
        val_test_ids, test_size=(1.0 - val_ratio_adj), random_state=seed
    )

    logger.info(
        f"Subject split: train={len(train_ids)}, "
        f"val={len(val_ids)}, test={len(test_ids)}"
    )

    return (
        [file_map[s] for s in train_ids  if s in file_map],
        [file_map[s] for s in val_ids    if s in file_map],
        [file_map[s] for s in test_ids   if s in file_map],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject window extraction
# ─────────────────────────────────────────────────────────────────────────────

def process_subject(
    file_path: Path,
    window_samples: int,
    min_windows: int = 10,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    Load one CSV and extract all valid 30-second windows.

    Returns
    -------
    (bvp, acc, ibi, labels, subject_ids)  or  None if subject is skipped.

    bvp    : (N, window_samples)    float32
    acc    : (N, 3, window_samples) float32
    ibi    : (N, 5)                 float32
    labels : (N,)                   int8
    """
    df = pd.read_csv(file_path, low_memory=False)
    df = canonicalize_columns(df)

    required = ["bvp", "acc_x", "acc_y", "acc_z", "stage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"{file_path.name}: missing columns {missing} — skipped")
        return None

    # Convert to numeric
    for col in ["bvp", "acc_x", "acc_y", "acc_z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ibi" not in df.columns:
        df["ibi"] = np.nan
    else:
        df["ibi"] = pd.to_numeric(df["ibi"], errors="coerce")

    df["stage_id"] = df["stage"].apply(map_stage)

    n_windows = len(df) // window_samples
    if n_windows < min_windows:
        logger.warning(
            f"{file_path.name}: only {n_windows} windows (min={min_windows}) — skipped"
        )
        return None

    subject_id = infer_subject_id(file_path)

    # Extract raw arrays once (avoids repeated DataFrame slicing)
    bvp_raw   = df["bvp"].to_numpy(dtype=np.float32)
    acc_x_raw = df["acc_x"].to_numpy(dtype=np.float32)
    acc_y_raw = df["acc_y"].to_numpy(dtype=np.float32)
    acc_z_raw = df["acc_z"].to_numpy(dtype=np.float32)
    ibi_raw   = df["ibi"].to_numpy(dtype=np.float32)
    stage_raw = df["stage_id"].to_numpy(dtype=np.int16)

    out_bvp, out_acc, out_ibi, out_labels = [], [], [], []

    for win in range(n_windows):
        t0 = win * window_samples
        t1 = t0 + window_samples

        stages = stage_raw[t0:t1]
        valid  = stages[stages >= 0]
        if len(valid) == 0:
            continue

        # Majority-vote label
        unique, counts = np.unique(valid, return_counts=True)
        label = int(unique[np.argmax(counts)])

        bvp_win = np.nan_to_num(bvp_raw[t0:t1], nan=0.0)
        acc_win = np.stack([
            np.nan_to_num(acc_x_raw[t0:t1], nan=0.0),
            np.nan_to_num(acc_y_raw[t0:t1], nan=0.0),
            np.nan_to_num(acc_z_raw[t0:t1], nan=0.0),
        ], axis=0)
        ibi_win = compute_ibi_features(ibi_raw[t0:t1])

        out_bvp.append(bvp_win)
        out_acc.append(acc_win)
        out_ibi.append(ibi_win)
        out_labels.append(label)

    if not out_bvp:
        logger.warning(f"{file_path.name}: no valid windows — skipped")
        return None

    n = len(out_bvp)
    return (
        np.stack(out_bvp).astype(np.float32),    # (N, W)
        np.stack(out_acc).astype(np.float32),    # (N, 3, W)
        np.stack(out_ibi).astype(np.float32),    # (N, 5)
        np.array(out_labels, dtype=np.int8),     # (N,)
        [subject_id] * n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    t_start = time.time()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_samples = int(args.fs * args.window_sec)
    logger.info(f"Window: {args.window_sec}s × {args.fs}Hz = {window_samples} samples")

    # ── 1. Subject-level split ──────────────────────────────────────────────
    all_files = discover_files(data_dir)
    if not all_files:
        logger.error("No CSV files found. Check --data_dir.")
        sys.exit(1)

    train_files, val_files, test_files = subject_split(
        all_files, args.train_ratio, args.val_ratio, args.seed
    )

    # ── 2. Process subjects in split order: train → val → test ─────────────
    # Windows are stored contiguously: [train | val | test]
    # The DataLoader selects the right slice via split_boundaries.json

    all_bvp, all_acc, all_ibi, all_labels, all_sids = [], [], [], [], []
    split_boundaries = {}
    current_idx = 0

    for split_name, files in [
        ("train", train_files),
        ("val",   val_files),
        ("test",  test_files),
    ]:
        split_start = current_idx
        n_skipped   = 0

        for file_path in tqdm(files, desc=f"Extracting [{split_name:5s}]", unit="subj"):
            result = process_subject(file_path, window_samples, args.min_windows)
            if result is None:
                n_skipped += 1
                continue
            bvp, acc, ibi, labels, sids = result
            all_bvp.append(bvp)
            all_acc.append(acc)
            all_ibi.append(ibi)
            all_labels.append(labels)
            all_sids.extend(sids)
            current_idx += len(bvp)

        split_end = current_idx
        split_boundaries[split_name] = {"start": split_start, "end": split_end}
        n_windows = split_end - split_start
        logger.info(
            f"[{split_name:5s}] {len(files) - n_skipped} subjects, "
            f"{n_windows:,} windows  ({n_skipped} subjects skipped)"
        )

    if not all_bvp:
        logger.error("No windows extracted. Check data directory and CSV format.")
        sys.exit(1)

    # ── 3. Concatenate ──────────────────────────────────────────────────────
    logger.info("Concatenating arrays ...")
    bvp_all    = np.concatenate(all_bvp,    axis=0)
    acc_all    = np.concatenate(all_acc,    axis=0)
    ibi_all    = np.concatenate(all_ibi,    axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    sids_all   = np.array(all_sids, dtype="<U10")

    n_total = len(labels_all)
    logger.info(
        f"Total: {n_total:,} windows | "
        f"BVP {bvp_all.nbytes/1e9:.2f} GB | "
        f"ACC {acc_all.nbytes/1e9:.2f} GB"
    )

    # Validate no label corruption
    assert bvp_all.shape == (n_total, window_samples),    f"BVP shape mismatch: {bvp_all.shape}"
    assert acc_all.shape == (n_total, 3, window_samples), f"ACC shape mismatch: {acc_all.shape}"
    assert ibi_all.shape == (n_total, 5),                 f"IBI shape mismatch: {ibi_all.shape}"
    assert labels_all.shape == (n_total,),                f"Labels shape mismatch: {labels_all.shape}"
    assert np.all((labels_all >= 0) & (labels_all <= 5)), "Labels out of range [0,5]"

    # ── 4. Save arrays ──────────────────────────────────────────────────────
    logger.info(f"Saving to {output_dir} ...")

    np.save(output_dir / "bvp.npy",      bvp_all)
    logger.info(f"  bvp.npy      {bvp_all.shape}  {bvp_all.nbytes/1e9:.2f} GB")

    np.save(output_dir / "acc.npy",      acc_all)
    logger.info(f"  acc.npy      {acc_all.shape}  {acc_all.nbytes/1e9:.2f} GB")

    np.save(output_dir / "ibi.npy",      ibi_all)
    logger.info(f"  ibi.npy      {ibi_all.shape}")

    np.save(output_dir / "labels.npy",   labels_all)
    logger.info(f"  labels.npy   {labels_all.shape}")

    np.save(output_dir / "subjects.npy", sids_all)
    logger.info(f"  subjects.npy {sids_all.shape}")

    # ── 5. Split boundaries ─────────────────────────────────────────────────
    with open(output_dir / "split_boundaries.json", "w") as f:
        json.dump(split_boundaries, f, indent=2)
    logger.info(f"  split_boundaries.json  {split_boundaries}")

    # ── 6. IBI normalization stats (train split only) ───────────────────────
    t_start_idx = split_boundaries["train"]["start"]
    t_end_idx   = split_boundaries["train"]["end"]
    train_ibi   = ibi_all[t_start_idx:t_end_idx]
    ibi_mean    = train_ibi.mean(axis=0).tolist()
    ibi_std     = (train_ibi.std(axis=0) + 1e-8).tolist()

    ibi_stats = {"mean": ibi_mean, "std": ibi_std,
                 "feature_names": ["mean_ibi", "std_ibi", "rmssd", "hr_mean", "n_beats"]}
    with open(output_dir / "ibi_stats.json", "w") as f:
        json.dump(ibi_stats, f, indent=2)
    logger.info(f"  ibi_stats.json  mean={[f'{v:.4f}' for v in ibi_mean]}")

    # ── 7. Class distribution per split ─────────────────────────────────────
    stage_names = {0: "P", 1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}
    class_dist = {}
    for split_name, bounds in split_boundaries.items():
        s, e = bounds["start"], bounds["end"]
        split_labels = labels_all[s:e]
        unique, counts = np.unique(split_labels, return_counts=True)
        class_dist[split_name] = {stage_names.get(int(u), str(u)): int(c)
                                   for u, c in zip(unique, counts)}
        logger.info(f"  [{split_name}] class dist: {class_dist[split_name]}")

    # ── 8. Metadata ─────────────────────────────────────────────────────────
    metadata = {
        "fs": args.fs,
        "window_sec": args.window_sec,
        "window_samples": window_samples,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "seed": args.seed,
        "min_windows": args.min_windows,
        "n_total": n_total,
        "split_boundaries": split_boundaries,
        "class_distribution": class_dist,
        "data_dir": str(data_dir),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t_start
    logger.info(
        f"\nDone in {elapsed/60:.1f} min. "
        f"Output: {output_dir}  "
        f"({sum(f.stat().st_size for f in output_dir.glob('*.npy'))/1e9:.2f} GB)"
    )
    logger.info("Upload the output directory to Google Drive, then set")
    logger.info("  data.preprocessed_dir: /content/drive/MyDrive/DREAMT/processed")
    logger.info("in config_triple_stream.yaml to use it for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-extract DREAMT windows to numpy arrays for efficient Colab training"
    )
    parser.add_argument(
        "--data_dir",
        default=r"C:\Users\SagiLevi\dreamt\dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.1.0\dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-2.1.0\data_64Hz",
        help="Path to folder containing subject CSV files",
    )
    parser.add_argument(
        "--output_dir",
        default="./dreamt_processed",
        help="Where to save the numpy arrays",
    )
    parser.add_argument("--fs",          type=float, default=64.0)
    parser.add_argument("--window_sec",  type=float, default=30.0)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--min_windows", type=int,   default=10)
    args = parser.parse_args()

    main(args)
