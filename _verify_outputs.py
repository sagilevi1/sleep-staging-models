"""Quick sanity check for preprocessed numpy files."""
import json
import sys
import numpy as np
from pathlib import Path

base = Path("dreamt_processed")

print("=== File shapes ===")
for fname in ["bvp.npy", "acc.npy", "ibi.npy", "labels.npy", "subjects.npy"]:
    arr = np.load(str(base / fname), mmap_mode="r")
    print(f"  {fname:20s} shape={str(arr.shape):30s} dtype={arr.dtype}")

with open(base / "split_boundaries.json") as f:
    bounds = json.load(f)
print("\n=== Split boundaries ===")
for split, b in bounds.items():
    n = b["end"] - b["start"]
    print(f"  {split:5s}: [{b['start']:6d} – {b['end']:6d}]  = {n:,} windows")

with open(base / "ibi_stats.json") as f:
    stats = json.load(f)
print("\n=== IBI stats (train only) ===")
for name, mean, std in zip(stats["feature_names"], stats["mean"], stats["std"]):
    print(f"  {name:12s}: mean={mean:.4f}, std={std:.6f}")

labels = np.load(str(base / "labels.npy"))
stage_names = {0: "P", 1: "W", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}
unique, counts = np.unique(labels, return_counts=True)
print("\n=== Label distribution ===")
for u, c in zip(unique, counts):
    print(f"  {stage_names[int(u)]}: {c:,}")
print(f"  Total: {labels.shape[0]:,}")

assert int(labels.min()) >= 0 and int(labels.max()) <= 5, "Labels out of range!"
print("\n  Label range check: PASS")

# Check alignment: bvp and acc same count
bvp = np.load(str(base / "bvp.npy"), mmap_mode="r")
acc = np.load(str(base / "acc.npy"), mmap_mode="r")
assert bvp.shape[0] == acc.shape[0] == labels.shape[0], "Array length mismatch!"
assert bvp.shape[1] == 1920, f"Unexpected window samples: {bvp.shape[1]}"
assert acc.shape[1] == 3 and acc.shape[2] == 1920, f"Unexpected ACC shape: {acc.shape}"
print("  Shape alignment check: PASS")

# Quick dataset test
print("\n=== Dataset __getitem__ test ===")
sys.path.insert(0, ".")
from dreamt_numpy_dataset import DreamtNumpyDataset
ds = DreamtNumpyDataset("dreamt_processed", split="train", verbose=False)
bvp_t, acc_t, ibi_t, label = ds[0]
print(f"  BVP : {bvp_t.shape}  min={bvp_t.min():.3f} max={bvp_t.max():.3f}  (z-scored)")
print(f"  ACC : {acc_t.shape}  min={acc_t.min():.3f} max={acc_t.max():.3f}  (z-scored)")
print(f"  IBI : {ibi_t.shape}  {ibi_t.numpy().round(3)}")
print(f"  Label: {label} ({stage_names.get(label, '?')})")
assert bvp_t.shape == (1, 1920)
assert acc_t.shape == (3, 1920)
assert ibi_t.shape == (5,)
print("  Shape check: PASS")

print("\nAll checks passed. Ready for training.")
