# Colab Quickstart — DREAMT Triple-Stream Training

Three steps total: preprocess locally → upload to Drive → open notebook.

---

## Step 1 — Run preprocessing locally (one-time, ~25 min)

This extracts all 105,902 windows from the raw CSVs and saves ~3.2 GB of numpy arrays.
Run from inside the project folder:

```powershell
python preprocess_dreamt.py --output_dir ./dreamt_processed
```

When it finishes you will see output like:

```
[train]  70 subjects, 74185 windows
[val]    15 subjects, 15855 windows
[test]   15 subjects, 15862 windows
Done in 22.3 min. Output: dreamt_processed  (3.24 GB)
```

Contents of `dreamt_processed/`:

| File | Shape | Size |
|------|-------|------|
| `bvp.npy` | (105902, 1920) float32 | ~0.81 GB |
| `acc.npy` | (105902, 3, 1920) float32 | ~2.43 GB |
| `ibi.npy` | (105902, 5) float32 | ~2 MB |
| `labels.npy` | (105902,) int8 | ~0.1 MB |
| `subjects.npy` | (105902,) str | ~1 MB |
| `split_boundaries.json` | — | tiny |
| `ibi_stats.json` | — | tiny |
| `metadata.json` | — | tiny |

---

## Step 2 — Upload to Google Drive

Upload the entire `dreamt_processed/` folder to your Google Drive.
Target location: `My Drive/DREAMT/processed/`

You can do this via the Drive web UI (drag-and-drop the folder) or:

```bash
# If you have rclone configured:
rclone copy ./dreamt_processed gdrive:DREAMT/processed --progress
```

The upload is ~3.2 GB and takes 5–20 minutes depending on your connection.

---

## Step 3 — Open the Colab notebook

1. Open **`DREAMT_Colab_Training.ipynb`** in Google Colab
2. Set runtime to **GPU (T4)**
3. Run cells top to bottom — the notebook handles everything:
   - Mounts Drive
   - Copies processed files to local Colab disk (fast I/O)
   - Writes a Colab-specific config
   - Runs a sanity check (one forward pass)
   - Starts training

---

## Architecture of the new pipeline

```
preprocess_dreamt.py       (run once locally)
        │
        ▼
dreamt_processed/
  bvp.npy  acc.npy  ibi.npy  labels.npy  ...
        │
        ▼
dreamt_numpy_dataset.py    (loaded with mmap_mode='r')
  DreamtNumpyDataset
  - BVP/ACC: memory-mapped (not in Python heap)
  - Per-window z-score normalization (BVP, ACC)
  - Global z-score for IBI (train stats)
        │
        ▼
train_triple_stream.py     (unchanged model, updated _create_dataloaders)
  - auto-detects numpy backend when preprocessed_dir is set
  - falls back to CSV backend when preprocessed_dir is empty
```

---

## Config reference

**Local (CSV backend — current behaviour):**
```yaml
data:
  preprocessed_dir: ""
  data_dir: 'C:\path\to\data_64Hz'
  num_workers: 0
```

**Colab (numpy backend — fast):**
```yaml
data:
  preprocessed_dir: /content/dreamt_processed
  num_workers: 2
training:
  batch_size: 64
hardware:
  device: cuda
```

---

## Expected training speed

| Setup | Batches/epoch | Time/epoch | 50 epochs |
|-------|--------------|------------|-----------|
| CPU, CSV, workers=0 | 4636 (bs=16) | ~16 hours | impractical |
| Colab T4, numpy, bs=64 | 1159 (bs=64) | ~3–5 min | ~3–4 hours |
| Colab A100, numpy, bs=128 | 580 (bs=128) | ~1–2 min | ~1–1.5 hours |

---

## RAM usage (numpy backend)

| Component | RAM |
|-----------|-----|
| Model + optimizer (21M params, AdamW) | ~340 MB |
| BVP array (mmap, OS pages on demand) | ~50 MB active |
| ACC array (mmap, OS pages on demand) | ~150 MB active |
| IBI + labels + subjects (fully loaded) | ~20 MB |
| PyTorch activations (batch=64) | ~200 MB |
| **Total** | **< 1 GB** |

Colab Free (12 GB RAM): ✅ comfortable
Colab Pro (25 GB RAM): ✅ very comfortable

---

## Troubleshooting

**`FileNotFoundError: ... is missing: ['bvp.npy', ...]`**
→ Preprocessing hasn't run or the Drive path is wrong.
→ Check `DRIVE_PROCESSED` path in Cell 4 of the notebook.

**`CUDA out of memory`**
→ Lower `batch_size` to 32 in Cell 5.

**Slow training (>10 min/epoch)**
→ Make sure you copied files from Drive to local disk (`/content/`) in Cell 4.
→ Drive I/O is 5-10× slower than local disk.

**`ModuleNotFoundError: dreamt_numpy_dataset`**
→ Run `%cd /content/sleep-staging-models` (or wherever the code lives) before Cell 6.

**Training crashed after 12 hours**
→ Colab Free has a 12-hour session limit.
→ Checkpoints are saved to Drive every 5 epochs — resume from the last checkpoint.
→ Or use Colab Pro for uninterrupted sessions.
