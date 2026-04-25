# Training Guide

This repo contains two trainers that share the same DREAMT npy backend:

| Trainer | Architecture | Config |
|---|---|---|
| `train_triple_stream.py` | Per-window: PPG + ACC + IBI cross-attention | `config_triple_stream.yaml` |
| `train_sequence_stream.py` | Sequence of L windows → BiLSTM → per-window labels | `config_sequence.yaml` |

Both trainers automatically write a tracked experiment under
`outputs/experiments/<date>_<NNN>_<name>/` and append a row to
[`EXPERIMENTS.md`](./EXPERIMENTS.md).

---

## 0. Prepare the data once

```bash
python preprocess_dreamt.py \
    --data_dir "C:/Users/<you>/dreamt/.../data_64Hz" \
    --output_dir ./dreamt_processed
```

This produces `bvp.npy`, `acc.npy`, `ibi.npy`, `labels.npy`, `subjects.npy`,
plus `split_boundaries.json`, `ibi_stats.json`, `metadata.json`.

The npy backend is **memory-mapped** — only the windows in the current batch
are pulled into RAM. Colab Free (12 GB) is fine for 100 subjects.

---

## 1. Triple-stream baseline (per-window)

```bash
python train_triple_stream.py \
    --config config_triple_stream.yaml \
    --run-name triple_v2_focal_aug \
    --notes "dropout=0.4, focal gamma=2, augs on, scheduler patience=3, wd=1e-4" \
    --main-changes "Focal+augs+stronger reg vs baseline"
```

Important config knobs (in `config_triple_stream.yaml`):

- `model.dropout`           : 0.4 (raised from 0.2)
- `training.weight_decay`   : 1e-4 (raised from 1e-5)
- `training.loss`           : `"focal"` (or `"ce"`)
- `training.focal_gamma`    : 2.0
- `training.label_smoothing`: 0.0 (try 0.05–0.1 if the model is over-confident)
- `training.scheduler_patience`: 3
- `training.augmentation.*` : Gaussian noise / time shift / amplitude jitter / modality dropout

Disable the tracker with `--no-track` for ad-hoc debug runs.

---

## 2. Sequence model (BiLSTM over L windows)

```bash
python train_sequence_stream.py \
    --config config_sequence.yaml \
    --run-name sequence_v1_bilstm \
    --notes "L=20, stride=5, BiLSTM(256, 1 layer, bi)" \
    --main-changes "Add temporal context with sequence model"
```

Sequence-specific config (`config_sequence.yaml`):

- `sequence.length`        : 20 windows × 30 s = 10 min of context
- `sequence.stride`        : 5 windows between sequences
- `sequence.model`         : `"bilstm"` or `"transformer"`
- `sequence.hidden_size`   : 256
- `sequence.num_layers`    : 1 (start simple)
- `sequence.bidirectional` : true
- `training.batch_size`    : 4 (each sample is L×window data; smaller batch needed)

Sequences never cross subject boundaries. Loss is computed across all L
positions per sample (mean reduction). Validation/Test metrics flatten over
(B, L) so they are directly comparable to the per-window kappa from the
triple-stream baseline.

---

## 3. Where do results go?

Per-run artifacts (always):

```
outputs/experiments/<YYYY-MM-DD>_<NNN>_<run_name>/
  config.yaml                     # exact snapshot
  metadata.json                   # git, env, dataset, model, hyperparameters, notes
  metrics.csv                     # per-epoch curves
  summary.md                      # human-readable overview
  classification_report.json      # final per-class P/R/F1
  confusion_matrix.png            # test-set confusion matrix
  best_model.pt                   # copied from the trainer's checkpoint dir
  git_diff.patch                  # working-tree diff at run start
```

Training-side artifacts (the trainer's `output.save_dir`):

```
<save_dir>/<arch>_<timestamp>/
  checkpoints/best_model.pth, checkpoint_epoch_*.pth
  logs/<TensorBoard>
  results/training_curves.png, confusion_matrix_*.png, test_results.json
```

The tracker copies the best checkpoint and test confusion matrix into the
`outputs/experiments/...` run dir so the experiment is fully self-contained.

---

## 4. Reproducing a run

```bash
# 1. Look up the row in EXPERIMENTS.md → get the run_id and commit
RUN_ID=2026-04-22_001_triple_v2_focal_aug
COMMIT=$(jq -r '.git.commit' outputs/experiments/$RUN_ID/metadata.json)

# 2. Restore code state
git checkout $COMMIT
git apply outputs/experiments/$RUN_ID/git_diff.patch  # optional, if dirty

# 3. Re-run with the captured config
python train_triple_stream.py --config outputs/experiments/$RUN_ID/config.yaml
```

The seed is captured in `metadata.extra.seed`.

---

## 5. Colab launch (one-cell)

```python
%cd /content/sleep-staging-models
!git pull
!python train_triple_stream.py \
    --config config_triple_stream.yaml \
    --run-name triple_v2_focal_aug \
    --main-changes "Focal+augs+stronger reg" \
    --notes "first run with experiment tracker"
```

For the sequence model:

```python
!python train_sequence_stream.py \
    --config config_sequence.yaml \
    --run-name sequence_v1_bilstm \
    --main-changes "Add BiLSTM over 20 windows"
```

---

## 6. Adding a new experiment

1. Edit the config (e.g. drop `weight_decay` to `1e-3`, raise `dropout` to `0.5`).
2. Run with a descriptive `--run-name` and `--main-changes`.
3. The tracker appends an `EXPERIMENTS.md` row automatically.
4. Compare across runs by sorting/filtering `EXPERIMENTS.md` or by reading
   the `metadata.json` files.

Never commit anything in `outputs/` — they're git-ignored and can be huge.
The `EXPERIMENTS.md` table at the repo root is the persistent record.
