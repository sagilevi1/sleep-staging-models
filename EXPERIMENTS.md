# Experiments Log

Each row corresponds to one run under `outputs/experiments/<run_id>/`.

The tracker (`experiment_tracker.py`) appends a row automatically on every
training run that uses `train_triple_stream.py` or `train_sequence_stream.py`
with experiment tracking enabled (the default).

| Run ID | Commit | Config | Main changes | Best Val Kappa | Test Kappa | Notes |
|--------|--------|--------|--------------|----------------|------------|-------|
| baseline_pre_tracking | 722edda | config_triple_stream.yaml | Triple-stream (PPG+ACC+IBI), CE+class weights, dropout=0.2, no aug, scheduler patience=5, 50 epochs | ~0.40 (estimated) | 0.33 | Retroactive entry. Predates the tracker — exact metrics live in `outputs/triple_stream/triple_stream_<ts>/results/test_results.json` on the training machine. Severe overfitting motivated the tracker + sequence work. |

---

## Baseline traceability

The pre-tracker baseline corresponds to commit **`722edda`** (`increase epochs to 50`) on `main`.
Reproducing it:

```bash
git checkout 722edda
python train_triple_stream.py --config config_triple_stream.yaml
```

⚠️ The pre-tracker `config_triple_stream.yaml` had `dropout: 0.2`,
`weight_decay: 1.0e-5`, no `augmentation`, no `loss: focal`. The current
config has been updated with the new defaults (dropout 0.4, focal loss, etc.) —
`git checkout 722edda -- config_triple_stream.yaml` to get the original.

The on-disk training artifacts (`outputs/triple_stream/triple_stream_<timestamp>/`)
are not committed (excluded by `.gitignore`); copy `test_results.json` and
`config.json` from there into `outputs/experiments/baseline_pre_tracking/`
for permanent archival.

## How rows are generated

The tracker appends to this file from `experiment_tracker.py::_append_experiments_md`.
Every run produces:

```
outputs/experiments/<YYYY-MM-DD>_<NNN>_<run_name>/
  config.yaml
  metadata.json
  metrics.csv
  summary.md
  classification_report.json
  confusion_matrix.png
  best_model.pt
  git_diff.patch
```

The `<NNN>` counter is monotonic across all dates — never reused.
| 2026-04-25_002_smoke2 | f9cff45+dirty |  | smoke | 0.4000 | 0.5000 |  |
