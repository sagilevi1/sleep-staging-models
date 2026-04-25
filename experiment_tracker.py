"""
Experiment Tracker for Sleep Staging Models
============================================

A reproducibility-first tracker that captures, for every training run, enough
information to fully reconstruct what produced a given result.

Captured artifacts (under outputs/experiments/<date>_<NNN>_<name>/):
    - config.yaml              : exact config snapshot
    - metadata.json            : run id, timestamp, git, env, dataset, splits,
                                 model summary, hyperparameters, notes
    - metrics.csv              : per-epoch training curves
    - summary.md               : human-readable overview (best epoch, final test)
    - confusion_matrix.png     : test-set confusion matrix (saved by trainer)
    - classification_report.json
    - best_model.pt            : best checkpoint (copied from trainer)
    - git_diff.patch           : working-tree diff at run start

Plus an `EXPERIMENTS.md` table at the project root, appended to on every run.

Usage:
    tracker = ExperimentTracker(
        config=config,
        run_name="triple_stream_baseline",
        notes="Baseline before sequence modeling",
    )
    tracker.start()
    tracker.log_dataset_info(train_ds, val_ds, test_ds)
    tracker.log_model(model)

    for epoch in range(...):
        tracker.log_epoch(epoch, train_loss, train_acc, val_metrics)

    tracker.log_test(test_metrics, classification_report_dict)
    tracker.finalize(best_epoch=best_epoch, best_val_kappa=best_kappa)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_git(args: List[str], cwd: Path) -> Optional[str]:
    """Run a git command, returning stripped stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (FileNotFoundError, OSError):
        return None


def _capture_git_state(repo_root: Path) -> Dict[str, Any]:
    """Capture branch, commit hash, dirty flag, and working-tree diff."""
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root) or "unknown"
    commit = _run_git(["rev-parse", "HEAD"], repo_root) or "unknown"
    short = _run_git(["rev-parse", "--short", "HEAD"], repo_root) or "unknown"
    status = _run_git(["status", "--porcelain"], repo_root) or ""
    dirty = bool(status)
    diff = _run_git(["diff", "HEAD"], repo_root) or ""
    return {
        "branch": branch,
        "commit": commit,
        "commit_short": short,
        "dirty": dirty,
        "status": status,
        "diff": diff,
    }


def _capture_environment() -> Dict[str, Any]:
    """Capture python version, torch, cuda, GPU, and pip freeze."""
    env: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["cudnn_version"] = torch.backends.cudnn.version()
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            try:
                props = torch.cuda.get_device_properties(0)
                env["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                pass
    except ImportError:
        pass

    # pip freeze (best effort)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        if result.returncode == 0:
            env["pip_freeze"] = result.stdout.splitlines()
    except Exception:
        pass

    return env


def _slugify(name: str) -> str:
    """Make a filesystem-safe slug."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_") or "run"


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackerPaths:
    repo_root: Path
    experiments_root: Path     # outputs/experiments
    run_dir: Path              # outputs/experiments/<date>_<NNN>_<name>
    config_path: Path
    metadata_path: Path
    metrics_csv_path: Path
    summary_path: Path
    classification_report_path: Path
    best_model_path: Path
    confusion_matrix_path: Path
    git_diff_path: Path


class ExperimentTracker:
    """Local, reproducible experiment tracker."""

    def __init__(
        self,
        config: Dict[str, Any],
        run_name: str = "run",
        notes: str = "",
        repo_root: Optional[Path] = None,
        experiments_root: Optional[Path] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.run_name = run_name
        self.notes = notes
        self.extra_metadata = dict(extra_metadata or {})

        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent
        self.experiments_root = (
            Path(experiments_root)
            if experiments_root
            else self.repo_root / "outputs" / "experiments"
        )

        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.run_id: Optional[str] = None
        self.paths: Optional[TrackerPaths] = None

        self._epoch_rows: List[Dict[str, Any]] = []
        self._dataset_info: Dict[str, Any] = {}
        self._model_info: Dict[str, Any] = {}
        self._test_info: Dict[str, Any] = {}
        self._git_info: Dict[str, Any] = {}
        self._env_info: Dict[str, Any] = {}
        self._best_epoch: Optional[int] = None
        self._best_val_kappa: Optional[float] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> "ExperimentTracker":
        """Allocate the run directory and snapshot config + git + env."""
        self.start_time = datetime.now()
        self.run_id = self._allocate_run_dir()
        self._git_info = _capture_git_state(self.repo_root)
        self._env_info = _capture_environment()

        # Snapshot config as YAML
        with open(self.paths.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

        # Save git diff (may be empty)
        with open(self.paths.git_diff_path, "w", encoding="utf-8") as f:
            f.write(self._git_info.get("diff", "") or "")

        # Initialize CSV header (epoch metrics)
        with open(self.paths.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_acc",
                "val_loss", "val_acc",
                "val_kappa", "val_f1_weighted", "val_f1_macro",
                "lr",
            ])

        logger.info("=" * 70)
        logger.info(f"[ExperimentTracker] Run ID:  {self.run_id}")
        logger.info(f"[ExperimentTracker] Dir:     {self.paths.run_dir}")
        logger.info(f"[ExperimentTracker] Branch:  {self._git_info.get('branch')}")
        logger.info(f"[ExperimentTracker] Commit:  {self._git_info.get('commit_short')}"
                    f"{' (DIRTY)' if self._git_info.get('dirty') else ''}")
        logger.info("=" * 70)
        return self

    def log_dataset_info(self, train_ds, val_ds, test_ds) -> None:
        """Capture split sizes, subjects, and class distributions."""
        def _summary(ds):
            try:
                subjects = sorted(ds.get_subjects())
            except Exception:
                subjects = []
            try:
                dist = ds.get_class_distribution()
            except Exception:
                dist = {}
            return {
                "n_windows": len(ds),
                "n_subjects": len(subjects),
                "subjects": subjects,
                "class_distribution": {str(k): int(v) for k, v in dist.items()},
            }

        data_cfg = self.config.get("data", {})
        self._dataset_info = {
            "preprocessed_dir": data_cfg.get("preprocessed_dir", ""),
            "data_dir": data_cfg.get("data_dir", ""),
            "fs": data_cfg.get("fs"),
            "window_sec": data_cfg.get("window_sec"),
            "train_ratio": data_cfg.get("train_ratio"),
            "val_ratio": data_cfg.get("val_ratio"),
            "test_ratio": data_cfg.get("test_ratio"),
            "seed": data_cfg.get("seed"),
            "splits": {
                "train": _summary(train_ds),
                "val": _summary(val_ds),
                "test": _summary(test_ds),
            },
        }

    def log_preprocessing_info(self, info: Dict[str, Any]) -> None:
        """Caller-supplied preprocessing version info (filtered/normalized/etc)."""
        self._dataset_info["preprocessing"] = dict(info)

    def log_model(self, model) -> None:
        """Capture parameter counts and a string summary of the model."""
        try:
            n_total = sum(p.numel() for p in model.parameters())
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception:
            n_total = n_trainable = -1
        self._model_info = {
            "class_name": type(model).__name__,
            "n_parameters_total": n_total,
            "n_parameters_trainable": n_trainable,
            "summary": str(model),
        }

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_metrics: Dict[str, Any],
        lr: Optional[float] = None,
    ) -> None:
        """Append one epoch of metrics to metrics.csv and in-memory list."""
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_metrics.get("loss", float("nan"))),
            "val_acc": float(val_metrics.get("accuracy", float("nan"))),
            "val_kappa": float(val_metrics.get("kappa", float("nan"))),
            "val_f1_weighted": float(val_metrics.get("f1_weighted", float("nan"))),
            "val_f1_macro": float(val_metrics.get("f1_macro", float("nan"))),
            "lr": float(lr) if lr is not None else float("nan"),
        }
        self._epoch_rows.append(row)
        with open(self.paths.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                row["epoch"], row["train_loss"], row["train_acc"],
                row["val_loss"], row["val_acc"],
                row["val_kappa"], row["val_f1_weighted"], row["val_f1_macro"],
                row["lr"],
            ])

    def log_test(
        self,
        test_metrics: Dict[str, Any],
        classification_report: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture final test metrics and per-class report."""
        cm = test_metrics.get("confusion_matrix")
        if hasattr(cm, "tolist"):
            cm = cm.tolist()
        self._test_info = {
            "loss": float(test_metrics.get("loss", float("nan"))),
            "accuracy": float(test_metrics.get("accuracy", float("nan"))),
            "kappa": float(test_metrics.get("kappa", float("nan"))),
            "f1_weighted": float(test_metrics.get("f1_weighted", float("nan"))),
            "f1_macro": float(test_metrics.get("f1_macro", float("nan"))),
            "confusion_matrix": cm,
        }
        if classification_report is not None:
            with open(self.paths.classification_report_path, "w", encoding="utf-8") as f:
                json.dump(classification_report, f, indent=2)

    def copy_best_checkpoint(self, src_path: Path) -> None:
        """Copy the best checkpoint into the run directory."""
        src = Path(src_path)
        if src.exists():
            try:
                shutil.copy2(src, self.paths.best_model_path)
            except Exception as e:
                logger.warning(f"Could not copy best checkpoint: {e}")

    def copy_confusion_matrix(self, src_path: Path) -> None:
        """Copy the test confusion matrix PNG into the run directory."""
        src = Path(src_path)
        if src.exists():
            try:
                shutil.copy2(src, self.paths.confusion_matrix_path)
            except Exception as e:
                logger.warning(f"Could not copy confusion matrix: {e}")

    def finalize(
        self,
        best_epoch: Optional[int] = None,
        best_val_kappa: Optional[float] = None,
    ) -> None:
        """Write metadata.json, summary.md, and append to EXPERIMENTS.md."""
        self.end_time = datetime.now()
        self._best_epoch = best_epoch
        self._best_val_kappa = best_val_kappa

        metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "notes": self.notes,
            "started_at": self.start_time.isoformat() if self.start_time else None,
            "ended_at": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if (self.start_time and self.end_time) else None
            ),
            "git": self._git_info,
            "environment": self._env_info,
            "config": self.config,
            "dataset": self._dataset_info,
            "model": self._model_info,
            "best_epoch": self._best_epoch,
            "best_val_kappa": self._best_val_kappa,
            "test": self._test_info,
            "extra": self.extra_metadata,
        }
        with open(self.paths.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        self._write_summary_md(metadata)
        self._append_experiments_md(metadata)

        logger.info("=" * 70)
        logger.info(f"[ExperimentTracker] Run finalized: {self.paths.run_dir}")
        logger.info(f"[ExperimentTracker] Best Val Kappa: {self._best_val_kappa}")
        if self._test_info:
            logger.info(f"[ExperimentTracker] Test Kappa:     {self._test_info.get('kappa')}")
        logger.info("=" * 70)

    # ── Internal ────────────────────────────────────────────────────────────

    def _allocate_run_dir(self) -> str:
        """Allocate <date>_<NNN>_<name> with a monotonically-incrementing NNN."""
        self.experiments_root.mkdir(parents=True, exist_ok=True)
        date_str = self.start_time.strftime("%Y-%m-%d")
        slug = _slugify(self.run_name)

        # Find max NNN across all dates so run numbers are globally unique.
        existing = [p.name for p in self.experiments_root.iterdir() if p.is_dir()]
        max_n = 0
        pat = re.compile(r"^\d{4}-\d{2}-\d{2}_(\d{3})_")
        for n in existing:
            m = pat.match(n)
            if m:
                max_n = max(max_n, int(m.group(1)))
        run_n = max_n + 1
        run_id = f"{date_str}_{run_n:03d}_{slug}"

        run_dir = self.experiments_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        self.paths = TrackerPaths(
            repo_root=self.repo_root,
            experiments_root=self.experiments_root,
            run_dir=run_dir,
            config_path=run_dir / "config.yaml",
            metadata_path=run_dir / "metadata.json",
            metrics_csv_path=run_dir / "metrics.csv",
            summary_path=run_dir / "summary.md",
            classification_report_path=run_dir / "classification_report.json",
            best_model_path=run_dir / "best_model.pt",
            confusion_matrix_path=run_dir / "confusion_matrix.png",
            git_diff_path=run_dir / "git_diff.patch",
        )
        return run_id

    def _write_summary_md(self, metadata: Dict[str, Any]) -> None:
        lines: List[str] = []
        lines.append(f"# Run {self.run_id}")
        lines.append("")
        lines.append(f"- **Name:** {self.run_name}")
        lines.append(f"- **Started:** {metadata['started_at']}")
        lines.append(f"- **Ended:** {metadata['ended_at']}")
        lines.append(f"- **Duration (s):** {metadata['duration_seconds']}")
        lines.append("")
        lines.append("## Git")
        lines.append(f"- Branch: `{self._git_info.get('branch')}`")
        lines.append(f"- Commit: `{self._git_info.get('commit')}`")
        lines.append(f"- Dirty:  `{self._git_info.get('dirty')}`")
        lines.append("")
        lines.append("## Environment")
        env = self._env_info
        lines.append(f"- Python:  `{env.get('python_version')}`")
        lines.append(f"- Torch:   `{env.get('torch_version')}`")
        lines.append(f"- CUDA:    `{env.get('cuda_version')}`")
        lines.append(f"- GPU:     `{env.get('gpu_name')}`")
        lines.append("")
        lines.append("## Dataset")
        ds = self._dataset_info
        for split_name, info in (ds.get("splits") or {}).items():
            lines.append(
                f"- **{split_name}**: {info['n_windows']:,} windows, "
                f"{info['n_subjects']} subjects, "
                f"classes={info['class_distribution']}"
            )
        if ds.get("preprocessing"):
            lines.append(f"- Preprocessing: `{ds['preprocessing']}`")
        lines.append("")
        lines.append("## Model")
        m = self._model_info
        lines.append(f"- Class: `{m.get('class_name')}`")
        n_total = m.get("n_parameters_total")
        n_trainable = m.get("n_parameters_trainable")
        n_total_s = f"{n_total:,}" if isinstance(n_total, int) and n_total >= 0 else "—"
        n_train_s = f"{n_trainable:,}" if isinstance(n_trainable, int) and n_trainable >= 0 else "—"
        lines.append(f"- Parameters: {n_total_s} total, {n_train_s} trainable")
        lines.append("")
        lines.append("## Training")
        lines.append(f"- Best epoch: `{self._best_epoch}`")
        lines.append(f"- Best val kappa: `{self._best_val_kappa}`")
        lines.append("")
        lines.append("## Test")
        if self._test_info:
            t = self._test_info
            def _fmt(v):
                return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            lines.append(f"- Loss: `{_fmt(t.get('loss'))}`")
            lines.append(f"- Accuracy: `{_fmt(t.get('accuracy'))}`")
            lines.append(f"- Kappa: `{_fmt(t.get('kappa'))}`")
            lines.append(f"- F1 (weighted): `{_fmt(t.get('f1_weighted'))}`")
            lines.append(f"- F1 (macro): `{_fmt(t.get('f1_macro'))}`")
        lines.append("")
        lines.append("## Notes")
        lines.append(self.notes or "_(none)_")
        lines.append("")

        with open(self.paths.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _append_experiments_md(self, metadata: Dict[str, Any]) -> None:
        """Append a one-row entry to EXPERIMENTS.md at repo root."""
        path = self.repo_root / "EXPERIMENTS.md"
        header = (
            "| Run ID | Commit | Config | Main changes | Best Val Kappa | Test Kappa | Notes |\n"
            "|--------|--------|--------|--------------|----------------|------------|-------|\n"
        )
        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Experiments Log\n\n")
                f.write(
                    "Each row corresponds to one run under "
                    "`outputs/experiments/<run_id>/`.\n\n"
                )
                f.write(header)

        # Detect config name from the current run (best-effort)
        cfg_name = self.extra_metadata.get("config_path", "")
        if cfg_name:
            cfg_name = Path(cfg_name).name

        commit = self._git_info.get("commit_short", "")
        if self._git_info.get("dirty"):
            commit += "+dirty"

        bvk = self._best_val_kappa
        tk = (self._test_info or {}).get("kappa")
        bvk_s = f"{bvk:.4f}" if isinstance(bvk, (int, float)) else "—"
        tk_s = f"{tk:.4f}" if isinstance(tk, (int, float)) else "—"

        notes = (self.notes or "").replace("|", "/").replace("\n", " ")[:120]
        main_changes = self.extra_metadata.get("main_changes", "").replace("|", "/")[:120]

        row = f"| {self.run_id} | {commit} | {cfg_name} | {main_changes} | {bvk_s} | {tk_s} | {notes} |\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(row)
