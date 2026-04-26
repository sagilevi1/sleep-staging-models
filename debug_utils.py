"""
NaN/Inf debugging instrumentation for sleep-staging trainers.

Design goals:
    - Zero overhead when debug_mode is False (every check is gated by
      `DebugConfig.enabled`).
    - Logs ONLY on anomaly (NaN, Inf, or |x| > extreme_threshold).
    - Reports the FIRST module that produces NaN/Inf in a batch.
    - Provides intermediate-tensor capture so the trainer can check
      encoder vs BiLSTM vs final logits separately without modifying the
      model code.
    - Includes (epoch, batch_idx) in every log line.

Wire-up (see train_sequence_stream.py):

    self.debug = DebugConfig.from_config(cfg["training"])

    if self.debug.enabled:
        self._nan_hook = FirstNaNHook(model).attach()
        self._capture = IntermediateCapture()
        self._capture.watch("encoder", model.embedder)
        self._capture.watch("bilstm",  model.seq_head)

    # per-batch:
    if self.debug.enabled:
        self._nan_hook.reset(epoch, batch_idx)
        self._capture.clear()

    # ... forward ...
    if self.debug.enabled:
        check_tensor("encoder_out", self._capture["encoder"], epoch, batch_idx, self.debug)
        check_tensor("bilstm_out",  self._capture["bilstm"],  epoch, batch_idx, self.debug)
        check_tensor("logits",      logits,                   epoch, batch_idx, self.debug)

    # loss finite check + skip
    if not loss_is_finite(loss, epoch, batch_idx, self.debug):
        optimizer.zero_grad(); continue

    # ... backward ...
    if self.debug.enabled:
        gs = grad_stats(model, epoch, batch_idx, self.debug)
        if gs["has_nan"]:
            optimizer.zero_grad(); continue
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DebugConfig:
    enabled: bool = False
    extreme_abs_threshold: float = 1.0e4   # warn if |x| exceeds this
    grad_norm_warn_threshold: float = 1.0e3  # warn if grad norm exceeds this

    @classmethod
    def from_config(cls, training_cfg: Optional[dict]) -> "DebugConfig":
        if not training_cfg:
            return cls()
        return cls(
            enabled=bool(training_cfg.get("debug_mode", False)),
            extreme_abs_threshold=float(
                training_cfg.get("debug_extreme_threshold", 1.0e4)
            ),
            grad_norm_warn_threshold=float(
                training_cfg.get("debug_grad_norm_warn", 1.0e3)
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tensor checks
# ─────────────────────────────────────────────────────────────────────────────

def _has_nan_inf(t: torch.Tensor) -> Tuple[bool, bool]:
    if not torch.is_tensor(t) or t.numel() == 0:
        return False, False
    return (
        bool(torch.isnan(t).any().item()),
        bool(torch.isinf(t).any().item()),
    )


def log_input_stats(
    epoch: int,
    bvp: torch.Tensor,
    acc: torch.Tensor,
    ibi: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """One-shot input validation. Always logs (called only at start of training)."""
    def _stats(name: str, t: torch.Tensor) -> str:
        x = t.detach().float()
        nan, inf = _has_nan_inf(x)
        return (
            f"{name:6s} shape={tuple(x.shape)}  "
            f"min={x.min().item():+.3e}  max={x.max().item():+.3e}  "
            f"mean={x.mean().item():+.3e}  std={x.std().item():.3e}  "
            f"NaN={nan}  Inf={inf}"
        )
    logger.info(f"[DEBUG epoch={epoch}] === Input batch stats (one-shot) ===")
    logger.info(f"[DEBUG]   {_stats('bvp', bvp)}")
    logger.info(f"[DEBUG]   {_stats('acc', acc)}")
    logger.info(f"[DEBUG]   {_stats('ibi', ibi)}")
    try:
        uniq = torch.unique(labels).tolist()
        logger.info(f"[DEBUG]   labels unique={uniq}  shape={tuple(labels.shape)}")
    except Exception:
        pass


def check_tensor(
    name: str,
    t: Optional[torch.Tensor],
    epoch: int,
    batch_idx: int,
    cfg: DebugConfig,
) -> bool:
    """Return True if tensor is healthy. Logs only on anomaly."""
    if not cfg.enabled or t is None or not torch.is_tensor(t):
        return True
    nan, inf = _has_nan_inf(t)
    if nan or inf:
        logger.error(
            f"[DEBUG epoch={epoch} batch={batch_idx}] "
            f"!! {name}: NaN={nan} Inf={inf} shape={tuple(t.shape)}"
        )
        return False
    abs_max = t.detach().abs().max().item()
    if abs_max > cfg.extreme_abs_threshold:
        logger.warning(
            f"[DEBUG epoch={epoch} batch={batch_idx}] "
            f"!  {name}: extreme |x|={abs_max:.3e} "
            f"(>{cfg.extreme_abs_threshold:.0e}) shape={tuple(t.shape)}"
        )
        return False
    return True


def loss_is_finite(
    loss: torch.Tensor,
    epoch: int,
    batch_idx: int,
    cfg: DebugConfig,
) -> bool:
    """Return True if loss is finite. Logs (always) when not — even with debug off."""
    if not torch.is_tensor(loss):
        return True
    val = loss.detach().float().item()
    if math.isfinite(val):
        return True
    # Always log non-finite loss (this is a real failure, not just debug noise)
    logger.error(
        f"[DEBUG epoch={epoch} batch={batch_idx}] "
        f"!! Non-finite loss = {val} (NaN or Inf). Skipping batch."
    )
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Gradient stats
# ─────────────────────────────────────────────────────────────────────────────

def grad_stats(
    model: nn.Module,
    epoch: int,
    batch_idx: int,
    cfg: DebugConfig,
) -> Dict:
    """
    Compute total grad norm + detect NaN/Inf gradients.
    Logs ONLY on anomaly (NaN/Inf or norm > grad_norm_warn_threshold).
    Returns dict with keys: norm, has_nan, n_nan_params.

    Call AFTER scaler.unscale_ in AMP mode, AFTER backward in fp32 mode.
    """
    if not cfg.enabled:
        return {"norm": float("nan"), "has_nan": False, "n_nan_params": 0}

    total_sq = 0.0
    has_nan = False
    nan_params: List[str] = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        # Detect NaN/Inf without materializing the full tensor twice
        if torch.isnan(g).any().item() or torch.isinf(g).any().item():
            has_nan = True
            nan_params.append(name)
            continue
        # Use local reduction to keep memory low
        total_sq += float(g.detach().pow(2).sum().item())

    norm = math.sqrt(total_sq) if not has_nan else float("nan")

    if has_nan:
        logger.error(
            f"[DEBUG epoch={epoch} batch={batch_idx}] "
            f"NaN/Inf gradients in {len(nan_params)} params. "
            f"first 5: {nan_params[:5]}"
        )
    elif norm > cfg.grad_norm_warn_threshold:
        logger.warning(
            f"[DEBUG epoch={epoch} batch={batch_idx}] "
            f"Large gradient norm: {norm:.3e} "
            f"(>{cfg.grad_norm_warn_threshold:.0e})"
        )

    return {"norm": norm, "has_nan": has_nan, "n_nan_params": len(nan_params)}


# ─────────────────────────────────────────────────────────────────────────────
# First-NaN forward hook
# ─────────────────────────────────────────────────────────────────────────────

class FirstNaNHook:
    """
    Forward hooks on every leaf module. When a forward pass produces a NaN/Inf,
    the FIRST such module's name is logged once per batch. Subsequent NaNs in
    the same batch are suppressed (downstream modules will all be NaN once
    upstream is — only the source matters).

    Call .reset(epoch, batch_idx) at the start of each batch.
    Call .remove() during teardown if you need to detach.
    """

    def __init__(self, model: nn.Module):
        self._model = model
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._reported = False
        self._epoch = 0
        self._batch_idx = 0

    def attach(self) -> "FirstNaNHook":
        for name, module in self._model.named_modules():
            # Skip the root module and non-leaf modules (children handle them)
            if name == "" or any(True for _ in module.children()):
                continue
            self._handles.append(
                module.register_forward_hook(self._make_hook(name))
            )
        return self

    def reset(self, epoch: int, batch_idx: int) -> None:
        self._epoch = epoch
        self._batch_idx = batch_idx
        self._reported = False

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            if self._reported:
                return
            tensors: List[torch.Tensor] = []
            if torch.is_tensor(out):
                tensors.append(out)
            elif isinstance(out, (tuple, list)):
                for o in out:
                    if torch.is_tensor(o):
                        tensors.append(o)
            for t in tensors:
                nan, inf = _has_nan_inf(t)
                if nan or inf:
                    logger.error(
                        f"[DEBUG epoch={self._epoch} batch={self._batch_idx}] "
                        f"FIRST NaN/Inf module: '{name}' "
                        f"({type(module).__name__})  NaN={nan}  Inf={inf}  "
                        f"out_shape={tuple(t.shape)}"
                    )
                    self._reported = True
                    return
        return hook


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate output capture (for encoder vs LSTM vs logits checks)
# ─────────────────────────────────────────────────────────────────────────────

class IntermediateCapture:
    """
    Cache the most recent forward output of selected sub-modules so the
    trainer can run check_tensor() on them after model(...) returns.

    Captures only the FIRST tensor of the module's output (handles tuple
    outputs from LSTM by taking out[0]). Stored tensors are detached views.
    """

    def __init__(self):
        self.outputs: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def watch(self, name: str, module: nn.Module) -> None:
        def hook(_m, _inp, out):
            if torch.is_tensor(out):
                self.outputs[name] = out.detach()
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                self.outputs[name] = out[0].detach()
        self._handles.append(module.register_forward_hook(hook))

    def __getitem__(self, name: str) -> Optional[torch.Tensor]:
        return self.outputs.get(name)

    def clear(self) -> None:
        self.outputs.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
