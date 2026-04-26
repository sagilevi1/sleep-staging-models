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
# Compact tensor formatting helpers (used by dump_failing_batch)
# ─────────────────────────────────────────────────────────────────────────────


def tensor_health_line(name: str, t: Optional[torch.Tensor]) -> str:
    """Multi-line-safe single-line health summary for a tensor."""
    if t is None:
        return f"{name}: <None>"
    if not torch.is_tensor(t):
        return f"{name}: <not a tensor: {type(t).__name__}>"
    if t.numel() == 0:
        return f"{name}: shape={tuple(t.shape)} <empty>"
    x = t.detach().float()
    nan_mask = torch.isnan(x)
    inf_mask = torch.isinf(x)
    n_nan = int(nan_mask.sum().item())
    n_inf = int(inf_mask.sum().item())
    finite_mask = ~(nan_mask | inf_mask)
    if finite_mask.any():
        xf = x[finite_mask]
        fmin = xf.min().item()
        fmax = xf.max().item()
        fmean = xf.mean().item()
        fstd = xf.std().item() if xf.numel() > 1 else 0.0
        amax = xf.abs().max().item()
        return (
            f"{name}: shape={tuple(t.shape)} "
            f"min={fmin:+.3e} max={fmax:+.3e} mean={fmean:+.3e} "
            f"std={fstd:.3e} |x|max={amax:.3e} NaN={n_nan} Inf={n_inf}"
        )
    return (
        f"{name}: shape={tuple(t.shape)} <no finite values> "
        f"NaN={n_nan} Inf={n_inf}"
    )


def tensor_summary(t: Optional[torch.Tensor]) -> str:
    """Tight single-line summary for compact in-line use."""
    if t is None:
        return "<None>"
    if not torch.is_tensor(t) or t.numel() == 0:
        return "<empty>"
    x = t.detach().float()
    nan_mask = torch.isnan(x)
    inf_mask = torch.isinf(x)
    finite_mask = ~(nan_mask | inf_mask)
    n_nan = int(nan_mask.sum().item())
    n_inf = int(inf_mask.sum().item())
    if finite_mask.any():
        xf = x[finite_mask]
        return (
            f"min={xf.min().item():+.3e} max={xf.max().item():+.3e} "
            f"mean={xf.mean().item():+.3e} NaN={n_nan} Inf={n_inf}"
        )
    return f"<no finite> NaN={n_nan} Inf={n_inf}"


def label_distribution_line(
    labels: torch.Tensor, num_classes: int
) -> str:
    """Returns 'shape=... unique=[...] counts=[c0,c1,...]'."""
    try:
        flat = labels.detach().reshape(-1)
        uniq = torch.unique(flat).tolist()
        counts = []
        for c in range(num_classes):
            counts.append(int((flat == c).sum().item()))
        n_invalid = int((flat < 0).sum().item())
        return (
            f"shape={tuple(labels.shape)} unique={uniq} "
            f"counts_per_class={counts} invalid(<0)={n_invalid}"
        )
    except Exception as e:
        return f"<label dist failed: {e}>"


def predicted_class_distribution_line(
    logits: torch.Tensor, num_classes: int
) -> str:
    """argmax over logits (with nan_to_num to survive bad logits)."""
    try:
        x = torch.nan_to_num(
            logits.detach(), nan=0.0, posinf=0.0, neginf=0.0
        )
        flat = x.reshape(-1, x.shape[-1]) if x.dim() > 1 else x
        preds = flat.argmax(dim=-1)
        counts = [int((preds == c).sum().item()) for c in range(num_classes)]
        return f"pred_counts_per_class={counts}"
    except Exception as e:
        return f"<pred dist failed: {e}>"


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


# ─────────────────────────────────────────────────────────────────────────────
# Failing-batch forensic dump
# ─────────────────────────────────────────────────────────────────────────────


def first_nan_param(model: nn.Module) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """Return (name, shape) of the first parameter with NaN/Inf grad, or None."""
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if torch.isnan(g).any().item() or torch.isinf(g).any().item():
            return name, tuple(g.shape)
    return None


def _fmt_seq(vals, fmt: str = "{:+.3e}", limit: int = 8) -> str:
    out = [fmt.format(v) for v in vals[:limit]]
    if len(vals) > limit:
        out.append(f"...(+{len(vals) - limit})")
    return "[" + ", ".join(out) + "]"


def dump_failing_batch(
    *,
    epoch: int,
    batch_idx: int,
    reason: str,
    bvp: Optional[torch.Tensor],
    acc: Optional[torch.Tensor],
    ibi: Optional[torch.Tensor],
    labels: torch.Tensor,
    logits: Optional[torch.Tensor],
    loss: Optional[torch.Tensor],
    capture: Optional["IntermediateCapture"],
    criterion: Optional[nn.Module],
    num_classes: int,
    model: Optional[nn.Module] = None,
) -> None:
    """
    Forensic dump for a batch that failed (loss NaN, forward anomaly, or
    grad NaN). Each section is wrapped in try/except so a diagnostic bug
    never crashes training. Logs answer one of:
        bad input / encoder NaN / BiLSTM NaN / logits extreme/NaN /
        Focal Loss numerical bug / backward gradient explosion
    """
    tag = f"[DUMP epoch={epoch} batch={batch_idx} reason={reason}]"
    logger.error(f"{tag} ====== BEGIN FAILING-BATCH DUMP ======")

    # ── (1) Input stats ────────────────────────────────────────────────────
    try:
        logger.error(f"{tag} -- inputs --")
        logger.error(f"{tag}   {tensor_health_line('bvp', bvp)}")
        logger.error(f"{tag}   {tensor_health_line('acc', acc)}")
        logger.error(f"{tag}   {tensor_health_line('ibi', ibi)}")
        logger.error(
            f"{tag}   labels: {label_distribution_line(labels, num_classes)}"
        )
    except Exception as e:
        logger.error(f"{tag}   <inputs section failed: {e}>")

    # ── (2) Model stages ───────────────────────────────────────────────────
    try:
        logger.error(f"{tag} -- model stages --")
        enc = capture["encoder"] if capture is not None else None
        lst = capture["bilstm"] if capture is not None else None
        logger.error(f"{tag}   {tensor_health_line('encoder_out', enc)}")
        logger.error(f"{tag}   {tensor_health_line('bilstm_out',  lst)}")
        logger.error(f"{tag}   {tensor_health_line('logits',      logits)}")
        if logits is not None and torch.is_tensor(logits):
            logger.error(
                f"{tag}   "
                f"{predicted_class_distribution_line(logits, num_classes)}"
            )
    except Exception as e:
        logger.error(f"{tag}   <model-stages section failed: {e}>")

    # ── (3) Loss internals (focal-loss specific) ───────────────────────────
    try:
        logger.error(f"{tag} -- loss --")
        if loss is not None and torch.is_tensor(loss):
            lv = loss.detach().float().item()
            logger.error(f"{tag}   loss_value = {lv}")
        if (
            logits is not None
            and torch.is_tensor(logits)
            and criterion is not None
            and hasattr(criterion, "compute_internals")
        ):
            C = logits.shape[-1]
            logits_flat = logits.reshape(-1, C)
            labels_flat = labels.reshape(-1)
            internals = criterion.compute_internals(logits_flat, labels_flat)
            ce = internals["ce"]
            p_t = internals["p_t"]
            fw = internals["focal_weight"]
            at = internals["alpha_t"]
            lps = internals["loss_per_sample"]

            logger.error(f"{tag}   focal internals (standard formulation):")
            logger.error(f"{tag}     ce              {tensor_summary(ce)}")
            logger.error(f"{tag}     p_t             {tensor_summary(p_t)}")
            logger.error(f"{tag}     focal_weight    {tensor_summary(fw)}")
            logger.error(f"{tag}     alpha_t         {tensor_summary(at)}")
            logger.error(f"{tag}     loss_per_sample {tensor_summary(lps)}")

            def _bad_count(t: torch.Tensor) -> int:
                return int((torch.isnan(t) | torch.isinf(t)).sum().item())

            stage_bad = {
                "logits":          _bad_count(logits_flat.float()),
                "ce":              _bad_count(ce),
                "p_t":             _bad_count(p_t),
                "focal_weight":    _bad_count(fw),
                "alpha_t":         _bad_count(at),
                "loss_per_sample": _bad_count(lps),
            }
            logger.error(
                f"{tag}     NaN/Inf count per stage: {stage_bad}"
            )

            try:
                alpha_buf = getattr(criterion, "alpha", None)
                if alpha_buf is not None and torch.is_tensor(alpha_buf):
                    logger.error(
                        f"{tag}     class_weights(alpha)="
                        f"{_fmt_seq(alpha_buf.detach().float().tolist())}"
                    )
            except Exception:
                pass

            try:
                bad_mask = ~torch.isfinite(lps)
                n_bad = int(bad_mask.sum().item())
                logger.error(
                    f"{tag}     non-finite loss_per_sample count = {n_bad}"
                )
                if n_bad > 0:
                    bad_idx = bad_mask.nonzero(as_tuple=False).flatten()[:8]
                    logger.error(
                        f"{tag}     first {len(bad_idx)} bad samples:"
                    )
                    logger.error(
                        f"{tag}       labels ={labels_flat[bad_idx].tolist()}"
                    )
                    logger.error(
                        f"{tag}       p_t    ={_fmt_seq(p_t[bad_idx].tolist())}"
                    )
                    logger.error(
                        f"{tag}       focal_w={_fmt_seq(fw[bad_idx].tolist())}"
                    )
                    logger.error(
                        f"{tag}       ce     ={_fmt_seq(ce[bad_idx].tolist())}"
                    )
                    logger.error(
                        f"{tag}       alpha_t={_fmt_seq(at[bad_idx].tolist())}"
                    )
            except Exception as e:
                logger.error(f"{tag}     <bad-sample slice failed: {e}>")

            try:
                fwd_loss = criterion(logits_flat, labels_flat)
                logger.error(
                    f"{tag}   criterion.forward() recomputed = "
                    f"{fwd_loss.detach().float().item()}"
                )
            except Exception as e:
                logger.error(f"{tag}   <recompute forward failed: {e}>")
        else:
            logger.error(
                f"{tag}   <criterion has no compute_internals; "
                "skipping focal internals>"
            )
    except Exception as e:
        logger.error(f"{tag}   <loss section failed: {e}>")

    # ── (4) Backward / first NaN parameter ─────────────────────────────────
    try:
        if model is not None:
            first = first_nan_param(model)
            if first is not None:
                pname, pshape = first
                logger.error(
                    f"{tag} -- backward -- first NaN/Inf grad param: "
                    f"'{pname}' shape={pshape}"
                )
    except Exception as e:
        logger.error(f"{tag}   <backward section failed: {e}>")

    logger.error(f"{tag} ====== END FAILING-BATCH DUMP ======")
