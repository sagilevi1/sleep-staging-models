"""
Sequence Sleep Staging Model
============================
Wraps the existing TripleStreamSleepNet as a per-window encoder and adds a
sequence head (BiLSTM by default; a Transformer variant is also provided)
that learns temporal context across L consecutive 30s windows.

Forward signature (sequence-aware):
    bvp_seq : (B, L, 1, T)
    acc_seq : (B, L, 3, T)
    ibi_seq : (B, L, F)
    -->
    logits  : (B, L, C)

Training: compute loss over all L positions (the trainer flattens to (B*L, C)).

The inner encoder produces a per-window embedding of size `embed_dim`; we
take it from the global-avg-pooled temporal features (i.e. the same point
where the original classifier head lived).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from triple_stream_model import TripleStreamSleepNet, create_triple_stream_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-window embedder
# ─────────────────────────────────────────────────────────────────────────────

class WindowEmbedder(nn.Module):
    """
    Wraps a TripleStreamSleepNet and exposes per-window embeddings instead
    of class logits. We replace the original `classifier` (which ends at
    n_classes) with the prefix that goes up to the d_model embedding.
    """

    def __init__(self, base: TripleStreamSleepNet):
        super().__init__()
        self.base = base
        self.d_model = base.d_model
        # Override the classifier with an identity-like head:
        #  - keep AdaptiveAvgPool1d + Flatten
        #  - drop the (Linear -> Linear) tail
        # The base.forward goes through `self.classifier` whole, so we must
        # re-implement the embed path here without touching base.classifier.

    @property
    def embed_dim(self) -> int:
        return self.d_model

    def forward(
        self,
        bvp: torch.Tensor,           # (B, 1, T)
        acc: torch.Tensor,           # (B, 3, T)
        ibi_features: torch.Tensor,  # (B, F)
    ) -> torch.Tensor:
        """Run the base network up to the temporal block output, then GAP."""
        b = self.base
        P = b.ppg_encoder(bvp)
        A = b.acc_encoder(acc)
        I = b.ibi_encoder(ibi_features)

        T = P.size(2)
        I = I.expand(-1, -1, T)

        P = P + b.positional_encoding[:, :, :T]
        A = A + b.positional_encoding[:, :, :T]
        I = I + b.positional_encoding[:, :, :T]

        for stage_idx in range(b.n_fusion_blocks):
            P_t = P.transpose(1, 2)
            A_t = A.transpose(1, 2)
            I_t = I.transpose(1, 2)
            P_t, A_t = b.fusion_p_a[stage_idx](P_t, A_t)
            P_t, I_t = b.fusion_p_i[stage_idx](P_t, I_t)
            A_t, I_t = b.fusion_a_i[stage_idx](A_t, I_t)
            P = P_t.transpose(1, 2)
            A = A_t.transpose(1, 2)
            I = I_t.transpose(1, 2)

        w_p, w_a, w_i = b.modality_weighting(P, A, I)
        P = P * w_p.unsqueeze(-1)
        A = A * w_a.unsqueeze(-1)
        I = I * w_i.unsqueeze(-1)

        fused = torch.cat([P, A, I], dim=1)
        fused = b.feature_aggregation(fused)
        temporal = b.temporal_blocks(fused)            # (B, d_model, T')

        # Global average pool over time → (B, d_model)
        emb = temporal.mean(dim=2)
        return emb


# ─────────────────────────────────────────────────────────────────────────────
# Sequence head
# ─────────────────────────────────────────────────────────────────────────────

class _BiLSTMHead(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_in)
        out, _ = self.lstm(x)
        return out  # (B, L, out_dim)


class _TransformerHead(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_len: int = 200,
    ):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        # Learned positional encoding (sequence is short, e.g. 20)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        L = x.size(1)
        x = x + self.pos[:, :L]
        return self.encoder(x)


class SequenceSleepNet(nn.Module):
    """
    Sequence-aware sleep stager.

    Pipeline:
        per-window: TripleStreamSleepNet → embedding (d_model)
        sequence : BiLSTM | Transformer over (B, L, d_model)
        per-step : Linear → n_classes
    """

    def __init__(
        self,
        n_classes: int = 6,
        d_model: int = 256,
        n_heads: int = 8,
        n_fusion_blocks: int = 3,
        dropout: float = 0.4,
        ibi_n_features: int = 5,
        seq_model: str = "bilstm",
        seq_hidden: int = 256,
        seq_layers: int = 1,
        seq_bidirectional: bool = True,
        seq_dropout: Optional[float] = None,
    ):
        super().__init__()

        base = create_triple_stream_model(
            n_classes=n_classes,
            d_model=d_model,
            n_heads=n_heads,
            n_fusion_blocks=n_fusion_blocks,
            dropout=dropout,
        )
        # The base has ibi_n_features baked in via default; rebuild if needed.
        if ibi_n_features != 5:
            base = TripleStreamSleepNet(
                n_classes=n_classes,
                d_model=d_model,
                n_heads=n_heads,
                n_fusion_blocks=n_fusion_blocks,
                dropout=dropout,
                ibi_n_features=ibi_n_features,
            )
        self.embedder = WindowEmbedder(base)
        self.embed_dim = self.embedder.embed_dim

        seq_dropout = dropout if seq_dropout is None else seq_dropout
        seq_model = (seq_model or "bilstm").lower()
        if seq_model in ("bilstm", "lstm"):
            self.seq_head = _BiLSTMHead(
                d_in=self.embed_dim,
                hidden_size=seq_hidden,
                num_layers=seq_layers,
                bidirectional=(seq_model == "bilstm" or seq_bidirectional),
                dropout=seq_dropout,
            )
        elif seq_model == "transformer":
            self.seq_head = _TransformerHead(
                d_in=self.embed_dim,
                d_model=seq_hidden,
                n_heads=n_heads,
                num_layers=max(1, seq_layers),
                dropout=seq_dropout,
            )
        else:
            raise ValueError(f"Unknown seq_model '{seq_model}'. Use 'bilstm' or 'transformer'.")

        self.classifier = nn.Sequential(
            nn.Dropout(seq_dropout),
            nn.Linear(self.seq_head.out_dim, n_classes),
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        bvp_seq: torch.Tensor,        # (B, L, 1, T)
        acc_seq: torch.Tensor,        # (B, L, 3, T)
        ibi_seq: torch.Tensor,        # (B, L, F)
    ) -> torch.Tensor:
        B, L = bvp_seq.shape[:2]
        T = bvp_seq.shape[-1]

        # Flatten (B, L) → (B*L) for the per-window encoder
        bvp_flat = bvp_seq.reshape(B * L, 1, T)
        acc_flat = acc_seq.reshape(B * L, 3, T)
        ibi_flat = ibi_seq.reshape(B * L, ibi_seq.shape[-1])

        emb = self.embedder(bvp_flat, acc_flat, ibi_flat)   # (B*L, d_model)
        emb = emb.reshape(B, L, -1)

        seq_out = self.seq_head(emb)                        # (B, L, H)
        logits = self.classifier(seq_out)                   # (B, L, C)
        return logits


def create_sequence_model(
    n_classes: int = 6,
    d_model: int = 256,
    n_heads: int = 8,
    n_fusion_blocks: int = 3,
    dropout: float = 0.4,
    ibi_n_features: int = 5,
    seq_model: str = "bilstm",
    seq_hidden: int = 256,
    seq_layers: int = 1,
    seq_bidirectional: bool = True,
    seq_dropout: Optional[float] = None,
) -> SequenceSleepNet:
    return SequenceSleepNet(
        n_classes=n_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_fusion_blocks=n_fusion_blocks,
        dropout=dropout,
        ibi_n_features=ibi_n_features,
        seq_model=seq_model,
        seq_hidden=seq_hidden,
        seq_layers=seq_layers,
        seq_bidirectional=seq_bidirectional,
        seq_dropout=seq_dropout,
    )
