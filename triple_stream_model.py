"""
Triple-Stream Sleep Staging Model with Pairwise Cross-Attention

Architecture:
- PPG Encoder: ResConv-based (1 channel)
- ACC Encoder: ResConv-based (3 channels)
- IBI Encoder: MLP for HRV features (5 -> d_model)
- Pairwise Cross-Attention Fusion with Sequential UPDATE RULE
- Temporal Modeling: Dilated TCN blocks
- Classifier: 6-class output (P, W, N1, N2, N3, REM)

Fusion Update Rule:
In each fusion stage, modality representations are UPDATED in place:
1. P <-> A (bidirectional cross-attention, update both P and A)
2. P <-> I (update both P and I)
3. A <-> I (update both A and I)
Repeat for N fusion stages.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# =============================================================================
# Building Blocks (Reused/Adapted from existing codebase)
# =============================================================================

class ResConvBlock(nn.Module):
    """
    Residual Convolutional Block.
    Adapted from multimodal_model_crossattn.py to support variable input channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.MaxPool1d(kernel_size=stride, stride=stride)
            )
        else:
            self.residual = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        if self.residual is not None:
            identity = self.residual(identity)
        
        return x + identity


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention Mechanism.
    Reused from multimodal_model_crossattn.py
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, T_q, d_model)
            key: (B, T_k, d_model)
            value: (B, T_v, d_model)
        
        Returns:
            output: (B, T_q, d_model)
            attention_weights: (B, n_heads, T_q, T_k)
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        
        # Merge heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        output = self.dropout(output)
        
        # Residual + LayerNorm
        output = self.layer_norm(output + query)
        
        return output, attention_weights


class CrossModalFusionBlock(nn.Module):
    """
    Cross-Modal Fusion Block.
    Performs bidirectional cross-attention between two modalities.
    
    Both inputs are UPDATED (in-place semantically):
    - modality_a attends to modality_b -> updated modality_a
    - modality_b attends to modality_a -> updated modality_b
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # A attends to B
        self.cross_attn_a = MultiHeadCrossAttention(d_model, n_heads, dropout)
        
        # B attends to A
        self.cross_attn_b = MultiHeadCrossAttention(d_model, n_heads, dropout)
        
        # Feed-forward networks
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.ffn_b = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.layer_norm_a = nn.LayerNorm(d_model)
        self.layer_norm_b = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_a: (B, T, d_model) - First modality features
            feat_b: (B, T, d_model) - Second modality features
        
        Returns:
            updated_a: (B, T, d_model) - Updated first modality
            updated_b: (B, T, d_model) - Updated second modality
        """
        # A attends to B (A = query, B = key/value)
        attended_a, _ = self.cross_attn_a(feat_a, feat_b, feat_b)
        
        # B attends to A (B = query, A = key/value)
        attended_b, _ = self.cross_attn_b(feat_b, feat_a, feat_a)
        
        # FFN + residual + LayerNorm
        updated_a = self.layer_norm_a(attended_a + self.dropout(self.ffn_a(attended_a)))
        updated_b = self.layer_norm_b(attended_b + self.dropout(self.ffn_b(attended_b)))
        
        return updated_a, updated_b


class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated convolutions.
    Reused from multimodal_model_crossattn.py
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        
        if self.residual is not None:
            x = self.residual(x)
        
        return self.relu(out + x)


# =============================================================================
# IBI Feature Encoder
# =============================================================================

class IBIFeatureEncoder(nn.Module):
    """
    Encode HRV feature vector to d_model embedding via MLP.
    
    Input: (B, n_features) where n_features = 5 [mean_ibi, std_ibi, rmssd, hr_mean, n_beats]
    Output: (B, d_model, 1) to match temporal format of other encoders
    """
    
    def __init__(self, n_features: int = 5, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, ibi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ibi_features: (B, n_features)
        
        Returns:
            embedding: (B, d_model, 1)
        """
        emb = self.mlp(ibi_features)  # (B, d_model)
        return emb.unsqueeze(-1)  # (B, d_model, 1)


# =============================================================================
# Triple Modality Weighting
# =============================================================================

class TripleModalityWeighting(nn.Module):
    """
    Adaptive weighting for 3 modalities using softmax normalization.
    
    Learns to weight PPG (P), ACC (A), and IBI (I) based on their features.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # One weight network per modality
        self.weight_net_p = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.weight_net_a = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.weight_net_i = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(
        self,
        feat_p: torch.Tensor,
        feat_a: torch.Tensor,
        feat_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_p: (B, d_model, T) - PPG features
            feat_a: (B, d_model, T) - ACC features
            feat_i: (B, d_model, T) - IBI features
        
        Returns:
            w_p, w_a, w_i: (B, 1) each - Softmax-normalized weights
        """
        score_p = self.weight_net_p(feat_p)  # (B, 1)
        score_a = self.weight_net_a(feat_a)  # (B, 1)
        score_i = self.weight_net_i(feat_i)  # (B, 1)
        
        # Stack and apply softmax
        scores = torch.cat([score_p, score_a, score_i], dim=1)  # (B, 3)
        weights = F.softmax(scores, dim=1)  # (B, 3)
        
        w_p = weights[:, 0:1]  # (B, 1)
        w_a = weights[:, 1:2]  # (B, 1)
        w_i = weights[:, 2:3]  # (B, 1)
        
        return w_p, w_a, w_i


# =============================================================================
# Signal Encoder Factory
# =============================================================================

def create_signal_encoder(in_channels: int, d_model: int = 256) -> nn.Sequential:
    """
    Create ResConv-based encoder for PPG or ACC signals.
    
    Downsamples from 1920 samples to a smaller temporal dimension.
    1920 -> 960 -> 480 -> 240 -> 120 -> 60 -> 30 (6 blocks with stride 2)
    """
    return nn.Sequential(
        ResConvBlock(in_channels, 32, stride=2),   # 1920 -> 960
        ResConvBlock(32, 64, stride=2),            # 960 -> 480
        ResConvBlock(64, 128, stride=2),           # 480 -> 240
        ResConvBlock(128, 256, stride=2),          # 240 -> 120
        ResConvBlock(256, 256, stride=2),          # 120 -> 60
        ResConvBlock(256, d_model, stride=2),      # 60 -> 30
    )


# =============================================================================
# Main Model: TripleStreamSleepNet
# =============================================================================

class TripleStreamSleepNet(nn.Module):
    """
    Triple-Stream Sleep Staging Network with Pairwise Cross-Attention.
    
    Streams:
    - P (PPG): BVP signal, 1 channel
    - A (ACC): Accelerometer, 3 channels
    - I (IBI): HRV features, 5-dim vector
    
    Fusion Update Rule (per stage):
    1. P <-> A (bidirectional, update both)
    2. P <-> I (bidirectional, update both)
    3. A <-> I (bidirectional, update both)
    """
    
    def __init__(
        self,
        n_classes: int = 6,
        d_model: int = 256,
        n_heads: int = 8,
        n_fusion_blocks: int = 3,
        dropout: float = 0.2,
        ibi_n_features: int = 5
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_fusion_blocks = n_fusion_blocks
        
        # === ENCODERS ===
        self.ppg_encoder = create_signal_encoder(in_channels=1, d_model=d_model)
        self.acc_encoder = create_signal_encoder(in_channels=3, d_model=d_model)
        self.ibi_encoder = IBIFeatureEncoder(n_features=ibi_n_features, d_model=d_model, dropout=dropout)
        
        # === POSITIONAL ENCODING ===
        self.register_buffer(
            "positional_encoding",
            self._create_positional_encoding(d_model, max_len=100)
        )
        
        # === PAIRWISE FUSION BLOCKS ===
        # Each fusion stage has 3 CrossModalFusionBlocks
        self.fusion_p_a = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, dropout) for _ in range(n_fusion_blocks)
        ])
        self.fusion_p_i = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, dropout) for _ in range(n_fusion_blocks)
        ])
        self.fusion_a_i = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, dropout) for _ in range(n_fusion_blocks)
        ])
        
        # === MODALITY WEIGHTING ===
        self.modality_weighting = TripleModalityWeighting(d_model)
        
        # === FEATURE AGGREGATION ===
        self.feature_aggregation = nn.Sequential(
            nn.Conv1d(d_model * 3, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # === TEMPORAL MODELING ===
        self.temporal_blocks = nn.Sequential(
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=1, dropout=dropout),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=2, dropout=dropout),
            TemporalConvBlock(d_model, d_model, kernel_size=7, dilation=4, dropout=dropout),
        )
        
        # === CLASSIFIER (6 classes) ===
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global pooling
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0).transpose(1, 2)  # (1, d_model, max_len)
    
    def _init_weights(self):
        """Initialize weights for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        bvp: torch.Tensor,
        acc: torch.Tensor,
        ibi_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            bvp: (B, 1, 1920) - PPG signal
            acc: (B, 3, 1920) - Accelerometer signal
            ibi_features: (B, 5) - HRV feature vector
        
        Returns:
            logits: (B, 6) - Class logits for 6 sleep stages
        """
        batch_size = bvp.size(0)
        
        # === ENCODE ===
        P = self.ppg_encoder(bvp)  # (B, d_model, T) where T=30
        A = self.acc_encoder(acc)  # (B, d_model, T)
        I = self.ibi_encoder(ibi_features)  # (B, d_model, 1)
        
        # Broadcast IBI embedding to match temporal dimension
        T = P.size(2)
        I = I.expand(-1, -1, T)  # (B, d_model, T)
        
        # === ADD POSITIONAL ENCODING ===
        P = P + self.positional_encoding[:, :, :T]
        A = A + self.positional_encoding[:, :, :T]
        I = I + self.positional_encoding[:, :, :T]
        
        # === FUSION STAGES WITH UPDATE RULE ===
        for stage_idx in range(self.n_fusion_blocks):
            # Convert to (B, T, d_model) for attention
            P_t = P.transpose(1, 2)
            A_t = A.transpose(1, 2)
            I_t = I.transpose(1, 2)
            
            # Step 1: P <-> A (bidirectional, update both)
            P_t, A_t = self.fusion_p_a[stage_idx](P_t, A_t)
            
            # Step 2: P <-> I (bidirectional, update both)
            P_t, I_t = self.fusion_p_i[stage_idx](P_t, I_t)
            
            # Step 3: A <-> I (bidirectional, update both)
            A_t, I_t = self.fusion_a_i[stage_idx](A_t, I_t)
            
            # Convert back to (B, d_model, T)
            P = P_t.transpose(1, 2)
            A = A_t.transpose(1, 2)
            I = I_t.transpose(1, 2)
        
        # === MODALITY WEIGHTING ===
        w_p, w_a, w_i = self.modality_weighting(P, A, I)
        
        P = P * w_p.unsqueeze(-1)
        A = A * w_a.unsqueeze(-1)
        I = I * w_i.unsqueeze(-1)
        
        # === AGGREGATION ===
        fused = torch.cat([P, A, I], dim=1)  # (B, 3*d_model, T)
        fused = self.feature_aggregation(fused)  # (B, d_model, T)
        
        # === TEMPORAL MODELING ===
        temporal = self.temporal_blocks(fused)  # (B, d_model, T)
        
        # === CLASSIFICATION ===
        logits = self.classifier(temporal)  # (B, n_classes)
        
        return logits
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Model Variants
# =============================================================================

def create_triple_stream_model(
    n_classes: int = 6,
    d_model: int = 256,
    n_heads: int = 8,
    n_fusion_blocks: int = 3,
    dropout: float = 0.2
) -> TripleStreamSleepNet:
    """Factory function to create TripleStreamSleepNet."""
    return TripleStreamSleepNet(
        n_classes=n_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_fusion_blocks=n_fusion_blocks,
        dropout=dropout
    )


# =============================================================================
# Sanity Check
# =============================================================================

def sanity_check():
    """Run sanity check on the model."""
    print("\n" + "=" * 70)
    print("Triple-Stream Sleep Model Sanity Check")
    print("=" * 70)
    
    # Create model
    model = TripleStreamSleepNet(
        n_classes=6,
        d_model=256,
        n_heads=8,
        n_fusion_blocks=3,
        dropout=0.2
    )
    
    print(f"\nModel created:")
    print(f"  Total parameters: {model.get_num_parameters():,}")
    print(f"  Trainable parameters: {model.get_num_trainable_parameters():,}")
    
    # Create dummy inputs
    batch_size = 4
    bvp = torch.randn(batch_size, 1, 1920)     # PPG: 30s @ 64Hz
    acc = torch.randn(batch_size, 3, 1920)     # ACC: 3 channels
    ibi_features = torch.randn(batch_size, 5)  # HRV features
    
    print(f"\nInput shapes:")
    print(f"  BVP: {bvp.shape}")
    print(f"  ACC: {acc.shape}")
    print(f"  IBI features: {ibi_features.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(bvp, acc, ibi_features)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"Output (first sample): {logits[0].tolist()}")
    
    # Check softmax probabilities
    probs = F.softmax(logits, dim=1)
    print(f"Probabilities sum: {probs.sum(dim=1).tolist()}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    model.train()
    logits = model(bvp, acc, ibi_features)
    loss = F.cross_entropy(logits, torch.randint(0, 6, (batch_size,)))
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    
    print(f"\nGradient norms (sample):")
    for name, norm in grad_norms[:5]:
        print(f"  {name}: {norm:.6f}")
    
    print("\n" + "=" * 70)
    print("[OK] Model sanity check passed!")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    sanity_check()

