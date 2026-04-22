# DREAMT Triple-Stream Sleep Staging Pipeline

## Overview

This pipeline implements a **Triple-Stream Cross-Attention Model** for 6-class sleep staging using the DREAMT wearable dataset.

| Component | Description |
|-----------|-------------|
| **Dataset** | DREAMT 64Hz wearable data (~100 subjects) |
| **Modalities** | PPG (BVP), ACC (3-channel), IBI (HRV features) |
| **Classes** | 6: P, W, N1, N2, N3, REM |
| **Architecture** | Triple-stream with pairwise cross-attention fusion |

---

## File Structure

```
sleep-staging-models/
├── dreamt_triple_dataset.py      # Dataset with subject-level split
├── triple_stream_model.py        # Model architecture
├── train_triple_stream.py        # Training script
├── config_triple_stream.yaml     # Configuration
└── DREAMT_TRIPLE_STREAM_PIPELINE.md  # This documentation
```

---

## Pipeline Stages

### Stage 1: Dataset (`dreamt_triple_dataset.py`)

**Purpose:** Load DREAMT data with strict subject-level train/val/test split to prevent data leakage.

#### Key Components

| Component | Description |
|-----------|-------------|
| `DatasetConfig` | Configuration dataclass (fs, window_sec, split ratios) |
| `DreamtTripleStreamDataset` | PyTorch Dataset class |
| `get_dataloaders()` | Factory function for train/val/test loaders |
| `compute_ibi_features()` | Extract HRV features from IBI values |

#### Data Flow

```
CSV Files (100 subjects)
    │
    ▼
Subject-Level Split (70/15/15)
    │
    ├── Train: 70 subjects
    ├── Val:   15 subjects
    └── Test:  15 subjects
    │
    ▼
30-second Window Segmentation
    │
    ▼
Per-Window Output:
    ├── BVP: (1, 1920) tensor
    ├── ACC: (3, 1920) tensor
    ├── IBI features: (5,) tensor
    └── Label: int (0-5)
```

#### IBI Feature Vector (5 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `mean_ibi` | Mean inter-beat interval |
| 1 | `std_ibi` | Standard deviation of IBI |
| 2 | `rmssd` | Root mean square of successive differences |
| 3 | `hr_mean` | Mean heart rate (60000/mean_ibi) |
| 4 | `n_beats` | Number of valid IBI values in window |

#### Verification Command

```bash
python dreamt_triple_dataset.py
```

#### Expected Output

```
======================================================================
DREAMT Triple-Stream Dataset Sanity Check
======================================================================
[INFO] Discovered 100 subject CSV files
[INFO] Subject split: train=70, val=15, test=15
[INFO] [OK] Subject-level split verified: No overlap between splits

--- Dataset Statistics ---
Train: ~74000 windows from 70 subjects
Val:   ~16000 windows from 15 subjects
Test:  ~16000 windows from 15 subjects

--- Sample Inspection ---
Sample 0:
  BVP shape: torch.Size([1, 1920]), dtype: torch.float32
  ACC shape: torch.Size([3, 1920]), dtype: torch.float32
  IBI features: [mean_ibi, std_ibi, rmssd, hr_mean, n_beats]
  Label: 0-5 (P/W/N1/N2/N3/REM)

--- Class Weights ---
  P: ~0.68
  W: ~0.83
  N1: ~1.93
  N2: ~0.46
  N3: ~6.78  (rarest class)
  REM: ~2.18

[OK] Sanity check passed!
======================================================================
```

#### Success Criteria

- [x] 100 subjects discovered
- [x] Subject split: 70/15/15 (no overlap)
- [x] BVP shape: (1, 1920)
- [x] ACC shape: (3, 1920)
- [x] IBI shape: (5,)
- [x] Labels: 0-5 (6 classes)
- [x] Class weights computed

---

### Stage 2: Model (`triple_stream_model.py`)

**Purpose:** Implement Triple-Stream architecture with pairwise cross-attention fusion.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  BVP (1, 1920)    ACC (3, 1920)    IBI Features (5,)               │
│       │                │                   │                        │
│       ▼                ▼                   ▼                        │
│  ┌─────────┐      ┌─────────┐        ┌──────────┐                  │
│  │ PPG     │      │ ACC     │        │ IBI MLP  │                  │
│  │ Encoder │      │ Encoder │        │ Encoder  │                  │
│  │(ResConv)│      │(ResConv)│        │ (5→256)  │                  │
│  └────┬────┘      └────┬────┘        └────┬─────┘                  │
│       │                │                   │                        │
│       ▼                ▼                   ▼                        │
│    P (256,30)      A (256,30)         I (256,30)                   │
├─────────────────────────────────────────────────────────────────────┤
│                    FUSION STAGES (x3)                               │
├─────────────────────────────────────────────────────────────────────┤
│  For each stage:                                                    │
│    Step 1: P ↔ A  (CrossModalFusionBlock) → Update P, A            │
│    Step 2: P ↔ I  (CrossModalFusionBlock) → Update P, I            │
│    Step 3: A ↔ I  (CrossModalFusionBlock) → Update A, I            │
├─────────────────────────────────────────────────────────────────────┤
│                   MODALITY WEIGHTING                                │
├─────────────────────────────────────────────────────────────────────┤
│  w_P, w_A, w_I = Softmax(score_P, score_A, score_I)                │
│  P = P * w_P, A = A * w_A, I = I * w_I                             │
├─────────────────────────────────────────────────────────────────────┤
│                    AGGREGATION                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Concat(P, A, I) → (768, 30)                                       │
│  Conv1d(768 → 256) → (256, 30)                                     │
├─────────────────────────────────────────────────────────────────────┤
│                  TEMPORAL MODELING                                  │
├─────────────────────────────────────────────────────────────────────┤
│  TemporalConvBlock (dilation=1)                                    │
│  TemporalConvBlock (dilation=2)                                    │
│  TemporalConvBlock (dilation=4)                                    │
├─────────────────────────────────────────────────────────────────────┤
│                    CLASSIFIER                                       │
├─────────────────────────────────────────────────────────────────────┤
│  AdaptiveAvgPool1d → Flatten → Linear(256→128) → Linear(128→6)    │
│       │                                                             │
│       ▼                                                             │
│  Output: (B, 6) logits                                              │
└─────────────────────────────────────────────────────────────────────┘
```

#### Key Classes

| Class | Purpose |
|-------|---------|
| `ResConvBlock` | Residual convolutional encoder block |
| `MultiHeadCrossAttention` | Multi-head cross-attention mechanism |
| `CrossModalFusionBlock` | Bidirectional cross-attention between two modalities |
| `TemporalConvBlock` | Dilated temporal convolution |
| `IBIFeatureEncoder` | MLP to encode IBI features to d_model |
| `TripleModalityWeighting` | Softmax-normalized modality weights |
| `TripleStreamSleepNet` | Main model class |

#### Fusion Update Rule (CRITICAL)

The fusion stages implement a **sequential update rule**:

```python
for stage_idx in range(n_fusion_blocks):  # 3 stages
    # Step 1: P ↔ A (bidirectional, update both)
    P, A = fusion_p_a[stage_idx](P, A)
    
    # Step 2: P ↔ I (bidirectional, update both)
    P, I = fusion_p_i[stage_idx](P, I)
    
    # Step 3: A ↔ I (bidirectional, update both)
    A, I = fusion_a_i[stage_idx](A, I)
```

#### Verification Command

```bash
python triple_stream_model.py
```

#### Expected Output

```
======================================================================
Triple-Stream Sleep Model Sanity Check
======================================================================

Model created:
  Total parameters: 21,324,809
  Trainable parameters: 21,324,809

Input shapes:
  BVP: torch.Size([4, 1, 1920])
  ACC: torch.Size([4, 3, 1920])
  IBI features: torch.Size([4, 5])

Output shape: torch.Size([4, 6])
Output (first sample): [0.37, -0.13, 0.25, 0.11, 0.12, 0.29]
Probabilities sum: [1.0, 1.0, 1.0, 1.0]

Testing backward pass...
Loss: ~1.5-2.5

Gradient norms (sample):
  ppg_encoder.0.conv1.weight: X.XXXX
  ...

[OK] Model sanity check passed!
======================================================================
```

#### Success Criteria

- [x] ~21M parameters
- [x] Input shapes: BVP (B,1,1920), ACC (B,3,1920), IBI (B,5)
- [x] Output shape: (B, 6)
- [x] Probabilities sum to 1.0
- [x] Backward pass successful
- [x] Gradients non-zero

---

### Stage 3: Training (`train_triple_stream.py`)

**Purpose:** Train the model with metrics tracking, checkpointing, and early stopping.

#### Training Features

| Feature | Description |
|---------|-------------|
| Mixed Precision (AMP) | Faster training on GPU |
| Class Weights | Handle imbalanced classes |
| Early Stopping | Patience-based stopping |
| Checkpointing | Save best and periodic models |
| Metrics | Loss, Accuracy, Kappa, F1 |
| TensorBoard | Optional logging |

#### Quick Pipeline Test

```bash
python -c "
from train_triple_stream import TripleStreamTrainer, load_config
import torch

config = load_config('config_triple_stream.yaml')
config['training']['num_epochs'] = 1
config['training']['batch_size'] = 8
config['data']['num_workers'] = 0

trainer = TripleStreamTrainer(config)
train_loader, val_loader, test_loader = trainer._create_dataloaders()
model = trainer._create_model()

bvp, acc, ibi_feat, labels = next(iter(train_loader))
bvp = bvp.to(trainer.device)
acc = acc.to(trainer.device)
ibi_feat = ibi_feat.to(trainer.device)
labels = labels.to(trainer.device)

model.train()
logits = model(bvp, acc, ibi_feat)
loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()

print(f'Batch shapes: BVP={bvp.shape}, ACC={acc.shape}, IBI={ibi_feat.shape}')
print(f'Logits shape: {logits.shape}')
print(f'Loss: {loss.item():.4f}')
print('[OK] Pipeline sanity check passed!')
"
```

#### Full Training Command

```bash
# Single run (50 epochs)
python train_triple_stream.py --config config_triple_stream.yaml

# Multiple runs for statistics
python train_triple_stream.py --config config_triple_stream.yaml --runs 5
```

#### Expected Training Output

```
======================================================================
Starting DREAMT Triple-Stream Training
======================================================================

[INFO] Subject split: train=70, val=15, test=15
[INFO] Class weights: [0.68, 0.83, 1.93, 0.46, 6.78, 2.18]
[INFO] Model parameters: 21,324,809

==================================================
Epoch 1/50
Learning rate: 1.00e-04
Epoch 1 [Train]: 100%|████| 4636/4636 [XX:XX<XX:XX, loss=X.XX, acc=X.XX]
Validation: 100%|████████████████| 991/991 [XX:XX<XX:XX]

Train Loss: X.XXXX, Train Acc: X.XXXX
Val Loss: X.XXXX, Val Acc: X.XXXX
Val Kappa: X.XXXX, Val F1: X.XXXX
Saved best model (kappa=X.XXXX)

...

======================================================================
Evaluating best model on test set
======================================================================

Test Results:
  Loss: X.XXXX
  Accuracy: X.XXXX
  Kappa: X.XXXX
  F1 (weighted): X.XXXX
  F1 (macro): X.XXXX

Classification Report:
              precision    recall  f1-score   support
           P       X.XX      X.XX      X.XX      XXXX
           W       X.XX      X.XX      X.XX      XXXX
          N1       X.XX      X.XX      X.XX      XXXX
          N2       X.XX      X.XX      X.XX      XXXX
          N3       X.XX      X.XX      X.XX      XXX
         REM       X.XX      X.XX      X.XX      XXXX

Results saved to: outputs/triple_stream/.../results/test_results.json

======================================================================
TRAINING COMPLETE!
======================================================================
```

#### Output Files

```
outputs/triple_stream/triple_stream_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── config.json           # Training configuration
│   ├── best_model.pth        # Best model checkpoint
│   └── checkpoint_epoch_X.pth # Periodic checkpoints
├── logs/                     # TensorBoard logs (if available)
└── results/
    ├── test_results.json     # Final metrics and history
    ├── confusion_matrix_test_epoch_X.png
    ├── confusion_matrix_best_epoch_X.png
    └── training_curves.png
```

#### Success Criteria

- [x] Training starts without errors
- [x] Loss decreases over epochs
- [x] Validation metrics computed (Kappa, F1)
- [x] Checkpoints saved
- [x] Test evaluation completed
- [x] Results JSON saved

---

## Configuration (`config_triple_stream.yaml`)

```yaml
# Data
data:
  data_dir: "C:/Users/sagil/data_64Hz"
  fs: 64.0
  window_sec: 30.0
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
  num_workers: 4

# Model
model:
  n_classes: 6
  d_model: 256
  n_heads: 8
  n_fusion_blocks: 3
  dropout: 0.2

# Training
training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  patience: 15

# Output
output:
  save_dir: "./outputs/triple_stream"
  save_frequency: 5

# Hardware
hardware:
  use_amp: true
  device: "cuda"
```

---

## Quick Verification Checklist

### Stage 1: Dataset
```bash
python dreamt_triple_dataset.py
```
- [ ] 100 subjects discovered
- [ ] Split: 70/15/15 subjects (no overlap)
- [ ] BVP shape: (1, 1920)
- [ ] ACC shape: (3, 1920)
- [ ] IBI shape: (5,)
- [ ] 6 classes preserved

### Stage 2: Model
```bash
python triple_stream_model.py
```
- [ ] ~21M parameters
- [ ] Output shape: (B, 6)
- [ ] Backward pass successful
- [ ] Gradients flowing

### Stage 3: Training Pipeline
```bash
python -c "from train_triple_stream import *; print('[OK]')"
```
- [ ] Imports without error
- [ ] Config loads
- [ ] Model initializes
- [ ] One batch processes

### Full Training (Optional)
```bash
python train_triple_stream.py --config config_triple_stream.yaml
```
- [ ] Training runs
- [ ] Metrics improve
- [ ] Checkpoints saved
- [ ] Test results generated

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: tensorboard` | Optional - training works without it |
| Out of memory | Reduce `batch_size` in config |
| CUDA not available | Will use CPU (slower) |
| NaN in IBI features | Expected when no IBI values in window |
| Low N3 (Deep) accuracy | Class is rare (~2.5% of data) |

### GPU Memory Tips

```yaml
# Reduce batch size
training:
  batch_size: 8

# Enable gradient accumulation
hardware:
  gradient_accumulation_steps: 2
```

---

## Key Design Decisions

### 1. Subject-Level Split (No Data Leakage)
Windows from the same subject are **never** split across train/val/test.

### 2. IBI as Feature Vector (Not Waveform)
IBI is irregularly sampled, so we extract 5 HRV features per window instead of trying to create a fixed-length waveform.

### 3. Sequential Fusion Update Rule
In each fusion stage, modalities are updated **in sequence**:
- P and A learn from each other first
- Then P and I (P has info from A now)
- Then A and I (A has info from P, I gets info from updated A)

### 4. 6 Classes (No Merging)
Labels: P (Preparation), W (Wake), N1, N2, N3, REM - exactly as in DREAMT.

### 5. Class Weighting
Inverse frequency weighting to handle class imbalance (N3 is ~7x rarer than N2).

---

## References

- DREAMT Dataset: Wearable sleep data at 64Hz
- Base Architecture: Adapted from SleepPPG-Net cross-attention model
- Fusion: Bidirectional pairwise cross-attention

---

## Version

- **Pipeline Version**: 1.0
- **Created**: December 2024
- **Python**: 3.8+
- **PyTorch**: 1.9+


