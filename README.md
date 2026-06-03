# ASTS: Attention-based Soft Temperature Scaling

Adversarial robustness training for Vision Transformers using learnable per-head attention temperatures.

## Overview

ASTS introduces learnable temperature parameters into the attention mechanism of Vision Transformers. During adversarial training, these temperatures are optimized alongside multiple loss components designed to improve robustness.

**Key Features:**
- Per-head learnable temperature scaling in attention layers
- TRADES-style logit KL loss for improved robustness
- Attention divergence loss (KL or JS) for attention consistency
- Per-layer attention weighting (uniform, linear, or exponential)
- Optional attention entropy regularization
- Support for curriculum adversarial training (progressive attack strength)
- Multiple attack types: PGD, FGSM, CW, AutoAttack
- Mixed precision (AMP) training for faster execution
- Stratified data sampling with class distribution preservation

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd asts

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA (recommended for training)

## Quick Start

```bash
# Run with default config (full dataset)
python run_experiment.py --config configs/default.yaml

# Quick test run with 10% of data
# Edit configs/default.yaml: set train_ratio: 0.1 and eval_ratio: 0.1
```

## Configuration

All settings are in YAML config files. See `configs/default.yaml` for the full reference.

### Key Configuration Sections

#### Experiment
```yaml
experiment:
  name: my_experiment    # Run name (used in output folder)
  seed: 42               # Random seed for reproducibility
  output_root: outputs   # Output directory
  device: auto           # auto|cpu|cuda
```

#### Data
```yaml
data:
  dataset: cifar10       # cifar10|cifar100|imagenet1k
  image_size: 224
  batch_size: 64
  num_workers: 4
  val_split: 0.1         # Fraction of train data for validation
  train_ratio: 1.0       # Stratified sampling ratio (0.0-1.0)
  eval_ratio: 1.0        # Stratified sampling ratio for test/val
  imagenet_dir: null     # Required if dataset=imagenet1k
```

**Stratified Sampling**: Set `train_ratio` and `eval_ratio` to values < 1.0 for faster experiments while preserving class distribution. For example, `eval_ratio: 0.1` uses 10% of test data with balanced classes.

#### Model
```yaml
model:
  hf_model_id: nateraw/vit-base-patch16-224-cifar10
  num_labels: 10
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

#### ASTS (Temperature Scaling)
```yaml
asts:
  enabled: true
  parameterization: exp   # exp|softplus
  init_temperature: 1.0   # Initial temperature value
  eps: 1.0e-6
```

#### Training
```yaml
training:
  epochs: 20
  lr: 5.0e-4
  weight_decay: 0.01
  grad_clip_norm: 1.0
  use_amp: true           # Mixed precision (FP16) - recommended
  torch_compile: false    # PyTorch 2.0+ compilation (experimental)
  eval_every_epochs: 1
  save_every_epochs: 1
```

#### Trainable Parameters
```yaml
trainable:
  train_temperatures: true      # Train ASTS temperature parameters
  train_classifier_head: true   # Train classification head
  train_layernorm: false        # Train LayerNorm parameters
  train_backbone: false         # Train full backbone (expensive)
```

#### Objective

```yaml
objective:
  lambda_attn: 0.5             # Weight for attention divergence loss
  ce_mode: adv_only            # adv_only|clean_only|mixed (CRITICAL: use adv_only for robustness)
  beta_clean: 0.5              # Weight for clean CE (if ce_mode=mixed)
  attn_divergence_mode: kl     # kl|js (js = symmetric Jensen-Shannon divergence)
  
  temperature_reg:             # Temperature regularization (pulls temps toward target)
    enabled: false
    weight: 0.0
    target: 1.0
    clamp_min: 0.1
    clamp_max: 10.0
  
  trades:                      # TRADES-style logit KL loss (RECOMMENDED for robustness)
    enabled: true
    lambda: 6.0                # Weight for TRADES loss (default from paper: 6.0)
    temperature: 1.0           # Temperature for softmax in KL computation
  
  attn_entropy:                # Attention entropy regularization (prevents attention collapse)
    enabled: false
    weight: 0.1
    target: 2.0                # Target entropy (higher = more uniform attention)
  
  layer_weighting:             # Per-layer attention loss weighting
    mode: linear               # uniform|linear|exponential (later layers weighted more)
    base: 1.5                  # Base for exponential mode
```

**Important Notes:**
- `ce_mode: adv_only` is **critical** for robustness - the model must learn to classify adversarial examples
- `trades.enabled: true` adds TRADES-style logit consistency loss from [Zhang et al., 2019]
- `layer_weighting: linear` weights later transformer layers more heavily (often more important for classification)

#### Attacks
```yaml
attacks:
  train:
    attacks:
      # Curriculum: start weak, increase strength (recommended)
      - name: pgd
        epochs: 5
        params:
          eps: 0.015686    # 4/255
          alpha: 0.003922  # 1/255
          steps: 7
      - name: pgd
        epochs: 7
        params:
          eps: 0.023529    # 6/255
          steps: 10
      - name: pgd
        epochs: 8
        params:
          eps: 0.031373    # 8/255 (matches eval)
          steps: 20

  eval:
    enabled: true
    attacks:
      - name: pgd
        params: { eps: 0.031373, steps: 20 }
      - name: autoattack
        params: { eps: 0.031373, norm: Linf, version: standard }
```

**Supported Attacks**: `pgd`, `fgsm`, `cw`, `autoattack`, `bim`, `mifgsm`

## Output Structure

```
outputs/
└── 2024-01-01_12-00-00_my_experiment/
    ├── config_resolved.yaml    # Saved config
    ├── logs/
    │   └── train.log           # Training logs
    ├── checkpoints/
    │   ├── best.pt             # Best validation checkpoint
    │   └── last.pt             # Latest checkpoint
    ├── metrics/
    │   ├── train_metrics.csv   # Per-epoch training losses + temperature stats
    │   ├── val_metrics.csv     # Validation accuracy
    │   ├── test_metrics.csv    # Final test results
    │   └── test_metrics_baseline.csv  # Pre-training baseline
    └── plots/
        ├── train_losses.png
        ├── val_accuracy.png
        ├── temperatures_heatmap.png
        └── ...
```

### Training Metrics CSV Columns

The `train_metrics.csv` now includes:
- `loss_total`, `loss_ce`, `loss_attn_div`, `loss_temp_reg`, `loss_trades`, `loss_entropy`
- `temp_mean`, `temp_std`, `temp_min`, `temp_max` (temperature statistics per epoch)

## Loss Function

The total loss combines multiple components:

```
L_total = L_ce + λ_attn * L_attn_div + L_temp_reg + λ_trades * L_trades + λ_entropy * L_entropy
```

Where:
- **L_ce**: Cross-entropy on adversarial examples (when `ce_mode: adv_only`)
- **L_attn_div**: KL or JS divergence between clean and adversarial attention patterns
- **L_temp_reg**: Optional L2 regularization on temperatures
- **L_trades**: KL(p_adv || p_clean) on logits (TRADES-style)
- **L_entropy**: Optional attention entropy regularization

## Performance Optimizations

This codebase includes several optimizations for faster training:

| Optimization | Speedup | Config |
|--------------|---------|--------|
| Mixed Precision (AMP) | 1.5-2x | `training.use_amp: true` |
| torch.compile() | 10-30% | `training.torch_compile: true` |
| Persistent DataLoader workers | 5-15% | Automatic |
| Per-layer attention computation | ~10% | Built-in |
| Cached module references | ~5% | Built-in |

## Example Configs

### Fast Debug Run
```yaml
data:
  train_ratio: 0.1
  eval_ratio: 0.1
training:
  epochs: 2
  use_amp: true
```

### Full Robust Training (Recommended)
```yaml
data:
  train_ratio: 1.0
  eval_ratio: 1.0
training:
  epochs: 20
  use_amp: true
objective:
  ce_mode: adv_only
  trades:
    enabled: true
    lambda: 6.0
  layer_weighting:
    mode: linear
```

### ImageNet-1K
```yaml
data:
  dataset: imagenet1k
  imagenet_dir: /path/to/imagenet
  train_ratio: 0.1  # Recommended for initial experiments
model:
  hf_model_id: google/vit-base-patch16-224
  num_labels: 1000
```

## References

- TRADES: [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573) (Zhang et al., 2019)
- QUEST: [Query-based Soft Token Pruning](https://arxiv.org/abs/2604.00199)
- AutoAttack: [Reliable evaluation of adversarial robustness](https://arxiv.org/abs/2003.01690)

## License

[Add your license here]

## Citation

[Add citation if applicable]
