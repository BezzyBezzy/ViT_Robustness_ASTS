# ASTS: Adaptive Self-Attention Temperature Scaling

Adversarial robustness training for Vision Transformers using learnable per-head attention temperatures.

Authors: Bezalel Spolter and Daniel Zafrir
Research Advisor: David Levin

## Overview

Vision Transformers (ViTs) are highly susceptible to **Attention Hijacking**, where an adversarial attack (such as a localized patch or gradient noise) forces the model to concentrate its attention mass on a corrupted region, effectively blinding the semantic pathways of the network.

ASTS combats this by introducing learnable, per-head temperature parameters into the self-attention mechanism. Using the **Clean-Adversarial Attention Alignment (CA3)** algorithm, these temperatures are dynamically optimized during adversarial training. By penalizing the divergence (via KL or JS) between a clean attention map and its adversarial counterpart, the model learns to independently self-regulate the sharpness of its attention heads, filtering out noise while preserving visual stability.

**Key Features:**

* Per-head learnable temperature scaling in attention layers
* CA3 Algorithm: Alignment of clean and adversarial attention maps
* Support for asymmetric (KL) and symmetric (Jensen-Shannon) divergence penalties
* Curriculum adversarial training (progressive attack strength, e.g., PGD to CW transitions)
* Automated extraction of temperature dynamics (IQR, Box plots, Top 10 head progressions)

---

## Key Findings & Empirical Results

Based on our evaluation using a ViT-B model fine-tuned on CIFAR-10, the ASTS methodology reveals several novel insights into how Vision Transformers defend themselves against adversarial attacks:

### 1. The Robustness-Accuracy Trade-off

By tuning the objective function, ASTS can navigate the fundamental trade-off between natural accuracy and adversarial robustness:

* **Maximum Robustness (Adv-Only Task Loss + KL Divergence):** By optimizing strictly on adversarial examples, the model achieved **48.11% on PGD-100** and **38.97% on AutoAttack**, but suffered a slight drop in clean accuracy (94.29%).
* **Optimal Balance (Mixed Task Loss + JS Divergence):** By preserving equal weight for clean examples and using a symmetric JS penalty, the model maintained a highly competitive **37.61% on AutoAttack** while restoring natural clean accuracy to an impressive **96.24%**.

### 2. Spatial Distribution of Defense

The choice of loss function radically alters *where* the model defends itself:

* When forced to preserve clean accuracy (Mixed Task Loss), the model cannot radically alter its deep semantic representations. Therefore, it **disperses its temperature scaling** throughout the early and middle layers to intercept adversarial noise before it reaches deep pathways.
* Conversely, models trained purely on adversarial data **concentrate their defenses** almost entirely in the deepest modules (e.g., Modules 10 and 11).

### 3. Threat-Specific Attention Dynamics (The Epoch 15 Pivot)

The model learns distinctly different attention distributions depending on the attack optimizer. During the initial PGD training curriculum, the model broadly flattens attention distributions (steadily increasing the temperature variance). However, upon transitioning to the **CW attack at Epoch 15**, the behavior sharply pivots: the model shrinks its overall temperature variance and relies entirely on a highly specialized, isolated subset of extreme outliers (e.g., temperatures scaling up to $T \approx 25$).

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd asts

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

```

## Quick Start

```bash
# Run with default config (full dataset)
python run_experiment.py --config configs/default.yaml

```

## Configuration

All settings are in YAML config files. See `configs/default.yaml` for the full reference.

### Objective (The CA3 Algorithm)

```yaml
objective:
  lambda_attn: 0.5             # Weight for CA3 attention divergence loss
  ce_mode: mixed               # adv_only|clean_only|mixed
  beta_clean: 0.5              # Weight for clean CE (if ce_mode=mixed)
  attn_divergence_mode: js     # kl|js (js = symmetric Jensen-Shannon divergence)
  
  trades:                      # TRADES-style logit KL loss
    enabled: true
    lambda: 6.0                # Weight for TRADES loss
    temperature: 1.0           

```

### Attacks (Curriculum Schedule)

To reproduce the curriculum from the ASTS paper, attacks progressively increase in strength before shifting optimization methods (PGD to CW) at Epoch 15:

```yaml
attacks:
  train:
    attacks:
      - name: pgd
        epochs: 4
        params: { eps: 0.015686, alpha: 0.003922, steps: 7 }   # 4/255 budget
      - name: pgd
        epochs: 5
        params: { eps: 0.023529, alpha: 0.003922, steps: 10 }  # 6/255 budget
      - name: pgd
        epochs: 6
        params: { eps: 0.031373, alpha: 0.003922, steps: 20 }  # 8/255 budget
      - name: cw
        epochs: 5
        params: { eps: 0.031373, steps: 15 }                   # 8/255 budget

```

## Reproducing Paper Results

To replicate the specific configurations detailed in the ASTS research paper:

### Config 2: Maximum Robustness (Adv Only + KL)

This configuration concentrates temperature scaling in the deepest layers of the network, achieving the highest robust accuracy at the cost of slight natural accuracy degradation.

```yaml
objective:
  ce_mode: adv_only
  attn_divergence_mode: kl

```

### Config 4: Optimal Trade-off (Mixed + JS)

This configuration disperses temperature scaling throughout the early and middle layers to protect clean semantic pathways. Using symmetric JS divergence, it preserves high natural accuracy while maintaining competitive robustness.

```yaml
objective:
  ce_mode: mixed
  beta_clean: 0.5
  attn_divergence_mode: js

```

## Output Structure

The codebase automatically tracks and plots the fine-grained temperature dynamics discussed in the paper:

```text
outputs/
└── 2026-01-01_12-00-00_my_experiment/
    ├── config_resolved.yaml
    ├── metrics/
    │   ├── train_metrics.csv      # Includes per-epoch temp_mean, temp_std, max/min
    │   └── test_metrics.csv       
    └── plots/
        ├── train_losses.png
        ├── temperatures_distribution_box.png  # Tracks IQR/variance over epochs
        ├── temperatures_progression.png       # Tracks the top 10 most scaled heads
        ├── temperatures_final_bar.png
        └── temperatures_heatmap_sorted.png

```

## Loss Function

The total objective function optimizes both the task and the CA3 alignment:

```text
L_total = [β * CE_clean + (1-β) * CE_adv] + λ_attn * L_align + λ_trades * L_trades

```

Where:

* **CE**: Cross-entropy on clean and/or adversarial examples based on `ce_mode`
* **L_align**: The CA3 penalty (KL or JS divergence) between clean and adversarial attention maps
* **L_trades**: KL(p_adv || p_clean) on logits to ensure prediction consistency

## References

* ASTS Methodology: *Adaptive Self-Attention Temperature Scaling for Robust Vision Transformers* (2026).
* TRADES: [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573) (Zhang et al., 2019)
* AutoAttack: [Reliable evaluation of adversarial robustness](https://arxiv.org/abs/2003.01690)



```
