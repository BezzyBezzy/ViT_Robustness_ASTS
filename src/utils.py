from __future__ import annotations

import os
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def setup_logger(log_path: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("asts")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def make_run_dir(output_root: str, experiment_name: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_root, f"{ts}_{experiment_name}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> ClassificationMetrics:
    # Vectorized confusion matrix computation
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)

    total = cm.sum()
    correct = int(np.trace(cm))
    accuracy = _safe_div(correct, int(total))

    # Vectorized precision/recall/f1 computation
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    supports = cm.sum(axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        precs = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        recs = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f1s = np.where((precs + recs) > 0, 2 * precs * recs / (precs + recs), 0.0)

    precision_macro = float(np.mean(precs))
    recall_macro = float(np.mean(recs))
    f1_macro = float(np.mean(f1s))

    weights = supports / supports.sum() if supports.sum() > 0 else np.ones_like(supports) / len(supports)
    precision_weighted = float(np.sum(precs * weights))
    recall_weighted = float(np.sum(recs * weights))
    f1_weighted = float(np.sum(f1s * weights))

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        f1_weighted=f1_weighted,
    )


def plot_curves(
    out_path: str,
    title: str,
    series: Dict[str, List[float]],
    xlabel: str = "Epoch",
    ylabel: str = "Value",
    tick_fontsize: int = 8,
    legend_order: Optional[List[str]] = None,
) -> None:
    plt.figure()
    ax = plt.gca()

    # Plot curves and keep handles so we can reorder legend later
    line_map = {}
    for name, values in series.items():
        (ln,) = ax.plot(range(1, len(values) + 1), values, label=name)
        line_map[name] = ln

    # Force integer epoch ticks + smaller font
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=tick_fontsize)

    # Light grid (vertical + horizontal)
    ax.grid(True, which="major", axis="both", alpha=0.18, linewidth=0.8)

    # Reordered legend
    if legend_order:
        handles = [line_map[n] for n in legend_order if n in line_map]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels)
    else:
        ax.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_temperature_heatmap(out_path: str, temps_matrix: np.ndarray, head_labels: List[str]) -> None:
    plt.figure(figsize=(max(8, temps_matrix.shape[1] * 0.25), max(4, temps_matrix.shape[0] * 0.35)))
    plt.imshow(temps_matrix, aspect="auto")
    plt.colorbar(label="Temperature")
    plt.yticks(range(temps_matrix.shape[0]), [f"{e+1}" for e in range(temps_matrix.shape[0])])
    plt.xticks(range(len(head_labels)), head_labels, rotation=90, fontsize=6)
    plt.title("Per-head temperature values over epochs")
    plt.xlabel("(Module,Head)")
    plt.ylabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_sorted_temperature_heatmap(
    out_path: str,
    temps_matrix: np.ndarray,
    head_labels: List[str],
    *,
    descending: bool = True,
    title: str = "Sorted heatmap (heads sorted by mean temperature)",
) -> Tuple[np.ndarray, List[str]]:
    """
    Plot a heatmap where columns (heads) are sorted by their mean temperature across epochs.

    Args:
        out_path: Path to save the figure.
        temps_matrix: Shape (num_epochs, num_heads).
        head_labels: List length = num_heads (e.g., ["(block0,head0)", ...]).
        descending: If True, hottest (highest mean) heads on the left.
        title: Plot title.

    Returns:
        order: np.ndarray of column indices used for sorting (len = num_heads).
        sorted_labels: head_labels reordered according to `order`.
    """
    if temps_matrix.ndim != 2:
        raise ValueError(f"temps_matrix must be 2D (epochs x heads), got shape {temps_matrix.shape}")
    if temps_matrix.shape[1] != len(head_labels):
        raise ValueError(
            f"head_labels length ({len(head_labels)}) must match num_heads ({temps_matrix.shape[1]})"
        )

    # Sort heads by mean temperature across epochs
    means = temps_matrix.mean(axis=0)  # (num_heads,)
    order = np.argsort(means)
    if descending:
        order = order[::-1]

    temps_sorted = temps_matrix[:, order]
    sorted_labels = [head_labels[i] for i in order]

    # Plot
    plt.figure(figsize=(max(8, temps_sorted.shape[1] * 0.25), max(4, temps_sorted.shape[0] * 0.35)))
    plt.imshow(temps_sorted, aspect="auto")
    plt.colorbar(label="Temperature")
    plt.yticks(range(temps_sorted.shape[0]), [f"{e+1}" for e in range(temps_sorted.shape[0])])
    plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=90, fontsize=6)
    plt.title(title)
    plt.xlabel("(Module,Head) sorted by mean temp")
    plt.ylabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_temperature_progression(
    out_path: str, 
    temps_matrix: np.ndarray, 
    head_labels: List[str], 
    top_n: int = 10
) -> None:
    """
    Plots the temperature progression over epochs for the top_n most dynamic heads.
    """
    num_epochs = temps_matrix.shape[0]
    num_heads = temps_matrix.shape[1]
    
    # Identify the top_n heads with the highest final temperature
    # (You could also use np.var(temps_matrix, axis=0) to find heads that change the most)
    final_temps = temps_matrix[-1, :]
    top_indices = np.argsort(final_temps)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    
    epochs = np.arange(1, num_epochs + 1)
    
    for idx in top_indices:
        plt.plot(
            epochs, 
            temps_matrix[:, idx], 
            label=head_labels[idx], 
            marker='o', 
            markersize=3, 
            linewidth=1.5
        )
    
    plt.title(f"Temperature Progression (Top {top_n} Heads)")
    plt.xlabel("Epoch")
    plt.ylabel("Temperature")
    
    # Force X-axis to show every epoch as an integer
    plt.xticks(epochs, fontsize=7) 
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_final_temperature_bar(
    out_path: str, 
    temps_matrix: np.ndarray, 
    head_labels: List[str]
) -> None:
    """
    Plots a bar chart of the temperature for every head at the last epoch.
    """
    final_temps = temps_matrix[-1, :]
    num_heads = len(head_labels)
    
    # Adjust width based on number of heads
    plt.figure(figsize=(max(10, num_heads * 0.2), 6))
    
    plt.bar(range(num_heads), final_temps, color='skyblue', edgecolor='navy', alpha=0.8)
    
    plt.title("Final Temperature per Head (Last Epoch)")
    plt.xlabel("Module/Head")
    plt.ylabel("Temperature")
    
    # Set x-ticks for every head with small font
    plt.xticks(range(num_heads), head_labels, rotation=90, fontsize=6)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_temperature_distribution(
    out_path: str, 
    temps_matrix: np.ndarray
) -> None:
    """
    Plots the distribution (box plot) of temperatures across all heads for each epoch.
    """
    num_epochs = temps_matrix.shape[0]
    epochs = np.arange(1, num_epochs + 1)
    
    plt.figure(figsize=(10, 6))
    
    # We transpose because boxplot expects a list of arrays (one per epoch)
    # 'labels' sets the x-axis integer values
    plt.boxplot(temps_matrix.T, labels=epochs)
    
    plt.title("Temperature Distribution per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Temperature Value")
    
    # Make epoch numbers smaller on x-axis
    plt.xticks(fontsize=7)
    
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()