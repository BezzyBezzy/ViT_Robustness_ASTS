from __future__ import annotations

import os
import csv
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import (
    compute_classification_metrics,
    plot_curves,
    plot_temperature_heatmap,
    plot_final_temperature_bar,
    plot_temperature_progression,
    plot_temperature_distribution,
    plot_sorted_temperature_heatmap
)


def build_epoch_attack_plan(train_attack_schedule, total_epochs: int, logger=None):
    """
    Build a list of Attack objects, one per epoch.
    -1 means use for all remaining epochs.
    """
    plan = []
    used = 0

    for item in train_attack_schedule:
        atk = item["attack"]
        e = int(item.get("epochs", -1))

        if e == -1:
            rem = total_epochs - used
            if rem > 0:
                plan.extend([atk] * rem)
                used += rem
            if logger:
                logger.info(f"Attack plan: using {atk.name} for remaining {rem} epochs (epochs=-1).")
            break

        take = min(e, total_epochs - used)
        if take > 0:
            plan.extend([atk] * take)
            used += take
        if used >= total_epochs:
            break

    if used < total_epochs:
        last = plan[-1] if plan else train_attack_schedule[-1]["attack"]
        rem = total_epochs - used
        plan.extend([last] * rem)
        if logger:
            logger.info(f"Attack plan shorter than training; repeating {last.name} for last {rem} epochs.")

    return plan


def attention_kl(attn_clean_list: List[torch.Tensor], attn_adv_list: List[torch.Tensor], 
                 eps: float = 1e-8, layer_weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    Compute KL divergence between clean and adversarial attention distributions.
    
    Args:
        attn_clean_list: List of attention tensors (B, H, T, T) per layer
        attn_adv_list: List of attention tensors (B, H, T, T) per layer  
        eps: Small value for numerical stability
        layer_weights: Optional per-layer weights (later layers can be weighted more)
    """
    if len(attn_clean_list) != len(attn_adv_list):
        raise ValueError(f"Attention list length mismatch: {len(attn_clean_list)} vs {len(attn_adv_list)}")
    
    if not attn_clean_list:
        return torch.tensor(0.0)
    
    num_layers = len(attn_clean_list)
    
    # Default to uniform weights if not provided
    if layer_weights is None:
        layer_weights = [1.0] * num_layers
    
    total_kl = 0.0
    total_weight = sum(layer_weights)
    
    for layer_idx, (P, Q) in enumerate(zip(attn_clean_list, attn_adv_list)):
        # Clamp and renormalize
        P = torch.clamp(P, min=eps, max=1.0)
        Q = torch.clamp(Q, min=eps, max=1.0)
        P = P / P.sum(dim=-1, keepdim=True)
        Q = Q / Q.sum(dim=-1, keepdim=True)
        
        # Compute KL divergence
        kl = torch.sum(P * (torch.log(P) - torch.log(Q)), dim=-1).mean()
        total_kl = total_kl + layer_weights[layer_idx] * kl
    
    return total_kl / total_weight


def temperature_l2_reg(temps: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    return torch.mean((temps - target) ** 2)


def attention_js(attn_clean_list: List[torch.Tensor], attn_adv_list: List[torch.Tensor], 
                 eps: float = 1e-8, layer_weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence between clean and adversarial attention distributions.
    Symmetric alternative to KL divergence: JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
    
    Args:
        layer_weights: Optional per-layer weights (later layers can be weighted more)
    """
    if len(attn_clean_list) != len(attn_adv_list):
        raise ValueError(f"Attention list length mismatch: {len(attn_clean_list)} vs {len(attn_adv_list)}")
    
    if not attn_clean_list:
        return torch.tensor(0.0)
    
    num_layers = len(attn_clean_list)
    
    # Default to uniform weights if not provided
    if layer_weights is None:
        layer_weights = [1.0] * num_layers
    
    total_js = 0.0
    total_weight = sum(layer_weights)
    
    for layer_idx, (P, Q) in enumerate(zip(attn_clean_list, attn_adv_list)):
        # Clamp and renormalize
        P = torch.clamp(P, min=eps, max=1.0)
        Q = torch.clamp(Q, min=eps, max=1.0)
        P = P / P.sum(dim=-1, keepdim=True)
        Q = Q / Q.sum(dim=-1, keepdim=True)
        
        # Compute midpoint distribution
        M = 0.5 * (P + Q)
        
        # Compute KL divergences
        kl_pm = torch.sum(P * (torch.log(P) - torch.log(M)), dim=-1).mean()
        kl_qm = torch.sum(Q * (torch.log(Q) - torch.log(M)), dim=-1).mean()
        
        # JS = 0.5 * (KL(P||M) + KL(Q||M))
        js = 0.5 * (kl_pm + kl_qm)
        total_js = total_js + layer_weights[layer_idx] * js
    
    return total_js / total_weight


def attention_entropy(attn_list: List[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """
    Compute mean entropy of attention distributions across all layers.
    Higher entropy = more uniform attention (less peaked).
    """
    if not attn_list:
        return torch.tensor(0.0)
    
    total_entropy = 0.0
    for attn in attn_list:  # attn: [B, H, T, T]
        # Entropy per query position: -sum(p * log(p))
        H = -torch.sum(attn * torch.log(attn + eps), dim=-1)  # [B, H, T]
        total_entropy = total_entropy + H.mean()
    
    return total_entropy / len(attn_list)


def logit_kl_loss(logits_clean: torch.Tensor, logits_adv: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    TRADES-style KL divergence loss between clean and adversarial logits.
    Encourages adversarial logits to match clean logits distribution.
    """
    p_clean = F.softmax(logits_clean.detach() / temperature, dim=-1)
    log_p_adv = F.log_softmax(logits_adv / temperature, dim=-1)
    return F.kl_div(log_p_adv, p_clean, reduction='batchmean') * (temperature * temperature)


@dataclass
class ObjectiveConfig:
    lambda_attn: float
    ce_mode: str
    beta_clean: float
    temp_reg_enabled: bool
    temp_reg_weight: float
    temp_reg_target: float
    temp_reg_clamp_min: float
    temp_reg_clamp_max: float
    # New TRADES-style logit KL loss
    trades_enabled: bool = False
    trades_lambda: float = 6.0
    trades_temperature: float = 1.0
    # Attention divergence mode
    attn_divergence_mode: str = "kl"  # "kl" or "js"
    # Attention entropy regularization
    attn_entropy_enabled: bool = False
    attn_entropy_weight: float = 0.1
    attn_entropy_target: float = 2.0  # Target entropy (log(num_tokens) for uniform)
    # Per-layer attention weighting
    layer_weighting: str = "uniform"  # "uniform", "linear", "exponential"
    layer_weight_base: float = 1.5  # Base for exponential weighting


def get_layer_weights(num_layers: int, mode: str = "uniform", base: float = 1.5) -> List[float]:
    """
    Compute per-layer weights for attention divergence loss.
    
    Args:
        num_layers: Number of transformer layers
        mode: "uniform" (all 1.0), "linear" (1,2,3,...), "exponential" (base^0, base^1, ...)
        base: Base for exponential weighting
    
    Returns:
        List of weights (normalized so sum = num_layers for comparable magnitude)
    """
    if mode == "uniform":
        return [1.0] * num_layers
    elif mode == "linear":
        # Later layers get higher weight: 1, 2, 3, ..., num_layers
        weights = [(i + 1) for i in range(num_layers)]
    elif mode == "exponential":
        # Exponential growth: base^0, base^1, ..., base^(n-1)
        weights = [base ** i for i in range(num_layers)]
    else:
        raise ValueError(f"Unknown layer_weighting mode: {mode}")
    
    # Normalize so sum = num_layers (keeps overall loss magnitude similar)
    total = sum(weights)
    return [w * num_layers / total for w in weights]


def train_step(
    model, 
    optimizer, 
    attack, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    cfg: ObjectiveConfig, 
    grad_clip_norm: float = 0.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single training step with adversarial training.
    
    Returns:
        loss_total, loss_ce, loss_attn_div, loss_temp, loss_trades, loss_entropy
    """
    device = x.device
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    
    # Attack generation (needs gradients, runs outside autocast for stability)
    x_adv = attack.generate(model, x, y)

    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        logits_clean, attn_clean = model(x, return_attn=True)
        attn_clean = [a.detach() for a in attn_clean]

        logits_adv, attn_adv = model(x_adv, return_attn=True)

        # Cross-entropy loss (clean, adversarial, or mixed)
        loss_ce_clean = F.cross_entropy(logits_clean, y.to(logits_clean.device))
        loss_ce_adv = F.cross_entropy(logits_adv, y.to(logits_adv.device))

        if cfg.ce_mode == "adv_only":
            loss_ce = loss_ce_adv
        elif cfg.ce_mode == "clean_only":
            loss_ce = loss_ce_clean
        elif cfg.ce_mode == "mixed":
            loss_ce = cfg.beta_clean * loss_ce_clean + (1.0 - cfg.beta_clean) * loss_ce_adv
        else:
            raise ValueError(f"Unknown ce_mode: {cfg.ce_mode}")

        # Attention divergence loss (KL or JS) with optional per-layer weighting
        num_layers = len(attn_clean)
        layer_weights = get_layer_weights(num_layers, cfg.layer_weighting, cfg.layer_weight_base)
        
        if cfg.attn_divergence_mode == "js":
            loss_attn_div = attention_js(attn_clean, attn_adv, layer_weights=layer_weights)
        else:
            loss_attn_div = attention_kl(attn_clean, attn_adv, layer_weights=layer_weights)

        # Temperature regularization
        loss_temp = torch.tensor(0.0, device=loss_ce.device)
        if cfg.temp_reg_enabled:
            _, temps = model.temperatures_flat()
            if temps.numel() > 0:
                temps = temps.to(loss_ce.device).clamp(cfg.temp_reg_clamp_min, cfg.temp_reg_clamp_max)
                loss_temp = temperature_l2_reg(temps, target=cfg.temp_reg_target) * cfg.temp_reg_weight

        # TRADES-style logit KL loss (encourages logit consistency)
        loss_trades = torch.tensor(0.0, device=loss_ce.device)
        if cfg.trades_enabled:
            loss_trades = logit_kl_loss(logits_clean, logits_adv, temperature=cfg.trades_temperature) * cfg.trades_lambda

        # Attention entropy regularization (prevents attention collapse)
        loss_entropy = torch.tensor(0.0, device=loss_ce.device)
        if cfg.attn_entropy_enabled:
            adv_entropy = attention_entropy(attn_adv)
            # Penalize deviation from target entropy
            loss_entropy = cfg.attn_entropy_weight * (adv_entropy - cfg.attn_entropy_target).abs()

        loss_total = loss_ce + cfg.lambda_attn * loss_attn_div + loss_temp + loss_trades + loss_entropy

    optimizer.zero_grad(set_to_none=True)
    
    if scaler is not None:
        scaler.scale(loss_total).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_total.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

    return (
        loss_total.detach(), 
        loss_ce.detach(), 
        loss_attn_div.detach(), 
        loss_temp.detach(),
        loss_trades.detach(),
        loss_entropy.detach()
    )


def evaluate(model, loader: DataLoader, num_classes: int, attack=None, desc: str = "Evaluating"):
    model.eval()
    t0 = time.time()

    y_true, y_pred = [], []
    n = 0
    device = next(model.parameters()).device

    for bi, (x, y) in enumerate(tqdm(loader, desc=desc)):
        x = x.to(device)
        y = y.to(device)

        if attack is not None and getattr(attack, "name", "none") != "none":
            with torch.enable_grad():
                x = attack.generate(model, x, y)

        with torch.no_grad():
            logits, _ = model(x, return_attn=False)
            pred = torch.argmax(logits, dim=-1)

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        n += y.numel()

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    metrics = compute_classification_metrics(y_true, y_pred, num_classes=num_classes)
    return metrics, (time.time() - t0), int(n)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Robust CSV writer:
    - Builds fieldnames from the UNION of keys across all rows (stable order).
    - Prevents DictWriter from crashing when later rows contain extra fields.
    """
    if not rows:
        return

    # Stable "first-seen" ordering of columns across all rows
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_checkpoint(path: str, model, optimizer, epoch: int, best_metric: float) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }, path)


def train_loop(
    run_dir: str,
    logger,
    model,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: DataLoader,
    num_classes: int,
    optimizer,
    train_attack_schedule,
    eval_attacks: List,
    obj_cfg: ObjectiveConfig,
    epochs: int,
    grad_clip_norm: float,
    log_every_steps: int,
    eval_every_epochs: int,
    save_every_epochs: int,
    eval_baseline_before_train: bool,
    use_amp: bool = False,
) -> None:
    device = next(model.parameters()).device
    logger.info(f"Device: {device}")
    logger.info(f"Injected attention modules: {getattr(model, 'num_injected_modules', 'unknown')}")
    
    # Initialize AMP GradScaler if enabled and on CUDA
    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        logger.info("Mixed precision (AMP) enabled with GradScaler")
    elif use_amp:
        logger.info("Mixed precision (AMP) enabled (no scaler on CPU/MPS)")

    epoch_attack_plan = build_epoch_attack_plan(train_attack_schedule, epochs, logger)
    logger.info("Per-epoch train attacks: " + ", ".join([f"e{i+1}:{a.name}" for i, a in enumerate(epoch_attack_plan[:min(epochs, 50)])]) + (" ..." if epochs > 50 else ""))

    # ---- Baseline evaluation BEFORE any training updates (epoch 0) ----
    run_baseline = bool(eval_baseline_before_train)
    if run_baseline:
        baseline_rows = []
        clean_m, clean_s, clean_n = evaluate(model, test_loader, num_classes=num_classes, attack=None, desc="Baseline (clean)")
        baseline_rows.append({
            "stage": "baseline", "attack": "clean", "accuracy": clean_m.accuracy, "precision_macro": clean_m.precision_macro,
            "recall_macro": clean_m.recall_macro, "f1_macro": clean_m.f1_macro, "precision_weighted": clean_m.precision_weighted,
            "recall_weighted": clean_m.recall_weighted, "f1_weighted": clean_m.f1_weighted, "seconds": clean_s, "num_samples": clean_n,
        })

        for idx, atk in enumerate(eval_attacks, start=1):
            logger.info(f"Evaluating baseline model on attack #{idx}/{len(eval_attacks)} started..")
            m, s, n = evaluate(model, test_loader, num_classes=num_classes, attack=atk, desc=f"Baseline ({atk.name})")
            baseline_rows.append({
                "stage": "baseline", "attack": atk.name, "accuracy": m.accuracy, "precision_macro": m.precision_macro,
                "recall_macro": m.recall_macro, "f1_macro": m.f1_macro, "precision_weighted": m.precision_weighted,
                "recall_weighted": m.recall_weighted, "f1_weighted": m.f1_weighted, "seconds": s, "num_samples": n,
            })
        _write_csv(os.path.join(run_dir, "metrics", "test_metrics_baseline.csv"), baseline_rows)
        logger.info("[BASELINE] wrote metrics/test_metrics_baseline.csv")

    train_rows, val_rows, test_rows = [], [], []
    train_loss_hist, train_ce_hist, train_kl_hist, train_trades_hist = [], [], [], []
    val_acc_hist = []
    temps_over_epochs, head_labels = [], None
    best_val_acc, global_step = -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        current_attack = epoch_attack_plan[epoch - 1]
        logger.info(f"[TRAIN] epoch={epoch} using attack={current_attack.name}")

        t0 = time.time()
        running = {"total": 0.0, "ce": 0.0, "attn_div": 0.0, "temp": 0.0, "trades": 0.0, "entropy": 0.0}
        count = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            loss_total, loss_ce, loss_attn_div, loss_temp, loss_trades, loss_entropy = train_step(
                model, optimizer, current_attack, x, y, obj_cfg, 
                grad_clip_norm, scaler=scaler, use_amp=use_amp
            )

            running["total"] += float(loss_total.item())
            running["ce"] += float(loss_ce.item())
            running["attn_div"] += float(loss_attn_div.item())
            running["temp"] += float(loss_temp.item())
            running["trades"] += float(loss_trades.item())
            running["entropy"] += float(loss_entropy.item())
            count += 1
            global_step += 1

            if global_step % log_every_steps == 0:
                log_msg = f"step={global_step} | loss={running['total']/count:.4f} | ce={running['ce']/count:.4f} | attn={running['attn_div']/count:.4f}"
                if obj_cfg.trades_enabled:
                    log_msg += f" | trades={running['trades']/count:.4f}"
                if obj_cfg.attn_entropy_enabled:
                    log_msg += f" | entropy={running['entropy']/count:.4f}"
                logger.info(log_msg)

        epoch_avg = {k: v / max(count, 1) for k, v in running.items()}
        
        # Get temperature statistics for this epoch
        labels, temps = model.temperatures_flat()
        temp_stats = {}
        if temps.numel() > 0:
            temp_stats = {
                "temp_mean": float(temps.mean().item()),
                "temp_std": float(temps.std().item()),
                "temp_min": float(temps.min().item()),
                "temp_max": float(temps.max().item()),
            }
        else:
            temp_stats = {"temp_mean": 1.0, "temp_std": 0.0, "temp_min": 1.0, "temp_max": 1.0}
        
        train_rows.append({
            "epoch": epoch, 
            "loss_total": epoch_avg["total"], 
            "loss_ce": epoch_avg["ce"], 
            "loss_attn_div": epoch_avg["attn_div"], 
            "loss_temp_reg": epoch_avg["temp"],
            "loss_trades": epoch_avg["trades"],
            "loss_entropy": epoch_avg["entropy"],
            **temp_stats,
            "seconds": time.time() - t0
        })
        train_loss_hist.append(epoch_avg["total"])
        train_ce_hist.append(epoch_avg["ce"])
        train_kl_hist.append(epoch_avg["attn_div"])
        train_trades_hist.append(epoch_avg["trades"])

        # Store temperature progression for plotting (reuse temps from stats)
        if head_labels is None: head_labels = labels
        if temps.numel() > 0: 
            temps_over_epochs.append(temps.numpy())
            logger.info(f"[TEMPS] epoch={epoch} mean={temp_stats['temp_mean']:.4f} std={temp_stats['temp_std']:.4f} min={temp_stats['temp_min']:.4f} max={temp_stats['temp_max']:.4f}")

        if val_loader is not None and (epoch % eval_every_epochs == 0):
            val_m, val_s, val_n = evaluate(model, val_loader, num_classes=num_classes, attack=None, desc=f"Val epoch {epoch}")
            val_rows.append({"epoch": epoch, "accuracy": val_m.accuracy, "f1_macro": val_m.f1_macro, "seconds": val_s, "num_samples": val_n})
            val_acc_hist.append(val_m.accuracy)
            logger.info(f"[VAL] epoch={epoch} acc={val_m.accuracy:.4f}")
            if val_m.accuracy > best_val_acc:
                best_val_acc = val_m.accuracy
                _save_checkpoint(os.path.join(run_dir, "checkpoints", "best.pt"), model, optimizer, epoch, best_val_acc)

        if epoch % save_every_epochs == 0:
            _save_checkpoint(os.path.join(run_dir, "checkpoints", "last.pt"), model, optimizer, epoch, best_val_acc)

    _write_csv(os.path.join(run_dir, "metrics", "train_metrics.csv"), train_rows)
    if val_rows: _write_csv(os.path.join(run_dir, "metrics", "val_metrics.csv"), val_rows)

    # Final test
    clean_m, clean_s, clean_n = evaluate(model, test_loader, num_classes=num_classes, attack=None, desc="Final test (clean)")
    # Make the "clean" row consistent with attacked rows (prevents mismatched columns and is nicer to analyze)
    test_rows.append({
        "attack": "clean",
        "accuracy": clean_m.accuracy,
        "precision_macro": clean_m.precision_macro,
        "recall_macro": clean_m.recall_macro,
        "f1_macro": clean_m.f1_macro,
        "precision_weighted": clean_m.precision_weighted,
        "recall_weighted": clean_m.recall_weighted,
        "f1_weighted": clean_m.f1_weighted,
        "seconds": clean_s,
        "num_samples": clean_n
    })

    for atk in eval_attacks:
        m, s, n = evaluate(model, test_loader, num_classes=num_classes, attack=atk, desc=f"Final test ({atk.name})")
        test_rows.append({
            "attack": atk.name,
            "accuracy": m.accuracy,
            "precision_macro": m.precision_macro,
            "recall_macro": m.recall_macro,
            "f1_macro": m.f1_macro,
            "precision_weighted": m.precision_weighted,
            "recall_weighted": m.recall_weighted,
            "f1_weighted": m.f1_weighted,
            "seconds": s,
            "num_samples": n
        })

    _write_csv(os.path.join(run_dir, "metrics", "test_metrics.csv"), test_rows)

    # Plots
    loss_dict = {"loss_attn": train_kl_hist, "loss_ce": train_ce_hist, "loss_total": train_loss_hist}
    if obj_cfg.trades_enabled and any(t > 0 for t in train_trades_hist):
        loss_dict["loss_trades"] = train_trades_hist
    plot_curves(
        os.path.join(run_dir, "plots", "train_losses.png"),
        "Training losses",
        loss_dict,
        ylabel="Loss"
    )
    if val_acc_hist:
        plot_curves(
            os.path.join(run_dir, "plots", "val_accuracy.png"),
            "Validation accuracy",
            {"val_acc": val_acc_hist},
            ylabel="Accuracy"
        )
    if temps_over_epochs and head_labels:
        temps_matrix = np.stack(temps_over_epochs, axis=0)  # (epochs, heads)

        plot_temperature_heatmap(
            os.path.join(run_dir, "plots", "temperatures_heatmap.png"),
            temps_matrix,
            head_labels
        )
        plot_temperature_progression(
            os.path.join(run_dir, "plots", "temperatures_progression.png"),
            temps_matrix,
            head_labels,
            top_n=10
        )
        plot_sorted_temperature_heatmap(
            os.path.join(run_dir, "plots", "temperatures_heatmap_sorted.png"),
            temps_matrix,
            head_labels,
            descending=True,  # hottest -> left
        )
        plot_final_temperature_bar(
            os.path.join(run_dir, "plots", "temperatures_final_bar.png"),
            temps_matrix,
            head_labels
        )
        plot_temperature_distribution(
            os.path.join(run_dir, "plots", "temperatures_distribution_box.png"),
            temps_matrix
        )
