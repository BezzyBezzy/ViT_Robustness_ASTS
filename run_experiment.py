from __future__ import annotations

import argparse
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.utils import load_yaml, save_yaml, seed_everything, make_run_dir, setup_logger
from src.data import DataModule
from src.asts_model import ASTSModel, NormalizeSpec, TemperatureSpec, TrainableSpec, set_trainable
from src.attacks import build_attack
from src.train import ObjectiveConfig, train_loop


@dataclass
class AttackSpec:
    base_name: str
    unique_name: str
    params: Dict[str, Any]
    epochs: int  # -1 means "use for all remaining epochs"


def _normalize_train_attack_cfg(attacks_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Backward compatible:
    - old format: attacks.train.name + attacks.train.params
    - new format: attacks.train.attacks = [...]
    """
    train_cfg = attacks_cfg.get("train", {})
    if "attacks" in train_cfg:
        return train_cfg["attacks"]
    if "name" in train_cfg:
        return [{
            "name": train_cfg.get("name", "none"),
            "epochs": -1,
            "params": train_cfg.get("params", {}),
        }]
    return [{"name": "none", "epochs": -1, "params": {}}]


def _normalize_eval_attack_cfg(attacks_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    eval_cfg = attacks_cfg.get("eval", {})
    if not eval_cfg.get("enabled", False):
        return []
    return eval_cfg.get("attacks", [])


def _assign_unique_attack_names(train_list: List[Dict[str, Any]], eval_list: List[Dict[str, Any]]) -> Tuple[List[AttackSpec], List[AttackSpec]]:
    """
    Assign pgd_0, pgd_1, ... across BOTH train and eval, in CONFIG ORDER:
      train attacks (in order) THEN eval attacks (in order)
    """
    counters: Dict[str, int] = {}

    def mk_spec(d: Dict[str, Any]) -> AttackSpec:
        base = str(d.get("name", "none")).lower()
        idx = counters.get(base, 0)
        counters[base] = idx + 1

        unique = f"{base}_{idx}"
        params = d.get("params", {}) or {}
        epochs = int(d.get("epochs", -1))  
        return AttackSpec(base_name=base, unique_name=unique, params=params, epochs=epochs)

    train_specs = [mk_spec(d) for d in train_list]
    eval_specs = [mk_spec(d) for d in eval_list]
    return train_specs, eval_specs


def pick_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))

    device = pick_device(cfg["experiment"].get("device", "auto"))
    run_dir = make_run_dir(cfg["experiment"]["output_root"], cfg["experiment"]["name"])

    logger = setup_logger(os.path.join(run_dir, "logs", "train.log"))
    save_yaml(cfg, os.path.join(run_dir, "config_resolved.yaml"))
    logger.info(f"Run dir: {run_dir}")

    dm = DataModule(
        dataset_name=cfg["data"]["dataset"],
        data_root=os.path.join(os.getcwd(), "data_cache"),
        image_size=int(cfg["data"]["image_size"]),
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        val_split=float(cfg["data"]["val_split"]),
        imagenet_dir=cfg["data"].get("imagenet_dir", None),
        train_ratio=float(cfg["data"].get("train_ratio", 1.0)),
        eval_ratio=float(cfg["data"].get("eval_ratio", 1.0)),
        seed=int(cfg["experiment"]["seed"]),
    )
    dm.setup()
    
    logger.info(f"Train samples: {len(dm.train_ds)}, Val samples: {len(dm.val_ds) if dm.val_ds else 0}, Test samples: {len(dm.test_ds)}")

    norm = NormalizeSpec(
        mean=cfg["model"]["normalize"]["mean"],
        std=cfg["model"]["normalize"]["std"],
    )

    temp_spec = TemperatureSpec(
        parameterization=cfg["asts"]["parameterization"],
        init_temperature=float(cfg["asts"]["init_temperature"]),
        eps=float(cfg["asts"]["eps"]),
    )

    model = ASTSModel(
        hf_model_id=cfg["model"]["hf_model_id"],
        num_labels=int(cfg["model"]["num_labels"]),
        normalize=norm,
        asts_enabled=bool(cfg["asts"]["enabled"]),
        temp_spec=temp_spec,
        device=device,
    )
    
    # Optional torch.compile() for PyTorch 2.0+ speedup
    if cfg["training"].get("torch_compile", False):
        try:
            compile_mode = cfg["training"].get("torch_compile_mode", "reduce-overhead")
            model = torch.compile(model, mode=compile_mode)
            logger.info(f"torch.compile() enabled with mode={compile_mode}")
        except Exception as e:
            logger.warning(f"torch.compile() failed, continuing without: {e}")

    trainable_spec = TrainableSpec(
        train_temperatures=bool(cfg["trainable"]["train_temperatures"]),
        train_classifier_head=bool(cfg["trainable"]["train_classifier_head"]),
        train_layernorm=bool(cfg["trainable"]["train_layernorm"]),
        train_backbone=bool(cfg["trainable"]["train_backbone"]),
    )
    trainable_params = set_trainable(model, trainable_spec)

    logger.info(f"Trainable tensors: {len(trainable_params)}")
    logger.info(f"Trainable scalars: {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    temp_reg = cfg["objective"]["temperature_reg"]
    
    # New TRADES config
    trades_cfg = cfg["objective"].get("trades", {})
    trades_enabled = trades_cfg.get("enabled", False)
    trades_lambda = float(trades_cfg.get("lambda", 6.0))
    trades_temperature = float(trades_cfg.get("temperature", 1.0))
    
    # New attention divergence config
    attn_divergence_mode = str(cfg["objective"].get("attn_divergence_mode", "kl"))
    
    # New attention entropy config
    attn_entropy_cfg = cfg["objective"].get("attn_entropy", {})
    attn_entropy_enabled = attn_entropy_cfg.get("enabled", False)
    attn_entropy_weight = float(attn_entropy_cfg.get("weight", 0.1))
    attn_entropy_target = float(attn_entropy_cfg.get("target", 2.0))
    
    # Per-layer weighting config
    layer_weighting_cfg = cfg["objective"].get("layer_weighting", {})
    layer_weighting = str(layer_weighting_cfg.get("mode", "uniform"))
    layer_weight_base = float(layer_weighting_cfg.get("base", 1.5))
    
    obj_cfg = ObjectiveConfig(
        lambda_attn=float(cfg["objective"]["lambda_attn"]),
        ce_mode=str(cfg["objective"]["ce_mode"]),
        beta_clean=float(cfg["objective"]["beta_clean"]),
        temp_reg_enabled=bool(temp_reg["enabled"]),
        temp_reg_weight=float(temp_reg["weight"]),
        temp_reg_target=float(temp_reg["target"]),
        temp_reg_clamp_min=float(temp_reg["clamp_min"]),
        temp_reg_clamp_max=float(temp_reg["clamp_max"]),
        trades_enabled=trades_enabled,
        trades_lambda=trades_lambda,
        trades_temperature=trades_temperature,
        attn_divergence_mode=attn_divergence_mode,
        attn_entropy_enabled=attn_entropy_enabled,
        attn_entropy_weight=attn_entropy_weight,
        attn_entropy_target=attn_entropy_target,
        layer_weighting=layer_weighting,
        layer_weight_base=layer_weight_base,
    )

    # Build attacks with unique names
    train_list = _normalize_train_attack_cfg(cfg["attacks"])
    eval_list = _normalize_eval_attack_cfg(cfg["attacks"])
    train_specs, eval_specs = _assign_unique_attack_names(train_list, eval_list)

    train_attack_schedule = []
    for s in train_specs:
        atk = build_attack(s.base_name, s.params, label=s.unique_name)
        train_attack_schedule.append({"attack": atk, "epochs": s.epochs, "name": s.unique_name})

    eval_attacks = []
    for s in eval_specs:
        eval_attacks.append(build_attack(s.base_name, s.params, label=s.unique_name))

    logger.info("Train attack schedule (in order): " + ", ".join([f"{x['name']}@{x['epochs']}" for x in train_attack_schedule]))
    logger.info("Eval attacks (in order): " + ", ".join([a.name for a in eval_attacks]))

    train_loop(
        run_dir=run_dir,
        logger=logger,
        model=model,
        train_loader=dm.train_loader(),
        val_loader=dm.val_loader(),
        test_loader=dm.test_loader(),
        num_classes=int(cfg["model"]["num_labels"]),
        optimizer=optimizer,
        train_attack_schedule=train_attack_schedule,
        eval_attacks=eval_attacks,
        obj_cfg=obj_cfg,
        epochs=int(cfg["training"]["epochs"]),
        grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
        log_every_steps=int(cfg["training"]["log_every_steps"]),
        eval_every_epochs=int(cfg["training"]["eval_every_epochs"]),
        save_every_epochs=int(cfg["training"]["save_every_epochs"]),
        eval_baseline_before_train=bool(cfg["training"].get("eval_baseline_before_train", True)),
        use_amp=bool(cfg["training"].get("use_amp", False)),
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()