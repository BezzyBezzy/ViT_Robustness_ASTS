from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageClassification


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    # softplus^{-1}(x) = log(exp(x) - 1)
    return torch.log(torch.exp(x) - 1.0)


@dataclass
class TemperatureSpec:
    parameterization: str  # exp|softplus
    init_temperature: float
    eps: float


class HeadTemperature(nn.Module):
    """
    Per-head learnable temperature.

    We keep an unconstrained parameter tau and map to a positive temperature T:
      - exp:     T = exp(tau)
      - softplus T = softplus(tau) + eps
    """
    def __init__(self, num_heads: int, spec: TemperatureSpec):
        super().__init__()
        self.spec = spec
        init_T = float(spec.init_temperature)

        if spec.parameterization == "exp":
            init_tau = torch.log(torch.tensor(init_T))
        elif spec.parameterization == "softplus":
            init_tau = inverse_softplus(torch.tensor(init_T - spec.eps).clamp_min(1e-6))
        else:
            raise ValueError(f"Unknown parameterization: {spec.parameterization}")

        self.tau = nn.Parameter(init_tau.repeat(num_heads))

    def temperature(self) -> torch.Tensor:
        if self.spec.parameterization == "exp":
            return torch.exp(self.tau)
        return F.softplus(self.tau) + self.spec.eps


def _is_vit_like_attention(m: nn.Module) -> bool:
    # Require Q/K/V and some indication of multi-head structure.
    has_qkv = all(hasattr(m, a) for a in ["query", "key", "value"])
    has_heads = hasattr(m, "num_attention_heads") or hasattr(m, "num_heads")
    # Dropout name varies across HF modules; we’ll handle multiple possibilities in patch.
    return bool(has_qkv and has_heads)



def _is_swin_attention(m: nn.Module) -> bool:
    # HF Swin attention module commonly has qkv and num_heads.
    return hasattr(m, "qkv") and hasattr(m, "num_heads")


def _patch_vit_attention_forward(attn_module: nn.Module) -> None:
    """
    Robust patch for HF ViT/DeiT-style self-attention modules.

    - Does NOT assume attn_module.dropout exists (might be dropout/attn_drop/attention_dropout)
    - Infers num_heads and head_dim dynamically from tensor shapes
    - Stores latest_attention (pre-dropout) for KL loss
    """
    if getattr(attn_module, "_asts_patched", False):
        return

    # Pick a dropout module if present (name differs across architectures/versions)
    drop = getattr(attn_module, "dropout", None)
    if drop is None:
        drop = getattr(attn_module, "attn_drop", None)
    if drop is None:
        drop = getattr(attn_module, "attention_dropout", None)
    # Dropout might not exist at all in some variants; in that case we just use identity.
    if drop is None:
        drop = nn.Identity()

    def get_num_heads() -> int:
        nh = getattr(attn_module, "num_attention_heads", None)
        if nh is None:
            nh = getattr(attn_module, "num_heads", None)
        if nh is None and hasattr(attn_module, "asts_temp"):
            nh = int(attn_module.asts_temp.tau.numel())
        if nh is None:
            raise RuntimeError("Cannot infer num_heads for attention module.")
        return int(nh)

    def forward(hidden_states, head_mask=None, output_attentions=False):
        # hidden_states: [B, T, D]
        B, T, _ = hidden_states.shape

        # Project to Q,K,V: [B, T, H*Hd] (standard HF format)
        q = attn_module.query(hidden_states)
        k = attn_module.key(hidden_states)
        v = attn_module.value(hidden_states)

        H = get_num_heads()
        if q.shape[-1] % H != 0:
            raise RuntimeError(f"Query last dim {q.shape[-1]} not divisible by num_heads {H}.")

        Hd = q.shape[-1] // H
        scale = 1.0 / math.sqrt(float(Hd))

        # Reshape to [B, H, T, Hd]
        q = q.view(B, T, H, Hd).permute(0, 2, 1, 3)
        k = k.view(B, T, H, Hd).permute(0, 2, 1, 3)
        v = v.view(B, T, H, Hd).permute(0, 2, 1, 3)

        # Attention logits: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        # Per-head temperature: [H]
        Tvec = attn_module.asts_temp.temperature().to(dtype=scores.dtype, device=scores.device)
        scores = scores / Tvec.view(1, -1, 1, 1)

        probs = torch.softmax(scores, dim=-1)

        if head_mask is not None:
            probs = probs * head_mask

        # Store for ASTS KL (pre-dropout)
        attn_module.latest_attention = probs

        probs_drop = drop(probs)

        # Context: [B, H, T, Hd] -> [B, T, H*Hd]
        ctx = torch.matmul(probs_drop, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, H * Hd)

        if output_attentions:
            return (ctx, probs)
        return (ctx, None)

    attn_module.forward = forward
    attn_module._asts_patched = True


def _patch_swin_attention_forward(attn_module: nn.Module) -> None:
    if getattr(attn_module, "_asts_patched", False):
        return

    def forward(hidden_states, mask=None, output_attentions=False):
        B_, N, C = hidden_states.shape
        qkv = attn_module.qkv(hidden_states).reshape(B_, N, 3, attn_module.num_heads, C // attn_module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * attn_module.scale
        attn = (q @ k.transpose(-2, -1))

        T = attn_module.asts_temp.temperature().to(attn.dtype).to(attn.device)
        attn = attn / T.view(1, -1, 1, 1)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, attn_module.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, attn_module.num_heads, N, N)

        attn = torch.softmax(attn, dim=-1)
        attn_module.latest_attention = attn
        attn = attn_module.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)

        if output_attentions:
            return (x, attn)
        return (x,)

    attn_module.forward = forward
    attn_module._asts_patched = True


def inject_asts_temperatures(model: nn.Module, spec: TemperatureSpec) -> int:
    injected = 0
    model_device = next(model.parameters()).device

    for _, m in model.named_modules():
        if _is_vit_like_attention(m):
            # infer num heads
            num_heads = getattr(m, "num_attention_heads", None) or getattr(m, "num_heads", None)
            if num_heads is None:
                continue
            if hasattr(m, "asts_temp"):
                continue

            m.asts_temp = HeadTemperature(int(num_heads), spec).to(model_device)
            _patch_vit_attention_forward(m)
            injected += 1

        elif _is_swin_attention(m):
            if hasattr(m, "asts_temp"):
                continue
            m.asts_temp = HeadTemperature(int(m.num_heads), spec).to(model_device)
            _patch_swin_attention_forward(m)
            injected += 1

    return injected


@dataclass
class NormalizeSpec:
    mean: List[float]
    std: List[float]


class ASTSModel(nn.Module):
    """
    Wraps a HF image classifier model:
      - expects pixel-space inputs in [0,1]
      - normalizes internally
      - can return attention matrices collected from patched modules
    """
    def __init__(
        self,
        hf_model_id: str,
        num_labels: int,
        normalize: NormalizeSpec,
        asts_enabled: bool,
        temp_spec: TemperatureSpec,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.normalize_spec = normalize
        
        # Register normalization tensors as buffers (avoids recreation per forward)
        self.register_buffer(
            "_norm_mean", 
            torch.tensor(normalize.mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "_norm_std", 
            torch.tensor(normalize.std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        )

        self.model = AutoModelForImageClassification.from_pretrained(
            hf_model_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        ).to(self.device)

        self.num_injected_modules = 0
        self._asts_modules: List[nn.Module] = []  # Cache patched modules
        if asts_enabled:
            self.num_injected_modules = inject_asts_temperatures(self.model, temp_spec)
            # Cache references to patched modules for fast access
            for m in self.model.modules():
                if hasattr(m, "asts_temp"):
                    self._asts_modules.append(m)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._norm_mean.to(x.dtype)) / self._norm_std.to(x.dtype)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = x.to(self.device)
        x_norm = self.normalize(x)

        out = self.model(pixel_values=x_norm, output_attentions=False)
        logits = out.logits

        if not return_attn:
            return logits, None

        # Use cached module list for fast attention collection
        attn_list: List[torch.Tensor] = []
        for m in self._asts_modules:
            if hasattr(m, "latest_attention"):
                attn_list.append(m.latest_attention)
        return logits, attn_list

    def temperatures_flat(self) -> Tuple[List[str], torch.Tensor]:
        labels: List[str] = []
        temps: List[torch.Tensor] = []
        # Use cached module list for fast temperature access
        for midx, m in enumerate(self._asts_modules):
            T = m.asts_temp.temperature().detach().cpu()
            for h in range(T.numel()):
                labels.append(f"m{midx}_h{h}")
            temps.append(T)
        if not temps:
            return labels, torch.empty(0)
        return labels, torch.cat(temps, dim=0)


@dataclass
class TrainableSpec:
    train_temperatures: bool
    train_classifier_head: bool
    train_layernorm: bool
    train_backbone: bool


def set_trainable(model: nn.Module, spec: TrainableSpec) -> List[nn.Parameter]:
    """
    Freeze/unfreeze parameters to isolate the temperature effect.

    Typical ablation:
      - temps only
      - temps + classifier head
      - temps + head + LayerNorm
      - full fine-tune (train_backbone=True)
    """
    trainable: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        n = name.lower()

        is_temp = ("asts_temp" in n) or (n.endswith(".tau"))
        is_head = ("classifier" in n) or (".head." in n)
        is_ln = ("layernorm" in n) or ("layer_norm" in n) or (".ln" in n)

        if spec.train_backbone:
            p.requires_grad = True
            trainable.append(p)
            continue

        should_train = False
        if spec.train_temperatures and is_temp:
            should_train = True
        if spec.train_classifier_head and is_head:
            should_train = True
        if spec.train_layernorm and is_ln:
            should_train = True

        p.requires_grad = should_train
        if should_train:
            trainable.append(p)

    return trainable
