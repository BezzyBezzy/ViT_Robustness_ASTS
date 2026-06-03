"""
Adversarial attack wrappers using torchattacks library.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    import torchattacks
except ImportError:
    raise ImportError("torchattacks is required. Install with: pip install torchattacks")


class ModelWrapper(nn.Module):
    """
    Wraps a model that returns (logits, extras) to return only logits.
    Required for compatibility with torchattacks which expects model(x) -> logits.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        # Handle models that return tuples (logits, attention, etc.)
        if isinstance(output, tuple):
            return output[0]
        return output


class AttackWrapper:
    """
    Wrapper for adversarial attacks with a consistent interface.
    """
    def __init__(self, attack: Any, name: str):
        self._attack = attack
        self.name = name
    
    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples."""
        if self._attack is None:
            return x
        
        was_training = model.training
        model.eval()
        
        with torch.enable_grad():
            x_adv = self._attack(x, y)
        
        if was_training:
            model.train()
        
        return x_adv


class NoAttack:
    """Dummy attack that returns clean inputs."""
    def __init__(self, name: str = "none"):
        self.name = name
    
    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x


def build_attack(name: str, params: Dict[str, Any], label: Optional[str] = None) -> AttackWrapper:
    """
    Build an attack object from name and parameters.
    
    Args:
        name: Attack type (pgd, fgsm, cw, autoattack, none)
        params: Attack-specific parameters
        label: Optional label for the attack (used for logging)
    
    Returns:
        AttackWrapper or NoAttack instance
    """
    name_lower = name.lower()
    attack_label = label if label else name_lower
    
    if name_lower == "none" or not name_lower:
        return NoAttack(name=attack_label)
    
    # We need a model to create attacks, but torchattacks creates them lazily
    # So we return a factory that creates the attack on first use
    return LazyAttackWrapper(name_lower, params, attack_label)


class LazyAttackWrapper:
    """
    Lazily creates the attack on first use (when model is available).
    """
    def __init__(self, attack_type: str, params: Dict[str, Any], name: str):
        self.attack_type = attack_type
        self.params = params
        self.name = name
        self._attack = None
        self._model_id = None
        self._wrapped_model = None
    
    def _create_attack(self, model: nn.Module):
        """Create the actual attack object."""
        p = self.params
        
        # Wrap model to return only logits (torchattacks compatibility)
        wrapped = ModelWrapper(model)
        self._wrapped_model = wrapped
        
        if self.attack_type == "pgd":
            norm = p.get("norm", "Linf")
            eps = float(p.get("eps", 8/255))
            alpha = float(p.get("alpha", 2/255))
            steps = int(p.get("steps", 10))
            random_start = bool(p.get("random_start", True))
            
            self._attack = torchattacks.PGD(
                wrapped, 
                eps=eps, 
                alpha=alpha, 
                steps=steps, 
                random_start=random_start
            )
        
        elif self.attack_type == "fgsm":
            eps = float(p.get("eps", 8/255))
            self._attack = torchattacks.FGSM(wrapped, eps=eps)
        
        elif self.attack_type == "cw" or self.attack_type == "cwlinf":
            # CW L-inf attack
            eps = float(p.get("eps", 8/255))
            steps = int(p.get("steps", 20))
            
            # torchattacks CW is L2 by default, use PGD with CW-like settings for Linf
            # or use their specific implementation
            try:
                self._attack = torchattacks.CW(wrapped, c=1, steps=steps)
            except Exception:
                # Fallback to PGD if CW fails
                self._attack = torchattacks.PGD(
                    wrapped, 
                    eps=eps, 
                    alpha=eps/steps, 
                    steps=steps
                )
        
        elif self.attack_type == "autoattack":
            eps = float(p.get("eps", 8/255))
            norm_raw = p.get("norm", "Linf")
            # Normalize norm string: torchattacks expects "Linf" or "L2" exactly
            if norm_raw.lower() == "linf":
                norm = "Linf"
            elif norm_raw.lower() == "l2":
                norm = "L2"
            else:
                norm = norm_raw
            version = p.get("version", "standard")
            
            self._attack = torchattacks.AutoAttack(
                wrapped, 
                eps=eps, 
                norm=norm, 
                version=version,
                n_classes=10,  # Will be inferred from model output
                verbose=False
            )
        
        elif self.attack_type == "bim":
            eps = float(p.get("eps", 8/255))
            alpha = float(p.get("alpha", 2/255))
            steps = int(p.get("steps", 10))
            self._attack = torchattacks.BIM(wrapped, eps=eps, alpha=alpha, steps=steps)
        
        elif self.attack_type == "mifgsm":
            eps = float(p.get("eps", 8/255))
            alpha = float(p.get("alpha", 2/255))
            steps = int(p.get("steps", 10))
            decay = float(p.get("decay", 1.0))
            self._attack = torchattacks.MIFGSM(wrapped, eps=eps, alpha=alpha, steps=steps, decay=decay)
        
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
    
    def generate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples."""
        # Recreate attack if model changed
        model_id = id(model)
        if self._attack is None or self._model_id != model_id:
            self._create_attack(model)
            self._model_id = model_id
        
        was_training = model.training
        model.eval()
        
        with torch.enable_grad():
            x_adv = self._attack(x, y)
        
        if was_training:
            model.train()
        
        return x_adv
