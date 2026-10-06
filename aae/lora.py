"""Minimal LoRA for frozen linear layers, and the merge that turns it back into plain weights."""
from __future__ import annotations

import re

import torch
from torch import nn


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen; B starts at zero so training starts from the original model."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, dtype=torch.float32, device=base.weight.device) / r)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.t() @ self.B.t()) * self.scale).to(x.dtype)

    def merged_weight(self) -> torch.Tensor:
        return (self.base.weight.float() + self.scale * (self.B @ self.A)).to(self.base.weight.dtype)


def apply(model: nn.Module, pattern: str, r: int, alpha: float) -> dict:
    """Freeze every parameter, then wrap each nn.Linear whose qualified name matches ``pattern``.

    Returns {qualified name: LoRALinear}."""
    for p in model.parameters():
        p.requires_grad_(False)
    rx = re.compile(pattern)
    wrapped = {}
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and rx.search(full):
                wrapped[full] = LoRALinear(child, r, alpha)
                setattr(mod, child_name, wrapped[full])
    if not wrapped:
        raise ValueError(f"LoRA pattern {pattern!r} matched no nn.Linear")
    return wrapped


@torch.no_grad()
def merged_state(wrapped: dict) -> dict:
    """{"<name>.weight": merged weight on the CPU}: what a checkpoint stores. Training can continue afterwards."""
    return {f"{name}.weight": m.merged_weight().detach().cpu().clone() for name, m in wrapped.items()}


class disabled:
    """Context manager that switches every adapter off, turning the wrapped model back into the original model."""

    def __init__(self, wrapped: dict):
        self.wrapped = wrapped

    def __enter__(self):
        self.saved = {k: m.scale for k, m in self.wrapped.items()}
        for m in self.wrapped.values():
            m.scale = 0.0

    def __exit__(self, *exc):
        for k, m in self.wrapped.items():
            m.scale = self.saved[k]
