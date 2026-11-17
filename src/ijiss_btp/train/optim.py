"""Layer-wise learning-rate groups, warm-up cosine schedule and epoch selection for the control encoder."""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import torch

_NO_DECAY = ("cls_token", "mask_token", "position_embeddings")


def _layer_id(name: str, num_layers: int) -> int:
    if name.startswith("embeddings."):
        return 0
    m = re.match(r"encoder\.layer\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if name.startswith("layernorm."):
        return num_layers + 1
    raise ValueError(f"unexpected Dinov2 parameter name: {name}")


def vit_param_groups(named: Iterable[tuple[str, torch.nn.Parameter]], num_layers: int, lr: float,
                     layer_decay: float, weight_decay: float) -> list[dict]:
    groups: dict[tuple[int, bool], dict] = {}
    for name, p in named:
        lid = _layer_id(name, num_layers)
        decay = not (p.ndim <= 1 or any(k in name for k in _NO_DECAY))
        g = groups.setdefault((lid, decay), {"params": [], "lr": lr * layer_decay ** (num_layers + 1 - lid),
                                             "weight_decay": weight_decay if decay else 0.0})
        g["params"].append(p)
    for g in groups.values():
        g["base_lr"] = g["lr"]
    return [groups[k] for k in sorted(groups)]


def warmup_cosine(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def best_epoch(val_eers: Sequence[float]) -> int:
    best = None
    for i, v in enumerate(val_eers):
        if math.isfinite(v) and (best is None or v < val_eers[best]):
            best = i
    if best is None:
        raise ValueError("no finite validation EER")
    return best
