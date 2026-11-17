"""ArcFace head (additive angular margin) for the control encoder."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class ArcFaceHead(torch.nn.Module):
    def __init__(self, in_features: int, n_classes: int, s: float = 30.0, m: float = 0.3):
        super().__init__()
        self.s, self.m = float(s), float(m)
        self.weight = torch.nn.Parameter(torch.empty(n_classes, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float())).clamp(-1.0, 1.0)
        sin = torch.sqrt((1.0 - cos * cos).clamp_min(1e-12))
        phi = cos * math.cos(self.m) - sin * math.sin(self.m)
        phi = torch.where(cos > math.cos(math.pi - self.m), phi, cos - math.sin(math.pi - self.m) * self.m)
        one_hot = F.one_hot(labels, cos.shape[1]).to(cos.dtype)
        return self.s * (one_hot * phi + (1.0 - one_hot) * cos)
