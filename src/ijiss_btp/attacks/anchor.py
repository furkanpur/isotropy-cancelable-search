"""Collusion (anchor) attacks T1/T1': decode stolen templates from r known (embedding, template) pairs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RidgeDecoder:
    mf: np.ndarray
    my: np.ndarray
    w: np.ndarray
    lam: float

    def __call__(self, f: np.ndarray) -> np.ndarray:
        return (f - self.mf) @ self.w + self.my


def fit_ridge_gcv(f: np.ndarray, y: np.ndarray, n_grid: int = 16) -> RidgeDecoder:
    """Ridge regression; lambda chosen by generalized cross-validation on the anchors only."""
    mf, my = f.mean(0), y.mean(0)
    fc, yc = f - mf, y - my
    u, s, vt = np.linalg.svd(fc, full_matrices=False)
    uty = u.T @ yc
    s2 = s ** 2
    n = len(f)
    best: tuple[float, float] | None = None
    for lam in max(float(s2.max()), 1e-12) * np.logspace(-6, 1, n_grid):
        fac = s2 / (s2 + lam)
        resid = yc - u @ (fac[:, None] * uty)
        g = float(np.mean(resid ** 2)) / max((1 - fac.sum() / n) ** 2, 1e-12)
        if best is None or g < best[0]:
            best = (g, float(lam))
    lam = best[1]
    w = vt.T @ ((s / (s2 + lam))[:, None] * uty)
    return RidgeDecoder(mf, my, w, lam)


def fit_projection(f_anchor: np.ndarray, z_known: np.ndarray) -> Callable:
    """Minimum-norm least-squares map without intercept: W = pinv(F) Z (the optimal T1 decoder of the theory)."""
    w = np.linalg.pinv(f_anchor) @ z_known
    return lambda f: f @ w


def fit_mlp(f: np.ndarray, y: np.ndarray, seed: int, hidden=(512,), max_iter: int = 400) -> Callable:
    """Adaptive non-linear decoder (scikit-learn MLP) on standardized features and targets."""
    fs, ys = StandardScaler().fit(f), StandardScaler().fit(y)
    model = MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-3, early_stopping=len(f) >= 50,
                         max_iter=max_iter, random_state=seed)
    model.fit(fs.transform(f), ys.transform(y))
    return lambda q: ys.inverse_transform(model.predict(fs.transform(q)))


def anchor_decode(f_anchor: np.ndarray, z_known: np.ndarray, f_target: np.ndarray,
                  decoder: str = "ridge", seed: int = 0) -> np.ndarray:
    if decoder == "ridge":
        return fit_ridge_gcv(f_anchor, z_known)(f_target)
    if decoder == "proj":
        return fit_projection(f_anchor, z_known)(f_target)
    if decoder == "mlp":
        return fit_mlp(f_anchor, z_known, seed)(f_target)
    raise ValueError(f"unknown decoder {decoder!r}")
