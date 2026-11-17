"""Coordinatewise quadratic projection baseline.

This experimental baseline is not an implementation of published PolyProtect.
The module and public function names are retained as legacy identifiers.
"""
from __future__ import annotations

import numpy as np
from ijiss_btp.protect.linear import projection_matrix


def polyprotect_coefficients(dim_out: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate coefficients for the coordinatewise quadratic baseline."""
    rng = np.random.default_rng(seed + 999)
    # Non-zero polynomial coefficients
    a = rng.uniform(-1.0, 1.0, size=dim_out).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=dim_out).astype(np.float32)
    # Ensure non-trivial linear and quadratic terms
    a[np.abs(a) < 0.1] = 0.5
    b[np.abs(b) < 0.1] = 0.5
    return a, b


def polyprotect_protect(emb: np.ndarray, dim_out: int, seed: int) -> np.ndarray:
    """Project embeddings and apply independent coordinatewise quadratics.

    For u = emb @ Q, compute t = a*u**2 + b*u and normalize each row.
    This baseline does not implement PolyProtect's multivariate block mapping.
    """
    q = projection_matrix(emb.shape[1], dim_out, seed)
    u = emb @ q
    a, b = polyprotect_coefficients(dim_out, seed)
    t = a * (u ** 2) + b * u
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    return (t / np.where(norm == 0, 1.0, norm)).astype(np.float32)
