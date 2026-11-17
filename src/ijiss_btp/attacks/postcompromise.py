"""Key-compromise attacks (T2): the attacker holds the shared key and the stolen templates."""
from __future__ import annotations

import numpy as np


def lrp_lmmse_inverse(t: np.ndarray, key: np.ndarray, cov_z: np.ndarray | None = None) -> np.ndarray:
    """LMMSE preimage z = t (K^T S K)^-1 K^T S; minimum-norm t K^T when no prior covariance is given."""
    if cov_z is None:
        return t @ key.T
    a = key.T @ cov_z @ key
    return t @ np.linalg.solve(a, key.T @ cov_z)


def biohash_inverse(bits: np.ndarray, key: np.ndarray, sigma_u: float) -> np.ndarray:
    """E[u | sign(u)] = sqrt(2/pi) sigma_u sign(u) for Gaussian projections, mapped back with K^T.

    With orthonormal key columns the result reproduces every bit exactly.
    """
    return (bits * (np.sqrt(2.0 / np.pi) * sigma_u)) @ key.T


def iom_preimage(codes: np.ndarray, key: np.ndarray, block: int, n_iter: int = 500, lr: float = 0.1,
                 margin: float = 0.05, z0: np.ndarray | None = None) -> np.ndarray:
    """Hinge-loss gradient descent to a z whose block argmax under `key` reproduces `codes`."""
    n, nb = codes.shape
    m, k = key.shape
    z = np.zeros((n, m)) if z0 is None else np.array(z0, dtype=float)
    code_cols = np.arange(nb)[None, :] * block + codes
    kt = key.T
    rows, blocks = np.arange(n)[:, None], np.arange(nb)[None, :]
    for _ in range(n_iter):
        u = z @ key
        win = np.take_along_axis(u, code_cols, axis=1)
        viol = (u.reshape(n, nb, block) + margin - win[..., None]) > 0
        viol[rows, blocks, codes] = False
        if not viol.any():
            break
        grad = viol.reshape(n, k).astype(float) @ kt - np.einsum("nb,nbm->nm", viol.sum(2).astype(float), kt[code_cols])
        z -= lr * grad
    return z
