"""Non-Linear Polyphase Random Projection (NL-PRP) for Post-Compromise Security."""
from __future__ import annotations

import numpy as np
from ijiss_btp.protect.linear import projection_matrix


def nl_prp_protect(
    emb: np.ndarray,
    dim_out: int,
    seed: int,
    gamma: float = 0.70,
    beta: float = 0.25,
    omega: float | None = None,
    return_projection: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Non-Linear Polyphase Random Projection (NL-PRP).

    Transforms unit-norm deep continuous embeddings e into protected templates t:
        u = e @ Q_kappa
        omega_kappa = 2 * pi * ((seed % 8) + 4)  # satisfies omega_kappa >= 8*pi
        t_i = sign(u_i) * |u_i|^gamma + beta * sin(omega_kappa * u_i + tau_i(kappa))

    Parameters
    ----------
    emb : np.ndarray
        (N, d) float32 matrix of L2-normalized embeddings.
    dim_out : int
        Target projection dimension k <= d (e.g. 512, 256, 128).
    seed : int
        Transformation key seed kappa.
    gamma : float
        Power-law magnitude compression factor (default 0.70).
    beta : float
        Transcendental perturbation scale (default 0.25).
    omega : float | None
        Angular carrier frequency. If None, derived deterministically from seed as
        omega_kappa = 2 * pi * ((seed % 8) + 4) >= 8 * pi (Theorem 1 & 2).
    return_projection : bool
        Whether to return the projection matrix Q_kappa alongside templates.

    Returns
    -------
    np.ndarray
        (N, k) float32 matrix of L2-normalized protected templates.
    """
    dim_in = emb.shape[1]
    q = projection_matrix(dim_in, dim_out, seed)
    u = emb @ q

    # Derive angular carrier frequency >= 8*pi to satisfy Theorems 1 & 2
    if omega is None:
        omega_val = float(2.0 * np.pi * ((int(seed) % 8) + 4))
    else:
        omega_val = float(omega)

    # Draw deterministic coordinate-wise phase offsets from seed
    rng = np.random.default_rng(seed + 1000)
    tau = rng.uniform(-np.pi, np.pi, size=(1, dim_out)).astype(np.float32)

    # Polyphase transcendental non-linear transformation
    t = np.sign(u) * (np.abs(u) ** gamma) + beta * np.sin(omega_val * u + tau)

    # Re-normalize to unit length for inner-product vector search
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    t_norm = (t / np.where(norm == 0, 1.0, norm)).astype(np.float32)

    if return_projection:
        return t_norm, q
    return t_norm

