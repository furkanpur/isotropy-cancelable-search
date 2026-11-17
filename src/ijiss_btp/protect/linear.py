"""Linear cancelable template transforms: Keyed Random Projection and BioHashing."""
from __future__ import annotations

import hashlib
import numpy as np


def projection_matrix(dim_in: int, dim_out: int, seed: int) -> np.ndarray:
    """Key-seeded orthonormal projection (dim_in x dim_out), columns orthonormal."""
    if dim_out > dim_in:
        raise ValueError(f"dim_out {dim_out} exceeds dim_in {dim_in}")
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((dim_in, dim_out))
    q, r = np.linalg.qr(g)
    # Fix the sign convention so the matrix is a deterministic function of the seed.
    q *= np.sign(np.diag(r))
    return q.astype(np.float32)


def rp_protect(emb: np.ndarray, dim_out: int, seed: int) -> np.ndarray:
    """Float cancelable template: project and re-normalize to unit length."""
    q = projection_matrix(emb.shape[1], dim_out, seed)
    y = emb @ q
    n = np.linalg.norm(y, axis=1, keepdims=True)
    return (y / np.where(n == 0, 1.0, n)).astype(np.float32)


def biohash_protect(emb: np.ndarray, n_bits: int, seed: int) -> np.ndarray:
    """BioHashing: project, sign-binarize, pack to uint8 (N, n_bits/8)."""
    if n_bits % 8:
        raise ValueError("n_bits must be a multiple of 8 for packed storage")
    q = projection_matrix(emb.shape[1], n_bits, seed)
    bits = (emb @ q) >= 0
    return np.packbits(bits, axis=1)


def user_seed(base_seed: int, identity: str) -> int:
    """Stable per-identity key seed."""
    h = hashlib.sha256(f"{base_seed}:{identity}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def rp_protect_per_user(
    emb: np.ndarray, identities: np.ndarray, dim_out: int, base_seed: int
) -> np.ndarray:
    """Protect each row under its own identity key (user-specific model)."""
    out = np.empty((emb.shape[0], dim_out), dtype=np.float32)
    for ident in np.unique(identities):
        mask = identities == ident
        seed = user_seed(base_seed, str(ident))
        out[mask] = rp_protect(emb[mask], dim_out, seed)
    return out

