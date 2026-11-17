"""Permutation-Scrambled Index-of-Max (PS-IoM) Hashing for Compact Protected Indexing."""
from __future__ import annotations

import numpy as np
from ijiss_btp.protect.linear import projection_matrix


def ps_iom_protect(
    emb: np.ndarray,
    dim_out: int,
    seed: int,
    block_size: int = 8,
) -> np.ndarray:
    """Permutation-Scrambled Index-of-Max (PS-IoM) Hashing.

    Parameters
    ----------
    emb : np.ndarray
        (N, d) float32 matrix of continuous embeddings.
    dim_out : int
        Intermediate projected dimension k (must be divisible by block_size).
    seed : int
        Transformation key seed kappa.
    block_size : int
        Length L of each sub-block (default 8).

    Returns
    -------
    np.ndarray
        (N, dim_out // 8) uint8 bitset matrix of packed one-hot indices.
    """
    if dim_out % block_size != 0:
        raise ValueError(f"dim_out {dim_out} must be divisible by block_size {block_size}")

    num_blocks = dim_out // block_size
    dim_in = emb.shape[1]
    q = projection_matrix(dim_in, dim_out, seed)
    u = emb @ q  # (N, k)

    # Reshape into blocks (N, num_blocks, block_size)
    u_blocked = u.reshape(-1, num_blocks, block_size)
    max_indices = np.argmax(u_blocked, axis=2)  # (N, num_blocks) in [0, block_size-1]

    # Block-specific key-dependent pseudorandom permutations \pi_\kappa^{(m)} \in S_L
    rng = np.random.default_rng(seed + 2000)
    perms = np.empty((num_blocks, block_size), dtype=np.int64)
    for m in range(num_blocks):
        perms[m] = rng.permutation(block_size)

    # Scramble maximum indices using block-specific permutations
    scrambled = np.empty_like(max_indices)
    for m in range(num_blocks):
        scrambled[:, m] = perms[m, max_indices[:, m]]

    # Convert to one-hot binary representation of length (N, dim_out)
    one_hot = np.zeros((emb.shape[0], num_blocks, block_size), dtype=bool)
    n_idx = np.arange(emb.shape[0])[:, None]
    b_idx = np.arange(num_blocks)[None, :]
    one_hot[n_idx, b_idx, scrambled] = True
    one_hot_flat = one_hot.reshape(emb.shape[0], dim_out)

    return np.packbits(one_hot_flat, axis=1)

