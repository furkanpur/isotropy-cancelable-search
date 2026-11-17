"""Informed non-linear coordinate-wise estimation and reconstruction attack under compromised key."""
from __future__ import annotations

import numpy as np
from ijiss_btp.protect.linear import projection_matrix


def coordinate_wise_inversion_attack(
    train_templates: np.ndarray,
    train_embeddings: np.ndarray,
    test_templates: np.ndarray,
    dim_in: int,
    seed: int,
    n_bins: int = 200,
) -> np.ndarray:
    """Informed coordinate-wise non-linear inversion attack (Threat Model 2b).

    An adversary with full key custody (Q_kappa, omega, tau) and access to an
    auxiliary development set computes the non-parametric conditional expectation
    E[u_j | t_j] coordinate-by-coordinate via binned empirical regression,
    interpolates held-out query templates, and backprojects through the known
    orthonormal projection matrix Q_kappa.

    Parameters
    ----------
    train_templates : np.ndarray
        (M, k) protected templates from the auxiliary development set.
    train_embeddings : np.ndarray
        (M, d) continuous authentic embeddings from the auxiliary development set.
    test_templates : np.ndarray
        (N, k) target protected templates to reconstruct.
    dim_in : int
        Ambient embedding dimension d (e.g. 768).
    seed : int
        Compromised key seed kappa.
    n_bins : int
        Number of discretization bins per coordinate (default 200).

    Returns
    -------
    np.ndarray
        (N, d) float32 reconstructed proxy embeddings on the unit hypersphere.
    """
    dim_out = test_templates.shape[1]
    q = projection_matrix(dim_in, dim_out, seed)
    train_u = train_embeddings @ q

    n_test = test_templates.shape[0]
    pred_u = np.zeros((n_test, dim_out), dtype=np.float32)

    for j in range(dim_out):
        t_col = train_templates[:, j]
        u_col = train_u[:, j]

        t_min = float(t_col.min())
        t_max = float(t_col.max())
        if t_min == t_max:
            continue

        bins = np.linspace(t_min, t_max, n_bins + 1)
        bin_idx = np.clip(np.digitize(t_col, bins) - 1, 0, n_bins - 1)

        counts = np.bincount(bin_idx, minlength=n_bins)
        sums = np.bincount(bin_idx, weights=u_col, minlength=n_bins)
        valid = counts > 0

        bin_means = np.zeros(n_bins, dtype=np.float32)
        bin_means[valid] = sums[valid] / counts[valid]

        # Fill empty bins via linear interpolation
        if not np.all(valid):
            if np.any(valid):
                xp = np.where(valid)[0]
                fp = bin_means[valid]
                bin_means = np.interp(np.arange(n_bins), xp, fp)

        test_bin_idx = np.clip(np.digitize(test_templates[:, j], bins) - 1, 0, n_bins - 1)
        pred_u[:, j] = bin_means[test_bin_idx]

    # Backproject estimated subspace vector through Q^T into ambient space
    pred_e = pred_u @ q.T
    norm = np.linalg.norm(pred_e, axis=1, keepdims=True)
    return (pred_e / np.where(norm == 0, 1.0, norm)).astype(np.float32)


def ps_iom_known_key_inversion(
    train_templates: np.ndarray,
    train_embeddings: np.ndarray,
    test_templates: np.ndarray,
    dim_in: int,
    dim_out: int,
    seed: int,
    block_size: int = 8,
) -> np.ndarray:
    """Informed known-key winner/loser backprojection attack against PS-IoM.

    The adversary uses knowledge of the key seed to reverse the block-specific
    permutations pi_kappa^(m), identifies the winning coordinate in each block,
    assigns +1.0 to the winning coordinate and -1/(L-1) to the losing coordinates,
    and backprojects through the known orthonormal projection matrix Q_kappa.

    Parameters
    ----------
    train_templates : np.ndarray
        (M, k // 8) packed uint8 bitset templates (or one-hot float).
    train_embeddings : np.ndarray
        (M, d) float32 authentic embeddings (unused in training-free attack).
    test_templates : np.ndarray
        (N, k // 8) packed uint8 target templates.
    dim_in : int
        Original dimension d.
    dim_out : int
        Intermediate projection dimension k.
    seed : int
        Compromised key seed.
    block_size : int
        Block size L (default 8).

    Returns
    -------
    np.ndarray
        (N, d) float32 reconstructed proxy embeddings.
    """
    q = projection_matrix(dim_in, dim_out, seed)
    n = test_templates.shape[0]
    num_blocks = dim_out // block_size

    # Unpack bits if packed uint8
    if test_templates.dtype == np.uint8 and test_templates.shape[1] == num_blocks:
        bits = np.unpackbits(test_templates, axis=1)[:, :dim_out].astype(np.float32)
    elif test_templates.shape[1] == dim_out:
        bits = test_templates.astype(np.float32)
    else:
        bits = np.unpackbits(test_templates, axis=1)[:, :dim_out].astype(np.float32)

    bits_blocked = bits.reshape(n, num_blocks, block_size)

    # Reverse key-dependent block permutations pi_kappa^(m)
    rng = np.random.default_rng(seed + 2000)
    unperm_blocked = np.zeros_like(bits_blocked)
    for m in range(num_blocks):
        perm = rng.permutation(block_size)
        for l in range(block_size):
            unperm_blocked[:, m, l] = bits_blocked[:, m, perm[l]]

    unperm_flat = unperm_blocked.reshape(n, dim_out)

    # Optimal expectation centering: winner gets +1.0, losers get -1 / (block_size - 1)
    neg_val = -1.0 / float(block_size - 1)
    centered = np.where(unperm_flat > 0.5, 1.0, neg_val).astype(np.float32)

    pred_e = centered @ q.T
    norm = np.linalg.norm(pred_e, axis=1, keepdims=True)
    return (pred_e / np.where(norm == 0, 1.0, norm)).astype(np.float32)

