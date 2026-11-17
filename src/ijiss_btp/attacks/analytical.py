"""Analytical reconstruction attack under full key compromise (Threat Model 2)."""
from __future__ import annotations

import numpy as np
from ijiss_btp.protect.linear import projection_matrix


def invert_linear_template(
    protected_templates: np.ndarray, dim_in: int, seed: int
) -> np.ndarray:
    """Moore-Penrose pseudo-inversion of linear random projection templates.

    Given t = (e @ Q) / ||e @ Q|| and secret key seed, the minimum-norm
    preimage is:
        e_hat = (t @ Q^T) / ||t @ Q^T||

    Parameters
    ----------
    protected_templates : np.ndarray
        (N, k) float32 matrix of protected templates.
    dim_in : int
        Original embedding dimension d (e.g. 768).
    seed : int
        Compromised key seed.

    Returns
    -------
    np.ndarray
        (N, d) float32 reconstructed proxy embeddings.
    """
    dim_out = protected_templates.shape[1]
    q = projection_matrix(dim_in, dim_out, seed)
    e_hat = protected_templates @ q.T
    norm = np.linalg.norm(e_hat, axis=1, keepdims=True)
    return (e_hat / np.where(norm == 0, 1.0, norm)).astype(np.float32)


def evaluate_proxy_match_rate(
    gallery_raw: np.ndarray,
    gallery_ids: np.ndarray,
    reconstructed_probes: np.ndarray,
    probe_ids: np.ndarray,
    batch_size: int = 1024,
) -> float:
    """Computes exact Rank-1 identity hit rate of reconstructed probes against raw gallery.

    Parameters
    ----------
    gallery_raw : np.ndarray
        (N_g, d) authentic raw gallery embeddings.
    gallery_ids : np.ndarray
        (N_g,) identity labels for gallery.
    reconstructed_probes : np.ndarray
        (N_p, d) adversary-reconstructed proxy embeddings.
    probe_ids : np.ndarray
        (N_p,) ground-truth identity labels for probes.
    batch_size : int
        Chunk size for matrix multiplication.

    Returns
    -------
    float
        Rank-1 proxy match rate in [0, 1].
    """
    hits = 0
    num_probes = len(reconstructed_probes)
    for i in range(0, num_probes, batch_size):
        chunk_probes = reconstructed_probes[i : i + batch_size]
        chunk_ids = probe_ids[i : i + batch_size]
        sim = chunk_probes @ gallery_raw.T
        best_match_idx = sim.argmax(axis=1)
        matched_ids = gallery_ids[best_match_idx]
        hits += int((matched_ids == chunk_ids).sum())
    return float(hits / num_probes)


def evaluate_verification_match_rate(
    gallery_raw: np.ndarray,
    gallery_ids: np.ndarray,
    reconstructed_probes: np.ndarray,
    probe_ids: np.ndarray,
    threshold: float = 0.58,
) -> float:
    """Computes verification match rate (fraction of reconstructed probes accepted above threshold).

    In biometric verification (1:1 matching), an adversary succeeds if the reconstructed proxy
    embedding matches the claimed authentic identity with similarity >= threshold.

    Parameters
    ----------
    gallery_raw : np.ndarray
        (N_g, d) authentic enrolled raw gallery embeddings.
    gallery_ids : np.ndarray
        (N_g,) identity labels for gallery.
    reconstructed_probes : np.ndarray
        (N_p, d) adversary-reconstructed proxy embeddings.
    probe_ids : np.ndarray
        (N_p,) ground-truth identity labels for probes.
    threshold : float
        Decision threshold tau (default 0.58 for FAR <= 0.1%).

    Returns
    -------
    float
        Fraction of probes whose similarity to target enrolled embedding >= threshold.
    """
    id_to_idx = {gid: idx for idx, gid in enumerate(gallery_ids)}
    valid_pairs = [(r, id_to_idx[pid]) for r, pid in zip(reconstructed_probes, probe_ids) if pid in id_to_idx]
    if not valid_pairs:
        return 0.0

    probes_arr = np.stack([p[0] for p in valid_pairs])
    target_gallery = gallery_raw[[p[1] for p in valid_pairs]]
    sims = np.sum(probes_arr * target_gallery, axis=1)
    return float(np.mean(sims >= threshold))

