"""Biometric verification and identification evaluation metrics."""
from __future__ import annotations

import numpy as np


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> tuple[float, float]:
    """Calculates Equal Error Rate (EER) and operating threshold using exact ROC linear interpolation.

    Evaluates False Match Rate (FMR) and False Non-Match Rate (FNMR) across sorted unique
    score thresholds and linearly interpolates the exact crossing point on the ROC curve.

    Parameters
    ----------
    genuine_scores : np.ndarray
        Similarity scores of authentic mated pairs (higher = more similar).
    impostor_scores : np.ndarray
        Similarity scores of impostor non-mated pairs.

    Returns
    -------
    tuple[float, float]
        (EER, threshold)
    """
    genuine = np.sort(np.asarray(genuine_scores, dtype=np.float64))
    impostor = np.sort(np.asarray(impostor_scores, dtype=np.float64))

    if len(genuine) == 0 or len(impostor) == 0:
        return 0.5, 0.0
    # Strict separation shortcuts only (non-strict cases require tie handling)
    if genuine.min() > impostor.max():
        return 0.0, float((genuine.min() + impostor.max()) / 2.0)
    if genuine.max() < impostor.min():
        return 1.0, float((genuine.max() + impostor.min()) / 2.0)

    all_scores = np.unique(np.concatenate([genuine, impostor]))
    if len(all_scores) == 1:
        return 0.5, float(all_scores[0])

    thresholds = np.concatenate([
        [all_scores[0] - 1e-6],
        all_scores,
        [all_scores[-1] + 1e-6],
    ])

    fnmr = np.searchsorted(genuine, thresholds, side="left") / len(genuine)
    fmr = (len(impostor) - np.searchsorted(impostor, thresholds, side="left")) / len(impostor)

    diff = fmr - fnmr
    idx = np.where(diff <= 0)[0]
    if len(idx) == 0:
        return float(fmr[-1]), float(thresholds[-1])
    i = idx[0]
    if i == 0:
        return float(fnmr[0]), float(thresholds[0])

    d1 = diff[i - 1]
    d2 = diff[i]
    alpha = d1 / (d1 - d2) if d1 != d2 else 0.0

    eer = (1.0 - alpha) * fnmr[i - 1] + alpha * fnmr[i]
    opt_thresh = (1.0 - alpha) * thresholds[i - 1] + alpha * thresholds[i]
    return float(eer), float(opt_thresh)


def compute_identification_metrics(
    retrieved_indices: np.ndarray,
    gallery_ids: np.ndarray,
    probe_ids: np.ndarray,
) -> dict[str, float]:
    """Calculates Rank-1, hit@5, and hit@10 retrieval accuracy.

    Parameters
    ----------
    retrieved_indices : np.ndarray
        (N_probes, top_k) matrix of retrieved gallery indices.
    gallery_ids : np.ndarray
        (N_gallery,) identity array.
    probe_ids : np.ndarray
        (N_probes,) identity array.

    Returns
    -------
    dict[str, float]
        Dictionary containing 'rank1', 'hit5', and 'hit10'.
    """
    valid_mask = retrieved_indices >= 0
    safe_indices = np.where(valid_mask, retrieved_indices, 0)
    matched_ids = gallery_ids[safe_indices]  # (N_probes, top_k)
    is_match = (matched_ids == probe_ids[:, None]) & valid_mask

    rank1 = float(np.mean(is_match[:, 0]))
    hit5 = float(np.mean(np.any(is_match[:, :5], axis=1))) if retrieved_indices.shape[1] >= 5 else rank1
    hit10 = float(np.mean(np.any(is_match[:, :10], axis=1))) if retrieved_indices.shape[1] >= 10 else hit5

    return {"rank1": rank1, "hit5": hit5, "hit10": hit10}

