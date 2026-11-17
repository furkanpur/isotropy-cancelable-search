"""Gómez-Barrero et al. (2018) Unlinkability Evaluation Framework (motivated by ISO/IEC 24745 / 30136 objectives)."""
from __future__ import annotations

import numpy as np


def compute_unlinkability_metric(
    mated_scores: np.ndarray,
    non_mated_scores: np.ndarray,
    num_bins: int = 100,
) -> tuple[float, float, float]:
    """Computes the Gómez-Barrero et al. (2018) system-level unlinkability metric D_sys.

    D_sys = integral p(s | Hm) * D_local(s) ds
    where D_local(s) = max(0, 2 * LR(s) / (LR(s) + 1) - 1) = max(0, (p_m - p_nm) / (p_m + p_nm))

    Parameters
    ----------
    mated_scores : np.ndarray
        Linkage similarity scores from the same subject under different keys.
    non_mated_scores : np.ndarray
        Linkage similarity scores from different subjects under different keys.
    num_bins : int
        Number of histogram bins.

    Returns
    -------
    tuple[float, float, float]
        (D_sys, mated_mean, non_mated_mean)
        D_sys in [0, 1]. D_sys -> 0 indicates full unlinkability.
    """
    min_score = min(float(mated_scores.min()), float(non_mated_scores.min()))
    max_score = max(float(mated_scores.max()), float(non_mated_scores.max()))
    if np.isclose(min_score, max_score):
        min_score -= 1e-4
        max_score += 1e-4
    bins = np.linspace(min_score, max_score, num_bins + 1)

    hist_mated, _ = np.histogram(mated_scores, bins=bins, density=True)
    hist_non_mated, _ = np.histogram(non_mated_scores, bins=bins, density=True)

    # Bin widths
    bin_width = bins[1] - bins[0]

    # Gomez-Barrero et al. (2018) likelihood-ratio local linkability measure:
    # D_local(s) = max(0, 2 * LR / (LR + 1) - 1) = max(0, (p_m - p_nm) / (p_m + p_nm))
    denom = hist_mated + hist_non_mated
    mask = denom > 1e-12
    local_d = np.zeros_like(hist_mated)
    local_d[mask] = np.maximum(0.0, (hist_mated[mask] - hist_non_mated[mask]) / denom[mask])

    # Global system unlinkability metric D_sys = integral p(s | Hm) * D_local(s) ds
    d_sys = float(np.sum(hist_mated * local_d) * bin_width)
    return d_sys, float(mated_scores.mean()), float(non_mated_scores.mean())

