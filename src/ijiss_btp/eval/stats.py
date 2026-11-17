"""Pairs, operating points and uncertainty for biometric evaluations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ijiss_btp.eval.metrics import compute_eer


def make_pairs(rows: np.ndarray, ids: np.ndarray, rng: np.random.Generator, n_imp: int = 200_000):
    """All within-identity pairs among `rows` plus `n_imp` random cross-identity pairs."""
    rows = np.asarray(rows)
    ids = np.asarray(ids)
    if len(np.unique(ids[rows])) < 2:
        raise ValueError("make_pairs needs rows from at least two identities")
    gen: list[tuple[int, int]] = []
    for _, g in pd.DataFrame({"r": rows, "l": ids[rows]}).groupby("l"):
        rr = g["r"].to_numpy()
        for i in range(len(rr)):
            for j in range(i + 1, len(rr)):
                gen.append((int(rr[i]), int(rr[j])))
    chunks, total = [], 0
    while total < n_imp:
        a = rng.choice(rows, size=2 * n_imp)
        b = rng.choice(rows, size=2 * n_imp)
        keep = ids[a] != ids[b]
        chunk = np.stack([a[keep], b[keep]], 1)
        chunks.append(chunk)
        total += len(chunk)
    imp = np.concatenate(chunks)[:n_imp]
    return np.array(gen, dtype=np.int64).reshape(-1, 2), imp


def threshold_at_fmr(impostor: np.ndarray, fmr: float) -> float:
    """Smallest threshold with accept-iff-score>=tau such that at most fmr*n impostors pass."""
    s = np.sort(np.asarray(impostor, dtype=float))[::-1]
    k = int(np.floor(fmr * len(s)))
    if k >= len(s):
        return float(s[-1])
    return float(np.nextafter(s[k], np.inf))


def rates_at_threshold(genuine: np.ndarray, impostor: np.ndarray, tau: float) -> dict:
    return {"tmr": float(np.mean(genuine >= tau)), "fmr": float(np.mean(impostor >= tau))}


def rank1(sim: np.ndarray, probe_labels: np.ndarray, gallery_labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(gallery_labels)[np.argmax(sim, 1)] == np.asarray(probe_labels)))


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    return compute_eer(genuine, impostor)[0]


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def bootstrap_ratio_ci(num, den, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of mean(num)/mean(den) under a paired bootstrap over units; NaN replicates (den mean 0) are ignored."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    n = len(num)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    d = den[idx].mean(1)
    ratio = np.where(d > 0, num[idx].mean(1) / np.where(d > 0, d, 1.0), np.nan)
    if np.isnan(ratio).all():
        return float("nan"), float("nan")
    return float(np.nanquantile(ratio, alpha / 2)), float(np.nanquantile(ratio, 1 - alpha / 2))


def clopper_pearson_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) upper confidence bound for a binomial proportion."""
    if successes >= trials:
        return 1.0
    from scipy import stats as sstats  # lazy: the GPU container image has no scipy and only needs make_pairs/eer
    return float(sstats.beta.ppf(1 - alpha, successes + 1, trials - successes))
