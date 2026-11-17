"""Retrieval metrics that do not depend on the ANN backend."""
from __future__ import annotations

import numpy as np


def tie_aware_recall(scheme, gallery, probes, exact, found, k: int = 10) -> float:
    """Fraction of HNSW results whose score reaches the exact k-th score (ties count as hits)."""
    hits = []
    for i in range(len(probes)):
        kth = scheme.sim_matrix(probes[i:i + 1], gallery[exact[i]])[0].min()
        valid = np.asarray(found[i])
        valid = valid[valid >= 0]
        fs = scheme.sim_matrix(probes[i:i + 1], gallery[valid])[0] if len(valid) else np.empty(0)
        hits.append(min(int(np.sum(fs >= kth - 1e-12)), k) / k)
    return float(np.mean(hits))
