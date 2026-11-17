"""Attack success endpoints: E1 (raw space, cross-system) and E2 (inside the system after key renewal)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _l2n(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class EndpointResult:
    rec_cos: float
    genuine: float
    proxy: float
    rel: float
    per_target_proxy: np.ndarray  # mean acceptance over probes per target; in memory only, never written
    n_success: int
    n_trials: int
    per_target_genuine: np.ndarray  # mean genuine acceptance over probes per target; in memory only


def _result(rec: float, gen_acc: np.ndarray, prx_acc: np.ndarray) -> EndpointResult:
    g, p = float(gen_acc.mean()), float(prx_acc.mean())
    return EndpointResult(rec, g, p, p / max(g, 1e-12), prx_acc.mean(1), int(prx_acc.sum()), int(prx_acc.size),
                          gen_acc.mean(1))


def e1(xn: np.ndarray, xhat: np.ndarray, enrolled_rows: np.ndarray, probe_rows: np.ndarray, tau: float) -> EndpointResult:
    """Proxies scored against the victims' fresh captures in the centered, normalized raw space."""
    xh = _l2n(xhat)
    gen = np.einsum("nd,npd->np", xn[enrolled_rows], xn[probe_rows]) >= tau
    prx = np.einsum("nd,npd->np", xh, xn[probe_rows]) >= tau
    rec = float(np.mean(np.sum(xh * xn[enrolled_rows], 1)))
    return _result(rec, gen.astype(float), prx.astype(float))


def e2(scheme, pre, key2, codes2, xhat, enrolled_rows, probe_rows, tau2) -> EndpointResult:
    """Proxies submitted to the system after renewal to key2 (re-enrolled templates codes2)."""
    pc2 = scheme.protect(pre.transform(_l2n(xhat)), key2)
    cols = range(probe_rows.shape[1])
    gen = np.stack([scheme.sim_pairs(codes2[enrolled_rows], codes2[probe_rows[:, j]]) >= tau2 for j in cols], 1)
    prx = np.stack([scheme.sim_pairs(pc2, codes2[probe_rows[:, j]]) >= tau2 for j in cols], 1)
    return _result(float("nan"), gen.astype(float), prx.astype(float))
