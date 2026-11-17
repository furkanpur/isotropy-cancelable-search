"""Closed-form quantities from the theory section (spec Section 4)."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def collusion_cos_optimum(r, m):
    """Proposition 2: exact optimal expected cosine with r exact anchors in an isotropic m-dim pre-key space.

    E sqrt(B), B ~ Beta(r/2, (m - r)/2), equals Gamma((r+1)/2) Gamma(m/2) / (Gamma(r/2) Gamma((m+1)/2)) and is <= sqrt(r/m).
    Returns 1.0 where r >= m. Scalar in, float out; array in, array out.
    """
    r_a = np.asarray(r, dtype=float)
    m_a = np.asarray(m, dtype=float)
    val = np.exp(gammaln((r_a + 1) / 2) + gammaln(m_a / 2) - gammaln(r_a / 2) - gammaln((m_a + 1) / 2))
    val = np.where(r_a >= m_a, 1.0, val)
    return float(val) if val.ndim == 0 else val


def uniform_abs_cos_mean(m):
    """E|theta_1| for theta uniform on S^{m-1}: Gamma(m/2) / (sqrt(pi) Gamma((m+1)/2)).

    The expected |cos| between a fixed direction and an independent uniform direction (sign-agnostic chance level).
    """
    m_a = np.asarray(m, dtype=float)
    val = np.exp(gammaln(m_a / 2) - gammaln((m_a + 1) / 2)) / np.sqrt(np.pi)
    return float(val) if val.ndim == 0 else val
