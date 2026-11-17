"""Zero-pair key-free attacks (T0): decode a stolen shared-key database with public auxiliary embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

EPS = 1e-12


def _cov(x: np.ndarray):
    mu = x.mean(0)
    xc = x - mu
    return mu, xc.T @ xc / max(len(x) - 1, 1)


def _eigh_desc(s: np.ndarray):
    w, v = np.linalg.eigh(s)
    return w[::-1], v[:, ::-1]


@dataclass(frozen=True, eq=False)
class EigenBasis:
    """Rows, sample mean and descending covariance eigendecomposition of one sample; computed once, shareable.
    Instances compare and hash by identity (array fields have no element-free equality)."""

    x: np.ndarray
    mu: np.ndarray
    lam: np.ndarray
    vec: np.ndarray

    @classmethod
    def of(cls, x: np.ndarray) -> EigenBasis:
        mu, s = _cov(x)
        lam, vec = _eigh_desc(s)
        return cls(x, mu, lam, vec)


def check_anchors(f_anchor: np.ndarray, z_known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate r >= 1 anchor pairs: protected templates (r, k) and the matching pre-key vectors (r, m), finite."""
    f_anchor, z_known = np.asarray(f_anchor, dtype=float), np.asarray(z_known, dtype=float)
    if f_anchor.ndim != 2 or z_known.ndim != 2:
        raise ValueError(f"anchors must be 2-D (r, dim) arrays; got shapes {f_anchor.shape} and {z_known.shape}")
    if len(f_anchor) != len(z_known):
        raise ValueError(f"anchors need the same number of templates and pre-key vectors; got {len(f_anchor)} "
                         f"and {len(z_known)}")
    if len(f_anchor) == 0:
        raise ValueError("at least one anchor pair is needed")
    if not (np.isfinite(f_anchor).all() and np.isfinite(z_known).all()):
        raise ValueError("anchors must be finite")
    return f_anchor, z_known


class EigenAligner:
    """Second-order alignment of stolen-template and auxiliary eigenbases (eigendecompositions cached).

    `stolen` and `aux` are row arrays or precomputed EigenBasis objects (to share one decomposition across aligners).
    """

    def __init__(self, stolen: np.ndarray | EigenBasis, aux: np.ndarray | EigenBasis) -> None:
        c = stolen if isinstance(stolen, EigenBasis) else EigenBasis.of(stolen)
        a = aux if isinstance(aux, EigenBasis) else EigenBasis.of(aux)
        self.stolen, self.aux = c.x, a.x
        self.mu_c, self.lc, self.vc = c.mu, c.lam, c.vec
        self.mu_a, self.la, self.va = a.mu, a.lam, a.vec

    def signs(self, p: int, oracle_key: np.ndarray | None = None,
              anchors: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """Component signs from skewness or mean evidence; oracle_key gives the true signs (diagnostic only).

        anchors = (f_anchor, z_known), r >= 1 colluders' templates and pre-key vectors: the sign of axis i is
        sign(sum_a <vc_i, f_a - mu_c> <va_i, z_a - mu_a>), ties to +1. With exact population moments and distinct
        eigenvalues, one anchor with a non-zero projection on every axis gives the true signs.
        """
        if p < 0:
            raise ValueError(f"p must be >= 0; got {p}")
        if oracle_key is not None and anchors is not None:
            raise ValueError("give either oracle_key or anchors, not both")
        vc, va = self.vc[:, :p], self.va[:, :p]
        if anchors is not None:
            f_a, z_a = check_anchors(*anchors)
            if f_a.shape[1] != self.vc.shape[0] or z_a.shape[1] != self.va.shape[0]:
                raise ValueError(f"anchor dimensions {f_a.shape[1]}, {z_a.shape[1]} do not match the stolen and "
                                 f"auxiliary dimensions {self.vc.shape[0]}, {self.va.shape[0]}")
            s = np.sign(np.einsum("ai,ai->i", (f_a - self.mu_c) @ vc, (z_a - self.mu_a) @ va))
        elif oracle_key is not None:
            s = np.sign(np.einsum("mi,mi->i", va, oracle_key @ vc))
        else:
            pc = (self.stolen - self.mu_c) @ vc
            pa = (self.aux - self.mu_a) @ va
            sk_c = np.mean(pc ** 3, 0) / (np.std(pc, 0) ** 3 + EPS)
            sk_a = np.mean(pa ** 3, 0) / (np.std(pa, 0) ** 3 + EPS)
            z_sk = np.minimum(np.abs(sk_c) * np.sqrt(len(pc) / 6), np.abs(sk_a) * np.sqrt(len(pa) / 6))
            mc, ma = self.mu_c @ vc, self.mu_a @ va
            z_mu = np.minimum(np.abs(mc) / np.sqrt(np.clip(self.lc[:p], EPS, None) / len(pc)),
                              np.abs(ma) / np.sqrt(np.clip(self.la[:p], EPS, None) / len(pa)))
            s = np.where(z_sk >= z_mu, np.sign(sk_c * sk_a), np.sign(mc * ma))
        s[s == 0] = 1.0
        return s

    def decoder(self, p: int, oracle_key: np.ndarray | None = None,
                anchors: tuple[np.ndarray, np.ndarray] | None = None) -> Callable[[np.ndarray], np.ndarray]:
        """Map a template to a pre-key estimate through the top p matched axes, variance-rescaled, with the signs of
        `signs` (skewness/mean rule, oracle_key or anchors)."""
        if p < 0:
            raise ValueError(f"p must be >= 0; got {p}")
        p = min(p, self.vc.shape[1], self.va.shape[1])
        s = self.signs(p, oracle_key, anchors)
        scale = np.sqrt(np.clip(self.la[:p], EPS, None) / np.clip(self.lc[:p], EPS, None))
        vc, va, mu_c, mu_a = self.vc[:, :p], self.va[:, :p], self.mu_c, self.mu_a
        return lambda cq: mu_a + (((cq - mu_c) @ vc) * (s * scale)) @ va.T


def _whiten_stats(x: np.ndarray):
    mu, s = _cov(x)
    w, v = _eigh_desc(s)
    w = np.clip(w, EPS, None)
    return mu, (v * w ** -0.5) @ v.T, (v * w ** 0.5) @ v.T


def fastica(z: np.ndarray, rng: np.random.Generator, iters: int = 300, tol: float = 1e-5) -> np.ndarray:
    """Symmetric FastICA (logcosh) on whitened data; rows of the result are unmixing vectors."""
    n, m = z.shape
    q, r = np.linalg.qr(rng.standard_normal((m, m)))
    w = (q * np.sign(np.diag(r))).T
    for _ in range(iters):
        g = np.tanh(z @ w.T)
        wn = (g.T @ z) / n - (1 - g ** 2).mean(0)[:, None] * w
        u, _, vt = np.linalg.svd(wn)
        wn = u @ vt
        done = np.max(np.abs(np.abs(np.einsum("ij,ij->i", wn, w)) - 1)) < tol
        w = wn
        if done:
            break
    return w


def ica_align_decoder(stolen: np.ndarray, aux: np.ndarray, seed: int, iters: int = 300) -> Callable:
    """Recover the rotation from independent components matched by quantile signatures."""
    rng = np.random.default_rng(seed)
    qlev = np.linspace(0.02, 0.98, 25)
    mu_c, w_c, _ = _whiten_stats(stolen)
    mu_a, w_a, w_a_inv = _whiten_stats(aux)
    u_c = fastica((stolen - mu_c) @ w_c, rng, iters)
    u_a = fastica((aux - mu_a) @ w_a, rng, iters)

    def sig(s: np.ndarray) -> np.ndarray:
        s = (s - s.mean(0)) / (s.std(0) + EPS)
        return np.quantile(s, qlev, axis=0).T

    qc = sig(((stolen - mu_c) @ w_c) @ u_c.T)
    qa = sig(((aux - mu_a) @ w_a) @ u_a.T)
    d_pos = ((qa[:, None, :] - qc[None, :, :]) ** 2).sum(-1)
    d_neg = ((qa[:, None, :] + qc[None, :, ::-1]) ** 2).sum(-1)
    ra, rc = linear_sum_assignment(np.minimum(d_pos, d_neg))
    sign = np.where(d_pos[ra, rc] <= d_neg[ra, rc], 1.0, -1.0)

    def decode(cq: np.ndarray) -> np.ndarray:
        sq = ((cq - mu_c) @ w_c) @ u_c.T
        sx = np.zeros_like(sq)
        sx[:, ra] = sq[:, rc] * sign
        return mu_a + (sx @ u_a) @ w_a_inv

    return decode


def wasserstein_procrustes_decoder(stolen: np.ndarray, aux: np.ndarray, seed: int, n_iter: int = 30,
                                   n_sub: int = 1000, reg: float = 0.05) -> Callable:
    """Unsupervised orthogonal alignment (Grave et al., 2019); requires k == m continuous templates."""
    rng = np.random.default_rng(seed)
    mu_c, mu_a = stolen.mean(0), aux.mean(0)
    c, a = stolen - mu_c, aux - mu_a
    al = EigenAligner(stolen, aux)
    p = min(c.shape[1], a.shape[1])
    m_map = al.vc[:, :p] @ np.diag(al.signs(p)) @ al.va[:, :p].T
    cs = c[rng.choice(len(c), size=min(n_sub, len(c)), replace=False)]
    as_ = a[rng.choice(len(a), size=min(n_sub, len(a)), replace=False)]
    wc = np.full(len(cs), 1.0 / len(cs))
    wa = np.full(len(as_), 1.0 / len(as_))
    for _ in range(n_iter):
        cost = ot.dist(cs @ m_map, as_)
        plan = ot.sinkhorn(wc, wa, cost / max(float(cost.max()), EPS), reg)
        u, _, vt = np.linalg.svd(cs.T @ plan @ as_)
        m_map = u @ vt
    return lambda cq: mu_a + (cq - mu_c) @ m_map
