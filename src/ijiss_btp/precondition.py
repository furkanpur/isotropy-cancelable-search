"""Public pre-conditioners applied before the shared-key transform (fitted on DEV identities only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special

EPS = 1e-12


def _cov(x: np.ndarray):
    mu = x.mean(0)
    xc = x - mu
    return mu, xc.T @ xc / max(len(x) - 1, 1)


def _eigh_desc(s: np.ndarray):
    w, v = np.linalg.eigh(s)
    return w[::-1], v[:, ::-1]


def haar_orthogonal(m: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((m, m)))
    return q * np.sign(np.diag(r))


class Identity:
    name = "none"

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x

    def inverse(self, z):
        return z


class PCAWhiten:
    """Top-m principal axes; alpha = 0 keeps variances, alpha = 1 whitens the total covariance."""

    def __init__(self, m: int, alpha: float):
        self.m, self.alpha = int(m), float(alpha)
        self.name = f"pca{self.m}_a{self.alpha:g}"

    def fit(self, x, y=None):
        if self.m > x.shape[1]:
            raise ValueError(f"m={self.m} exceeds the input dimension {x.shape[1]}")
        self.mu, s = _cov(x)
        w, v = _eigh_desc(s)
        self.u = v[:, : self.m]
        self.scale = np.clip(w[: self.m], EPS, None) ** (self.alpha / 2.0)
        return self

    def transform(self, x):
        return ((x - self.mu) @ self.u) / self.scale

    def inverse(self, z):
        return (z * self.scale) @ self.u.T + self.mu


class FisherIsotropic:
    """Fisher-optimal (trace criterion under the key-hiding constraint) linear pre-conditioner (spec Proposition 3).

    Whitens the total covariance after a PCA to the numerical-rank subspace of the
    DEV covariance (eigenvalues above rtol * largest, capped at p0), then keeps the
    m directions with the smallest within-class variance, i.e. the largest Fisher
    ratio. Directions below the rank floor are dropped rather than whitened, so
    W^T S_t W = I holds on the fitting data. inverse() is the LMMSE back-map E[x | z].
    """

    def __init__(self, m: int, p0: int, rtol: float = 1e-8):
        self.m, self.p0, self.rtol = int(m), int(p0), float(rtol)
        if not 1 <= self.m <= self.p0:
            raise ValueError("FisherIsotropic needs 1 <= m <= p0")
        self.name = f"fi{self.m}_p{self.p0}"

    def fit(self, x, y=None):
        if y is None:
            raise ValueError("FisherIsotropic needs identity labels")
        if self.p0 > x.shape[1]:
            raise ValueError(f"p0={self.p0} exceeds the input dimension {x.shape[1]}")
        self.mu, s = _cov(x)
        w, v = _eigh_desc(s)
        keep = int(min(self.p0, np.sum(w > self.rtol * w[0])))
        if keep < self.m:
            raise ValueError(f"DEV covariance has numerical rank {keep} < m={self.m}")
        p = v[:, :keep] / np.sqrt(w[:keep])
        z = (x - self.mu) @ p
        means = pd.DataFrame(z).groupby(np.asarray(y)).transform("mean").to_numpy()
        zw = z - means
        sw = zw.T @ zw / max(len(z) - 1, 1)  # same normalization as _cov: within + between = I
        lw, vw = np.linalg.eigh(sw)  # ascending within-class variance = descending Fisher ratio
        self.proj = p @ vw[:, : self.m]
        self.back = self.proj.T @ s
        self.within_var = lw[: self.m]
        self.p0_effective = keep
        if not np.allclose(self.proj.T @ s @ self.proj, np.eye(self.m), atol=1e-6):
            raise RuntimeError("FisherIsotropic lost W^T S_t W = I; the DEV covariance is ill-conditioned")
        return self

    def transform(self, x):
        return (x - self.mu) @ self.proj

    def inverse(self, z):
        return z @ self.back + self.mu


class MarginalGauss:
    """Per-dimension empirical CDF followed by the standard normal quantile."""

    def __init__(self, n_knots: int = 1025):
        self.n_knots = n_knots

    def fit(self, z, y=None):
        n = len(z)
        self.p = np.linspace(0.5 / n, 1 - 0.5 / n, self.n_knots)
        knots = np.quantile(z, self.p, axis=0)
        self.knots = np.maximum.accumulate(knots + np.arange(self.n_knots)[:, None] * 1e-12, axis=0)
        return self

    def transform(self, z):
        out = np.empty_like(z, dtype=float)
        for j in range(z.shape[1]):
            out[:, j] = special.ndtri(np.interp(z[:, j], self.knots[:, j], self.p))
        return out

    def inverse(self, g):
        u = special.ndtr(g)
        out = np.empty_like(g, dtype=float)
        for j in range(g.shape[1]):
            out[:, j] = np.interp(u[:, j], self.p, self.knots[:, j])
        return out


class FixedRotation:
    """Public, seeded orthogonal mixing used between Gaussianization steps."""

    def __init__(self, m: int, seed: int):
        self.r = haar_orthogonal(m, seed)

    def fit(self, z, y=None):
        return self

    def transform(self, z):
        return z @ self.r

    def inverse(self, z):
        return z @ self.r.T


class Chain:
    def __init__(self, name: str, steps: list):
        self.name, self.steps = name, steps

    def fit(self, x, y=None):
        z = x
        for s in self.steps:
            s.fit(z, y)
            z = s.transform(z)
        return self

    def transform(self, x):
        for s in self.steps:
            x = s.transform(x)
        return x

    def inverse(self, z):
        for s in reversed(self.steps):
            z = s.inverse(z)
        return z


def _gaussianized(name: str, first, m: int, iters: int) -> Chain:
    if iters < 1:
        raise ValueError("iters must be >= 1")
    steps = [first, MarginalGauss()]
    for i in range(iters - 1):
        steps += [FixedRotation(m, 1000 + i), MarginalGauss()]
    return Chain(name, steps)


def make_preconditioner(spec: str):
    """none | pca:<m>:<alpha> | fi:<m>:<p0> | rbig:<m>:<iters> | firbig:<m>:<p0>:<iters>"""
    if spec == "none":
        return Identity()
    kind, *args = spec.split(":")
    if kind == "pca" and len(args) == 2:
        return PCAWhiten(int(args[0]), float(args[1]))
    if kind == "fi" and len(args) == 2:
        return FisherIsotropic(int(args[0]), int(args[1]))
    if kind == "rbig" and len(args) == 2:
        m, iters = int(args[0]), int(args[1])
        return _gaussianized(f"rbig{m}_i{iters}", PCAWhiten(m, 1.0), m, iters)
    if kind == "firbig" and len(args) == 3:
        m, p0, iters = int(args[0]), int(args[1]), int(args[2])
        return _gaussianized(f"firbig{m}_p{p0}_i{iters}", FisherIsotropic(m, p0), m, iters)
    raise ValueError(f"unknown pre-conditioner spec: {spec!r}")


def fisher_traces(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(tr W^T S_b W, tr W^T S_t W) with 1/n-normalized between and total scatter.

    The 1/n normalization differs from _cov's ddof=1 by n/(n-1); ratios are unaffected.
    """
    xc = x - x.mean(0)
    st = xc.T @ xc / len(x)
    means = pd.DataFrame(xc).groupby(np.asarray(y)).transform("mean").to_numpy()
    sb = means.T @ means / len(x)
    return float(np.trace(w.T @ sb @ w)), float(np.trace(w.T @ st @ w))
