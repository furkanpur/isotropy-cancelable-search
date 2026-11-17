"""Shared-key cancelable transforms with a common interface: LRP, BioHashing, GRP-IoM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ijiss_btp.protect.linear import projection_matrix

KINDS = ("lrp", "biohash", "iom")


def _l2n(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class KeyedScheme:
    kind: str
    k_rule: str
    block: int = 8

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown scheme kind {self.kind!r}; expected one of {KINDS}")

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.k_rule}"

    def k_for(self, m: int) -> int:
        k = m if self.k_rule == "full" else m // 2 if self.k_rule == "half" else int(self.k_rule)
        if self.kind == "iom" and k % self.block:
            raise ValueError(f"GRP-IoM needs k divisible by block {self.block}; got {k}")
        if self.kind != "iom" and k > m:
            raise ValueError(f"{self.kind} needs k <= m; got k={k}, m={m}")
        return k

    def keygen(self, m: int, seed: int) -> np.ndarray:
        """Haar orthonormal columns for LRP/BioHashing; Gaussian projections for GRP-IoM."""
        k = self.k_for(m)
        if self.kind == "iom":
            return np.random.default_rng(seed).standard_normal((m, k)) / np.sqrt(m)
        return projection_matrix(m, k, seed).astype(np.float64)

    def protect(self, z: np.ndarray, key: np.ndarray) -> np.ndarray:
        u = z @ key
        if self.kind == "lrp":
            return u
        if self.kind == "biohash":
            return np.where(u >= 0, 1.0, -1.0)
        return u.reshape(len(z), -1, self.block).argmax(2)

    def features(self, codes: np.ndarray) -> np.ndarray:
        """Numeric matrix for decoders: identity for LRP/BioHashing, one-hot for GRP-IoM."""
        if self.kind != "iom":
            return codes.astype(np.float64)
        n, nb = codes.shape
        f = np.zeros((n, nb * self.block))
        f[np.arange(n)[:, None], np.arange(nb)[None, :] * self.block + codes] = 1.0
        return f

    def sim_pairs(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.kind == "iom":
            return (a == b).mean(1)
        return np.sum(_l2n(a) * _l2n(b), 1)

    def sim_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.kind == "iom":
            return self.features(a) @ self.features(b).T / a.shape[1]
        return _l2n(a) @ _l2n(b).T

    def index_codes(self, codes: np.ndarray) -> tuple[np.ndarray, bool]:
        """Array for faiss: normalized float32 (inner product) or packed bits (Hamming)."""
        if self.kind == "lrp":
            return np.ascontiguousarray(_l2n(codes), dtype=np.float32), False
        bits = codes > 0 if self.kind == "biohash" else self.features(codes) > 0
        return np.packbits(bits, axis=1), True


def parse_scheme(spec: str) -> KeyedScheme:
    """'lrp:full' | 'lrp:half' | 'biohash:half' | 'iom:256' ..."""
    kind, k_rule = spec.split(":")
    return KeyedScheme(kind, k_rule)
