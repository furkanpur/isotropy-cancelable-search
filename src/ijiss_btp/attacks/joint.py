"""Joint collusion decoder: the projection decoder on the anchors' span plus an aligned estimate outside it.

With exact anchors (T1) under LRP-full, the projection decoder W = pinv(F) Z (attacks.anchor.fit_projection) returns
the exact component of a victim on the span of the anchors' pre-key vectors. The joint decoder keeps that component
and takes the remainder from the anchor-signed eigen alignment (EigenAligner.decoder with anchors):

    z_hat = f W + (I - P_A) j(f),    P_A the orthogonal projector onto the span of the anchors' pre-key vectors.

Restricted to the anchor span it equals the projection decoder; with p = 0 aligned axes, j(f) is the AUX mean, so the
decoder adds (I - P_A) mu_aux and equals the projection decoder when that mean lies in the anchor span (e.g. is 0).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ijiss_btp.attacks.anchor import fit_projection
from ijiss_btp.attacks.keyfree import EigenAligner, check_anchors


PINV_RCOND = 1e-15   # numpy.linalg.pinv's default relative cut-off, which fit_projection uses


def span_basis(z_known: np.ndarray) -> np.ndarray:
    """Orthonormal basis (m, rank) of the row space of z_known, with the projection decoder's rank cut-off, so that
    P_A spans the directions fit_projection inverts."""
    _, sv, vt = np.linalg.svd(z_known, full_matrices=False)
    if sv.size == 0 or sv[0] == 0.0:
        return np.zeros((z_known.shape[1], 0))
    return vt[: int(np.sum(sv > PINV_RCOND * sv[0]))].T


class AnchorSpan:
    """r >= 1 anchor pairs fitted once: the projection decoder and an orthonormal basis of the anchors' pre-key span."""

    def __init__(self, f_anchor: np.ndarray, z_known: np.ndarray) -> None:
        f_anchor, z_known = check_anchors(f_anchor, z_known)
        self.project: Callable[[np.ndarray], np.ndarray] = fit_projection(f_anchor, z_known)
        self.basis = span_basis(z_known)

    def complement(self, z: np.ndarray) -> np.ndarray:
        """(I - P_A) applied to the rows of z."""
        return z - (z @ self.basis) @ self.basis.T

    def joint(self, projected: np.ndarray, aligned: np.ndarray) -> np.ndarray:
        """Joint estimate from the projection decoder's output and an aligned estimate of the same rows."""
        return projected + self.complement(aligned)


def joint_decoder(f_anchor: np.ndarray, z_known: np.ndarray, aligner: EigenAligner, p: int) -> Callable:
    """The joint decoder with the top p axes of `aligner`, their signs resolved by the same anchors."""
    span = AnchorSpan(f_anchor, z_known)
    aligned = aligner.decoder(p, anchors=(f_anchor, z_known))
    return lambda f: span.joint(span.project(f), aligned(f))
