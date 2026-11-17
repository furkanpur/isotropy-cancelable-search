"""Identity-disjoint, seeded DEV / TGT / AUX splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IdentitySplit:
    dev_ids: np.ndarray
    tgt_ids: np.ndarray
    aux_ids: np.ndarray
    rows_by_id: dict

    def rows(self, ids) -> np.ndarray:
        return np.concatenate([self.rows_by_id[i] for i in ids])

    def enrolled_row(self, ident) -> int:
        return int(self.rows_by_id[ident][0])

    def known_row(self, ident) -> int:
        return int(self.rows_by_id[ident][1])

    def probe_rows(self, ident) -> np.ndarray:
        return self.rows_by_id[ident][1:3]


def group_rows(ids: np.ndarray) -> dict:
    """Map each identity to its sorted row indices."""
    ids = np.asarray(ids)
    order = np.argsort(ids, kind="stable")
    s = ids[order]
    bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    return {s[a]: np.sort(order[a:b]) for a, b in zip(bounds[:-1], bounds[1:])}


def make_split(ids: np.ndarray, n_dev: int, n_tgt: int, seed: int, min_tgt_images: int = 3) -> IdentitySplit:
    """Targets come from identities with >= min_tgt_images rows (enrolled, colluder/probe, second probe)."""
    rows_by_id = group_rows(ids)
    uniq = np.array(sorted(rows_by_id))
    np.random.default_rng(seed).shuffle(uniq)
    eligible = np.array([i for i in uniq if len(rows_by_id[i]) >= min_tgt_images])
    if len(eligible) < n_tgt:
        raise ValueError(f"only {len(eligible)} identities have >= {min_tgt_images} images; need {n_tgt}")
    tgt = eligible[:n_tgt]
    taken = set(tgt.tolist())
    rest = np.array([i for i in uniq if i not in taken])
    if len(rest) <= n_dev:
        raise ValueError(f"need more than {n_dev} non-target identities for DEV plus a non-empty AUX")
    return IdentitySplit(dev_ids=rest[:n_dev], tgt_ids=tgt, aux_ids=rest[n_dev:], rows_by_id=rows_by_id)


def corpus_counts(ids: np.ndarray, min_tgt_images: int) -> dict:
    """Image and identity counts of a loaded corpus (aggregates only)."""
    _, per_id = np.unique(ids, return_counts=True)
    return {"images": int(len(ids)), "identities": int(len(per_id)),
            "identities_ge_min_tgt_images": int((per_id >= min_tgt_images).sum())}


def split_counts(split: IdentitySplit) -> dict:
    """Identity and image counts of the DEV / TGT / AUX partitions of one split."""
    parts = (("dev", split.dev_ids), ("tgt", split.tgt_ids), ("aux", split.aux_ids))
    out = {f"{name}_identities": int(len(ids)) for name, ids in parts}
    out.update({f"{name}_images": int(sum(len(split.rows_by_id[i]) for i in ids)) for name, ids in parts})
    return out
