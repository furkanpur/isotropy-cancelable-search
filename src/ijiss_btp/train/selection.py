"""Training-set selection, validation split and D2 disjointness checks for the control encoder (pandas only).

The training set is the set of compilation images of the fingers of the first corpus (D1).  The checks measure how it
relates to the evaluation corpus D2: shared fingers, shared exact image content, and shared subjects.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def subject_key(identity: pd.Series) -> pd.Series:
    """The compilation's subject link: the identity without its last ':' component."""
    return identity.astype(str).str.rsplit(":", n=1).str[0]


def select_training_rows(manifest: pd.DataFrame, identities: pd.DataFrame, train_identities: Iterable[str],
                         keys: pd.DataFrame, partition: str = "development") -> tuple[pd.DataFrame, dict]:
    """Manifest rows of `train_identities` with exclusion == "none", in `partition` (IDENTITIES.csv), with exact
    duplicates within a finger collapsed (the first path in sorted order is kept).  Adds a `key` column from `keys`
    (joined on `path`; a missing key is an error).  Returns (rows, counts)."""
    wanted = set(train_identities)
    rows = manifest[manifest["identity"].isin(wanted) & (manifest["exclusion"] == "none")]
    rows = rows.sort_values("path", kind="stable")
    k = keys.drop_duplicates("path").set_index("path")["key"]
    key = rows["path"].map(k)
    if key.isna().any():
        raise ValueError(f"{int(key.isna().sum())} selected rows have no content key")
    rows = rows.assign(key=key.to_numpy())
    counts = {"rows_before": int(len(rows)), "identities_requested": len(wanted),
              "identities_found": int(rows["identity"].nunique())}
    in_partition = set(identities.loc[identities["partition"] == partition, "identity"])
    dropped = ~rows["identity"].isin(in_partition)
    counts["identities_dropped_partition"] = int(rows.loc[dropped, "identity"].nunique())
    counts["rows_dropped_partition"] = int(dropped.sum())
    rows = rows[~dropped]
    dup = rows.duplicated(["identity", "key"], keep="first")
    counts["duplicate_rows_collapsed"] = int(dup.sum())
    rows = rows[~dup].reset_index(drop=True)
    shared = rows.groupby("key")["identity"].transform("nunique") > 1   # one key under several fingers: kept
    counts["keys_shared_across_fingers"] = int(rows.loc[shared, "key"].nunique())
    counts["rows_with_shared_keys"] = int(shared.sum())
    counts["rows"] = int(len(rows))
    counts["identities"] = int(rows["identity"].nunique())
    return rows, counts


def validation_split(rows: pd.DataFrame, frac: float = 0.1, seed: int = 20261115) -> pd.Series:
    """Boolean per row (True = validation), constant within a finger.  Per source, floor(frac * n + 0.5) fingers are
    drawn with numpy.random.default_rng(seed) from that source's sorted identity list; sources are visited in sorted
    order and share one generator."""
    rng = np.random.default_rng(seed)
    chosen: set[str] = set()
    for _, g in sorted(rows.groupby("source"), key=lambda kv: kv[0]):
        ids = np.array(sorted(g["identity"].unique()))
        n_val = int(np.floor(frac * len(ids) + 0.5))
        chosen.update(rng.choice(ids, size=n_val, replace=False).tolist())
    return rows["identity"].isin(chosen).astype(bool)


def _by_source(d: pd.DataFrame, flag: pd.Series) -> dict:
    """Distinct fingers (identities) of `d` where `flag`, per D2 source."""
    sel = d[flag.to_numpy()].drop_duplicates("identity")
    return {str(s): int(n) for s, n in sorted(sel.groupby("source").size().items())}


def disjointness(train: pd.DataFrame, d2: pd.DataFrame, keys: pd.DataFrame) -> dict:
    """identity_overlap, content_overlap (exact keys; raises when a D2 row has no key), d2_rows, d2_keys_checked, subjects {train, d2, shared} by subject_key and by
    (source, subject), d2_fingers_with_shared_subject {total, by_source}, d2_fingers_without_shared_subject
    {total, by_source}."""
    identity_overlap = len(set(train["identity"]) & set(d2["identity"]))
    d2_key = d2["path"].map(keys.drop_duplicates("path").set_index("path")["key"])
    if d2_key.isna().any():   # an unchecked D2 image would make the gate pass vacuously
        raise ValueError(f"{int(d2_key.isna().sum())} D2 rows have no content key (path missing or null key)")
    d2_keys = set(d2_key)
    content_overlap = len(set(train["key"]) & d2_keys)
    tr_sub, d2_sub = set(subject_key(train["identity"])), set(subject_key(d2["identity"]))
    tr_ss = set(zip(train["source"].astype(str), train["subject"].astype(str)))
    d2_ss = set(zip(d2["source"].astype(str), d2["subject"].astype(str)))
    shared = subject_key(d2["identity"]).isin(tr_sub)
    with_, without = _by_source(d2, shared), _by_source(d2, ~shared)
    return {
        "identity_overlap": int(identity_overlap),
        "content_overlap": int(content_overlap),
        "d2_rows": int(len(d2)), "d2_keys_checked": len(d2_keys),
        "subjects": {"by_subject_key": {"train": len(tr_sub), "d2": len(d2_sub), "shared": len(tr_sub & d2_sub)},
                     "by_source_subject": {"train": len(tr_ss), "d2": len(d2_ss), "shared": len(tr_ss & d2_ss)}},
        "d2_fingers_with_shared_subject": {"total": sum(with_.values()), "by_source": with_},
        "d2_fingers_without_shared_subject": {"total": sum(without.values()), "by_source": without},
    }


def subject_flags(train: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    """One row per D2 identity: identity, source, subject_shared (bool)."""
    f = d2.drop_duplicates("identity")[["identity", "source"]].reset_index(drop=True)
    f["subject_shared"] = subject_key(f["identity"]).isin(set(subject_key(train["identity"]))).to_numpy()
    return f
