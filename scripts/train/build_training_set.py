"""Training set of the control encoder: the compilation images of the first corpus's fingers (D1), a per-finger
validation split, and the checks that it is disjoint from the second corpus (D2).  Runs inside a container on the data
host; the image-level outputs stay there and only the aggregate JSON comes back.

Selection (ijiss_btp.train.selection): rows of the identities of the D1 relabel map with `exclusion == "none"` whose
identity is in the `development` partition, exact duplicates (content key) within a finger collapsed; the selected
images are then probed (both loaders must decode them) and unusable ones are dropped and counted.  The validation
split holds out 10% of the fingers per source.

Outputs in --out (data host only): `train_manifest.parquet` (path, identity, source, subject, finger, device, key, val)
and `d2_subject_flags.parquet` (identity, source, subject_shared).  --json receives the aggregates: counts by source,
devices, images per finger, selection counts, the disjointness result, input SHA-256 hashes and provenance.

The script exits non-zero (after writing the JSON) when the identity overlap or the content overlap with D2 is not 0, or a probed key differs from the key table
(a D2 row without a content key raises before any output).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from extract_d2 import environment, probe_images, sha256_file  # noqa: E402  (same repository, scripts/data)

from ijiss_btp.train.selection import (disjointness, select_training_rows, subject_flags,  # noqa: E402
                                       validation_split)

SEED = 20261115
VAL_FRAC = 0.1
TRAIN_COLUMNS = ["path", "identity", "source", "subject", "finger", "device", "key", "val"]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _per_source(rows: pd.DataFrame) -> dict:
    out = {}
    for s, g in sorted(rows.groupby("source")):
        tr, va = g[~g["val"]], g[g["val"]]
        out[str(s)] = {"images": int(len(g)), "fingers": int(g["identity"].nunique()),
                       "train_images": int(len(tr)), "train_fingers": int(tr["identity"].nunique()),
                       "val_images": int(len(va)), "val_fingers": int(va["identity"].nunique())}
    return out


def build(a: argparse.Namespace) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    compilation = Path(a.compilation)
    manifest = _read_csv(compilation / "MANIFEST.csv")
    identities = _read_csv(compilation / "IDENTITIES.csv")
    d1_ids = set(_read_csv(Path(a.d1_map))["identity"]) - {""}
    keys = pd.read_parquet(a.compilation_keys, columns=["path", "key"])
    d2 = pd.read_parquet(a.d2_manifest)

    rows, counts = select_training_rows(manifest, identities, d1_ids, keys)
    probe, pstats = probe_images([str(compilation / p) for p in rows["path"]], a.workers)
    usable = probe["usable"].to_numpy(dtype=bool)
    mismatch = int((probe["key"].to_numpy() != rows["key"].to_numpy())[usable].sum())
    dropped_ids = int(rows.loc[~usable, "identity"].nunique())
    rows = rows[usable].reset_index(drop=True)
    rows["val"] = validation_split(rows, VAL_FRAC, SEED).to_numpy()
    train = rows[TRAIN_COLUMNS]

    per_finger = train.groupby("identity").size()
    res = {
        "data_source": ("real: compilation real part, the fingers of the first corpus's relabel map (development "
                        "partition, exclusion none, exact duplicates within a finger collapsed); D2 is the second "
                        "corpus of the study"),
        "config": {"seed": SEED, "validation_fraction": VAL_FRAC, "partition": "development"},
        "inputs_sha256": {name: sha256_file(path) for name, path in
                          [("MANIFEST.csv", compilation / "MANIFEST.csv"), ("IDENTITIES.csv", compilation / "IDENTITIES.csv"),
                           ("d1_relabel.csv", a.d1_map), ("compilation_keys.parquet", a.compilation_keys),
                           ("d2_manifest.parquet", a.d2_manifest)]},
        "selection": dict(counts, d1_identities=len(d1_ids)),
        "probe": {"probed": pstats["probed"], "unusable": pstats["unusable"],
                  "unusable_by_mode": pstats["unusable_by_mode"], "fingers_with_an_unusable_image": dropped_ids,
                  "probe_key_differs_from_key_table": mismatch},
        "images": int(len(train)), "fingers": int(train["identity"].nunique()),
        "train_images": int((~train["val"]).sum()), "val_images": int(train["val"].sum()),
        "train_fingers": int(train.loc[~train["val"], "identity"].nunique()),
        "val_fingers": int(train.loc[train["val"], "identity"].nunique()),
        "by_source": _per_source(train),
        "devices": {"distinct": int(train["device"].replace("", pd.NA).dropna().nunique()),
                    "images_by_device": {str(k): int(v) for k, v in train["device"].value_counts().sort_index().items()}},
        "images_per_finger": {"min": int(per_finger.min()), "median": float(per_finger.median()),
                              "max": int(per_finger.max())},
    }
    res["disjointness"] = disjointness(train, d2, keys)
    res["provenance"] = {"script_sha256": sha256_file(Path(__file__).resolve()), "env": environment(),
                         "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    return res, train, subject_flags(train, d2)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compilation", required=True, help="compilation `real` directory (MANIFEST.csv, IDENTITIES.csv, images/)")
    ap.add_argument("--d1-map", required=True, help="d1_relabel.csv")
    ap.add_argument("--compilation-keys", required=True, help="compilation_keys.parquet (path, key)")
    ap.add_argument("--d2-manifest", required=True, help="manifest.parquet of the second corpus")
    ap.add_argument("--out", required=True, help="output directory (image-level tables)")
    ap.add_argument("--json", required=True, help="aggregate JSON file")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    res, train, flags = build(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train_manifest.parquet", index=False)
    flags.to_parquet(out / "d2_subject_flags.parquet", index=False)
    Path(a.json).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    d = res["disjointness"]
    print(f"training set: {res['images']} images, {res['fingers']} fingers "
          f"({res['train_fingers']} train, {res['val_fingers']} validation); identity overlap "
          f"{d['identity_overlap']}, content overlap {d['content_overlap']}", flush=True)
    if d["identity_overlap"] != 0 or d["content_overlap"] != 0:
        print("REFUSED: the training set overlaps D2", file=sys.stderr)
        return 1
    if res["probe"]["probe_key_differs_from_key_table"] != 0:
        print("REFUSED: probed content keys differ from the key table", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
