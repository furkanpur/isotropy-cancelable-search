"""Check the path-keyed secondary relabel map on the images the secondary corpus shares with the first corpus.

The secondary map was built by opening each secondary manifest path at its copy in the labelled corpus through
path-map entries, so a content match there only shows that the mapped file exists in the labelled corpus. The mapping
itself is checked here on the secondary rows that duplicate a first-corpus row (same image and model: uncentred cosine
>= 0.999 between the two corpora's embeddings); for those rows the first corpus's own map (built from its own image
files, without path maps) gives an independent identity.

Per path-map entry, labelled by the secondary manifest's `dataset` values of its rows (never by a path prefix), and per
model: rows in the manifest, rows matched by the secondary map, rows with a duplicate first-corpus row, of those the
rows to which both maps assign an identity (comparable) and the rows whose identities agree; the same counts at the
looser threshold (default 0.95) and over all rows. Entries with no exact duplicate are listed; rows outside every entry
are reported per dataset label.

Usage:
  python scripts/data/check_secondary_map.py --sec-root DIR --sec-map secondary_relabel.csv --d1-root DIR \
      --d1-map d1_relabel.csv --path-map HOST_PREFIX=CONTAINER_PREFIX [--path-map ...] --out FILE \
      [--models base large] [--thresholds 0.999 0.95]
`--path-map` takes the entries used to build the secondary map; only their host prefixes are used here. Both roots are
in the SIN layout (manifest.parquet with `path`, the secondary one also with `dataset`, and emb_<model>.npy). Run it
inside the container on the data host: it reads embeddings. The output holds counts, labels and hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from build_relabel_map import match_index, parse_path_map


def entry_index(paths: Sequence[str], maps: list[tuple[str, str]]) -> np.ndarray:
    """Path-map entry of each path (longest host prefix on a component boundary), -1 outside every entry."""
    out = np.full(len(paths), -1, dtype=int)
    for k, p in enumerate(paths):
        i = match_index(str(p), maps)
        if i is not None:
            out[k] = i
    return out


def nearest(sec: np.ndarray, d1: np.ndarray, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Highest uncentred cosine of each secondary row to any first-corpus row, and the index of that row."""
    s = sec / np.maximum(np.linalg.norm(sec, axis=1, keepdims=True), 1e-12)
    p = d1 / np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-12)
    best, arg = np.empty(len(s)), np.empty(len(s), dtype=int)
    for i in range(0, len(s), block):
        m = s[i:i + block] @ p.T
        arg[i:i + block] = m.argmax(1)
        best[i:i + block] = m[np.arange(len(m)), arg[i:i + block]]
    return best, arg


def _counts(rows: np.ndarray, cos: np.ndarray, sec_id: np.ndarray, d1_id: np.ndarray, thr: float) -> dict:
    dup = rows & (cos >= thr)
    comparable = dup & (sec_id != "") & (d1_id != "")
    same = comparable & (sec_id == d1_id)
    return {"duplicate_rows": int(dup.sum()), "comparable": int(comparable.sum()), "same_identity": int(same.sum())}


def aggregate(dataset: Sequence[str], entry: Sequence[int], matched: Sequence[bool], sec_id: Sequence[str],
              nn: Mapping[str, tuple[np.ndarray, np.ndarray]], n_entries: int,
              thresholds: Sequence[float] = (0.999, 0.95)) -> dict:
    """Aggregate the check from per-row inputs (no identity value or path reaches the output).

    dataset: the secondary manifest's `dataset` value per row; entry: path-map entry per row (-1: none); matched: row
    has a secondary-map identity; sec_id: that identity ("" if none); nn: per model, (highest cosine to a first-corpus
    row, the first-corpus map identity of that row or ""). The first threshold defines an exact duplicate.
    """
    dataset = np.asarray(dataset, dtype=str)
    entry = np.asarray(entry, dtype=int)
    matched = np.asarray(matched, dtype=bool)
    sec_id = np.asarray(sec_id, dtype=str)
    n = len(dataset)
    if len(entry) != n or len(matched) != n or len(sec_id) != n or any(
            len(c) != n or len(i) != n for c, i in nn.values()):
        raise ValueError("all per-row inputs must have the same number of rows")
    keys = [f"{t:g}" for t in thresholds]
    per_model = {m: (np.asarray(c, dtype=float), np.asarray(i, dtype=str)) for m, (c, i) in nn.items()}

    def stats(rows: np.ndarray) -> dict:
        return {m: {k: _counts(rows, c, sec_id, i, t) for k, t in zip(keys, thresholds)}
                for m, (c, i) in per_model.items()}

    entries = []
    for e in range(n_entries):
        rows = entry == e
        models = stats(rows)
        entries.append({"label": "+".join(sorted(set(dataset[rows]))) or f"(entry {e}: no rows)",
                        "rows": int(rows.sum()), "matched": int((rows & matched).sum()), "models": models,
                        "no_exact_duplicate_coverage": all(v[keys[0]]["duplicate_rows"] == 0 for v in models.values())})
    outside = entry < 0
    unmapped = [{"label": lab, "rows": int((outside & (dataset == lab)).sum()),
                 "matched": int((outside & (dataset == lab) & matched).sum())} for lab in sorted(set(dataset[outside]))]
    return {"thresholds": [float(t) for t in thresholds], "models": list(per_model), "entries": entries,
            "entries_without_exact_duplicate_coverage": [x["label"] for x in entries if x["no_exact_duplicate_coverage"]],
            "unmapped": unmapped,
            "totals": {"rows": n, "matched": int(matched.sum()), "models": stats(np.ones(n, dtype=bool))}}


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(root: Path, model: str, n_rows: int) -> np.ndarray:
    x = np.load(root / f"emb_{model}.npy").astype(np.float64)
    if len(x) != n_rows:
        raise ValueError(f"emb_{model}.npy has {len(x)} rows for {n_rows} manifest rows")
    return x


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sec-root", required=True, help="secondary corpus, SIN layout")
    ap.add_argument("--sec-map", required=True, help="path-keyed relabel map of the secondary corpus (file,identity)")
    ap.add_argument("--d1-root", required=True, help="first corpus, SIN layout")
    ap.add_argument("--d1-map", required=True, help="basename-keyed relabel map of the first corpus (file,identity)")
    ap.add_argument("--path-map", action="append", required=True, metavar="HOST_PREFIX=CONTAINER_PREFIX")
    ap.add_argument("--models", nargs="+", default=["base", "large"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.999, 0.95])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    maps = parse_path_map(a.path_map)
    sec_root, d1_root = Path(a.sec_root), Path(a.d1_root)
    sm = pd.read_parquet(sec_root / "manifest.parquet")
    paths = sm["path"].astype(str).tolist()
    if len(set(paths)) != len(paths):
        raise ValueError("the secondary manifest repeats a path")
    smap = pd.read_csv(a.sec_map, dtype=str, keep_default_na=False)
    sec_lookup = dict(zip(smap["file"], smap["identity"]))
    sec_id = np.array([sec_lookup.get(p, "") for p in paths], dtype=object).astype(str)
    dm = pd.read_parquet(d1_root / "manifest.parquet")
    dmap = pd.read_csv(a.d1_map, dtype=str, keep_default_na=False)
    d1_lookup = dict(zip(dmap["file"], dmap["identity"]))
    d1_id = np.array([d1_lookup.get(os.path.basename(str(p)), "") for p in dm["path"]], dtype=object).astype(str)
    nn = {}
    for m in a.models:
        cos, arg = nearest(_load(sec_root, m, len(sm)), _load(d1_root, m, len(dm)))
        nn[m] = (cos, d1_id[arg])
    res = aggregate(sm["dataset"].astype(str), entry_index(paths, maps), sec_id != "", sec_id, nn, len(maps),
                    a.thresholds)
    res["provenance"] = {
        "secondary_map_sha256": _sha256(a.sec_map),
        "d1_map_sha256": _sha256(a.d1_map),
        "secondary_manifest_sha256": _sha256(sec_root / "manifest.parquet"),
        "d1_manifest_sha256": _sha256(d1_root / "manifest.parquet"),
        "secondary_embeddings_sha256": {m: _sha256(sec_root / f"emb_{m}.npy") for m in a.models},
        "d1_embeddings_sha256": {m: _sha256(d1_root / f"emb_{m}.npy") for m in a.models},
        "models": list(a.models),
        "thresholds": [float(t) for t in a.thresholds],
        "path_map_entries": len(maps),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "git_sha": os.environ.get("CANVEC_GIT_SHA", "unknown"),
        "image": os.environ.get("CANVEC_IMAGE"),
        "image_id": os.environ.get("CANVEC_IMAGE_ID"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    t = res["totals"]
    print(f"rows={t['rows']} matched={t['matched']} entries={len(maps)} "
          f"without_exact_duplicates={len(res['entries_without_exact_duplicate_coverage'])}")
    return res


if __name__ == "__main__":
    main()
