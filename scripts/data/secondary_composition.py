"""Composition of the secondary corpus after its relabel and after removing the fingers it shares with the first corpus.

Counts over the secondary manifest after the path-keyed relabel with the secondary map, against the identities of the
first corpus's map, per manifest `dataset` value, per family (FVC by edition, SD301a, SD302b, SD302d), per source
(FVC, SD301, SD302 with SD302b and SD302d together) and over all rows:
- rows, and rows dropped by the relabel (no unique identity in the secondary map), split into rows outside every
  path-map entry (never opened) and rows inside one (unreadable, unmatched or ambiguous);
- rows and fingers (identities) after the relabel, shared with the first corpus, and kept.
Finger counts are unions within each group, so they do not add up across groups that share fingers (SD302b and SD302d
hold the same fingers); the totals are unions over all rows.

Built-in checks; the script stops before writing anything when one fails:
- against a committed run's metadata.json (an isotropy run with a shifted AUX or a scale run with secondary
  distractors): shared rows = `secondary_removed_shared_identity`; kept rows = the rows left before the near-duplicate
  filter (isotropy: shifted AUX rows + near-duplicate removals; scale: distractors loaded - shared rows); rows dropped by
  the relabel = the `dropped=` count of the run's provenance string;
- against the secondary map's summary (optional): rows after the relabel = `matched`, fingers after the relabel =
  `true_fingers`, dropped rows = `unmatched` + `ambiguous`, rows outside every path-map entry = `paths_unmapped`.

Usage:
  python scripts/data/secondary_composition.py --sec-manifest manifest.parquet --sec-map secondary_relabel.csv \
      --d1-map d1_relabel.csv --path-map HOST_PREFIX=CONTAINER_PREFIX [--path-map ...] --run-metadata metadata.json \
      [--relabel-summary secondary_relabel.summary.json] --out composition.json
`--path-map` takes the entries used to build the secondary map (only the host prefixes are used). Run it inside the
container on the data host: it reads the corpus manifest and the maps. The output holds labels, counts and hashes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from build_relabel_map import parse_path_map
from check_secondary_map import entry_index

_FVC = re.compile(r"^fvc(\d{4})/DB(\d+)$")
_SD = re.compile(r"^sd(\d{3})([a-z]?)$")


def dataset_label(value: str) -> str:
    """Public dataset name of a manifest `dataset` value: fvc2000/DB1 -> FVC2000 DB1, sd301a -> SD301a."""
    if m := _FVC.match(value):
        return f"FVC{m[1]} DB{m[2]}"
    if m := _SD.match(value):
        return f"SD{m[1]}{m[2]}"
    return value


def family_of(value: str) -> str:
    """Family: FVC by edition (FVC2000), each NIST subset on its own (SD301a, SD302b, SD302d)."""
    if m := _FVC.match(value):
        return f"FVC{m[1]}"
    return dataset_label(value)


def source_of(value: str) -> str:
    """Source: FVC (all editions), SD301, SD302 (all subsets)."""
    if _FVC.match(value):
        return "FVC"
    if m := _SD.match(value):
        return f"SD{m[1]}"
    return value


def _counts(rows: np.ndarray, matched: np.ndarray, outside: np.ndarray, shared: np.ndarray, sec_id: np.ndarray) -> dict:
    after, sh, kept = rows & matched, rows & shared, rows & matched & ~shared
    return {"rows": int(rows.sum()), "dropped_by_relabel": int((rows & ~matched).sum()),
            "dropped_outside_path_map": int((rows & ~matched & outside).sum()),
            "dropped_inside_path_map": int((rows & ~matched & ~outside).sum()),
            "rows_after_relabel": int(after.sum()), "fingers_after_relabel": int(len(np.unique(sec_id[after]))),
            "rows_shared": int(sh.sum()), "fingers_shared": int(len(np.unique(sec_id[sh]))),
            "rows_kept": int(kept.sum()), "fingers_kept": int(len(np.unique(sec_id[kept])))}


def composition(dataset: Sequence[str], entry: Sequence[int], sec_id: Sequence[str], d1_ids: Sequence[str]) -> dict:
    """Per-dataset, per-family, per-source and total counts (no identity value or path reaches the output).

    dataset: the secondary manifest's `dataset` value per row; entry: path-map entry per row (-1: none); sec_id: the
    secondary-map identity per row ("" when the relabel drops the row); d1_ids: the identities of the first corpus's map.
    """
    dataset = np.asarray(dataset, dtype=str)
    entry = np.asarray(entry, dtype=int)
    sec_id = np.asarray(sec_id, dtype=str)
    if len(entry) != len(dataset) or len(sec_id) != len(dataset):
        raise ValueError("dataset, entry and sec_id must have the same number of rows")
    matched = sec_id != ""
    shared = matched & np.isin(sec_id, np.unique(np.asarray(d1_ids, dtype=str)))
    outside = entry < 0
    order = sorted(set(dataset), key=lambda v: (source_of(v), family_of(v), dataset_label(v)))

    def counts(mask: np.ndarray) -> dict:
        return _counts(mask, matched, outside, shared, sec_id)

    fam = {v: family_of(v) for v in order}
    src = {v: source_of(v) for v in order}
    fam_src = {fam[v]: src[v] for v in order}
    families, sources = list(dict.fromkeys(fam.values())), list(dict.fromkeys(src.values()))
    by_dataset = [{"label": dataset_label(v), "family": fam[v], "source": src[v], **counts(dataset == v)}
                  for v in order]
    by_family = [{"label": f, "source": fam_src[f], "datasets": [dataset_label(v) for v in order if fam[v] == f],
                  **counts(np.isin(dataset, [v for v in order if fam[v] == f]))} for f in families]
    by_source = [{"label": s, "families": [f for f in families if fam_src[f] == s],
                  **counts(np.isin(dataset, [v for v in order if src[v] == s]))} for s in sources]
    return {"by_dataset": by_dataset, "by_family": by_family, "by_source": by_source,
            "totals": counts(np.ones(len(dataset), dtype=bool))}


def _dropped(source: str | None) -> int | None:
    m = re.search(r":dropped=(\d+)", source or "")
    return int(m[1]) if m else None


def run_metadata_counts(meta: Mapping) -> dict:
    """Shared rows, kept rows before the near-duplicate filter and relabel-dropped rows recorded by a committed run."""
    shared = meta.get("secondary_removed_shared_identity")
    if shared is None:
        raise ValueError("run metadata has no secondary_removed_shared_identity (no exclusion in this run)")
    run = {"config": (meta.get("config") or {}).get("name"), "git_sha": meta.get("git_sha"),
           "started_utc": meta.get("started_utc")}
    if "shifted" in (meta.get("aux_sizes") or {}) and meta.get("shifted_aux_removed") is not None:
        return {"kind": "isotropy", "shared_rows": int(shared),
                "kept_rows_before_near_duplicate_filter": int(meta["aux_sizes"]["shifted"] + meta["shifted_aux_removed"]),
                "relabel_dropped": _dropped(meta.get("shifted_aux_source")), "run": run}
    if meta.get("distractors_loaded") is not None:
        return {"kind": "scale", "shared_rows": int(shared),
                "kept_rows_before_near_duplicate_filter": int(meta["distractors_loaded"] - shared),
                "relabel_dropped": _dropped(meta.get("distractor_source")), "run": run}
    raise ValueError("run metadata is neither an isotropy run with a shifted AUX nor a scale run with distractors")


def check(totals: Mapping, run_counts: Mapping, summary: Mapping | None = None) -> dict:
    """Compare the totals with the run metadata and the map summary; raise ValueError naming every mismatch."""
    pairs = {"shared_rows_vs_run_metadata": (totals["rows_shared"], run_counts["shared_rows"]),
             "kept_rows_vs_run_metadata": (totals["rows_kept"], run_counts["kept_rows_before_near_duplicate_filter"])}
    if run_counts.get("relabel_dropped") is not None:
        pairs["relabel_dropped_vs_run_metadata"] = (totals["dropped_by_relabel"], run_counts["relabel_dropped"])
    if summary is not None:
        pairs.update({
            "rows_after_relabel_vs_map_summary": (totals["rows_after_relabel"], summary["matched"]),
            "fingers_after_relabel_vs_map_summary": (totals["fingers_after_relabel"], summary["true_fingers"]),
            "dropped_vs_map_summary": (totals["dropped_by_relabel"], summary["unmatched"] + summary["ambiguous"]),
            "outside_path_map_vs_map_summary": (totals["dropped_outside_path_map"], summary["paths_unmapped"]),
        })
    res = {k: {"computed": int(c), "reference": int(r), "match": int(c) == int(r)} for k, (c, r) in pairs.items()}
    bad = [f"{k}: computed {v['computed']}, reference {v['reference']}" for k, v in res.items() if not v["match"]]
    if bad:
        raise ValueError("composition does not match the committed aggregates: " + "; ".join(bad))
    return res


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sec-manifest", required=True, help="secondary manifest (parquet with path and dataset)")
    ap.add_argument("--sec-map", required=True, help="path-keyed relabel map of the secondary corpus (file,identity)")
    ap.add_argument("--d1-map", required=True, help="relabel map of the first corpus (file,identity)")
    ap.add_argument("--path-map", action="append", required=True, metavar="HOST_PREFIX=CONTAINER_PREFIX")
    ap.add_argument("--run-metadata", required=True, help="metadata.json of a committed run with the exclusion")
    ap.add_argument("--relabel-summary", default=None, help="summary JSON of the secondary map (optional check)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    man = pd.read_parquet(a.sec_manifest)
    paths = man["path"].astype(str).tolist()
    if len(set(paths)) != len(paths):
        raise ValueError("the secondary manifest repeats a path")
    smap = pd.read_csv(a.sec_map, dtype=str, keep_default_na=False)
    lookup = dict(zip(smap["file"], smap["identity"]))
    sec_id = np.array([lookup.get(p, "") for p in paths], dtype=object).astype(str)
    d1_ids = pd.read_csv(a.d1_map, dtype=str, keep_default_na=False)["identity"].to_numpy(dtype=str)
    maps = parse_path_map(a.path_map)
    res = composition(man["dataset"].astype(str), entry_index(paths, maps), sec_id, d1_ids)
    meta = json.loads(Path(a.run_metadata).read_text())
    run_counts = run_metadata_counts(meta)
    summary = json.loads(Path(a.relabel_summary).read_text()) if a.relabel_summary else None
    res["checks"] = check(res["totals"], run_counts, summary)   # raises before anything is written
    res["provenance"] = {
        "secondary_map_sha256": _sha256(a.sec_map),
        "d1_map_sha256": _sha256(a.d1_map),
        "secondary_manifest_sha256": _sha256(a.sec_manifest),
        "run_metadata_sha256": _sha256(a.run_metadata),
        "relabel_summary_sha256": _sha256(a.relabel_summary) if a.relabel_summary else None,
        "run": {"kind": run_counts["kind"], **run_counts["run"]},
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
    print(f"rows={t['rows']} after_relabel={t['rows_after_relabel']}/{t['fingers_after_relabel']} "
          f"shared={t['rows_shared']}/{t['fingers_shared']} kept={t['rows_kept']}/{t['fingers_kept']} (rows/fingers)")
    return res


if __name__ == "__main__":
    main()
