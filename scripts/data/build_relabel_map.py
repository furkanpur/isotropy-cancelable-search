"""Build a relabel map (file,identity) for a corpus by exact pixel-content matching against a labelled corpus.

Usage:
  python scripts/data/build_relabel_map.py --manifest M.parquet --images DIR --compilation COMPILATION_ROOT --out map.csv [--workers 8]
  python scripts/data/build_relabel_map.py --manifest M.parquet --key path --path-map HOST_PREFIX=CONTAINER_PREFIX \
      [--path-map ...] --compilation COMPILATION_ROOT --out map.csv [--workers 8]

Key: SHA-1 of the image size plus its 8-bit grayscale pixel bytes (insensitive to the file container, so a
re-encoded PNG of a BMP matches).  Only labelled-corpus rows with a non-empty identity are used; the
`exclusion` column is a quality flag and is not filtered on.  An image is kept only if its content maps to
exactly one identity.

Two key modes for the output column `file`:
- basename (default): the basename of the manifest path; images are opened as DIR/<basename>, and a manifest with a
  repeated basename raises.
- path: the exact manifest `path` string, for corpora whose basenames repeat across folders (`load_sin_corpus` with
  relabel_key "path" matches it).  A repeated path raises.  Absolute manifest paths are opened through `--path-map`
  (repeatable; the longest matching prefix wins, on a path-component boundary); with maps given, a path that matches
  none of them is not opened and counts as unmapped (and unmatched).  Without maps, paths are opened as written,
  relative ones under DIR when --images is given.

The output CSV has columns `file,identity`; `<out>.summary.json` holds aggregates only (no file names, no
prefixes).  Run it inside the container on the data host; never commit its outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image


def content_key(path: str | Path) -> str | None:
    """SHA-1 of repr(size) + grayscale bytes; None if the file cannot be read as an image."""
    try:
        with Image.open(path) as im:
            g = im.convert("L")
            return hashlib.sha1(repr(g.size).encode() + g.tobytes()).hexdigest()
    except Exception:
        return None


def _hash_all(paths: list[str], workers: int) -> list[str | None]:
    if workers <= 1:
        return [content_key(p) for p in paths]
    with Pool(workers) as pool:
        return pool.map(content_key, paths, chunksize=64)


KEY_MODES = ("basename", "path")


def parse_path_map(entries: list[str] | None) -> list[tuple[str, str]]:
    """`HOST_PREFIX=CONTAINER_PREFIX` strings -> (host, container) pairs, trailing slashes removed."""
    maps = []
    for e in entries or []:
        host, sep, cont = e.partition("=")
        host, cont = host.rstrip("/"), cont.rstrip("/")
        if not sep or not host or not cont:
            raise ValueError(f"path map entry must be HOST_PREFIX=CONTAINER_PREFIX, got {e!r}")
        maps.append((host, cont))
    return maps


def match_index(path: str, maps: list[tuple[str, str]]) -> int | None:
    """Index of the longest host prefix that matches `path` on a path-component boundary; None if none matches."""
    best = None
    for i, (host, _) in enumerate(maps):
        if (path == host or path.startswith(host + "/")) and (best is None or len(host) > len(maps[best][0])):
            best = i
    return best


def map_path(path: str, maps: list[tuple[str, str]]) -> str | None:
    """Replace the longest matching host prefix (on a path-component boundary); None if no prefix matches."""
    i = match_index(path, maps)
    return None if i is None else maps[i][1] + path[len(maps[i][0]):]


def build(manifest: str | Path, images: str | Path | None, compilation: str | Path, out: str | Path, workers: int = 4,
          key: str = "basename", path_map: list[str] | None = None) -> dict:
    if key not in KEY_MODES:
        raise ValueError(f"key must be one of {KEY_MODES}, got {key!r}")
    if key == "basename" and images is None:
        raise ValueError("basename mode needs the images directory")
    compilation, out = Path(compilation), Path(out)
    images = Path(images) if images is not None else None
    maps = parse_path_map(path_map)
    gm = pd.read_csv(compilation / "MANIFEST.csv", dtype=str, keep_default_na=False)
    gm = gm[gm["identity"].str.strip() != ""].reset_index(drop=True)
    paths = [str(p) for p in pd.read_parquet(manifest)["path"]]
    basenames = [os.path.basename(p) for p in paths]
    if key == "basename":
        names = basenames
        if len(set(names)) != len(names):
            raise ValueError(f"manifest contains {len(names) - len(set(names))} duplicate basenames")
        locations: list[str | None] = [str(images / n) for n in names]
    else:
        names = paths
        if len(set(names)) != len(names):
            raise ValueError(f"manifest contains {len(names) - len(set(names))} duplicate paths")
        locations = [map_path(p, maps) for p in paths] if maps else list(paths)
        if images is not None:
            locations = [str(images / loc) if loc is not None else None for loc in locations]
    gkeys = _hash_all([str(compilation / p) for p in gm["path"]], workers)
    sources = gm["source"] if "source" in gm else pd.Series([""] * len(gm))
    key_ids: dict[str, set] = defaultdict(set)
    ident_source: dict[str, str] = {}
    for k, ident, src in zip(gkeys, gm["identity"], sources):
        if k is None:
            continue
        key_ids[k].add(ident)
        ident_source.setdefault(ident, src)

    opened = [i for i, loc in enumerate(locations) if loc is not None]   # unmapped paths (path mode) are not opened
    dkeys: list[str | None] = [None] * len(names)
    for i, k in zip(opened, _hash_all([locations[i] for i in opened], workers)):
        dkeys[i] = k
    per_file: dict[str, set] = defaultdict(set)
    n_ambiguous = n_unmatched = 0
    for n, k in zip(names, dkeys):
        ids = key_ids.get(k) if k is not None else None
        if not ids:
            n_unmatched += 1
        elif len(ids) > 1:
            n_ambiguous += 1
        else:
            per_file[n] |= ids
    # A key (basename or path) that resolves to two different identities is ambiguous too.
    rows = []
    for n in sorted(per_file):
        if len(per_file[n]) == 1:
            rows.append((n, next(iter(per_file[n]))))
        else:
            n_ambiguous += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["file", "identity"]).to_csv(out, index=False)

    per_finger = Counter(i for _, i in rows)
    summary = {
        "n_images": len(names),
        "n_compilation_labelled_rows": len(gm),
        "matched": len(rows),
        "unmatched": n_unmatched,
        "compilation_unreadable": sum(k is None for k in gkeys),
        "images_unreadable": sum(dkeys[i] is None for i in opened),
        "ambiguous": n_ambiguous,
        "true_fingers": len(per_finger),
        "images_per_finger_hist": {str(k): v for k, v in sorted(Counter(per_finger.values()).items())},
        "per_source": dict(sorted(Counter(ident_source.get(i, "") for _, i in rows).items())),
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "compilation_manifest_sha256": hashlib.sha256((compilation / "MANIFEST.csv").read_bytes()).hexdigest(),
    }
    if key == "path":   # the basename-mode summary keeps its earlier fields
        summary.update({"key": key, "path_map_entries": len(maps), "paths_unmapped": len(names) - len(opened),
                        "duplicate_basenames": len(basenames) - len(set(basenames))})
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"matched={summary['matched']} unmatched={n_unmatched} ambiguous={n_ambiguous} "
          f"fingers={summary['true_fingers']}")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", default=None, help="image directory (required in basename mode)")
    ap.add_argument("--compilation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--key", choices=KEY_MODES, default="basename",
                    help="output key: the manifest basename (default) or the exact manifest path")
    ap.add_argument("--path-map", action="append", default=None, metavar="HOST_PREFIX=CONTAINER_PREFIX",
                    help="path mode: open manifest paths under HOST_PREFIX at CONTAINER_PREFIX (repeatable)")
    a = ap.parse_args(argv)
    if a.key == "basename" and a.images is None:
        ap.error("--images is required in basename mode")
    build(a.manifest, a.images, a.compilation, a.out, a.workers, key=a.key, path_map=a.path_map)


if __name__ == "__main__":
    main()
