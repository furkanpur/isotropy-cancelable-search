"""Worst-case training-overlap measurement for the evaluation encoders (aggregates only).

The encoders' training data was not recorded with their weights.  This script measures how much of each evaluation
corpus occurs in the most plausible candidate training pool, as a worst case: the encoder is assumed to have been
trained on the whole pool.  Every step runs inside a container on the data host, with the data and the code mounted
read-only, `--network none` and the invoking user.  File lists, per-image keys and matches, embeddings and per-image
maxima stay in the output directory on the host; only the JSON aggregates (no path, file name or identity) leave it.

  # (a) first corpus (D1) against candidate pool A, by exact content key (CPU image):
  python scripts/data/training_overlap.py exact --pool /data/pool --expect-files N \
      --d1-manifest /data/d1/manifest.parquet --d1-images /data/d1_images --d1-map /data/relabel/d1_relabel.csv \
      --compilation /data/compilation --compilation-keys /data/keys/compilation_keys.parquet --out /out --json /out/<a>.json

  # (b1) coarse keys (CPU image): calibration pairs (an old test image, packaged like pool B, whose coarse key
  #      occurs on exactly one compilation image: its original) and the match table of pool B against compilation:
  python scripts/data/training_overlap.py coarse --pool /data/pool --expect-files N \
      --old-images /data/old_d2_test --compilation /data/compilation --compilation-keys /data/keys/compilation_keys.parquet --out /out

  # (b2) finger-vit embeddings (GPU image; encode_fingervit of extract_d2.py) of pool B, of both sides of every
  #      calibration pair and of the D1 images:
  python scripts/data/training_overlap.py embed --pool /data/pool --old-images /data/old_d2_test --compilation /data/compilation \
      --models /models --d1-manifest /data/d1/manifest.parquet --d1-images /data/d1_images --out /out

  # (b3) calibration and measurement (CPU image): the threshold from the calibration pairs against the second corpus's
  #      (D2) genuine and impostor pairs, then the maximum cosine of every D2 (and D1) image to any pool image:
  python scripts/data/training_overlap.py measure --d2 /data/corpus_d2 \
      --old-fingervit /data/old_results/finger-vit_result/features.npz --compilation /data/compilation \
      --compilation-keys /data/keys/compilation_keys.parquet --d1-manifest /data/d1/manifest.parquet \
      --d1-map /data/relabel/d1_relabel.csv --out /out --json /out/<b>.json

  # (c) DeepPrint: no measurement; the D2 sources against the training sets named in the flx documentation:
  python scripts/data/training_overlap.py deepprint --d2 /data/corpus_d2 --json /out/<c>.json

(a) Image level: D1 images whose exact content key occurs in the pool.  Finger level: relabel-map identities with at
least one labelled compilation image (any impression) whose exact key occurs in the pool; a pool key that occurs under several
compilation identities is ambiguous and counted apart.  (b) Pool B was re-encoded (224 x 224 RGB) during packaging, so exact
keys fail and finger-vit serves as the near-duplicate detector: a D2 image counts when its maximum cosine to any pool
image is at least the threshold.  The threshold tau is the 1st percentile of the calibration pairs' cosines (both sides
embedded now by the measurement's code path), used only if it exceeds the 99.9th percentile of the D2 genuine pairs;
the counts are reported at tau, at two neighbouring thresholds and at that genuine percentile either way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from build_relabel_map import content_key  # same directory: the exact content key of the relabel maps
from check_secondary_map import nearest
from extract_d2 import (FINGERVIT_DIR, PREPROCESSING, SEED, code_hashes, cosine_rows, encode_fingervit, environment,
                          l2_normalize, load_compilation_keys, sha256_file, software_versions, weight_hashes)

POOL_A_LABEL = "candidate training pool A: extended dataset train split"
POOL_B_LABEL = "candidate training pool B: packaged benchmark train split"
LIST_RULE = ("SHA-256 of the sorted list of file paths relative to the pool folder, one per line, each followed by a "
             "newline (UTF-8); the list stays on the data host")
EXACT_RULE = ("exact content key (build_relabel_map.content_key): SHA-1 of the image size and its 8-bit grayscale "
              "bytes; insensitive to the file container, not to resizing")
COARSE_RULE = ("coarse key: PIL convert('L'), resize to 16 x 16 with Image.BILINEAR, uint8 values shifted right by 4 "
               "bits, SHA-1 of the 256 bytes")
POSITIVE_RULE = ("an old test image (packaged 224 x 224 RGB, like pool B) whose coarse key occurs on exactly one compilation "
                 "file (every MANIFEST path with a file, labelled or not), paired with that file, taken as its original; "
                 "both sides embedded now with encode_fingervit, as pool B and D2 are")
TAU_RULE = ("tau = 1st percentile of the calibration pairs' cosines, used only if it exceeds the 99.9th percentile of "
            "the D2 genuine pairs (separable); otherwise the overlap is reported at several thresholds and flagged as "
            "not separable")
DETECTOR = ("maximum cosine of each image's finger-vit embedding to any pool image's; an image counts at a threshold "
            "when that maximum is at least the threshold. The detector targets re-encoded copies of the corpus images; "
            "other impressions of a finger in the pool are expected to score like genuine pairs, of which at most 0.1% "
            "reach a separable tau")
COPY_CHECK_RULE = ("D2 images whose coarse key is on no other compilation file and occurs in the pool, so that a pool image is "
                   "very likely a re-encoded copy; these copies are selected by the same coarse key as the calibration "
                   "positives, so the check measures recall only for copies whose coarse key survived packaging and says "
                   "nothing about recall for copies the coarse key misses")
PER_PAIR_NOTE = ("one-sided 95% Clopper-Pearson upper bound on the rate per pair, given only where no pair reaches the "
                 "threshold; it is not the detector's per-image false-positive rate, because the detector takes the "
                 "maximum over all pool embeddings")
CHAIN_KEYS = ("calibration_pairs", "compilation_manifest", "pool_file_list")
QUANTILE_LEVELS = (0.1, 1, 5, 50, 95, 99, 99.9)
QUANTILE_METHOD = "numpy.percentile, linear interpolation"
IMPOSTOR_PAIRS = 1_000_000
IMPORTED = ("build_relabel_map.py", "check_secondary_map.py", "extract_d2.py")
FLX_TRAINING_SETS = {"SFinGe synthetic": ("sfinge",), "MCYT330": ("mcyt",)}
FLX_DOCUMENTATION = ("flx README, section 'Databases': the paper's models were trained on SFinGe synthetic fingerprints "
                     "(6,000 subjects x 10 impressions; validation 2,000 x 4) and on MCYT330 optical and capacitive "
                     "(first 2,000 subjects); the flx experiment definitions build DeepPrint_TexMinu with 2000 + 6000 "
                     "training subjects, the configuration the released weights are loaded with "
                     "(num_training_subjects=8000, num_dims=256)")
FLX_README_SECTION = "Databases"
FLX_README_SOURCE = ("README.md of the flx repository whose package directory the extraction image holds; the image "
                     "itself contains no README")

# files in the output directory; they stay on the data host
POOL_A_LIST, POOL_A_KEYS, D1_MATCHES = "pool_a_files.txt", "pool_a_keys.parquet", "d1_exact_matches.parquet"
POOL_B_LIST, POOL_B_KEYS, COMPILATION_COARSE = "pool_b_files.txt", "pool_b_keys.parquet", "compilation_coarse.parquet"
PAIRS, COARSE_SUMMARY, EMBED_SUMMARY, POOL_NAMES = ("calibration_pairs.csv", "coarse_summary.json",
                                                    "embed_summary.json", "emb_pool_names.txt")
EMB = {"pool": "emb_pool.npy", "pair_packaged": "emb_pair_packaged.npy", "pair_original": "emb_pair_original.npy",
       "d1": "emb_d1.npy"}
D2_NEAREST, D1_NEAREST = "d2_pool_nearest.parquet", "d1_pool_nearest.parquet"


# ---- keys and file lists ----------------------------------------------------------------------------------------
def coarse_key(path: str | Path) -> str | None:
    """SHA-1 of the 16 x 16 bilinear grayscale thumbnail quantised to 4 bits; None if the file cannot be read."""
    try:
        with Image.open(path) as im:
            thumb = np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.uint8) >> 4
        return hashlib.sha1(thumb.tobytes()).hexdigest()
    except Exception:
        return None


def hash_paths(fn: Callable[[str], str | None], paths: Sequence[str], workers: int) -> list[str | None]:
    if workers <= 1:
        return [fn(p) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, paths, chunksize=64))


def list_files(root: str | Path) -> list[str]:
    """Sorted POSIX paths, relative to `root`, of the regular files below it."""
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def list_digest(names: Sequence[str]) -> str:
    """SHA-256 of the list as write_list writes it: one name per line, each followed by a newline."""
    return hashlib.sha256("".join(f"{n}\n" for n in names).encode("utf-8")).hexdigest()


def write_list(path: str | Path, names: Sequence[str]) -> str:
    Path(path).write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
    return list_digest(names)


def pool_listing(root: str | Path, expect: int | None) -> list[str]:
    """The pool's file list; refuses a pool whose file count differs from `expect`."""
    names = list_files(root)
    if expect is not None and len(names) != expect:
        raise SystemExit(f"the pool holds {len(names)} files, expected {expect}")
    return names


def _is_key(k) -> bool:
    return isinstance(k, str)


# ---- aggregation --------------------------------------------------------------------------------------------------
def _frac(k: int, n: int) -> float | None:
    return round(k / n, 6) if n else None


def tally(hit: Iterable[bool], groups: Iterable[str], word: str) -> dict:
    """Number of True values in `hit`, overall and per group (source), with the fractions."""
    hit, groups = np.asarray(list(hit), dtype=bool), np.asarray(list(groups), dtype=str)
    if hit.shape != groups.shape:
        raise ValueError("hit and groups differ in length")
    per = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        k, n = int(hit[m].sum()), int(m.sum())
        per[g] = {word: k, "of": n, "fraction": _frac(k, n)}
    k, n = int(hit.sum()), int(hit.size)
    return {word: k, "of": n, "fraction": _frac(k, n), "per_source": per}


def finger_hits(hit: Iterable[bool], ids: Iterable[str], sources: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per identity (sorted): whether any of its images is hit, and its source."""
    df = pd.DataFrame({"identity": np.asarray(list(ids), dtype=str), "hit": np.asarray(list(hit), dtype=bool),
                       "source": np.asarray(list(sources), dtype=str)})
    g = df.groupby("identity", sort=True).agg(hit=("hit", "any"), source=("source", "first"),
                                              n_sources=("source", "nunique"))
    if (g["n_sources"] > 1).any():
        raise ValueError(f"{int((g['n_sources'] > 1).sum())} identities span several sources")
    return g["hit"].to_numpy(dtype=bool), g["source"].to_numpy(dtype=str)


# ---- (a) exact overlap --------------------------------------------------------------------------------------------
def load_key_table(path: str | Path, compilation_manifest_sha256: str) -> tuple[pd.DataFrame, dict, dict]:
    """(identity, key) rows of the labelled compilation key table, identity -> source, and the table's provenance;
    load_compilation_keys refuses a table built from another MANIFEST.csv."""
    table, _, info = load_compilation_keys(path, compilation_manifest_sha256, set())
    src = pd.read_parquet(path, columns=["identity", "source"]).astype(str).drop_duplicates()
    several = src["identity"].duplicated(keep=False)
    if several.any():
        raise ValueError(f"{src.loc[several, 'identity'].nunique()} identities span several sources in the key table")
    return table[["identity", "key"]].reset_index(drop=True), dict(zip(src["identity"], src["source"])), info


def key_identities(table: pd.DataFrame) -> dict:
    """Content key -> the identities whose labelled compilation images have it."""
    t = table[table["key"].map(_is_key)]
    return {k: frozenset(v) for k, v in t.groupby("key")["identity"]}


def exact_overlap(d1: pd.DataFrame, d1_ids: Iterable[str], table: pd.DataFrame, id_source: Mapping[str, str],
                  pool_keys: Sequence[str | None]) -> dict:
    """Aggregates of (a).  d1: one row per D1 manifest image with its identity ('' when the relabel map assigns none)
    and its exact key (None if unreadable); d1_ids: the relabel-map identities; table: (identity, key) of every
    labelled compilation image; pool_keys: the exact key of every pool file (None if unreadable)."""
    key_ids = key_identities(table)
    pool = {k for k in pool_keys if _is_key(k)}
    d1_ids = set(d1_ids)
    missing = d1_ids - set(id_source)
    if missing:
        raise ValueError(f"{len(missing)} relabel-map identities are not in the compilation key table")
    ident = d1["identity"].astype(str).to_numpy()
    keys = d1["key"].to_numpy(dtype=object)
    in_pool = np.array([_is_key(k) and k in pool for k in keys], dtype=bool)
    assigned = ident != ""
    images = tally(in_pool[assigned], [id_source[i] for i in ident[assigned]], "in_pool")
    images["unassigned"] = {"images": int((~assigned).sum()), "in_pool": int(in_pool[~assigned].sum())}
    images["unreadable"] = int(sum(not _is_key(k) for k in keys))

    t = table[table["key"].map(_is_key) & table["identity"].isin(d1_ids)]
    t_pool = t["key"].isin(pool).to_numpy(dtype=bool)
    t_unique = t["key"].map(lambda k: len(key_ids[k]) == 1).to_numpy(dtype=bool)
    hit_unique, hit_any = set(t.loc[t_pool & t_unique, "identity"]), set(t.loc[t_pool, "identity"])
    order = sorted(d1_ids)
    f_src = [id_source[f] for f in order]
    fingers = tally([f in hit_unique for f in order], f_src, "in_pool")
    fingers["only_via_ambiguous_keys"] = len(hit_any - hit_unique)
    fingers["worst_case"] = tally([f in hit_any for f in order], f_src, "in_pool")
    fingers["compilation_images"], fingers["compilation_images_in_pool"] = int(len(t)), int(t_pool.sum())

    cat, per, ambiguous = Counter(), Counter(), set()
    for k in pool_keys:
        if not _is_key(k):
            cat["unreadable"] += 1
        elif k not in key_ids:
            cat["none"] += 1
        elif len(key_ids[k]) == 1:
            cat["exact"] += 1
            per[id_source[next(iter(key_ids[k]))]] += 1
        else:
            cat["ambiguous"] += 1
            ambiguous.add(k)
    pool_match = {"files": len(pool_keys), "exact": {"images": cat["exact"], "per_source": dict(sorted(per.items()))},
                  "exact_ambiguous": {"images": cat["ambiguous"], "keys": len(ambiguous)}, "none": cat["none"],
                  "unreadable": cat["unreadable"]}
    return {"d1_images": images, "d1_fingers": fingers, "pool_match": pool_match}


# ---- (b) coarse matching ------------------------------------------------------------------------------------------
def unique_matches(query: Sequence[str | None], ref: Sequence[str | None]) -> tuple[np.ndarray, list[str]]:
    """For each query key, the index of the single reference entry with that key (-1 otherwise) and the status:
    'unique', 'ambiguous' (several reference entries), 'unmatched' or 'unreadable' (no query key)."""
    where, several = {}, set()
    for i, k in enumerate(ref):
        if _is_key(k):
            if k in where:
                several.add(k)
            else:
                where[k] = i
    idx, status = np.full(len(query), -1, dtype=np.int64), []
    for j, k in enumerate(query):
        if not _is_key(k):
            status.append("unreadable")
        elif k in several:
            status.append("ambiguous")
        elif k in where:
            idx[j] = where[k]
            status.append("unique")
        else:
            status.append("unmatched")
    return idx, status


def pool_b_match(pool_exact: Sequence[str | None], pool_coarse: Sequence[str | None], exact_keys: set,
                 ref: pd.DataFrame) -> dict:
    """Pool images against compilation: exact (the exact key of a labelled compilation image), else a coarse key on exactly one compilation
    file (per source of that file), on several, or on none; unreadable files apart.  ref: coarse and source per file."""
    idx, status = unique_matches(list(pool_coarse), ref["coarse"].tolist())
    src = ref["source"].astype(str).to_numpy()
    c, per = Counter(), Counter()
    for e, i, s in zip(pool_exact, idx, status):
        if s == "unreadable":
            c["unreadable"] += 1
        elif _is_key(e) and e in exact_keys:
            c["exact"] += 1
        elif s == "unique":
            c["coarse_unique"] += 1
            per[src[i]] += 1
        elif s == "ambiguous":
            c["coarse_ambiguous"] += 1
        else:
            c["none"] += 1
    return {"files": len(status), "exact": c["exact"],
            "coarse_unique": {"images": c["coarse_unique"], "per_source": dict(sorted(per.items()))},
            "coarse_ambiguous": c["coarse_ambiguous"], "none": c["none"], "unreadable": c["unreadable"]}


# ---- (b) calibration and measurement --------------------------------------------------------------------------------
def quantiles(x: Iterable[float]) -> dict:
    x = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    q = {"n": int(x.size), "min": float(x.min())}
    q.update({f"p{lv:g}": float(np.percentile(x, lv)) for lv in QUANTILE_LEVELS})
    q["max"] = float(x.max())
    return q


def genuine_scores(x: np.ndarray, ids: Iterable[str]) -> np.ndarray:
    """Cosine of every pair of rows with the same identity."""
    x = l2_normalize(x)
    out = [np.empty(0)]
    for rows in pd.DataFrame({"identity": np.asarray(list(ids), dtype=str)}).groupby("identity").indices.values():
        if len(rows) > 1:
            out.append((x[rows] @ x[rows].T)[np.triu_indices(len(rows), 1)])
    return np.concatenate(out)


def impostor_pairs(ids: Iterable[str], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`n` distinct row pairs (i < j) with different identities, drawn uniformly at random (seeded)."""
    codes = pd.factorize(pd.Series(np.asarray(list(ids), dtype=str)))[0].astype(np.int64)
    m = len(codes)
    sizes = np.bincount(codes) if m else np.zeros(0, dtype=np.int64)
    available = m * (m - 1) // 2 - int((sizes * (sizes - 1) // 2).sum())
    if n > available:
        raise ValueError(f"{n} impostor pairs requested, {available} exist")
    rng = np.random.default_rng(seed)
    got = np.empty(0, dtype=np.int64)
    while len(got) < n:
        k = max(2 * (n - len(got)), 1024)
        i, j = rng.integers(0, m, size=k), rng.integers(0, m, size=k)
        ok = codes[i] != codes[j]
        got = np.concatenate([got, np.minimum(i, j)[ok] * m + np.maximum(i, j)[ok]])
        _, first = np.unique(got, return_index=True)
        got = got[np.sort(first)]                                    # distinct pairs, in the order drawn
    got = got[:n]
    return got // m, got % m


def pair_scores(x: np.ndarray, a: np.ndarray, b: np.ndarray, block: int = 16384) -> np.ndarray:
    x = l2_normalize(x)
    out = np.empty(len(a))
    for s in range(0, len(a), block):
        out[s:s + block] = np.einsum("ij,ij->i", x[a[s:s + block]], x[b[s:s + block]])
    return out


def tau_rule(pos_q: Mapping[str, float], gen_q: Mapping[str, float]) -> dict:
    tau, limit = float(pos_q["p1"]), float(gen_q["p99.9"])
    return {"rule": TAU_RULE, "tau": tau, "d2_genuine_p99.9": limit, "separable": bool(tau > limit)}


def report_thresholds(pos_q: Mapping[str, float], gen_q: Mapping[str, float]) -> list[dict]:
    """tau, its neighbours in the positive quantiles, and the genuine limit, in ascending order."""
    th = [{"name": "tau = positives p1", "value": float(pos_q["p1"])},
          {"name": "positives p0.1", "value": float(pos_q["p0.1"])},
          {"name": "positives p5", "value": float(pos_q["p5"])},
          {"name": "D2 genuine p99.9", "value": float(gen_q["p99.9"])}]
    return sorted(th, key=lambda t: t["value"])


def counts_above(maxcos: np.ndarray, ids: Iterable[str], sources: Iterable[str], thresholds: Sequence[dict]) -> list:
    """Per threshold: images whose maximum cosine is at least the threshold, and fingers with at least one such image,
    overall and per source."""
    maxcos, ids, sources = np.asarray(maxcos, dtype=np.float64), list(ids), list(sources)
    out = []
    for t in thresholds:
        hit = maxcos >= t["value"]
        f_hit, f_src = finger_hits(hit, ids, sources)
        out.append({"name": t["name"], "value": t["value"], "images": tally(hit, sources, "above"),
                    "fingers": tally(f_hit, f_src, "above")})
    return out


def coarse_copy_check(d2_coarse: Sequence[str | None], compilation_coarse_counts: Mapping[str, int], pool_coarse: set,
                      maxcos: np.ndarray, tau: float, sources: Iterable[str]) -> dict:
    """How many coarse-key copies of D2 images (COPY_CHECK_RULE) the detector places at or above tau, overall and per
    source (every source, also those without copies)."""
    copy = np.array([_is_key(k) and compilation_coarse_counts.get(k, 0) == 1 and k in pool_coarse for k in d2_coarse],
                    dtype=bool)
    maxcos, src = np.asarray(maxcos, dtype=np.float64), np.asarray(list(sources), dtype=str)
    m = maxcos[copy]
    above = int((m >= tau).sum())
    by_source = {}
    for s in sorted(set(src.tolist())):
        sel = copy & (src == s)
        k, n = int((maxcos[sel] >= tau).sum()), int(sel.sum())
        by_source[s] = {"copies": n, "at_or_above_tau": k, "fraction": _frac(k, n)}
    return {"definition": COPY_CHECK_RULE, "d2_images_with_a_coarse_copy_in_pool": int(copy.sum()),
            "at_or_above_tau": above, "fraction": _frac(above, int(copy.sum())),
            "min_max_cosine": float(m.min()) if m.size else None, "by_source": by_source}


def per_pair_upper_bound(n: int, alpha: float = 0.05) -> tuple[float, str]:
    """One-sided (1 - alpha) Clopper-Pearson upper bound for zero events in n pairs, and how it was computed:
    clopper_pearson_upper of the library when it imports, else the closed form 1 - alpha^(1/n) (equal for zero)."""
    try:
        from ijiss_btp.eval.stats import clopper_pearson_upper
    except ImportError:
        return 1.0 - alpha ** (1.0 / n), "closed form 1 - 0.05^(1/n)"
    return clopper_pearson_upper(0, n, alpha), "ijiss_btp.eval.stats.clopper_pearson_upper"


def pair_count(scores: np.ndarray, value: float) -> dict:
    """Pairs whose cosine is at least `value`, of n; with the per-pair upper bound when there are none."""
    scores = np.asarray(scores, dtype=np.float64)
    k, n = int((scores >= value).sum()), int(scores.size)
    out = {"at_or_above": k, "n": n, "fraction": k / n if n else None}
    if k == 0 and n:
        out["upper_95_per_pair"] = per_pair_upper_bound(n)[0]
    return out


def _by_source(values: np.ndarray, sources: np.ndarray, summary: Callable[[np.ndarray], dict]) -> dict:
    return {s: summary(values[sources == s]) for s in sorted(set(sources.tolist()))}


def calibrate(pos: np.ndarray, pos_sources: Iterable[str], gen: np.ndarray, gen_sources: Iterable[str],
              imp: np.ndarray) -> dict:
    """The calibration as a function of the three cosine distributions: their quantiles, the tau rule, the reported
    thresholds, the genuine and impostor pairs at or above each threshold (with the per-pair bound where there are
    none), and the per-source view of the positives and of the genuine pairs at tau."""
    pos, gen, imp = (np.asarray(v, dtype=np.float64) for v in (pos, gen, imp))
    for name, v in (("calibration pairs", pos), ("D2 genuine pairs", gen), ("D2 impostor pairs", imp)):
        if v.size == 0:
            raise SystemExit(f"no {name}: the threshold cannot be calibrated")
    pos_sources, gen_sources = np.asarray(list(pos_sources), dtype=str), np.asarray(list(gen_sources), dtype=str)
    if len(pos_sources) != len(pos) or len(gen_sources) != len(gen):
        raise ValueError("every pair needs a source")
    qp, qg, qi = quantiles(pos), quantiles(gen), quantiles(imp)
    rule = tau_rule(qp, qg)
    thresholds = report_thresholds(qp, qg)
    tau = rule["tau"]
    by_pos = _by_source(pos, pos_sources, lambda x: {
        "n": int(x.size), "min": float(x.min()), "p1": float(np.percentile(x, 1)), "p50": float(np.median(x)),
        "below_tau": int((x < tau).sum())})
    by_gen = _by_source(gen, gen_sources, lambda x: {
        "n": int(x.size), "at_or_above_tau": int((x >= tau).sum()), "p99.9": float(np.percentile(x, 99.9)),
        "max": float(x.max())})
    return {"quantiles": {"positives": qp, "d2_genuine": qg, "d2_impostor": qi}, "tau_rule": rule,
            "thresholds": thresholds,
            "pairs_at_or_above": [{"name": t["name"], "value": t["value"], "genuine": pair_count(gen, t["value"]),
                                   "impostor": pair_count(imp, t["value"])} for t in thresholds],
            "per_pair_bound": {"note": PER_PAIR_NOTE, "method": per_pair_upper_bound(1)[1]},
            "by_source": {"positives": by_pos, "d2_genuine": by_gen}}


def chain_problems(coarse_inputs: Mapping[str, str], embed_summary: Mapping, recomputed: Mapping[str, str]) -> list:
    """Mismatches between the hashes the coarse and embedding steps recorded and those recomputed now (a missing
    recorded or recomputed hash counts as a mismatch)."""
    problems = [f"the coarse step recorded another {k}" for k in CHAIN_KEYS if coarse_inputs.get(k) != recomputed.get(k)]
    for verb, block in (("wrote", "outputs_sha256"), ("read", "inputs_sha256")):
        for k, v in sorted((embed_summary.get(block) or {}).items()):
            if recomputed.get(k) != v:
                problems.append(f"the embedding step {verb} another {k}")
    return problems


# ---- (c) DeepPrint ------------------------------------------------------------------------------------------------
def deepprint_sources(summary: Mapping) -> dict:
    per = summary["composition"]["per_source"]
    sources = sorted(per)
    hits = [s for s in sources if any(tok in s.lower() for toks in FLX_TRAINING_SETS.values() for tok in toks)]
    named = ", ".join(FLX_TRAINING_SETS)
    statement = (f"The training sets named in the flx documentation ({named}) are not among the D2 sources."
                 if not hits else f"D2 sources {hits} match training sets named in the flx documentation ({named}).")
    return {"d2_sources": {s: per[s] for s in sources}, "flx_training_sets": list(FLX_TRAINING_SETS),
            "flx_documentation": FLX_DOCUMENTATION, "named_training_sets_among_d2_sources": hits,
            "statement": statement}


def fvc_databases(images_root: str | Path) -> dict:
    """Database folders (names starting DB<digit>) in an image tree, keyed by their parent folder (the FVC edition)."""
    found: dict = {}
    for dirpath, dirnames, _ in os.walk(images_root):
        for d in dirnames:
            if re.match(r"(?i)db\d", d):
                found.setdefault(os.path.basename(dirpath), []).append(d)
    return {edition: sorted(dbs) for edition, dbs in sorted(found.items())}


def fvc_statement(dbs: Mapping[str, Sequence[str]]) -> str:
    """The observed FVC databases in words; DB4 of every FVC edition is the SFinGe-generated database."""
    numbers = sorted({int(re.match(r"(?i)db(\d)", n).group(1)) for names in dbs.values() for n in names})
    if 4 in numbers:
        held = sorted(f"{e}/{n}" for e, names in dbs.items() for n in names if re.match(r"(?i)db4", n))
        return "compilation real part holds FVC DB4 (SFinGe-generated) folders: " + ", ".join(held)
    if not numbers:
        return "compilation real part: no FVC database folders"
    contiguous = numbers == list(range(numbers[0], numbers[-1] + 1))
    span = f"DB{numbers[0]}-DB{numbers[-1]}" if contiguous else ", ".join(f"DB{n}" for n in numbers)
    return f"compilation real part: FVC {span} only; no SFinGe-generated DB4"


def manifest_db4_rows(compilation: str | Path) -> dict:
    """compilation MANIFEST rows whose path has a DB4 folder: how many, how many have a file or an identity, and their
    exclusion values."""
    compilation = Path(compilation)
    man = pd.read_csv(compilation / "MANIFEST.csv", dtype=str, keep_default_na=False,
                      usecols=["path", "identity", "exclusion"])
    rows = man[man["path"].str.contains(r"(?i)(?:^|/)db4", regex=True)]
    return {"rows": int(len(rows)), "with_file": int(sum(os.path.isfile(compilation / p) for p in rows["path"])),
            "with_identity": int((rows["identity"].str.strip() != "").sum()),
            "exclusion": {str(k): int(v) for k, v in sorted(rows["exclusion"].value_counts().items())}}


def markdown_section(text: str, title: str) -> list[str]:
    """Lines under the heading `title`, up to the next heading of the same or a higher level ([] if absent)."""
    out, level = [], None
    for line in text.splitlines():
        m = re.match(r"^(#+)\s+(.*?)\s*$", line)
        if level is None:
            if m and m.group(2) == title:
                level = len(m.group(1))
        elif m and len(m.group(1)) <= level:
            break
        else:
            out.append(line)
    return out


# ---- inputs and provenance ------------------------------------------------------------------------------------------
def load_d1(manifest: str | Path, relabel_map: str | Path) -> tuple[pd.DataFrame, set, dict]:
    """First-corpus images in manifest order: file (basename, as the images are stored) and identity ('' when the
    relabel map assigns none)."""
    files = [os.path.basename(str(p)) for p in pd.read_parquet(manifest, columns=["path"])["path"]]
    if len(set(files)) != len(files):
        raise ValueError("the first-corpus manifest repeats a basename")
    rm = pd.read_csv(relabel_map, dtype=str, keep_default_na=False)
    if list(rm.columns[:2]) != ["file", "identity"] or rm["file"].duplicated().any():
        raise ValueError("the relabel map needs unique files in the columns file,identity")
    lookup = dict(zip(rm["file"], rm["identity"]))
    unknown = set(lookup) - set(files)
    if unknown:
        raise ValueError(f"{len(unknown)} relabel-map files are not in the first-corpus manifest")
    d1 = pd.DataFrame({"file": files, "identity": [lookup.get(f, "") for f in files]})
    info = {"manifest_images": len(files), "relabel_map_rows": int(len(rm)),
            "identities": int(rm["identity"].nunique()), "unassigned_images": int((d1["identity"] == "").sum())}
    return d1, set(rm["identity"]), info


def provenance() -> dict:
    import PIL
    import pyarrow
    here = Path(__file__).resolve()
    return {"script_sha256": sha256_file(here),
            "imported_sha256": {name: sha256_file(here.parent / name) for name in IMPORTED},
            "env": environment(),
            "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "PIL": PIL.__version__, "pyarrow": pyarrow.__version__}}


def _dump(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _pool_block(label: str, names: Sequence[str], digest: str) -> dict:
    return {"label": f"{label}, {len(names):,} files", "files": len(names), "file_list_sha256": digest,
            "file_list_rule": LIST_RULE}


# ---- commands ------------------------------------------------------------------------------------------------------
def cmd_exact(a: argparse.Namespace) -> int:
    t0 = time.time()
    out, compilation = Path(a.out), Path(a.compilation)
    out.mkdir(parents=True, exist_ok=True)
    names = pool_listing(a.pool, a.expect_files)
    digest = write_list(out / POOL_A_LIST, names)
    pool_keys = hash_paths(content_key, [str(Path(a.pool) / n) for n in names], a.workers)
    pd.DataFrame({"name": names, "key": pool_keys}).to_parquet(out / POOL_A_KEYS, index=False)
    manifest_sha256 = sha256_file(compilation / "MANIFEST.csv")
    table, id_source, key_info = load_key_table(a.compilation_keys, manifest_sha256)
    d1, d1_ids, d1_info = load_d1(a.d1_manifest, a.d1_map)
    d1["key"] = hash_paths(content_key, [str(Path(a.d1_images) / f) for f in d1["file"]], a.workers)
    res = exact_overlap(d1[["identity", "key"]], d1_ids, table, id_source, pool_keys)
    pool = {k for k in pool_keys if _is_key(k)}
    d1.assign(in_pool=[_is_key(k) and k in pool for k in d1["key"]]).to_parquet(out / D1_MATCHES, index=False)
    report = {
        "data_source": ("real: first-corpus (D1) images and candidate training pool A; identities and sources from the "
                        "labelled rows of compilation `real` (internal research corpus of real fingerprint images); no "
                        "synthetic data"),
        "question": ("worst case: the D1 encoders are assumed to have been trained on the whole of candidate training "
                     "pool A"),
        "pool": _pool_block(POOL_A_LABEL, names, digest),
        "match_rule": EXACT_RULE,
        "thresholds": {"rule": "exact content key equality; no numeric threshold"},
        "d1": d1_info,
        "finger_rule": ("a relabel-map identity counts when at least one of its labelled compilation images (any impression, "
                        "any exclusion or partition) has its exact key in the pool; a pool key under several compilation "
                        "identities is ambiguous: in_pool counts unique keys only, worst_case adds the ambiguous ones"),
        **res,
        "compilation_key_table": key_info,
        "inputs_sha256": {"d1_manifest": sha256_file(a.d1_manifest), "d1_relabel_map": sha256_file(a.d1_map),
                          "compilation_manifest": manifest_sha256, "compilation_key_table": key_info["table_sha256"],
                          "pool_file_list": digest},
        "outputs_sha256": {"pool_a_keys": sha256_file(out / POOL_A_KEYS), "d1_exact_matches": sha256_file(out / D1_MATCHES)},
        "provenance": provenance(),
        "seconds": round(time.time() - t0, 1),
    }
    _dump(a.json, report)
    img, fin = res["d1_images"], res["d1_fingers"]
    print(f"exact: pool {len(names)} files; D1 images in pool {img['in_pool']}/{img['of']}; D1 fingers in pool "
          f"{fin['in_pool']}/{fin['of']} (+{fin['only_via_ambiguous_keys']} only through ambiguous keys)", flush=True)
    return 0


def cmd_coarse(a: argparse.Namespace) -> int:
    t0 = time.time()
    out, compilation = Path(a.out), Path(a.compilation)
    out.mkdir(parents=True, exist_ok=True)
    names = pool_listing(a.pool, a.expect_files)
    digest = write_list(out / POOL_B_LIST, names)
    pool_paths = [str(Path(a.pool) / n) for n in names]
    pool_exact = hash_paths(content_key, pool_paths, a.workers)
    pool_coarse = hash_paths(coarse_key, pool_paths, a.workers)
    # object columns: the embed step reads this table with the extraction image's older pandas
    pd.DataFrame({"name": names, "exact": pool_exact, "coarse": pool_coarse}, dtype=object).to_parquet(
        out / POOL_B_KEYS, index=False)

    manifest_sha256 = sha256_file(compilation / "MANIFEST.csv")
    man = pd.read_csv(compilation / "MANIFEST.csv", dtype=str, keep_default_na=False)
    has_file = man["path"].map(lambda p: os.path.isfile(compilation / p)).to_numpy(dtype=bool)
    ref = man.loc[has_file, ["path", "identity", "source"]].drop_duplicates("path").reset_index(drop=True)
    ref["coarse"] = hash_paths(coarse_key, [str(compilation / p) for p in ref["path"]], a.workers)
    ref[["path", "coarse"]].to_parquet(out / COMPILATION_COARSE, index=False)
    per_key = ref["coarse"].dropna().value_counts()

    old = list_files(a.old_images)
    old_coarse = hash_paths(coarse_key, [str(Path(a.old_images) / n) for n in old], a.workers)
    idx, status = unique_matches(old_coarse, ref["coarse"].tolist())
    keep = idx >= 0
    pairs = pd.DataFrame({"old_image": np.asarray(old, dtype=object)[keep],
                          "compilation_path": ref["path"].to_numpy(dtype=object)[idx[keep]]})
    pairs.to_csv(out / PAIRS, index=False)
    sc = Counter(status)

    table, _, key_info = load_key_table(a.compilation_keys, manifest_sha256)
    match = pool_b_match(pool_exact, pool_coarse, set(table["key"]), ref[["coarse", "source"]])
    summary = {
        "pool": _pool_block(POOL_B_LABEL, names, digest),
        "coarse_rule": COARSE_RULE,
        "compilation_reference": {"manifest_rows": int(len(man)), "files": int(len(ref)),
                           "unreadable": int(ref["coarse"].isna().sum()), "distinct_coarse_keys": int(per_key.size),
                           "coarse_keys_on_several_files": int((per_key > 1).sum())},
        "calibration_candidates": {"old_test_images": len(old), "unique": sc["unique"],
                                   "ambiguous_excluded": sc["ambiguous"], "unmatched": sc["unmatched"],
                                   "unreadable": sc["unreadable"], "pairs": int(keep.sum()),
                                   "originals_labelled": int((ref["identity"].to_numpy()[idx[keep]] != "").sum())},
        "pool_match_to_compilation": match,
        "compilation_key_table": key_info,
        "inputs_sha256": {"compilation_manifest": manifest_sha256, "compilation_key_table": key_info["table_sha256"],
                          "pool_file_list": digest, "calibration_pairs": sha256_file(out / PAIRS),
                          "old_test_file_list": list_digest(old)},
        "provenance": provenance(),
        "seconds": round(time.time() - t0, 1),
    }
    _dump(out / COARSE_SUMMARY, summary)
    print(f"coarse: pool {len(names)} files; calibration pairs {int(keep.sum())} of {len(old)} old test images "
          f"({sc['ambiguous']} ambiguous excluded); pool match {match}", flush=True)
    return 0


def cmd_embed(a: argparse.Namespace) -> int:
    t0 = time.time()
    out = Path(a.out)
    if bool(a.d1_manifest) != bool(a.d1_images):
        raise SystemExit("--d1-manifest and --d1-images go together")
    keys = pd.read_parquet(out / POOL_B_KEYS)
    pool_names = keys.loc[keys["coarse"].map(_is_key), "name"].tolist()   # every readable pool file
    pairs = pd.read_csv(out / PAIRS, dtype=str, keep_default_na=False)
    groups = {"pool": [str(Path(a.pool) / n) for n in pool_names],
              "pair_packaged": [str(Path(a.old_images) / n) for n in pairs["old_image"]],
              "pair_original": [str(Path(a.compilation) / p) for p in pairs["compilation_path"]]}
    if a.d1_manifest:
        files = [os.path.basename(str(p)) for p in pd.read_parquet(a.d1_manifest, columns=["path"])["path"]]
        groups["d1"] = [str(Path(a.d1_images) / f) for f in files]
    paths = [p for g in groups.values() for p in g]
    x = np.asarray(encode_fingervit(a.models, paths, batch_size=a.fv_batch, workers=a.workers), dtype=np.float64)
    if x.ndim != 2 or len(x) != len(paths) or not np.isfinite(x).all():
        raise SystemExit(f"the encoder returned an array of shape {x.shape} for {len(paths)} images, or non-finite values")
    start, files_sha256 = 0, {}
    for name, g in groups.items():
        np.save(out / EMB[name], l2_normalize(x[start:start + len(g)]).astype(np.float32))
        files_sha256[f"emb_{name}"] = sha256_file(out / EMB[name])
        start += len(g)
    write_list(out / POOL_NAMES, pool_names)
    summary = {
        "rows": {name: len(g) for name, g in groups.items()},
        "batch_size": a.fv_batch, "loader_workers": a.workers,
        "preprocessing": PREPROCESSING["fingervit"], "stored": PREPROCESSING["stored"],
        "weights_sha256": {k: v for k, v in weight_hashes(a.models).items() if k.startswith(FINGERVIT_DIR)},
        "code": code_hashes(), "software": software_versions(), "outputs_sha256": files_sha256,
        "inputs_sha256": {"calibration_pairs": sha256_file(out / PAIRS), "pool_b_keys": sha256_file(out / POOL_B_KEYS)},
        "provenance": provenance(), "seconds": round(time.time() - t0, 1),
    }
    _dump(out / EMBED_SUMMARY, summary)
    print(f"embed: {summary['rows']} at batch size {a.fv_batch} in {summary['seconds']:.0f} s", flush=True)
    return 0


def _nearest_table(paths: Sequence[str], best: np.ndarray, arg: np.ndarray, pool_names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({"path": list(paths), "max_cosine": best,
                         "pool_name": np.asarray(pool_names, dtype=object)[arg]})


def cmd_measure(a: argparse.Namespace) -> int:
    t0 = time.time()
    out, d2, compilation = Path(a.out), Path(a.d2), Path(a.compilation)
    coarse = json.loads((out / COARSE_SUMMARY).read_text())
    embed = json.loads((out / EMBED_SUMMARY).read_text())
    pool = np.load(out / EMB["pool"])
    pool_names = (out / POOL_NAMES).read_text(encoding="utf-8").splitlines()
    man = pd.read_parquet(d2 / "manifest.parquet")
    x2 = np.load(d2 / "emb_fingervit.npy")
    pairs = pd.read_csv(out / PAIRS, dtype=str, keep_default_na=False)
    packaged, original = np.load(out / EMB["pair_packaged"]), np.load(out / EMB["pair_original"])
    if len(pool) != len(pool_names) or len(x2) != len(man) or not len(packaged) == len(original) == len(pairs):
        raise SystemExit("embedding rows do not match their lists")

    # the chain of hashes: what the coarse and embedding steps recorded must equal what is read here
    manifest_sha256 = sha256_file(compilation / "MANIFEST.csv")
    recomputed = {"calibration_pairs": sha256_file(out / PAIRS), "compilation_manifest": manifest_sha256,
                  "pool_file_list": sha256_file(out / POOL_B_LIST), "pool_b_keys": sha256_file(out / POOL_B_KEYS),
                  **{f"emb_{k}": sha256_file(out / f) for k, f in EMB.items() if (out / f).is_file()}}
    problems = chain_problems(coarse.get("inputs_sha256", {}), embed, recomputed)
    if problems:
        raise SystemExit("broken chain of hashes: " + "; ".join(problems))
    chain = {"checked": sorted({*CHAIN_KEYS, *(embed.get("outputs_sha256") or {}), *(embed.get("inputs_sha256") or {})}),
             "problems": problems, "embedding_step_recorded_its_inputs": bool(embed.get("inputs_sha256"))}

    # calibration: positives (both sides embedded now), their cross-checks, D2 genuine and impostor pairs
    pos = cosine_rows(packaged, original)
    compilation_source = pd.read_csv(compilation / "MANIFEST.csv", dtype=str, keep_default_na=False, usecols=["path", "source"])
    pos_sources = pairs["compilation_path"].map(dict(zip(compilation_source["path"], compilation_source["source"]))).fillna("unknown")
    stems = [Path(n).stem for n in pairs["old_image"]]
    with np.load(a.old_fingervit, allow_pickle=False) as z:
        stored_keys = set(z.files)
        have = np.array([s in stored_keys for s in stems], dtype=bool)
        stored = [np.asarray(z[s], dtype=np.float64).reshape(-1) for s, h in zip(stems, have) if h]
    stored_check = cosine_rows(packaged[have], np.stack(stored)) if stored else np.empty(0)
    d2_row = pd.Series(np.arange(len(man)), index=man["path"].astype(str))
    in_d2 = pairs["compilation_path"].isin(d2_row.index).to_numpy(dtype=bool)
    d2_check = (cosine_rows(original[in_d2], x2[d2_row[pairs.loc[in_d2, "compilation_path"]].to_numpy()])
                if in_d2.any() else np.empty(0))
    ids, sources = man["identity"].astype(str).to_numpy(), man["source"].astype(str).to_numpy()
    if (man.groupby("identity")["source"].nunique() > 1).any():
        raise SystemExit("a D2 identity spans several sources")
    gen_parts = [(s, genuine_scores(x2[sources == s], ids[sources == s])) for s in sorted(set(sources.tolist()))]
    gen = np.concatenate([g for _, g in gen_parts])
    gen_sources = np.concatenate([np.full(len(g), s, dtype=object) for s, g in gen_parts])
    ia, ib = impostor_pairs(ids, a.impostor_pairs, a.seed)
    imp = pair_scores(x2, ia, ib)
    cal = calibrate(pos, pos_sources, gen, gen_sources, imp)
    rule, thresholds = cal["tau_rule"], cal["thresholds"]

    # measurement: the maximum cosine of every D2 image to any pool image, in float64 like the calibration
    pool64 = pool.astype(np.float64)
    best, arg = nearest(x2.astype(np.float64), pool64)
    _nearest_table(man["path"].astype(str), best, arg, pool_names).to_parquet(out / D2_NEAREST, index=False)
    compilation_coarse = pd.read_parquet(out / COMPILATION_COARSE)
    counts = compilation_coarse["coarse"].dropna().value_counts().to_dict()
    path_coarse = dict(zip(compilation_coarse["path"].astype(str), compilation_coarse["coarse"]))
    pool_coarse = {k for k in pd.read_parquet(out / POOL_B_KEYS)["coarse"] if _is_key(k)}
    copy_check = coarse_copy_check([path_coarse.get(p) for p in man["path"].astype(str)], counts, pool_coarse, best,
                                   rule["tau"], sources)

    inputs = {"d2_manifest": sha256_file(d2 / "manifest.parquet"), "d2_emb_fingervit": sha256_file(d2 / "emb_fingervit.npy"),
              "old_fingervit_features": sha256_file(a.old_fingervit), "compilation_coarse_keys": sha256_file(out / COMPILATION_COARSE),
              "pool_coarse_keys": recomputed["pool_b_keys"], "pool_embedded_list": list_digest(pool_names),
              **{k: recomputed[k] for k in (*CHAIN_KEYS, "emb_pool", "emb_pair_packaged", "emb_pair_original")}}
    if (d2 / "extract_summary.json").is_file():
        inputs["d2_extract_summary"] = sha256_file(d2 / "extract_summary.json")

    d1_section = None
    if a.d1_manifest and a.d1_map and (out / EMB["d1"]).is_file():
        x1 = np.load(out / EMB["d1"])
        d1, _, d1_info = load_d1(a.d1_manifest, a.d1_map)
        if len(x1) != len(d1):
            raise SystemExit("D1 embedding rows do not match the first-corpus manifest")
        _, id_source, _ = load_key_table(a.compilation_keys, manifest_sha256)
        b1, g1 = nearest(x1.astype(np.float64), pool64)
        _nearest_table(d1["file"], b1, g1, pool_names).to_parquet(out / D1_NEAREST, index=False)
        assigned = (d1["identity"] != "").to_numpy()
        d1_ids = d1.loc[assigned, "identity"].to_numpy()
        d1_section = {
            "description": ("optional: first-corpus images against pool B with the same detector and thresholds (only "
                            "the D1 images needed a new extraction)"),
            "d1": d1_info,
            "by_threshold": counts_above(b1[assigned], d1_ids, [id_source[i] for i in d1_ids], thresholds),
            "unassigned": {"images": int((~assigned).sum()), "at_or_above_tau": int((b1[~assigned] >= rule["tau"]).sum())},
        }
        inputs.update({"d1_manifest": sha256_file(a.d1_manifest), "d1_relabel_map": sha256_file(a.d1_map),
                       "emb_d1": recomputed["emb_d1"]})

    report = {
        "data_source": ("real: candidate training pool B (packaged 224 x 224 RGB images), the second corpus D2 (compilation "
                        "`real` images, internal research corpus of real fingerprint images), old test images packaged "
                        "like pool B with their compilation originals (calibration) and the first-corpus (D1) images; "
                        "finger-vit embeddings; no synthetic data"),
        "question": "worst case: finger-vit is assumed to have been trained on the whole of candidate training pool B",
        "detector": DETECTOR,
        "pool": dict(coarse["pool"], embedded=embed["rows"]["pool"]),
        "pool_match_to_compilation": coarse["pool_match_to_compilation"],
        "coarse_rule": coarse.get("coarse_rule"),
        "compilation_reference": coarse["compilation_reference"],
        "calibration": {
            "positives": {"definition": POSITIVE_RULE, "selection": coarse["calibration_candidates"],
                          "originals_in_d2": int(in_d2.sum()), "quantiles": cal["quantiles"]["positives"],
                          "by_source": cal["by_source"]["positives"],
                          "by_source_rule": "compilation source of the original; below_tau counts cosines under tau",
                          "crosscheck_stored_features": {
                              "description": "packaged side embedded now against the features stored for it",
                              "quantiles": quantiles(stored_check)},
                          "crosscheck_d2_embeddings": {
                              "description": "original side embedded now against its stored D2 embedding",
                              "quantiles": quantiles(d2_check)}},
            "d2_genuine": {"definition": "every pair of D2 images of one finger (different impressions)",
                           "quantiles": cal["quantiles"]["d2_genuine"], "by_source": cal["by_source"]["d2_genuine"]},
            "d2_impostor": {"definition": ("distinct pairs of D2 images of different fingers, drawn uniformly at "
                                           "random"), "seed": a.seed, "quantiles": cal["quantiles"]["d2_impostor"]},
            "pairs_at_or_above_thresholds": cal["pairs_at_or_above"],
            "per_pair_bound": dict(cal["per_pair_bound"], pool_embeddings=int(len(pool))),
            "quantile_method": QUANTILE_METHOD,
        },
        "tau_rule": rule,
        "thresholds": thresholds,
        "d2": {"images": int(len(man)), "fingers": int(man["identity"].nunique()),
               "finger_rule": "a D2 finger counts when at least one of its D2 images counts",
               "maxima": "float64 (both sides upcast before the maximum, like the calibration cosines)",
               "by_threshold": counts_above(best, ids, sources, thresholds)},
        "d2_coarse_copy_check": copy_check,
        "d1_vs_pool": d1_section,
        "extraction": {k: embed.get(k) for k in ("batch_size", "loader_workers", "rows", "weights_sha256",
                                                  "preprocessing", "software", "code", "seconds")}
                      | {"env": embed["provenance"]["env"]},
        "inputs_sha256": dict(sorted({**coarse.get("inputs_sha256", {}), **inputs}.items())),
        "provenance_chain": chain,
        "provenance": {"coarse": coarse["provenance"], "embed": embed["provenance"], "measure": provenance()},
        "seconds": {"coarse": coarse.get("seconds"), "embed": embed.get("seconds"),
                    "measure": round(time.time() - t0, 1)},
    }
    _dump(a.json, report)
    at = next(e for e in report["d2"]["by_threshold"] if e["name"].startswith("tau"))
    print(f"measure: tau {rule['tau']:.6f} (D2 genuine p99.9 {rule['d2_genuine_p99.9']:.6f}, separable "
          f"{rule['separable']}); D2 images at or above tau {at['images']['above']}/{at['images']['of']}, fingers "
          f"{at['fingers']['above']}/{at['fingers']['of']}", flush=True)
    return 0


def cmd_deepprint(a: argparse.Namespace) -> int:
    d2 = Path(a.d2)
    summary = json.loads((d2 / "extract_summary.json").read_text())
    dbs = fvc_databases(Path(a.compilation) / "images")
    section = "\n".join(markdown_section(Path(a.flx_readme).read_text(encoding="utf-8"), FLX_README_SECTION)).lower()
    if not section.strip():
        raise SystemExit(f"the flx README has no section {FLX_README_SECTION!r}")
    report = {
        "data_source": ("real: composition of the second corpus D2 from its extraction summary, and the folders of the "
                        "compilation `real` image tree; no measurement"),
        **deepprint_sources(summary),
        "compilation_fvc_databases": dbs,
        "compilation_fvc_statement": fvc_statement(dbs),
        "compilation_manifest_db4_rows": manifest_db4_rows(a.compilation),
        "flx_readme": {"source": FLX_README_SOURCE, "sha256": sha256_file(a.flx_readme), "section": FLX_README_SECTION,
                       "training_sets_named_in_section": [name for name, tokens in FLX_TRAINING_SETS.items()
                                                          if any(t in section for t in tokens)]},
        "pool": None,
        "thresholds": None,
        "inputs_sha256": {"d2_extract_summary": sha256_file(d2 / "extract_summary.json"),
                          "d2_manifest": sha256_file(d2 / "manifest.parquet"), "flx_readme": sha256_file(a.flx_readme)},
        "provenance": provenance(),
    }
    _dump(a.json, report)
    print(f"deepprint: {report['statement']}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exact", help="(a) D1 against candidate pool A by exact content key")
    e.add_argument("--d1-manifest", required=True)
    e.add_argument("--d1-images", required=True, help="first-corpus images, stored under their manifest basenames")
    e.add_argument("--d1-map", required=True, help="relabel map of the first corpus (file,identity)")
    c = sub.add_parser("coarse", help="(b) calibration pairs and pool B against compilation by coarse key")
    c.add_argument("--old-images", required=True, help="old test images, packaged like pool B")
    for q in (e, c):
        q.add_argument("--pool", required=True)
        q.add_argument("--expect-files", type=int, default=None, help="refuse a pool with another file count")
        q.add_argument("--compilation", required=True, help="compilation `real` directory (MANIFEST.csv, images/)")
        q.add_argument("--compilation-keys", required=True, help="compilation_keys.parquet of extract_d2.py keys (json beside it)")
        q.add_argument("--workers", type=int, default=16)
    m = sub.add_parser("embed", help="(b) finger-vit embeddings of pool B, the calibration pairs and D1 (GPU)")
    m.add_argument("--pool", required=True)
    m.add_argument("--old-images", required=True)
    m.add_argument("--compilation", required=True)
    m.add_argument("--models", required=True, help="directory with finger-vit_giant/")
    m.add_argument("--d1-manifest", default=None)
    m.add_argument("--d1-images", default=None)
    m.add_argument("--fv-batch", type=int, default=1, help="finger-vit batch size (the D2 extraction used 1)")
    m.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("measure", help="(b) calibration, threshold and overlap counts")
    s.add_argument("--d2", required=True, help="SIN-layout D2 (manifest.parquet, emb_fingervit.npy)")
    s.add_argument("--old-fingervit", required=True, help="features.npz stored for the old test images")
    s.add_argument("--compilation", required=True)
    s.add_argument("--compilation-keys", required=True)
    s.add_argument("--d1-manifest", default=None)
    s.add_argument("--d1-map", default=None)
    s.add_argument("--impostor-pairs", type=int, default=IMPOSTOR_PAIRS)
    s.add_argument("--seed", type=int, default=SEED)
    d = sub.add_parser("deepprint", help="(c) D2 sources against the training sets named in the flx documentation")
    d.add_argument("--d2", required=True)
    d.add_argument("--compilation", required=True, help="compilation `real` directory: its images/ tree is searched for FVC DB folders")
    d.add_argument("--flx-readme", required=True, help="README.md of the flx repository behind the extraction image")
    for q in (e, c, m, s):
        q.add_argument("--out", required=True, help="output directory on the data host (intermediate files)")
    for q in (e, s, d):
        q.add_argument("--json", required=True, help="aggregate JSON written by this step")
    a = ap.parse_args(argv)
    return {"exact": cmd_exact, "coarse": cmd_coarse, "embed": cmd_embed, "measure": cmd_measure,
            "deepprint": cmd_deepprint}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
