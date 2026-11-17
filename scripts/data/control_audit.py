"""Overlap audit of the control encoder (ctrl-vit): its training images against the evaluation corpus D2 (aggregates only).

Inputs, all SIN-layout directories written by `train_control.py embed` (manifest.parquet, emb_ctrlvit.npy,
embed_summary.json) unless noted:
  --train     the training images of ctrl-vit (manifest with identity, source, subject, key; embeddings)
  --d2        D2 (manifest with identity, source, path; embeddings) under the same frozen weights
  --compilation-keys the content-key table of the compilation (path, key)
  --trainset  the training-set directory of Task C1 (d2_subject_flags.parquet, and training_set.json when present)

The JSON (--json) holds:
- identity overlap and exact-content overlap, recomputed with ijiss_btp.train.selection.disjointness (a D2 row without a
  content key raises);
- subject overlap, taken from Task C1 (d2_subject_flags.parquet; the C1 counts are copied from training_set.json);
- the near-duplicate scan: for every D2 image the maximum cosine to any training image (blockwise dot products of unit
  vectors), the 99.9th percentile of the cosines of all D2 genuine pairs (numpy.quantile, linear interpolation) as the
  reference level, the D2 images and fingers whose maximum is at least that level (overall and by source), and the
  quantiles of the maximum (overall and by source).  Counts are reported as measured, never thresholded away.
The JSON holds no path, identity or per-image value.  The per-image maxima are written next to it as max_cosine_d2.npy
(on the data host only).  Run it inside the container on the data host: it reads embeddings.

- the finger-level statistic (finger_overlap): per D2 finger, the maximum over training fingers of the mean cross cosine
  against its own genuine mean.  It neither bounds nor detects a physical finger held in training under another
  identity key (such a copy is expected to match the finger's own genuine mean about as often as it exceeds it); it is
  an additional check, not proof of finger-disjointness.
The recomputed subject counts are compared with the C1 flags and the C1 JSON and the comparison is recorded.

Exit status: 0 ok; 1 when the two embedding runs are not both ok with the same weights, or the identity or content
overlap is not 0, or the subject counts disagree with C1 (the JSON is written in the last two cases); 2 when an output
exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ijiss_btp.train.selection import disjointness, subject_key  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_check  # noqa: E402  (same directory)

GENUINE_PERCENTILE = 99.9
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)
UNIT_NORM_TOL = 1e-3
THRESHOLD_RULE = ("D2 genuine pairs' 99.9th percentile of the cosine under ctrl-vit (all genuine pairs; numpy.quantile, "
                  "linear interpolation); an image counts when its maximum cosine to any training image is at least it")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo: Path) -> str | None:
    """CANVEC_GIT_SHA, else the GIT_SHA file the sync script writes next to the code, else None."""
    if os.environ.get("CANVEC_GIT_SHA"):
        return os.environ["CANVEC_GIT_SHA"]
    f = Path(repo) / "GIT_SHA"
    return f.read_text().strip() if f.is_file() else None


def max_cosine(queries: np.ndarray, pool: np.ndarray, block: int = 2048) -> np.ndarray:
    """For each row of `queries` the maximum dot product with any row of `pool` (unit vectors: the cosine), computed
    over blocks of `block` query rows."""
    q = np.asarray(queries, dtype=np.float32)
    p = np.asarray(pool, dtype=np.float32)
    out = np.empty(len(q), dtype=np.float32)
    for i in range(0, len(q), block):
        out[i:i + block] = (q[i:i + block] @ p.T).max(axis=1)
    return out


def genuine_scores(emb: np.ndarray, identity: np.ndarray) -> np.ndarray:
    """Cosines of all within-identity pairs (each pair once)."""
    e = np.asarray(emb, dtype=np.float32)
    ident = pd.factorize(np.asarray(identity).astype(str))[0]
    order = np.argsort(ident, kind="stable")
    bounds = np.flatnonzero(np.diff(ident[order])) + 1
    chunks = []
    for g in np.split(order, bounds):
        if len(g) > 1:
            s = e[g] @ e[g].T
            chunks.append(s[np.triu_indices(len(g), k=1)])
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def genuine_percentile(emb: np.ndarray, identity: np.ndarray, pct: float = GENUINE_PERCENTILE) -> float:
    g = genuine_scores(emb, identity)
    if not len(g):
        raise ValueError("no genuine pairs")
    return float(np.quantile(g.astype(np.float64), pct / 100.0))


def _summary(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(len(v)), "min": float(v.min()), "mean": float(v.mean()), "max": float(v.max()),
            "quantiles": {str(q): float(np.quantile(v, q)) for q in QUANTILES}}


def _check_embeddings(name: str, man: pd.DataFrame, emb: np.ndarray) -> None:
    if len(emb) != len(man):
        raise ValueError(f"{name}: {len(emb)} embedding rows for {len(man)} manifest rows")
    if not np.isfinite(emb).all() or np.abs(np.linalg.norm(emb, axis=1) - 1.0).max(initial=0.0) > UNIT_NORM_TOL:
        raise ValueError(f"{name}: embeddings are not finite and unit-norm")


def _group(emb: np.ndarray, man: pd.DataFrame):
    """Fingers (identities, sorted by name): the sum of their unit embeddings (float64), image counts, the row order
    and group starts (to find a finger's images), and one metadata row per finger."""
    codes, _ = pd.factorize(man["identity"].astype(str).to_numpy(), sort=True)
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1]
    sums = np.add.reduceat(np.asarray(emb, dtype=np.float64)[order], starts, axis=0)
    counts = np.diff(np.r_[starts, len(order)])
    meta = man.iloc[order[starts]][["identity", "source", "subject", "finger"]].astype(str).reset_index(drop=True)
    meta["subject_key"] = subject_key(meta["identity"])
    return sums, counts, order, starts, meta


def finger_table(train: pd.DataFrame, train_emb: np.ndarray, d2: pd.DataFrame, d2_emb: np.ndarray) -> dict:
    """Per D2 finger: the maximum over training fingers of the mean cross cosine (`mx`) and the best training finger
    (`best`), its own genuine mean (`own`, NaN below two images), `gap = mx - own`, `valid` (>= 2 images) and
    `reach` (valid and mx >= own); plus both fingers' metadata and image positions.  For unit embeddings the mean
    cosine over all image pairs of two fingers equals the dot product of their mean embeddings."""
    ts, tc, torder, tstarts, tmeta = _group(train_emb, train)
    ds, dc, dorder, dstarts, dmeta = _group(d2_emb, d2)
    cross = (ds @ ts.T) / (dc[:, None] * tc[None, :])
    best, mx = cross.argmax(axis=1), cross.max(axis=1)
    valid = dc >= 2
    own = np.full(len(dc), np.nan)
    own[valid] = ((ds[valid] ** 2).sum(axis=1) - dc[valid]) / (dc[valid] * (dc[valid] - 1))
    return {"mx": mx, "best": best, "own": own, "gap": mx - own, "valid": valid, "reach": valid & (mx >= own),
            "dc": dc, "tc": tc, "dorder": dorder, "dstarts": dstarts, "torder": torder, "tstarts": tstarts,
            "dmeta": dmeta, "tmeta": tmeta}


def finger_overlap(train: pd.DataFrame, train_emb: np.ndarray, d2: pd.DataFrame, d2_emb: np.ndarray,
                   tab: dict | None = None) -> dict:
    """Finger-level statistic.  For every D2 finger with at least two images: its own genuine mean (mean cosine over its
    distinct image pairs, (|sum|^2 - n) / (n (n - 1))) and the maximum over all training fingers of the mean cross cosine.
    Reported overall and by source: quantiles of that maximum, quantiles of (maximum - own genuine mean), and the fingers
    whose maximum reaches their own genuine mean.  A source with no training source or subject (polyu_cross) is the null
    reference: its fingers cannot be held in training."""
    tab = tab or finger_table(train, train_emb, d2, d2_emb)
    mx, own, gap, valid, reach = (tab[k] for k in ("mx", "own", "gap", "valid", "reach"))
    dsrc = tab["dmeta"]["source"].to_numpy()
    sources = sorted(set(dsrc))

    def block(sel):
        v = sel & valid
        return {"fingers": int(v.sum()), "max_cross_mean_cosine": _summary(mx[v]) if v.any() else None,
                "max_minus_own_genuine_mean": _summary(gap[v]) if v.any() else None,
                "own_genuine_mean": _summary(own[v]) if v.any() else None,
                "reaching_own_genuine_mean": int((reach & sel).sum())}
    out = {"statistic": ("per D2 finger with >= 2 images: maximum over training fingers of the mean cross-cosine "
                         "(dot product of mean embeddings) against its own genuine mean cosine"),
           "d2_fingers": int(len(dsrc)), "d2_fingers_single_image_excluded": int((~valid).sum()),
           "train_fingers": int(len(tab["tc"])), "overall": block(np.ones(len(dsrc), bool)),
           "by_source": {s: block(dsrc == s) for s in sources}}
    if "polyu_cross" in out["by_source"]:
        out["null_reference_source"] = "polyu_cross"
    return out


# Source families whose subject ids are assumed to be comparable (that the same people appear in both editions is an
# unverified assumption) and finger-position vocabularies.  casia_ageing names its fingers L_index, L_middle, R_index, R_middle; casia_v5 codes <hand><0..3>.  Both
# are mapped to <hand>_<finger> names, reading 0 as thumb, 1 as index, 2 as middle and 3 as ring (CASIA-FingerprintV5
# documentation); this is an assumption from that documentation, not verified on the images.
FAMILIES = {"casia_v5": "casia", "casia_ageing": "casia", "sd301": "sd30x", "sd302": "sd30x"}
_FINGERS = ("thumb", "index", "middle", "ring")
POSITION_MAP = {
    "casia_v5": {f"{h}{i}": f"{name}_{_FINGERS[i]}" for h, name in (("L", "left"), ("R", "right")) for i in range(4)},
    "casia_ageing": {f"{h}_{_FINGERS[i]}": f"{name}_{_FINGERS[i]}" for h, name in (("L", "left"), ("R", "right"))
                     for i in (1, 2)}}
POSITION_NOTE = ("casia_v5 codes <hand>0..3 and casia_ageing names <hand>_index/<hand>_middle are mapped to "
                 "<hand>_<finger> (assumed from the CASIA-FingerprintV5 documentation: 0 thumb, 1 index, 2 middle, 3 ring; "
                 "not verified on the images); finger codes are written as <source>:<code>, the compilation's identity notation; positions are compared only within a source or between linked families; "
                 "numeric subject ids are compared after stripping leading zeros")


def position_map_json() -> dict:
    """POSITION_MAP with every source code written as <source>:<code>, the compilation's identity notation, so that a bare
    finger code never appears in the output."""
    return {src: {f"{src}:{code}": name for code, name in m.items()} for src, m in POSITION_MAP.items()}


def _position(meta: pd.DataFrame) -> np.ndarray:
    return np.array([POSITION_MAP.get(s, {}).get(f, f) for s, f in zip(meta["source"], meta["finger"])])


def _numeric_subject(meta: pd.DataFrame) -> np.ndarray:
    return np.array([x.lstrip("0") or ("0" if x else "") for x in meta["subject"]])


def _vocabulary(man: pd.DataFrame, cap: int = 12) -> dict:
    out = {}
    for s, g in sorted(man.groupby("source")):
        v = sorted(g["finger"].astype(str).unique())
        out[str(s)] = {"n": len(v), "values": [f"{s}:{x}" for x in v[:cap]]}
    return out


def _pair_counts(d_src: np.ndarray, t_src: np.ndarray) -> dict:
    return {k: int(n) for k, n in sorted(pd.Series([f"{a}->{b}" for a, b in zip(d_src, t_src)]).value_counts().items())}


def _linked(dm: pd.DataFrame, tm: pd.DataFrame) -> np.ndarray:
    """Boolean per row pair: different sources of one linked family."""
    fd, ft = dm["source"].map(FAMILIES), tm["source"].map(FAMILIES)
    return ((dm["source"].to_numpy() != tm["source"].to_numpy()) & fd.notna().to_numpy()
            & (fd.to_numpy() == ft.to_numpy()))


def characterise(tab: dict, train_emb: np.ndarray, d2_emb: np.ndarray) -> dict:
    """Counts and distributions for the best-matching training finger of each D2 finger with >= 2 images, for the flagged
    fingers (reaching their own genuine mean) and for all of them (base rates); overall and by D2 source.  Categories:
    same subject_key, same source, source pair, position comparable and equal, same numeric subject id across linked
    source families; quantiles of the finger's own genuine mean, image count, gap, the best image-pair cosine between
    the two fingers and the finger's best genuine image-pair cosine (and their difference)."""
    dm = tab["dmeta"]
    tm = tab["tmeta"].iloc[tab["best"]].reset_index(drop=True)
    valid = tab["valid"]
    same_sk = (dm["subject_key"].to_numpy() == tm["subject_key"].to_numpy())
    same_src = dm["source"].to_numpy() == tm["source"].to_numpy()
    fd, ft = dm["source"].map(FAMILIES), tm["source"].map(FAMILIES)
    comparable = same_src | (fd.notna().to_numpy() & (fd.to_numpy() == ft.to_numpy()))
    same_pos = comparable & (_position(dm) == _position(tm))
    linked_num = _linked(dm, tm) & (_numeric_subject(dm) == _numeric_subject(tm)) & (_numeric_subject(dm) != "")
    dorder, dstarts, dc = tab["dorder"], tab["dstarts"], tab["dc"]
    torder, tstarts, tc = tab["torder"], tab["tstarts"], tab["tc"]
    d2e, tre = np.asarray(d2_emb, dtype=np.float32), np.asarray(train_emb, dtype=np.float32)
    best_cross = np.full(len(dc), np.nan)
    best_gen = np.full(len(dc), np.nan)
    for f in np.flatnonzero(valid):
        di = dorder[dstarts[f]:dstarts[f] + dc[f]]
        t = tab["best"][f]
        ti = torder[tstarts[t]:tstarts[t] + tc[t]]
        best_cross[f] = float((d2e[di] @ tre[ti].T).max())
        g = d2e[di] @ d2e[di].T
        best_gen[f] = float(g[np.triu_indices(len(di), 1)].max())
    d_src, t_src = dm["source"].to_numpy(), tm["source"].to_numpy()

    def block(sel):
        v = sel & valid
        q = lambda x: _summary(x[v]) if v.any() else None   # noqa: E731
        return {"n": int(v.sum()), "same_subject_key": int((same_sk & v).sum()), "same_source": int((same_src & v).sum()),
                "source_pairs": _pair_counts(d_src[v], t_src[v]), "position_comparable": int((comparable & v).sum()),
                "same_finger_position": int((same_pos & v).sum()),
                "linked_family_same_numeric_subject": int((linked_num & v).sum()),
                "own_genuine_mean": q(tab["own"]), "images": q(dc.astype(float)), "gap": q(tab["gap"]),
                "best_cross_image_cos": q(best_cross), "best_genuine_image_cos": q(best_gen),
                "best_cross_minus_best_genuine": q(best_cross - best_gen)}
    sources = sorted(set(d_src))
    flagged = tab["reach"]
    return {"flagged": {"overall": block(flagged),
                        "by_source": {s: block(flagged & (d_src == s)) for s in sources if (flagged & (d_src == s)).any()}},
            "all": {"overall": block(np.ones(len(dc), bool)),
                    "by_source": {s: block(d_src == s) for s in sources if ((d_src == s) & valid).any()}},
            "position_normalisation": {"map": position_map_json(), "families": FAMILIES, "note": POSITION_NOTE},
            "position_vocabulary": {"d2": _vocabulary_of(tab, "dmeta"), "train": _vocabulary_of(tab, "tmeta")}}


def _vocabulary_of(tab: dict, key: str) -> dict:
    return _vocabulary(tab[key])


def label_structure(tab: dict) -> tuple[dict, set]:
    """Embedding-free scan: D2 fingers with a training finger in the other source of a linked family (casia_v5 and
    casia_ageing; sd301 and sd302) that has the same numeric subject id and the same finger position (candidate
    cross-source duplicates the compilation did not link).  Returns (aggregates, identities of the candidates)."""
    dm, tm = tab["dmeta"].copy(), tab["tmeta"].copy()
    for m in (dm, tm):
        m["num"], m["pos"], m["fam"] = _numeric_subject(m), _position(m), m["source"].map(FAMILIES)
    dm["i"], tm["j"] = np.arange(len(dm)), np.arange(len(tm))
    pairs = dm.dropna(subset=["fam"]).merge(tm.dropna(subset=["fam"]), on=["fam", "num"], suffixes=("_d", "_t"))
    pairs = pairs[(pairs["source_d"] != pairs["source_t"]) & (pairs["num"] != "")]
    flagged, best = tab["reach"], tab["best"]

    def summary(p):
        d = p.drop_duplicates(["i", "source_t"])
        return {"total": int(len(d)), "by_source_pair": _pair_counts(d["source_d"].to_numpy(), d["source_t"].to_numpy())}
    cand = pairs[pairs["pos_d"] == pairs["pos_t"]]
    ci = set(cand["i"])
    best_is = cand[cand["j"].to_numpy() == best[cand["i"].to_numpy()]]
    out = {"rule": ("D2 finger and a training finger in the other source of a linked family (casia_v5/casia_ageing, "
                    "sd301/sd302) with the same numeric subject id and the same finger position"),
           "position_normalisation_note": POSITION_NOTE,
           "candidates": summary(cand), "candidates_among_flagged": int(flagged[sorted(ci)].sum()) if ci else 0,
           "candidates_best_match_is_candidate": int(len(set(best_is["i"]))),
           "candidates_best_match_is_candidate_among_flagged": int(flagged[sorted(set(best_is["i"]))].sum())
           if len(best_is) else 0,
           "candidates_without_position_requirement": summary(pairs),
           "candidates_without_position_requirement_among_flagged": int(flagged[sorted(set(pairs["i"]))].sum())
           if len(pairs) else 0}
    return out, set(dm["identity"].iloc[sorted(ci)])


def exclusion_sets(tab: dict, candidates: set) -> dict:
    """D2 finger sets removed in the sensitivity subsets: (a) none, (b) the flagged fingers, (c) the flagged fingers and
    every D2 finger whose subject_key matches a flagged finger's, (d) the structural candidates."""
    dm = tab["dmeta"]
    flagged = set(dm["identity"][tab["reach"]])
    keys = set(dm["subject_key"][tab["reach"]])
    return {"a_as_is": set(), "b_without_flagged": flagged,
            "c_without_flagged_subjects": set(dm["identity"][dm["subject_key"].isin(keys)]) | flagged,
            "d_without_structural_candidates": set(candidates)}


def sensitivity(d2: pd.DataFrame, embeddings: dict, exclusions: dict, bins: int = 4000, block: int = 512) -> dict:
    """Genuine-versus-impostor AUC (scripts/data/label_check.py's method: overall, same-device, cross-device) of each
    encoder on D2 with the fingers of each exclusion set removed."""
    d2 = d2.reset_index(drop=True)
    out = {}
    for name, excluded in exclusions.items():
        keep = ~d2["identity"].astype(str).isin(excluded).to_numpy()
        man = d2[keep].reset_index(drop=True)
        entry = {"images": int(keep.sum()), "fingers": int(man["identity"].nunique()),
                 "removed_fingers": int(d2["identity"].nunique() - man["identity"].nunique()), "encoders": {}}
        for enc, x in embeddings.items():
            r = label_check._model_check(man, np.asarray(x)[keep], bins, 30, block)["all"]
            entry["encoders"][enc] = {"genuine_pairs": r["genuine_pairs"], "impostor_pairs": r["impostor_pairs"],
                                      "auc": r["auc"], "same_device": r["same_device"], "cross_device": r["cross_device"]}
        out[name] = entry
    return {"method": "scripts/data/label_check.py (histogram AUC, 4000 bins over [-1, 1], ties count one half)"
            if bins == 4000 else f"scripts/data/label_check.py ({bins} bins)", "subsets": out}


def audit(train: pd.DataFrame, train_emb: np.ndarray, d2: pd.DataFrame, d2_emb: np.ndarray, keys: pd.DataFrame,
          block: int = 2048) -> tuple[dict, np.ndarray, dict]:
    """Identity/content overlap (disjointness), the near-duplicate scan, the finger-level statistic with the
    characterisation of the flagged fingers and the label-structure scan of D2 against the training images.
    Returns (result, per-image maximum cosine of the D2 images, exclusion sets of the sensitivity subsets).
    `train` needs identity, source, subject, key, finger; `d2` identity, source, path, subject, finger; `keys` path, key."""
    train, d2 = train.reset_index(drop=True), d2.reset_index(drop=True)
    _check_embeddings("train", train, train_emb)
    _check_embeddings("d2", d2, d2_emb)
    dj = disjointness(train, d2, keys)   # raises when a D2 row has no content key
    ident = d2["identity"].astype(str).to_numpy()
    thr = genuine_percentile(d2_emb, ident, GENUINE_PERCENTILE)   # raises without genuine pairs
    n_genuine = int(len(genuine_scores(d2_emb, ident)))
    mx = max_cosine(d2_emb, train_emb, block)
    hit = mx >= thr
    src = d2["source"].astype(str).to_numpy()
    sources = sorted(set(src))
    nd = {"threshold_rule": THRESHOLD_RULE, "genuine_percentile": GENUINE_PERCENTILE, "threshold": thr,
          "genuine_pairs": n_genuine, "d2_images": int(len(d2)), "train_images": int(len(train)), "block": int(block),
          "images_at_or_above": {"total": int(hit.sum()),
                                 "by_source": {s: int((hit & (src == s)).sum()) for s in sources}},
          "fingers_at_or_above": {"total": int(len(set(ident[hit]))),
                                  "by_source": {s: int(len(set(ident[hit & (src == s)]))) for s in sources}},
          "max_cosine": {"overall": _summary(mx), "by_source": {s: _summary(mx[src == s]) for s in sources}}}
    res = {k: dj[k] for k in ("identity_overlap", "content_overlap", "d2_rows", "d2_keys_checked")}
    res["train_images"], res["train_fingers"] = int(len(train)), int(train["identity"].nunique())
    res["d2_fingers"] = int(len(set(ident)))
    res["near_duplicates"] = nd
    tab = finger_table(train, train_emb, d2, d2_emb)
    res["finger_level"] = finger_overlap(train, train_emb, d2, d2_emb, tab)
    res["characterisation"] = characterise(tab, train_emb, d2_emb)
    res["label_structure"], candidates = label_structure(tab)
    exclusions = exclusion_sets(tab, candidates)
    res["subjects_recomputed"] = {k: dj[k] for k in ("subjects", "d2_fingers_with_shared_subject",
                                                      "d2_fingers_without_shared_subject")}
    return res, mx, exclusions


def _read_embed_run(d: Path, name: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    s = json.loads((d / "embed_summary.json").read_text())
    if s.get("status") != "ok" or s.get("smoke"):
        raise ValueError(f"{name}: embed summary is not a complete ok run (status {s.get('status')!r}, "
                         f"smoke {s.get('smoke')!r})")
    man_sha, emb_sha = sha256_file(d / "manifest.parquet"), sha256_file(d / "emb_ctrlvit.npy")
    if s.get("manifest_sha256") != man_sha or s.get("emb_sha256") != emb_sha:
        raise ValueError(f"{name}: manifest or embeddings differ from the hashes in embed_summary.json")
    man = pd.read_parquet(d / "manifest.parquet")
    emb = np.load(d / "emb_ctrlvit.npy")
    info = {"weights_sha256": s["weights_sha256"], "manifest_sha256": man_sha,
            "input_manifest_sha256": s.get("input_manifest_sha256"), "emb_sha256": emb_sha,
            "rows": int(len(man)), "data_source": s.get("data_source"), "status": s["status"],
            "smoke": bool(s.get("smoke")), "complete": s.get("complete"), "self_consistency": s.get("self_consistency"),
            "input_images_sha256": (s.get("input_images") or {}).get("combined_sha256"),
            "embed_git_sha": ((s.get("environment") or {}).get("env") or {}).get("git_sha")}
    return man, emb, info


def _by_source(flags: pd.DataFrame, mask: pd.Series) -> dict:
    sel = flags[mask.to_numpy()]
    return {str(k): int(n) for k, n in sorted(sel.groupby("source").size().items())}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True, help="embed output directory of the training images")
    ap.add_argument("--d2", required=True, help="embed output directory of D2")
    ap.add_argument("--compilation-keys", required=True, help="compilation_keys.parquet (path, key)")
    ap.add_argument("--trainset", required=True, help="Task C1 output directory")
    ap.add_argument("--json", required=True, help="output JSON file (must not exist)")
    ap.add_argument("--label", required=True, help="data-source label written into the output")
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--d2-baselines", help="directory with D2's manifest.parquet (identical to --d2's) and "
                    "emb_fingervit.npy, emb_deepprint.npy (same row order): adds the sensitivity analysis")
    a = ap.parse_args(argv)
    out = Path(a.json)
    maxfile = out.parent / "max_cosine_d2.npy"
    for f in (out, maxfile):
        if f.exists():
            print(f"refusing to overwrite {f}", file=sys.stderr)
            return 2
    train_dir, d2_dir, ts = Path(a.train), Path(a.d2), Path(a.trainset)
    try:
        train, train_emb, tr_info = _read_embed_run(train_dir, "train")
        d2, d2_emb, d2_info = _read_embed_run(d2_dir, "d2")
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    if tr_info["weights_sha256"] != d2_info["weights_sha256"]:
        print("refused: the training images and D2 were embedded with different weights", file=sys.stderr)
        return 1
    keys = pd.read_parquet(a.compilation_keys, columns=["path", "key"])
    base_embs, base_hashes = {}, None
    if a.d2_baselines:
        bd = Path(a.d2_baselines)
        if sha256_file(bd / "manifest.parquet") != d2_info["manifest_sha256"]:
            print("refused: the baselines' manifest differs from D2's manifest", file=sys.stderr)
            return 1
        names = ["emb_fingervit.npy", "emb_deepprint.npy"]
        base_embs = {n[4:-4]: np.load(bd / n) for n in names}
        for n, x in base_embs.items():
            if len(x) != len(d2):
                print(f"refused: {n} has {len(x)} rows for {len(d2)} D2 images", file=sys.stderr)
                return 1
        base_hashes = {"manifest.parquet": d2_info["manifest_sha256"], **{n: sha256_file(bd / n) for n in names}}
    res, mx, exclusions = audit(train, train_emb, d2, d2_emb, keys, a.block)
    if base_embs:
        res["sensitivity"] = sensitivity(d2, {"ctrlvit": d2_emb, **base_embs}, exclusions)
        res["sensitivity"]["subset_rules"] = {
            "a_as_is": "all D2 fingers", "b_without_flagged": "without the fingers reaching their own genuine mean",
            "c_without_flagged_subjects": "without those and every D2 finger whose subject_key matches a flagged finger's",
            "d_without_structural_candidates": "without the label-structure candidates (label_structure)"}
    flags = pd.read_parquet(ts / "d2_subject_flags.parquet")
    shared = flags["subject_shared"].astype(bool)
    ts_json = ts / "training_set.json"
    c1_dj = json.loads(ts_json.read_text())["disjointness"] if ts_json.is_file() else None
    c1 = c1_dj["subjects"] if c1_dj else None
    rec = res["subjects_recomputed"]
    flags_with = {"total": int(shared.sum()), "by_source": _by_source(flags, shared)}
    flags_without = {"total": int((~shared).sum()), "by_source": _by_source(flags, ~shared)}
    agree = {"flags_vs_recomputed": rec["d2_fingers_with_shared_subject"] == flags_with
             and rec["d2_fingers_without_shared_subject"] == flags_without,
             "c1_json_vs_recomputed": None if c1 is None else c1 == rec["subjects"]}
    res["subjects"] = {"source": "Task C1 (d2_subject_flags.parquet; training_set.json for c1_subjects)",
                       "d2_fingers_with_shared_subject": flags_with,
                       "d2_fingers_without_shared_subject": flags_without,
                       "c1_subjects": c1, "recomputed_by_this_audit": rec, "agreement": agree}
    del res["subjects_recomputed"]
    inputs = {"train/manifest.parquet": tr_info["manifest_sha256"], "train/emb_ctrlvit.npy": tr_info["emb_sha256"],
              "d2/manifest.parquet": d2_info["manifest_sha256"], "d2/emb_ctrlvit.npy": d2_info["emb_sha256"],
              "compilation_keys.parquet": sha256_file(a.compilation_keys), "d2_subject_flags.parquet": sha256_file(ts / "d2_subject_flags.parquet")}
    if ts_json.is_file():
        inputs["training_set.json"] = sha256_file(ts_json)
    if base_hashes:
        inputs["d2_baselines"] = base_hashes
    here = Path(__file__).resolve()
    head = {"data_source": a.label, "weights_sha256": tr_info["weights_sha256"],
            "train_embed": tr_info, "d2_embed": d2_info, "inputs_sha256": inputs,
            "provenance": {"env": {"git_sha": git_sha(here.parents[2]), "image": os.environ.get("CANVEC_IMAGE"),
                                   "image_id": os.environ.get("CANVEC_IMAGE_ID")},
                           "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version(),
                           "script_sha256": sha256_file(here)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(maxfile, mx)
    out.write_text(json.dumps({**head, **res}, indent=1) + "\n")
    nd = res["near_duplicates"]
    print(f"audit: identity overlap {res['identity_overlap']}, content overlap {res['content_overlap']}, "
          f"threshold {nd['threshold']:.6f}, images at or above {nd['images_at_or_above']['total']}", flush=True)
    ok = (res["identity_overlap"] == 0 and res["content_overlap"] == 0 and agree["flags_vs_recomputed"]
          and agree["c1_json_vs_recomputed"] is not False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
