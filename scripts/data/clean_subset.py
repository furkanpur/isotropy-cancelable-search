"""Clean-subset sensitivity of the utility and isotropy statistics (aggregates only).

The overlap measurement (training_overlap.py) flags, as a worst case, the evaluation fingers that occur in the most
plausible candidate training pool of each encoder.  This script compares every encoder on the fingers absent from its
pool ("clean") with the fingers present in it ("flagged").  It runs no attack and changes no experiment.  It runs inside
a container on the data host, with the data mounted read-only, `--network none` and the invoking user; the per-finger
flags stay in the output directory on the host, and only the JSON aggregates (no path, file name or identity) leave it.

  python scripts/data/clean_subset.py --overlap /data/overlap --d1-overlap /data/overlap/<a>.json \
      --d2-overlap /data/overlap/<b>.json --d1 /data/d1 --d1-map /data/relabel/d1_relabel.csv \
      --compilation-keys /data/keys/compilation_keys.parquet --compilation-manifest /data/compilation/MANIFEST.csv --d2 /data/corpus_d2 \
      --out /out --json /out/<c>.json [--configs configs] [--label "<data source>"]

--overlap holds the overlap measurement's per-image results, <a> and <b> its aggregates for the first and the second
corpus.  The inputs it recorded must be the files read here, and the flags must reproduce its finger counts, before
anything else is computed:
- first corpus (D1; encoders base and large): a relabel-map finger is flagged when any of its labelled compilation images (the
  full key table) or any of its D1 images has its exact content key in candidate pool A;
- second corpus (D2; finger-vit and DeepPrint): a finger is flagged when the maximum finger-vit cosine of any of its
  images to candidate pool B is at least tau.  DeepPrint, whose documented training data does not include the D2
  sources, is evaluated on the same split as the control.
Fingers need at least --min-images embedding rows; clean = not flagged.  Strata: all sources pooled, and every source
with at least --min-clean clean fingers; every stratum needs at least 2 clean fingers and at least as many flagged as
clean ones, which is checked before any embedding is loaded.  Each clean subset is compared with --subsamples seeded
random subsamples of the stratum's flagged fingers of the same size.  Per encoder, stratum and subset, in two spaces (the
raw L2-normalised embeddings, and the experiments' pre-key embeddings: the DEV mean of the first seed's split of the
encoder's isotropy config subtracted, then L2-normalised): the genuine-versus-impostor AUC (hist_auc of label_check.py)
and EER (ijiss_btp.eval.stats.eer) over all genuine pairs and --impostor-pairs distinct impostor pairs (every impostor
pair when fewer exist), the mean genuine and impostor cosine, and the participation ratio of the centred embeddings
(ijiss_btp.experiments.isotropy.spectrum_stats) on one seeded image per finger and on all images.  The seeds do not
depend on the encoder or the space, so all of them are evaluated on the same fingers, images and pairs.  Per stratum,
the source composition (fingers and genuine pairs) of the clean subset and of the flagged subsamples is recorded.  The
script reports numbers; it tests no hypothesis.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import yaml

import training_overlap as to  # same directory: the overlap measurement's readers, finger counts and pair sampling
from extract_d2 import SEED, environment, l2_normalize, sha256_file
from ijiss_btp.data.loaders import center_normalize
from ijiss_btp.data.splits import make_split
from ijiss_btp.eval.stats import eer
from ijiss_btp.experiments.isotropy import load_data, spectrum_stats, stable_seed
from label_check import bin_index, hist_auc

CONFIGS = Path(__file__).resolve().parents[2] / "configs"   # isotropy_<corpus>_<encoder>.yaml: the main isotropy configs
D1_MODELS, D2_MODELS = ("base", "large"), ("fingervit", "deepprint")
IMPOSTOR_PAIRS, SUBSAMPLES, MIN_IMAGES, MIN_CLEAN, AUC_BINS = 200_000, 20, 2, 100, 4000
RANGE = (2.5, 97.5)
POOLED = "all_sources"
STATISTICS = ("auc", "eer", "mean_genuine", "mean_impostor", "pr_one_image_per_finger", "pr_all_images")
PR_KEYS = ("pr_one_image_per_finger", "pr_all_images")
COUNTS = ("fingers", "images", "genuine_pairs", "impostor_pairs") + tuple(f"{k}_samples" for k in PR_KEYS)
IMPORTED = ("training_overlap.py", "extract_d2.py", "label_check.py", "build_relabel_map.py", "check_secondary_map.py")
LIBRARY = ("ijiss_btp.eval.stats", "ijiss_btp.eval.metrics", "ijiss_btp.experiments.isotropy", "ijiss_btp.data.loaders",
           "ijiss_btp.data.splits")
SPACES = {"raw": "raw (uncentred) cosine space",
          "prekey": "experiments' pre-key space (DEV-mean centred, then L2-normalised)"}

DATA_SOURCE = ("real: first-corpus (D1) embeddings of the base and large encoders, relabelled with the D1 map, and "
               "second-corpus (D2; compilation `real` images, internal research corpus of real fingerprint images) embeddings "
               "of finger-vit and DeepPrint; finger-level flags from the per-image results of the worst-case "
               "training-overlap measurement; no synthetic data")
QUESTION = ("do the encoders behave differently on the fingers absent from their candidate training pool (clean) than "
            "on size-matched samples of the fingers present in it (flagged)? No attack is run and no experiment changed")
D1_FLAG_RULE = ("a relabel-map finger is flagged when any of its labelled compilation images (the full key table: any "
                "impression, exclusion or partition) or any of its D1 images has its exact content key in candidate pool "
                "A; a pool key under several compilation identities flags each of them (the worst case)")
D2_FLAG_RULE = ("a D2 finger is flagged when at least one of its D2 images has a maximum finger-vit cosine to candidate "
                "pool B (float64 maxima of the overlap measurement) of at least tau; DeepPrint uses the same split")
D1_RECORDED = "finger count overall and per source equal to d1_fingers.worst_case of the overlap measurement"
D2_RECORDED = "finger count overall and per source equal to the fingers at tau (d2.by_threshold) of the overlap measurement"
SEED_RULE = ("stable_seed (ijiss_btp.experiments.isotropy) of (seed, corpus, stratum, subset, purpose); the encoder is not "
             "part of it, so the encoders of a corpus see the same fingers, images and pairs")
SUBSAMPLE_RULE = ("the clean subset of a stratum against `subsamples` random subsamples, without replacement, of the "
                  "stratum's eligible flagged fingers, each of the clean subset's size")
PAIR_RULE = ("all genuine pairs (every pair of rows of one finger) and `impostor_pairs` distinct pairs of rows of "
             "different fingers drawn uniformly at random (impostor_pairs of training_overlap.py), or every such pair when "
             "no more exist; both within the subset")
AUC_RULE = (f"histogram AUC (label_check.hist_auc) over {AUC_BINS} bins on [-1, 1]: P(genuine > impostor) plus half the "
            "probability that both fall in one bin")
EER_RULE = "ijiss_btp.eval.stats.eer (exact ROC interpolation), as in the experiments' utility rows"
PR_RULE = ("participation ratio, (sum of eigenvalues)^2 / sum of squared eigenvalues of the covariance of the "
           "L2-normalised rows centred on their own mean (ijiss_btp.experiments.isotropy.spectrum_stats), on one seeded "
           "image per finger (samples = fingers) and on all images; computed in both spaces, because center_normalize "
           "L2-normalises every row after subtracting the DEV mean: the rows are rescaled by 1/||x - m|| instead of "
           "1/||x||, and the subset-mean centring inside spectrum_stats removes only a common shift, so the two spaces "
           "can give different values; with n samples it cannot exceed min(n - 1, dim), and bounded_by_samples marks the "
           "entries where n - 1 < dim for the clean subset or a flagged subsample")
EMBEDDING_RULE = ("embeddings loaded as the experiments load them (ijiss_btp.experiments.isotropy.load_data, kind sin; D1 "
                  "with its relabel map, rows without an identity dropped); raw space: each row L2-normalised; pre-key "
                  "space: see prekey_rule; cosine similarity in both")
PREKEY_RULE = ("for each encoder, the first seed's split of its main isotropy config (isotropy_<corpus>_<encoder>.yaml): "
               "make_split with the config's n_dev, n_tgt and min_tgt_images, as SeedContext of "
               "ijiss_btp.experiments.isotropy builds it; the mean of the split's DEV rows is subtracted from every "
               "embedding and each row is L2-normalised (center_normalize), which gives the experiments' pre-key "
               "embeddings; every statistic is then computed on the same fingers, images, pairs and seeds as in the raw "
               "space")
RANGE_RULE = ("the clean value against the mean of the flagged subsamples' values and their 2.5th to 97.5th percentiles "
              "(numpy.percentile, linear interpolation); clean_outside_flagged_range is descriptive, not a test")
SINGLE_DRAW_NOTE = ("each clean value is a single draw (one sample of impostor pairs and one seeded image per finger) "
                    "with its own sampling error, which the flagged range does not include; clean_outside_flagged_range "
                    "is descriptive, and a clean value just outside the range can fall inside it with another seed")
COMPOSITION_NOTE = ("the pooled strata are confounded by source: a finger with k images contributes C(k, 2) genuine "
                    "pairs, so sources with many images per finger dominate the genuine pairs, and the clean subset and "
                    "the flagged subsamples mix the sources differently; composition gives, per stratum and source, the "
                    "fingers and genuine pairs of the clean subset and their mean, minimum and maximum over the flagged "
                    "subsamples")
CONTROL_RULE = ("DeepPrint is evaluated on the finger-vit split of D2; its documented training data does not include "
                "the D2 sources")


# ---- flags --------------------------------------------------------------------------------------------------------
def d1_flags(matches: pd.DataFrame, table: pd.DataFrame, id_source: Mapping[str, str], pool_keys: Iterable,
             d1_ids: Iterable[str]) -> pd.DataFrame:
    """Per relabel-map identity (sorted): its source and whether any of its labelled compilation images (key table rows:
    identity, key) or any of its D1 images (matches rows: identity, in_pool) has its exact key in pool A, with both
    routes (via_compilation, via_d1).  A pool key under several compilation identities counts for each of them."""
    ids = sorted(set(d1_ids))
    missing = [i for i in ids if i not in id_source]
    if missing:
        raise ValueError(f"{len(missing)} relabel-map identities are not in the compilation key table")
    pool = {k for k in pool_keys if isinstance(k, str)}
    t = table[table["identity"].isin(ids)]
    m = matches[matches["identity"].isin(ids)]
    via_compilation, via_d1 = set(t.loc[t["key"].isin(pool), "identity"]), set(m.loc[m["in_pool"].astype(bool), "identity"])
    compilation = np.array([i in via_compilation for i in ids], dtype=bool)
    d1 = np.array([i in via_d1 for i in ids], dtype=bool)
    return pd.DataFrame({"identity": ids, "source": [id_source[i] for i in ids], "via_compilation": compilation, "via_d1": d1,
                         "flagged": compilation | d1})


def d2_flags(nearest: pd.DataFrame, manifest: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Per D2 identity (sorted): its source and whether the maximum cosine of any of its images to pool B is at least
    tau (finger_hits of the overlap measurement); the maxima must be aligned with the manifest rows."""
    if nearest["path"].astype(str).tolist() != manifest["path"].astype(str).tolist():
        raise ValueError("the per-image maxima are not aligned with the D2 manifest")
    ids = manifest["identity"].astype(str).to_numpy()
    flagged, source = to.finger_hits(nearest["max_cosine"].to_numpy(dtype=np.float64) >= tau, ids, manifest["source"])
    return pd.DataFrame({"identity": np.unique(ids), "source": source, "flagged": flagged})


def reproduction_problems(flags: pd.DataFrame, recorded: Mapping, word: str) -> list[str]:
    """Differences between the finger counts of the flags (tally of the overlap measurement) and a recorded tally
    ({word: k, 'of': n, 'per_source': {source: {word: k, 'of': n}}}), overall and per source."""
    got = to.tally(flags["flagged"], flags["source"], word)
    problems = []
    if (got[word], got["of"]) != (recorded.get(word), recorded.get("of")):
        problems.append(f"{got[word]} of {got['of']} flagged here, {recorded.get(word)} of {recorded.get('of')} recorded")
    rec = recorded.get("per_source") or {}
    for s in sorted(set(got["per_source"]) | set(rec)):
        a, b = got["per_source"].get(s), rec.get(s)
        if a is None or b is None or (a[word], a["of"]) != (b.get(word), b.get("of")):
            here = f"{a[word]} of {a['of']} flagged here" if a else "none here"
            problems.append(f"{s}: {here}, " + (f"{b.get(word)} of {b.get('of')} recorded" if b else "none recorded"))
    return problems


def hash_problems(recorded: Mapping[str, str], recomputed: Mapping[str, str]) -> list[str]:
    """The inputs whose SHA-256 differs from the one the overlap measurement recorded, or that it did not record."""
    return [f"the overlap measurement recorded {'another' if k in recorded else 'no'} {k}"
            for k in sorted(recomputed) if recorded.get(k) != recomputed[k]]


# ---- subsets and strata -------------------------------------------------------------------------------------------
def with_images(flags: pd.DataFrame, row_ids: Iterable[str], min_images: int) -> pd.DataFrame:
    """The flags with each finger's number of embedding rows and whether it has at least min_images (eligible); every
    embedding row must belong to a finger of the flags."""
    counts = pd.Series(np.asarray(list(row_ids), dtype=str)).value_counts()
    unknown = set(counts.index) - set(flags["identity"])
    if unknown:
        raise ValueError(f"{len(unknown)} embedding identities have no flag")
    images = flags["identity"].map(counts).fillna(0).astype(int).to_numpy()
    return flags.assign(images=images, eligible=images >= min_images)


def paired_rows(embs: Mapping) -> np.ndarray:
    """The identity of every embedding row; it must be the same for all encoders of a corpus (paired evaluation)."""
    ids = [np.asarray(e.ids, dtype=str) for e in embs.values()]
    if any(not np.array_equal(ids[0], other) for other in ids[1:]):
        raise ValueError("the encoders of a corpus have different rows")
    return ids[0]


def finger_counts(flags: pd.DataFrame) -> dict:
    """Clean and flagged fingers, pooled and per source: all of them, their embedding rows, the eligible ones and
    those the image rule drops."""
    def block(d: pd.DataFrame) -> dict:
        clean, ok = ~d["flagged"].to_numpy(dtype=bool), d["eligible"].to_numpy(dtype=bool)
        return {"fingers": int(len(d)), "clean": int(clean.sum()), "flagged": int((~clean).sum()),
                "images": int(d["images"].sum()),
                "eligible": {"clean": int((clean & ok).sum()), "flagged": int((~clean & ok).sum())},
                "dropped_by_image_rule": {"clean": int((clean & ~ok).sum()), "flagged": int((~clean & ~ok).sum())}}
    return {POOLED: block(flags), "per_source": {str(s): block(g) for s, g in flags.groupby("source", sort=True)}}


def strata(flags: pd.DataFrame, min_clean: int) -> list[str]:
    """The pooled stratum, then every source (sorted) with at least min_clean eligible clean fingers."""
    counts = flags.loc[flags["eligible"] & ~flags["flagged"], "source"].value_counts()
    return [POOLED] + sorted(str(s) for s, n in counts.items() if n >= min_clean)


def stratum_fingers(flags: pd.DataFrame, stratum: str) -> tuple[np.ndarray, np.ndarray]:
    """The eligible clean and the eligible flagged fingers of a stratum (each sorted)."""
    d = flags[flags["eligible"]]
    if stratum != POOLED:
        d = d[d["source"] == stratum]
    hit, ids = d["flagged"].to_numpy(dtype=bool), d["identity"].to_numpy(dtype=str)
    return np.sort(ids[~hit]), np.sort(ids[hit])


def stratum_seeds(seed: int, corpus: str, stratum: str, subsamples: int) -> dict:
    """Seeds of a stratum (stable_seed of the labels, never of the encoder): the clean subset's impostor pairs and
    image choice, and per flagged subsample its draw, impostor pairs and image choice."""
    def s(*parts) -> int:
        return stable_seed(seed, corpus, stratum, *parts)
    return {"clean": {"impostors": s("clean", "impostors"), "one_image_per_finger": s("clean", "one_image_per_finger")},
            "flagged": [{"subsample": s("flagged", k, "subsample"), "impostors": s("flagged", k, "impostors"),
                         "one_image_per_finger": s("flagged", k, "one_image_per_finger")} for k in range(subsamples)]}


def matched_subsamples(flagged: np.ndarray, size: int, seeds: Sequence[int]) -> list[np.ndarray]:
    """One subsample of `size` flagged fingers per seed, drawn without replacement (each sorted)."""
    flagged = np.asarray(flagged, dtype=str)
    if size > len(flagged):
        raise ValueError(f"fewer flagged fingers ({len(flagged)}) than clean ones ({size})")
    return [np.sort(flagged[np.random.default_rng(s).choice(len(flagged), size=size, replace=False)]) for s in seeds]


def stratum_problems(flags: pd.DataFrame, strata_: Sequence[str]) -> list[str]:
    """Strata that cannot be compared: fewer than 2 eligible clean fingers, or fewer eligible flagged than clean ones."""
    problems = []
    for s in strata_:
        clean, flagged = stratum_fingers(flags, s)
        if len(clean) < 2:
            problems.append(f"{s}: {len(clean)} clean (at least 2 needed)")
        if len(flagged) < len(clean):
            problems.append(f"{s}: {len(flagged)} flagged for {len(clean)} clean (at least as many needed)")
    return problems


def stratum_subsets(flags: pd.DataFrame, stratum: str, seeds: Mapping) -> tuple[np.ndarray, np.ndarray, list]:
    """The eligible clean and flagged fingers of a stratum and its size-matched flagged subsamples."""
    clean, flagged = stratum_fingers(flags, stratum)
    return clean, flagged, matched_subsamples(flagged, len(clean), [s["subsample"] for s in seeds["flagged"]])


def composition(flags: pd.DataFrame, clean: np.ndarray, subsamples: Sequence[np.ndarray]) -> dict:
    """Per source: the fingers and genuine pairs (C(k, 2) for a finger with k embedding rows) of the clean subset, and
    their mean, minimum and maximum over the flagged subsamples; no identity."""
    f = flags.set_index("identity")[["source", "images"]]

    def per_source(fingers) -> pd.DataFrame:
        d = f.loc[np.asarray(fingers, dtype=str)]
        return d.assign(pairs=d["images"] * (d["images"] - 1) // 2).groupby("source").agg(
            fingers=("images", "size"), genuine_pairs=("pairs", "sum"))

    parts = [per_source(clean)] + [per_source(s) for s in subsamples]
    sources = sorted(str(s) for s in set().union(*(p.index for p in parts)))
    parts = [p.reindex(sources, fill_value=0) for p in parts]

    def spread(col: str, s: str) -> dict:
        v = np.array([p.loc[s, col] for p in parts[1:]], dtype=np.int64)
        return {"mean": float(v.mean()), "min": int(v.min()), "max": int(v.max())}
    return {"clean": {s: {"fingers": int(parts[0].loc[s, "fingers"]), "genuine_pairs": int(parts[0].loc[s, "genuine_pairs"])}
                      for s in sources},
            "flagged_subsamples": {s: {"fingers": spread("fingers", s), "genuine_pairs": spread("genuine_pairs", s)}
                                   for s in sources}}


# ---- statistics ---------------------------------------------------------------------------------------------------
def impostor_sample(ids: Iterable[str], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n distinct cross-identity row pairs (i < j) drawn by impostor_pairs of the overlap measurement (seeded), or
    every cross-identity pair when no more than n exist."""
    ids = np.asarray(list(ids), dtype=str)
    m, sizes = len(ids), np.unique(ids, return_counts=True)[1]
    if m * (m - 1) // 2 - int((sizes * (sizes - 1) // 2).sum()) > n:
        return to.impostor_pairs(ids, n, seed)
    i, j = np.triu_indices(m, 1)
    keep = ids[i] != ids[j]
    return i[keep], j[keep]


def one_per_finger(ids: Iterable[str], seed: int) -> np.ndarray:
    """One row index per identity (identities sorted), chosen uniformly among its rows (seeded)."""
    rng = np.random.default_rng(seed)
    groups = pd.DataFrame({"identity": np.asarray(list(ids), dtype=str)}).groupby("identity", sort=True).indices
    return np.array([rng.choice(groups[k]) for k in sorted(groups)], dtype=np.int64)


def subset_stats(x: np.ndarray, ids: np.ndarray, fingers: np.ndarray, n_imp: int, imp_seed: int, one_seed: int,
                 bins: int) -> dict:
    """Statistics of the L2-normalised embedding rows of `fingers`: the AUC and EER of all genuine pairs against the
    impostor sample, the mean genuine and impostor cosine, and the participation ratio of the centred rows on one image
    per finger and on all images, with the counts."""
    ids = np.asarray(ids, dtype=str)
    rows = np.flatnonzero(np.isin(ids, fingers))
    sub = ids[rows]
    absent = set(np.asarray(fingers, dtype=str).tolist()) - set(sub.tolist())
    if absent:
        raise ValueError(f"{len(absent)} fingers have no rows")
    xs = l2_normalize(np.asarray(x)[rows])
    gen = to.genuine_scores(xs, sub)
    a, b = impostor_sample(sub, n_imp, imp_seed)
    imp = to.pair_scores(xs, a, b)
    one = one_per_finger(sub, one_seed)
    return {"fingers": int(len(one)), "images": int(len(rows)), "genuine_pairs": int(gen.size),
            "impostor_pairs": int(imp.size),
            "auc": hist_auc(*(np.bincount(bin_index(s, bins), minlength=bins) for s in (gen, imp))),
            "eer": float(eer(gen, imp)), "mean_genuine": float(gen.mean()), "mean_impostor": float(imp.mean()),
            "pr_one_image_per_finger": spectrum_stats(xs[one])["participation_ratio"],
            "pr_one_image_per_finger_samples": int(len(one)),
            "pr_all_images": spectrum_stats(xs)["participation_ratio"], "pr_all_images_samples": int(len(rows))}


def compare(clean: Mapping, flagged: Sequence[Mapping]) -> dict:
    """Per statistic: the clean value, the mean of the flagged subsamples' values, their RANGE percentiles and whether
    the clean value lies outside that range (descriptive, not a test); per count: the clean count and the flagged
    minimum, mean and maximum."""
    lo_key, hi_key = (f"flagged_p{p:g}" for p in RANGE)
    stats = {}
    for k in STATISTICS:
        v = np.array([f[k] for f in flagged], dtype=np.float64)
        lo, hi = (float(p) for p in np.percentile(v, RANGE))
        c = float(clean[k])
        stats[k] = {"clean": c, "flagged_mean": float(v.mean()), lo_key: lo, hi_key: hi,
                    "clean_outside_flagged_range": bool(c < lo or c > hi)}
    counts = {}
    for k in COUNTS:
        v = np.array([f[k] for f in flagged], dtype=np.int64)
        counts[k] = {"clean": int(clean[k]), "flagged_min": int(v.min()), "flagged_mean": float(v.mean()),
                     "flagged_max": int(v.max())}
    return {"statistics": stats, "counts": counts}


def evaluate(spaces: Mapping[str, np.ndarray], ids: np.ndarray, flags: pd.DataFrame, stratum: str, seeds: Mapping,
             n_imp: int) -> dict:
    """One encoder on one stratum, in every space (name -> embedding rows): the clean subset against its size-matched
    flagged subsamples, on the same fingers, images, pairs and seeds in each space; each participation ratio carries its
    sample counts and its bound min(n - 1, dim)."""
    clean, flagged, subs = stratum_subsets(flags, stratum, seeds)
    out = {"fingers": {"clean": int(len(clean)), "flagged": int(len(flagged)), "subsample_size": int(len(clean))},
           "spaces": {}}
    for name, x in spaces.items():
        c = subset_stats(x, ids, clean, n_imp, seeds["clean"]["impostors"], seeds["clean"]["one_image_per_finger"],
                         AUC_BINS)
        res = compare(c, [subset_stats(x, ids, sub, n_imp, s["impostors"], s["one_image_per_finger"], AUC_BINS)
                          for sub, s in zip(subs, seeds["flagged"])])
        dim = int(np.shape(x)[1])
        for k in PR_KEYS:
            n = res["counts"].pop(f"{k}_samples")
            res["statistics"][k].update(samples=n, max_possible_clean=min(n["clean"] - 1, dim),
                                        bounded_by_samples=bool(min(n["clean"], n["flagged_min"]) - 1 < dim))
        if out.setdefault("counts", res["counts"]) != res["counts"]:
            raise ValueError("the spaces differ in their fingers, images or pairs")
        out["spaces"][name] = {"label": SPACES[name], "statistics": res["statistics"]}
    return out


# ---- the experiments' pre-key space -------------------------------------------------------------------------------
def split_config(path: str | Path, spec: Mapping) -> tuple[dict, str]:
    """The split block of an encoder's isotropy config, which must describe the embeddings loaded here (kind, model,
    relabel map and key), and the config's SHA-256."""
    cfg = yaml.safe_load(Path(path).read_text())
    d = cfg.get("data") or {}
    same = (d.get("kind") == spec["kind"] and d.get("model") == spec["model"]
            and Path(d.get("relabel") or "").name == Path(spec.get("relabel") or "").name
            and d.get("relabel_key") == spec.get("relabel_key"))
    if not same:
        raise SystemExit(f"{Path(path).name} describes other embeddings than the {spec['model']} ones loaded here")
    return cfg["split"], sha256_file(path)


def prekey_space(x: np.ndarray, ids: np.ndarray, split: Mapping) -> tuple[np.ndarray, dict]:
    """The experiments' pre-key embeddings, as SeedContext of ijiss_btp.experiments.isotropy builds them: the first
    seed's split with the config's sizes (make_split), the mean of its DEV rows subtracted from every row and each row
    L2-normalised (center_normalize); with the record of the split."""
    x = np.asarray(x, dtype=np.float64)
    seed, n_dev, n_tgt = int(split["seeds"][0]), int(split["n_dev"]), int(split["n_tgt"])
    min_tgt = int(split.get("min_tgt_images", 3))
    s = make_split(np.asarray(ids), n_dev, n_tgt, seed, min_tgt)
    dev = s.rows(s.dev_ids)
    return center_normalize(x, x[dev].mean(0)), {"seed": seed, "n_dev": n_dev, "n_tgt": n_tgt, "min_tgt_images": min_tgt,
                                                 "dev_fingers": int(len(s.dev_ids)), "dev_images": int(len(dev))}


# ---- provenance and command -----------------------------------------------------------------------------------------
def provenance() -> dict:
    """Hashes of this script, of its neighbours (by file name) and of the library modules it uses (by module name, so
    that the JSON holds no path), the container environment and the software versions."""
    import pyarrow
    here = Path(__file__).resolve()
    return {"script_sha256": sha256_file(here),
            "imported_sha256": {**{n: sha256_file(here.parent / n) for n in IMPORTED},
                                **{m: sha256_file(importlib.import_module(m).__file__) for m in LIBRARY}},
            "env": environment(),
            "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "scipy": scipy.__version__, "pyarrow": pyarrow.__version__}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--overlap", required=True, help="output directory of the overlap measurement (per-image results)")
    ap.add_argument("--d1-overlap", required=True, help="its aggregate JSON for the first corpus")
    ap.add_argument("--d2-overlap", required=True, help="its aggregate JSON for the second corpus")
    ap.add_argument("--d1", required=True, help="first corpus in the SIN layout (manifest.parquet, emb_<model>.npy)")
    ap.add_argument("--d1-map", required=True, help="relabel map of the first corpus (file,identity)")
    ap.add_argument("--d1-models", nargs="+", default=list(D1_MODELS))
    ap.add_argument("--compilation-keys", required=True, help="compilation_keys.parquet of extract_d2.py keys (json beside it)")
    ap.add_argument("--compilation-manifest", required=True, help="the compilation MANIFEST.csv the key table was built from")
    ap.add_argument("--d2", required=True, help="second corpus in the SIN layout")
    ap.add_argument("--d2-models", nargs="+", default=list(D2_MODELS))
    ap.add_argument("--configs", default=str(CONFIGS), help="directory with isotropy_<corpus>_<encoder>.yaml")
    ap.add_argument("--out", required=True, help="output directory on the data host (per-finger flags)")
    ap.add_argument("--json", required=True, help="aggregate JSON written by this script")
    ap.add_argument("--label", default=DATA_SOURCE, help="data-source label written into the JSON")
    ap.add_argument("--min-images", type=int, default=MIN_IMAGES)
    ap.add_argument("--min-clean", type=int, default=MIN_CLEAN, help="clean fingers a source needs to be a stratum")
    ap.add_argument("--subsamples", type=int, default=SUBSAMPLES, help="size-matched flagged subsamples (B)")
    ap.add_argument("--impostor-pairs", type=int, default=IMPOSTOR_PAIRS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    t0 = time.time()
    overlap, d1, d2, out = Path(a.overlap), Path(a.d1), Path(a.d2), Path(a.out)
    rec1, rec2 = (json.loads(Path(p).read_text()) for p in (a.d1_overlap, a.d2_overlap))

    # the inputs the overlap measurement recorded must be the files read here
    manifest_sha256 = sha256_file(a.compilation_manifest)
    table, id_source, key_info = to.load_key_table(a.compilation_keys, manifest_sha256)
    now1 = {"d1_manifest": sha256_file(d1 / "manifest.parquet"), "d1_relabel_map": sha256_file(a.d1_map),
            "compilation_manifest": manifest_sha256, "compilation_key_table": key_info["table_sha256"]}
    now1_out = {"pool_a_keys": sha256_file(overlap / to.POOL_A_KEYS),
                "d1_exact_matches": sha256_file(overlap / to.D1_MATCHES)}
    now2 = {"d2_manifest": sha256_file(d2 / "manifest.parquet"), "d2_emb_fingervit": sha256_file(d2 / "emb_fingervit.npy")}
    problems = (hash_problems(rec1.get("inputs_sha256", {}), now1) + hash_problems(rec1.get("outputs_sha256", {}), now1_out)
                + hash_problems(rec2.get("inputs_sha256", {}), now2))
    if problems:
        raise SystemExit("the inputs differ from those of the overlap measurement: " + "; ".join(problems))

    # the flags, checked against the overlap measurement's finger counts before anything else is computed
    d1_rows, d1_ids, _ = to.load_d1(d1 / "manifest.parquet", a.d1_map)
    f1 = d1_flags(pd.read_parquet(overlap / to.D1_MATCHES, columns=["identity", "in_pool"]), table, id_source,
                  pd.read_parquet(overlap / to.POOL_A_KEYS, columns=["key"])["key"], d1_ids)
    tau = float(rec2["tau_rule"]["tau"])
    man2 = pd.read_parquet(d2 / "manifest.parquet", columns=["identity", "source", "path"])
    f2 = d2_flags(pd.read_parquet(overlap / to.D2_NEAREST, columns=["path", "max_cosine"]), man2, tau)
    at_tau = next((e for e in rec2["d2"]["by_threshold"] if e["value"] == tau), None)
    if at_tau is None:
        raise SystemExit("the second-corpus aggregates have no finger counts at tau")
    problems = ([f"D1 {p}" for p in reproduction_problems(f1, rec1["d1_fingers"]["worst_case"], "in_pool")]
                + [f"D2 {p}" for p in reproduction_problems(f2, at_tau["fingers"], "above")])
    if problems:
        raise SystemExit("the flags do not reproduce the overlap measurement's finger counts: " + "; ".join(problems))
    print(f"flags reproduced: D1 {int(f1['flagged'].sum())} of {len(f1)} fingers flagged; D2 {int(f2['flagged'].sum())} "
          f"of {len(f2)} at tau {tau}", flush=True)

    # the subsets and strata, from the manifests' rows, and the encoders' configs, checked before any embedding is loaded
    rows = {"d1": d1_rows.loc[d1_rows["identity"] != "", "identity"].to_numpy(dtype=str),
            "d2": man2["identity"].astype(str).to_numpy()}
    flags = {"d1": with_images(f1, rows["d1"], a.min_images), "d2": with_images(f2, rows["d2"], a.min_images)}
    strata_ = {c: strata(f, a.min_clean) for c, f in flags.items()}
    problems = [f"{c.upper()} {p}" for c, f in flags.items() for p in stratum_problems(f, strata_[c])]
    if problems:
        raise SystemExit("degenerate strata: " + "; ".join(problems))
    specs = {"d1": [{"kind": "sin", "root": str(d1), "model": m, "relabel": str(a.d1_map)} for m in a.d1_models],
             "d2": [{"kind": "sin", "root": str(d2), "model": m} for m in a.d2_models]}
    splits = {c: {s["model"]: split_config(Path(a.configs) / f"isotropy_{c}_{s['model']}.yaml", s) for s in specs[c]}
              for c in specs}

    # the embeddings, as the experiments load them; per corpus the statistics in both spaces and the composition
    out.mkdir(parents=True, exist_ok=True)
    fingers, loading, seeds, composition_, results, outputs = {}, {}, {}, {}, {}, {}
    for corpus, corpus_specs in specs.items():
        f = flags[corpus]
        f.to_parquet(out / f"{corpus}_finger_flags.parquet", index=False)
        outputs[f"{corpus}_finger_flags"] = sha256_file(out / f"{corpus}_finger_flags.parquet")
        fingers[corpus] = finger_counts(f)
        print(f"{corpus}: {json.dumps(fingers[corpus])}; strata {strata_[corpus]}", flush=True)
        seeds[corpus] = {s: stratum_seeds(a.seed, corpus, s, a.subsamples) for s in strata_[corpus]}
        composition_[corpus] = {}
        for s in strata_[corpus]:
            clean, _, subs = stratum_subsets(f, s, seeds[corpus][s])
            composition_[corpus][s] = composition(f, clean, subs)
        embs = {s["model"]: load_data(s) for s in corpus_specs}
        ids = paired_rows(embs)
        if not np.array_equal(ids, rows[corpus]):
            raise SystemExit(f"the {corpus} embeddings have other rows than its manifest")
        n = len(pd.read_parquet(Path(corpus_specs[0]["root"]) / "manifest.parquet", columns=["path"]))
        loading[corpus] = {"manifest_rows": n, "rows": int(len(ids)), "dropped_without_identity": n - len(ids)}
        results[corpus] = {}
        for model, e in embs.items():
            t1 = time.time()
            split, config_sha256 = splits[corpus][model]
            xc, record = prekey_space(e.x, ids, split)
            results[corpus][model] = {
                "dim": int(e.x.shape[1]),
                "prekey": {"config": f"isotropy_{corpus}_{model}.yaml", "config_sha256": config_sha256, **record},
                "strata": {s: evaluate({"raw": e.x, "prekey": xc}, ids, f, s, seeds[corpus][s], a.impostor_pairs)
                           for s in strata_[corpus]}}
            print(f"{corpus} {model}: {len(strata_[corpus])} strata in {time.time() - t1:.0f} s", flush=True)

    inputs = {**now1, **now1_out, **now2, "d2_pool_maxima": sha256_file(overlap / to.D2_NEAREST),
              "compilation_key_table_metadata": sha256_file(Path(a.compilation_keys).with_suffix(".json")),
              **{f"d1_emb_{m}": sha256_file(d1 / f"emb_{m}.npy") for m in a.d1_models},
              **{f"d2_emb_{m}": sha256_file(d2 / f"emb_{m}.npy") for m in a.d2_models},
              "d1_overlap_aggregates": sha256_file(a.d1_overlap), "d2_overlap_aggregates": sha256_file(a.d2_overlap)}
    report = {
        "data_source": a.label,
        "question": QUESTION,
        "pools": {"d1": to.POOL_A_LABEL, "d2": to.POOL_B_LABEL},
        "flags": {
            "d1": {"rule": D1_FLAG_RULE, "check": D1_RECORDED,
                   "reproduced": {"flagged": int(f1["flagged"].sum()), "of": int(len(f1))},
                   "routes": {"via_compilation_key_table": int(f1["via_compilation"].sum()), "via_d1_images": int(f1["via_d1"].sum()),
                              "only_via_d1_images": int((f1["via_d1"] & ~f1["via_compilation"]).sum())}},
            "d2": {"rule": D2_FLAG_RULE, "check": D2_RECORDED, "tau": tau,
                   "reproduced": {"flagged": int(f2["flagged"].sum()), "of": int(len(f2))}}},
        "fingers": fingers,
        "loading": loading,
        "strata": strata_,
        "composition": composition_,
        "settings": {
            "tau": tau, "seed": a.seed, "seed_rule": SEED_RULE, "subsamples": a.subsamples,
            "subsample_rule": SUBSAMPLE_RULE, "min_images": a.min_images,
            "min_images_rule": (f"fingers with at least {a.min_images} embedding rows, so that every finger has a genuine "
                                "pair; the others are left out of both subsets"),
            "min_clean_fingers": a.min_clean,
            "stratum_rule": (f"all sources pooled, and every source with at least {a.min_clean} eligible clean fingers; "
                             "every stratum needs at least 2 clean fingers and at least as many flagged as clean ones"),
            "impostor_pairs": a.impostor_pairs, "pair_rule": PAIR_RULE, "auc": AUC_RULE, "eer": EER_RULE,
            "participation_ratio": PR_RULE, "embeddings": EMBEDDING_RULE, "spaces": dict(SPACES),
            "prekey_rule": PREKEY_RULE, "range_percentiles": list(RANGE), "range_rule": RANGE_RULE,
            "single_draw": SINGLE_DRAW_NOTE, "composition_note": COMPOSITION_NOTE, "control": CONTROL_RULE},
        "seeds": seeds,
        "results": results,
        "inputs_sha256": dict(sorted(inputs.items())),
        "outputs_sha256": outputs,
        "provenance": provenance(),
        "seconds": round(time.time() - t0, 1),
    }
    Path(a.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"clean subset: {sum(len(v) for v in strata_.values())} strata, {len(a.d1_models) + len(a.d2_models)} "
          f"encoders in {report['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
