"""Subject-sensitivity check of the control encoder (ctrl-vit) on the casia_v5 stratum of D2 (aggregates only).

ctrl-vit's training set (Task C1) is finger- and image-disjoint from D2, but some D2 fingers belong to a subject (the
compilation's subject link: the identity without its last `:` part) who has other fingers in training.  Task C1 recorded
this per D2 finger (d2_subject_flags.parquet: identity, source, subject_shared).  In the casia_v5 stratum this script
compares the fingers whose subject has no training finger ("clean") with `--subsamples` size-matched random subsamples
of the fingers whose subject has one ("flagged"), with the statistics and conventions of the clean-subset analysis
(scripts/data/clean_subset.py: subset_stats, matched_subsamples through evaluate, compare, prekey_space, split_config;
raw and pre-key spaces; its default impostor-pair count).  ctrl-vit is the encoder of interest; finger-vit and DeepPrint
are evaluated on the same fingers, images, pairs and seeds as composition controls.  It also records the genuine-versus-
impostor AUC of the three D2 encoders on `polyu_cross` (label_check.py), a source ctrl-vit never saw in training.
The script reports numbers; it tests no hypothesis.

  python scripts/data/control_subject_check.py --d2-ctrl /data/d2_ctrl --d2-base /data/d2_base \
      --flags /data/trainset/d2_subject_flags.parquet --json /out/subject_check.json --label "<data source>"

--d2-ctrl holds manifest.parquet and emb_ctrlvit.npy, --d2-base the byte-identical manifest and emb_fingervit.npy,
emb_deepprint.npy.  Run it inside the container on the data host; the JSON holds no path, identity or per-image value.

Exit status: 0 ok; 1 when the two manifests differ, the rows disagree with the manifest, the stratum is degenerate or
its eligible sizes differ from --expect-clean / --expect-flagged (nothing is written); 2 when the output exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_subset as cs  # noqa: E402  (same directory)
import label_check  # noqa: E402
from extract_d2 import SEED, sha256_file  # noqa: E402
from ijiss_btp.experiments.isotropy import load_data  # noqa: E402

CORPUS = "d2_subject"          # seed label of the stratum's draws (stable_seed of the labels, never of the encoder)
STRATUM = "casia_v5"
UNSEEN_SOURCE = "polyu_cross"
ENCODERS = ("ctrlvit", "fingervit", "deepprint")
BASELINES = ("fingervit", "deepprint")
BINS_NOTE = f"histogram AUC over {cs.AUC_BINS} bins, as label_check.py"
FLAG_RULE = ("flagged = the D2 finger's subject (identity without its last ':' part) has at least one finger in the "
             "ctrl-vit training set (d2_subject_flags.parquet, Task C1); clean = it has none")
DATA_SOURCE = ("real: D2 embeddings of ctrl-vit, finger-vit and DeepPrint (second corpus), subject flags of the ctrl-vit "
               "training-set build")


def stratum_frame(flags: pd.DataFrame, row_ids, stratum: str, min_images: int) -> pd.DataFrame:
    """The subject flags of one source as the clean-subset analysis' flag table (identity, source, flagged, images,
    eligible); every embedding row must belong to a finger of the flags, and the source must have fingers."""
    frame = flags[["identity", "source"]].assign(flagged=flags["subject_shared"].astype(bool).to_numpy())
    f = cs.with_images(frame, row_ids, min_images)
    f = f[f["source"] == stratum].reset_index(drop=True)
    if f.empty:
        raise ValueError(f"no fingers in the stratum {stratum!r}")
    return f


def stratum_check(frame: pd.DataFrame, stratum: str) -> dict:
    """The stratum's finger counts; stops when it cannot be compared (fewer than 2 clean or fewer flagged than clean)."""
    problems = cs.stratum_problems(frame, [stratum])
    if problems:
        raise SystemExit("degenerate stratum: " + "; ".join(problems))
    return {"name": stratum, "fingers": cs.finger_counts(frame)[cs.POOLED]}


def check_encoder(x: np.ndarray, ids: np.ndarray, frame: pd.DataFrame, split: Mapping, stratum: str, subsamples: int,
                  n_imp: int, seed: int) -> dict:
    """One encoder on the stratum in the raw and pre-key spaces (clean_subset.evaluate), with the pre-key split record."""
    seeds = cs.stratum_seeds(seed, CORPUS, stratum, subsamples)
    xc, record = cs.prekey_space(x, ids, split)
    res = cs.evaluate({"raw": x, "prekey": xc}, ids, frame, stratum, seeds, n_imp)
    res["prekey_record"] = record
    res["dim"] = int(np.shape(x)[1])
    return res


def source_auc(man: pd.DataFrame, embs: Mapping[str, np.ndarray], source: str) -> dict:
    """Per encoder the label-check AUC and pair counts of the rows of one source (pairs within the source)."""
    keep = (man["source"].astype(str) == source).to_numpy()
    if not keep.any():
        raise ValueError(f"no rows of source {source!r}")
    sub = man.loc[keep, ["identity", "source", "device"]].reset_index(drop=True)
    res = label_check.label_check(sub, {m: np.asarray(x)[keep] for m, x in embs.items()})
    out = {}
    for m, r in res["models"].items():
        e = r["per_source"][source]
        out[m] = {"source": source, "auc": e["auc"], "genuine_pairs": e["genuine_pairs"],
                  "impostor_pairs": e["impostor_pairs"], "mean_genuine": e["mean_genuine"],
                  "mean_impostor": e["mean_impostor"], "images": int(keep.sum()),
                  "fingers": int(sub["identity"].nunique())}
    return out


def descriptive_seeds(seed: int, source: str) -> dict:
    """Seeds of a source's descriptive statistics (stable_seed of the labels, never of the encoder)."""
    return {"impostors": cs.stable_seed(seed, CORPUS, source, "descriptive", "impostors"),
            "one_image_per_finger": cs.stable_seed(seed, CORPUS, source, "descriptive", "one_image_per_finger")}


def descriptive_stats(x: np.ndarray, ids: np.ndarray, man: pd.DataFrame, split: Mapping, source: str, n_imp: int,
                      seed: int, min_images: int) -> dict:
    """clean_subset.subset_stats of all fingers of one source (at least min_images rows each) in the raw and pre-key
    spaces, with no comparison set; the pairs and image choice do not depend on the encoder or the space."""
    ids = np.asarray(ids, dtype=str)
    rows = (man["source"].astype(str) == source).to_numpy()
    if not rows.any():
        raise ValueError(f"no rows of source {source!r}")
    counts = pd.Series(ids[rows]).value_counts()
    fingers = np.sort(counts.index[counts >= min_images].to_numpy(dtype=str))
    if len(fingers) < 2:
        raise ValueError(f"fewer than 2 fingers with {min_images} images in source {source!r}")
    seeds = descriptive_seeds(seed, source)
    xc, record = cs.prekey_space(x, ids, split)
    spaces = {name: cs.subset_stats(v, ids, fingers, n_imp, seeds["impostors"], seeds["one_image_per_finger"],
                                    cs.AUC_BINS) for name, v in (("raw", x), ("prekey", xc))}
    counts_ = {k: spaces["raw"][k] for k in cs.COUNTS if k in spaces["raw"]}
    if any({k: v[k] for k in counts_} != counts_ for v in spaces.values()):
        raise ValueError("the spaces differ in their fingers, images or pairs")
    return {"source": source, "fingers": int(len(fingers)), "counts": counts_, "dim": int(np.shape(x)[1]),
            "spaces": spaces, "prekey_record": record}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--d2-ctrl", required=True, help="D2 under ctrl-vit (SIN layout: manifest.parquet, emb_ctrlvit.npy)")
    ap.add_argument("--d2-base", required=True, help="D2 under finger-vit and DeepPrint (same manifest)")
    ap.add_argument("--flags", required=True, help="d2_subject_flags.parquet of the training-set build")
    ap.add_argument("--configs", default=str(cs.CONFIGS), help="directory with isotropy_d2_<encoder>.yaml")
    ap.add_argument("--json", required=True, help="output JSON file (must not exist)")
    ap.add_argument("--label", default=DATA_SOURCE, help="data-source label written into the JSON")
    ap.add_argument("--stratum", default=STRATUM)
    ap.add_argument("--min-images", type=int, default=cs.MIN_IMAGES)
    ap.add_argument("--subsamples", type=int, default=cs.SUBSAMPLES, help="size-matched flagged subsamples (B)")
    ap.add_argument("--impostor-pairs", type=int, default=cs.IMPOSTOR_PAIRS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--expect-clean", type=int, help="stop unless the stratum has this many eligible clean fingers")
    ap.add_argument("--expect-flagged", type=int, help="stop unless it has this many eligible flagged fingers")
    a = ap.parse_args(argv)
    out = Path(a.json)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    t0 = time.time()
    ctrl, base = Path(a.d2_ctrl), Path(a.d2_base)
    man_sha = sha256_file(ctrl / "manifest.parquet")
    if sha256_file(base / "manifest.parquet") != man_sha:
        print("refused: the baselines' manifest differs from the ctrl-vit run's manifest", file=sys.stderr)
        return 1
    man = pd.read_parquet(ctrl / "manifest.parquet")
    specs = {m: {"kind": "sin", "root": str(ctrl if m == "ctrlvit" else base), "model": m} for m in ENCODERS}
    embs = {m: load_data(s) for m, s in specs.items()}
    ids = cs.paired_rows(embs)
    if not np.array_equal(ids, man["identity"].astype(str).to_numpy()):
        print("refused: the embeddings' rows differ from the manifest's", file=sys.stderr)
        return 1
    flags = pd.read_parquet(a.flags)
    frame = stratum_frame(flags, ids, a.stratum, a.min_images)
    try:
        stratum = stratum_check(frame, a.stratum)
    except SystemExit as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    eligible = stratum["fingers"]["eligible"]
    for name, want in (("clean", a.expect_clean), ("flagged", a.expect_flagged)):
        if want is not None and eligible[name] != want:
            print(f"refused: {eligible[name]} eligible {name} fingers in {a.stratum}, expected {want}", file=sys.stderr)
            return 1
    print(f"stratum {a.stratum}: {json.dumps(stratum['fingers'])}", flush=True)

    splits = {m: cs.split_config(Path(a.configs) / f"isotropy_d2_{m}.yaml", specs[m]) for m in ENCODERS}
    results = {}
    for m in ENCODERS:
        t1 = time.time()
        split, config_sha256 = splits[m]
        res = check_encoder(embs[m].x, ids, frame, split, a.stratum, a.subsamples, a.impostor_pairs, a.seed)
        res["prekey_record"].update(config=f"isotropy_d2_{m}.yaml", config_sha256=config_sha256)
        results[m] = res
        print(f"{m}: {time.time() - t1:.0f} s", flush=True)
    descriptive = {}
    for m in ENCODERS:
        d = descriptive_stats(embs[m].x, ids, man, splits[m][0], UNSEEN_SOURCE, a.impostor_pairs, a.seed, a.min_images)
        d["prekey_record"].update(config=f"isotropy_d2_{m}.yaml", config_sha256=splits[m][1])
        descriptive[m] = d
    polyu = source_auc(man, {m: embs[m].x for m in ENCODERS}, UNSEEN_SOURCE)

    inputs = {"d2_ctrl/manifest.parquet": man_sha, "d2_base/manifest.parquet": sha256_file(base / "manifest.parquet"),
              **{f"d2_{'ctrl' if m == 'ctrlvit' else 'base'}/emb_{m}.npy": sha256_file(
                  Path(specs[m]["root"]) / f"emb_{m}.npy") for m in ENCODERS},
              "d2_subject_flags.parquet": sha256_file(a.flags)}
    prov = cs.provenance()
    prov["clean_subset_sha256"] = prov.pop("script_sha256")
    prov["script_sha256"] = sha256_file(Path(__file__).resolve())
    report = {
        "data_source": a.label,
        "question": ("does ctrl-vit behave differently on the casia_v5 fingers whose subject has no training finger "
                     "(clean) than on size-matched random subsamples of those whose subject has one (flagged)?"),
        "flag_rule": FLAG_RULE,
        "stratum": stratum,
        "encoders": results,
        "polyu": polyu,
        "polyu_descriptive": descriptive,
        "polyu_descriptive_note": ("subset_stats of clean_subset.py on every polyu_cross finger with at least min_images "
                                   "images, in the raw and pre-key spaces; descriptive, no comparison set"),
        "polyu_note": ("polyu_cross is a D2 source absent from ctrl-vit's training data; AUC from label_check.py, "
                       f"{BINS_NOTE}"),
        "settings": {"seed": a.seed, "corpus_label": CORPUS, "seed_rule": cs.SEED_RULE, "subsamples": a.subsamples,
                     "subsample_rule": cs.SUBSAMPLE_RULE, "impostor_pairs": a.impostor_pairs,
                     "min_images": a.min_images, "pair_rule": cs.PAIR_RULE, "auc": cs.AUC_RULE, "eer": cs.EER_RULE,
                     "participation_ratio": cs.PR_RULE, "spaces": dict(cs.SPACES), "prekey_rule": cs.PREKEY_RULE,
                     "range_percentiles": list(cs.RANGE), "range_rule": cs.RANGE_RULE,
                     "single_draw": cs.SINGLE_DRAW_NOTE, "controls": "finger-vit and DeepPrint on the same flags"},
        "inputs_sha256": dict(sorted(inputs.items())),
        "provenance": prov,
        "seconds": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"subject check: {len(results)} encoders in {report['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
