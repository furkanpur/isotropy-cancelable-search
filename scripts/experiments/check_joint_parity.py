#!/usr/bin/env python3
"""Parity of a joint-stage run with the committed anchor stage of the same configuration.

The joint stage's `proj` rows reuse the anchor stage's keys, thresholds, colluders and victims, so at every budget of
the committed r_grid they must reproduce the committed anchor.csv rows with variant exact, decoder proj and scheme
lrp_full, for every seed, key and pre-conditioner of the joint stage: the E1/E2 relative success, successes, trials
and genuine acceptance (which fixes the thresholds), the number of victims and the pre-key cosine. The two runs must
also share the configuration (apart from the stage selection) and every metadata field that fixes the data the stages
see (DATA_FIELDS: data source and file hashes, corpus and split counts, the AUX pools). Differences in the recorded
environment (ENVIRONMENT) are reported but do not fail the check. Exit status 0 on parity, 1 otherwise.

Usage:
    python scripts/experiments/check_joint_parity.py <joint_run_dir> <isotropy_run_dir> [--atol 1e-9]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ijiss_btp.experiments.isotropy import JOINT_SCHEME, joint_preconditioners, valid_budgets
from ijiss_btp.precondition import make_preconditioner
from ijiss_btp.protect.keyed import parse_scheme

KEYS = ["seed", "key", "pre", "scheme", "r"]
EXACT = ["n_targets", "e1_successes", "e1_trials", "e2_successes", "e2_trials"]
CLOSE = ["e1_rel", "e2_rel", "e1_genuine", "e2_genuine", "rec_cos_z"]
DATA_FIELDS = ["data_source", "data_files", "corpus_counts", "split_counts", "aux_sizes", "shifted_aux_source",
               "shifted_aux_removed", "secondary_removed_shared_identity"]
ENVIRONMENT = ["image_freeze", "python", "numpy", "omp_num_threads", "cpu_model"]


def _config(meta: dict) -> dict:
    return {k: v for k, v in meta["config"].items() if k != "stages"}


def _short(value) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def parity(joint_dir, anchor_dir, atol: float = 1e-9) -> tuple[list[str], dict]:
    """(problems, summary): an empty problem list means parity; the summary holds the compared rows, the max |diff|
    per floating column and the environment fields that differ (informational)."""
    joint_dir, anchor_dir = Path(joint_dir), Path(anchor_dir)
    mj = json.loads((joint_dir / "metadata.json").read_text())
    ma = json.loads((anchor_dir / "metadata.json").read_text())
    cfg, cfg_j = _config(ma), _config(mj)
    problems = []
    if cfg_j != cfg:
        problems.append(f"config differs in {sorted(k for k in set(cfg) | set(cfg_j) if cfg.get(k) != cfg_j.get(k))}")
    problems += [f"{k} differs (joint run {_short(mj.get(k))}, committed run {_short(ma.get(k))})"
                 for k in DATA_FIELDS if mj.get(k) != ma.get(k)]
    if problems:
        return problems, {}
    pres = [make_preconditioner(s).name for s in joint_preconditioners(cfg["preconditioners"])]
    # Budgets from the configured n_tgt: a run whose split held fewer targets fails safe, with rows reported missing.
    budgets = valid_budgets(cfg["attacks"]["anchor"], cfg["split"]["n_tgt"])
    expected = pd.MultiIndex.from_product([cfg["split"]["seeds"], range(cfg["keys_per_seed"]), pres,
                                           [parse_scheme(JOINT_SCHEME).name], budgets], names=KEYS)
    anchor = pd.read_csv(anchor_dir / "anchor.csv")
    joint = pd.read_csv(joint_dir / "joint.csv")
    sides = {"committed anchor.csv": anchor[(anchor.variant == "exact") & (anchor.decoder == "proj")],
             "joint run": joint[joint.method == "proj"]}
    for name, df in sides.items():
        df = df.set_index(KEYS)
        df = df[df.index.isin(expected)]
        missing = expected.difference(df.index)
        if len(missing):
            problems.append(f"missing in the {name}: {len(missing)} of {len(expected)} rows, e.g. {missing[0]}")
        if df.index.duplicated().any():
            problems.append(f"duplicate rows in the {name}")
        sides[name] = df
    if problems:
        return problems, {}
    a, j = (df.loc[expected] for df in sides.values())
    summary = {"rows": len(expected), "environment": [k for k in ENVIRONMENT if mj.get(k) != ma.get(k)]}
    for col in EXACT:
        bad = a[col].to_numpy() != j[col].to_numpy()
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} of {len(bad)} rows differ, e.g. {expected[int(np.argmax(bad))]}")
    for col in CLOSE:
        x, y = a[col].to_numpy(dtype=float), j[col].to_numpy(dtype=float)
        bad = ~np.isclose(x, y, rtol=0.0, atol=atol, equal_nan=True)
        both = np.isfinite(x) & np.isfinite(y)
        summary[col] = float(np.max(np.abs(x - y)[both])) if both.any() else 0.0
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} of {len(bad)} rows differ by more than {atol:g} "
                            f"(max {summary[col]:.3g}), e.g. {expected[int(np.argmax(bad))]}")
    return problems, summary


def check(joint_dir, anchor_dir, atol: float = 1e-9) -> list[str]:
    return parity(joint_dir, anchor_dir, atol)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joint_run", help="run directory with joint.csv and metadata.json")
    ap.add_argument("isotropy_run", help="committed run directory of the same configuration (anchor.csv)")
    ap.add_argument("--atol", type=float, default=1e-9, help="absolute tolerance for the floating columns")
    args = ap.parse_args(argv)
    problems, summary = parity(args.joint_run, args.isotropy_run, args.atol)
    if problems:
        print("parity: FAILED")
        for p in problems:
            print(f"  {p}")
        return 1
    diffs = ", ".join(f"{c} {summary[c]:.2g}" for c in CLOSE)
    print(f"parity: OK ({summary['rows']} rows; exact {', '.join(EXACT)}; max |diff| {diffs}; "
          f"equal {', '.join(DATA_FIELDS)} and config)")
    if summary["environment"]:
        print(f"environment differs (informational): {', '.join(summary['environment'])}")
    else:
        print("environment: same image freeze, Python, NumPy, threads and CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
