"""Aggregate isotropy runs across seeds and keys: curves with seed-level 95% CIs and r50 values.

Two interval types are written for the identity-level endpoints (E1/E2 relative success):

* `<col>_lo/_hi`: t-interval over seed-level means. It reflects between-split variability over overlapping
  identity splits (the splits share identities) and is NOT an identity-level confidence interval.
* `<col>_cons_lo/_cons_hi` (e1_rel, e2_rel, best_e1_rel): conservative identity-level interval. The runner's
  per-row paired-bootstrap CI gives se_row = (hi - lo) / (2 * 1.959964); se_row is averaged over keys within a
  seed, then over seeds (SE_bar), and the interval is mean +/- 1.959964 * SE_bar. Averaging SEs instead of
  dividing by sqrt(number of splits) treats the overlapping splits as fully correlated, hence conservative.

Zero-pair "best" statistics are taken at the argmax-E1 row over p for each (seed, key): best_e1_rel, its SE,
best_rec_cos_z, best_abs_cos_z and best_cos2_z all come from that same row, then are averaged over keys.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ijiss_btp.theory import collusion_cos_optimum

TABLES = ("utility", "anchor", "zeropair", "postcompromise")


def r_at_level(r, values, level: float = 0.5):
    """Smallest r reaching `level` by log-linear interpolation; None if never reached."""
    order = np.argsort(np.asarray(r, dtype=float))
    rr = np.asarray(r, dtype=float)[order]
    vv = np.asarray(values, dtype=float)[order]
    hit = np.flatnonzero(vv >= level)
    if len(hit) == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return float(rr[0])
    if not np.isfinite(vv[i - 1]):
        return float(rr[i])
    lo, hi = np.log(rr[i - 1]), np.log(rr[i])
    frac = (level - vv[i - 1]) / max(vv[i] - vv[i - 1], 1e-12)
    return float(np.exp(lo + frac * (hi - lo)))


def seed_ci(values) -> tuple[float, float, float]:
    """Mean and t-based 95% CI across seed-level means; non-finite entries are ignored."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


Z95 = 1.959964


def _row_se(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-row standard error implied by the runner's bootstrap CI (NaN where the CI is not finite)."""
    return (df[f"{col}_hi"] - df[f"{col}_lo"]) / (2 * Z95)


def _add_cons(out: pd.DataFrame, per_seed: pd.DataFrame, keys: list, cols: list) -> pd.DataFrame:
    """Add <col>_cons_lo/_cons_hi columns to the curve table from the per-seed mean SEs (`<col>_se`)."""
    for c in cols:
        se_bar = per_seed.groupby(keys)[f"{c}_se"].apply(lambda v: float(np.nanmean(v)) if v.notna().any() else np.nan)
        se_bar = se_bar.rename("se_bar").reset_index()
        merged = out[keys + [c]].merge(se_bar, on=keys, how="left")
        out[f"{c}_cons_lo"] = (merged[c] - Z95 * merged["se_bar"]).to_numpy()
        out[f"{c}_cons_hi"] = (merged[c] + Z95 * merged["se_bar"]).to_numpy()
    return out


def load_runs(run_dirs) -> dict:
    tables: dict = {}
    for d in map(Path, run_dirs):
        name = json.loads((d / "metadata.json").read_text())["config"]["name"]
        for t in TABLES:
            p = d / f"{t}.csv"
            if p.exists() and p.stat().st_size > 1:
                df = pd.read_csv(p)
                if len(df):
                    df.insert(0, "encoder", name)
                    tables.setdefault(t, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}


def _ci_rows(df: pd.DataFrame, keys: list, cols: list) -> pd.DataFrame:
    rows = []
    for g, part in df.groupby(keys):
        row = dict(zip(keys, g if isinstance(g, tuple) else (g,)))
        for c in cols:
            m, lo, hi = seed_ci(part[c])
            row.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi})
        row["n_seeds"] = int(np.isfinite(part[cols[0]].to_numpy(dtype=float)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(tables: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a = tables["anchor"]
    keys = ["encoder", "pre", "scheme", "m", "k", "variant", "decoder", "r"]
    a = a.assign(e1_rel_se=_row_se(a, "e1_rel"), e2_rel_se=_row_se(a, "e2_rel"))
    a_cols = ["rec_cos", "rec_cos_z", "rec_abs_cos_z", "rec_cos2_z", "e1_rel", "e2_rel"]
    per_seed = a.groupby(keys + ["seed"], as_index=False)[a_cols + ["e1_rel_se", "e2_rel_se"]].mean()
    curves = _ci_rows(per_seed, keys, a_cols)
    curves = _add_cons(curves, per_seed, keys, ["e1_rel", "e2_rel"])
    curves["cstar"] = collusion_cos_optimum(curves["r"].to_numpy(), curves["m"].to_numpy())
    curves["sqrt_r_m"] = np.sqrt(curves["r"].to_numpy(dtype=float) / curves["m"].to_numpy(dtype=float))
    curves.to_csv(out_dir / "summary_anchor.csv", index=False)
    r50 = []
    for g, df in curves.groupby(keys[:-1]):
        r50.append({**dict(zip(keys[:-1], g)),
                    "r50_e1": r_at_level(df["r"], df["e1_rel"]),
                    "r50_e2": r_at_level(df["r"], df["e2_rel"]),
                    "r_cos08": r_at_level(df["r"], df["rec_cos"], 0.8)})
    pd.DataFrame(r50).to_csv(out_dir / "r50.csv", index=False)

    u = tables["utility"]
    ukeys = ["encoder", "pre", "scheme", "m", "k", "fmr_target"]
    _ci_rows(u.groupby(ukeys + ["seed"], as_index=False)[["eer", "tmr", "rank1"]].mean(), ukeys,
             ["eer", "tmr", "rank1"]).to_csv(out_dir / "summary_utility.csv", index=False)

    z = tables.get("zeropair")
    if z is not None and len(z):
        zk = ["encoder", "pre", "scheme", "aux", "method"]
        z = z.assign(e1_rel_se=_row_se(z, "e1_rel"))
        # argmax-E1 row over p for each (seed, key); NaN E1 rows are never selected unless all are NaN
        z = z.assign(_e1=z["e1_rel"].fillna(-np.inf))
        at_best = z.loc[z.groupby(zk + ["seed", "key"])["_e1"].idxmax()]
        zcols = {"e1_rel": "best_e1_rel", "e1_rel_se": "best_e1_rel_se", "rec_cos_z": "best_rec_cos_z",
                 "rec_abs_cos_z": "best_abs_cos_z", "rec_cos2_z": "best_cos2_z"}
        best = (at_best.groupby(zk + ["seed"], as_index=False)[list(zcols)].mean().rename(columns=zcols))
        out_z = _ci_rows(best, zk, list(zcols.values())[:1] + list(zcols.values())[2:])
        out_z = _add_cons(out_z, best, zk, ["best_e1_rel"])
        out_z.to_csv(out_dir / "summary_zeropair.csv", index=False)

    p = tables.get("postcompromise")
    if p is not None and len(p):
        pk = ["encoder", "pre", "scheme", "m", "k", "method"]
        p = p.assign(e1_rel_se=_row_se(p, "e1_rel"), e2_rel_se=_row_se(p, "e2_rel"))
        p_seed = p.groupby(pk + ["seed"], as_index=False)[["in_system_accept", "e1_rel", "e2_rel", "e1_rel_se", "e2_rel_se"]].mean()
        pc = _ci_rows(p_seed, pk, ["in_system_accept", "e1_rel", "e2_rel"])
        _add_cons(pc, p_seed, pk, ["e1_rel", "e2_rel"]).to_csv(out_dir / "summary_postcompromise.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    summarize(load_runs(args.runs), Path(args.out))


if __name__ == "__main__":
    main()
