"""Print the Markdown result tables of the isotropy study; every number in RESULTS.md comes from here.

Reads results/isotropy/summary/ (FMR 0.1% runs) and results/isotropy/summary_fmr1/ (FMR 1% runs), both from
ijiss_btp.experiments.aggregate, the per-run directories (metadata.json, spectra.json, stdout.log, raw CSVs) and
results/scale/*/scale.csv. The output is deterministic: run code commits and the date in the title (the latest run
start) come from metadata.json, never from git or the clock.

Usage:
    python scripts/figures/results_tables.py > tables.md
    python scripts/figures/results_tables.py --md results/isotropy/RESULTS.md   # full RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ijiss_btp.experiments.aggregate import r_at_level, seed_ci  # noqa: E402
from ijiss_btp.theory import uniform_abs_cos_mean  # noqa: E402
from isotropy_figures import best_decoder_status, enc_rank, seed_curves, seed_r50_range, wider_interval  # noqa: E402

BASE_ORDER = ["d1_base", "d1_large", "d2_fingervit", "d2_deepprint", "d2_ctrlvit"]
ENCODER_ORDER = BASE_ORDER + [f"{e}_fmr1" for e in BASE_ORDER]
D1_SOURCE = "SIN 2026 image set, identity labels re-derived from each source's native naming by content matching"
D2_SOURCE = ("development folds of an internal compilation with labels parsed from each source's native naming, "
             "finger-disjoint from D1")
CTRL_SOURCE = "ctrl-vit: a control ViT trained by us on the compilation images of the D1 fingers"
SECONDARY_CORPUS = "secondary corpus (FVC set B, SD301a, SD302b/d)"
SECONDARY_SOURCE = f"{SECONDARY_CORPUS}, fingers shared with D1 removed"
DATASET_LABEL = {
    "d1_base": f"D1: {D1_SOURCE} (real)",
    "d1_large": f"D1: {D1_SOURCE} (real)",
    "d2_fingervit": f"D2: {D2_SOURCE} (real)",
    "d2_deepprint": f"D2: {D2_SOURCE} (real)",
    "d2_ctrlvit": f"D2: {D2_SOURCE} (real), {CTRL_SOURCE}",
}
SCALE_LABELS = {
    "scale_d1_base_secondary": f"D1 targets + distractors from the {SECONDARY_SOURCE} (real)",
    "scale_d1_base_synthetic": ("D1 targets + SYNTHETIC distractors "
                                "(gazi_synthetic_fp images embedded with gazi_base_extended)"),
}
SPECTRUM_FLAG = ("distractor spectrum differs from the target population by more than 30%; interpret HNSW numbers "
                 "on these distractors as a stress test, not as a model of real galleries")


def scale_label(r: dict) -> str:
    name = r["meta"]["config"]["name"]
    if name not in SCALE_LABELS:
        raise ValueError(f"scale run {r['dir'].name}: unknown config name {name!r}; add an explicit data-source label")
    return SCALE_LABELS[name]
PRE_PREFIXES = ["none", "pca128_a1", "pca128_a0.5", "pca256_a1", "fi64", "fi128", "fi256", "rbig128", "firbig128"]
KEYFREE = ("eigen", "ica", "wp")
COMMON = ("eigen", "wp")   # key-free methods that run for every pre-conditioner (ICA needs m <= 128)
SCHEME = "lrp_full"
ALL_SCHEMES = ("lrp_full", "lrp_half", "biohash_half", "iom_256")
R_COLLUSION = (20, 40, 80, 160)
R_ANISO = (5, 10, 20, 40, 80, 160)


def base_name(enc: str) -> str:
    return enc[: -len("_fmr1")] if enc.endswith("_fmr1") else enc


def find_pre(names, prefix: str):
    """Actual pre-conditioner name for a prefix (p0 differs by encoder); None if absent."""
    names = list(names)
    if prefix in names:
        return prefix
    for n in names:
        if n.startswith(prefix + "_"):
            return n
    return None


def short_sha(s: str) -> str:
    return s[:7] + ("-dirty" if s.endswith("-dirty") else "")


def f3(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


# ---- joint alignment-plus-anchor stage (results/joint/) ------------------------------------------------------------
JOINT_METHODS = ("proj", "align_anchor", "joint")   # the evaluated T1 attacks of the stage; ties go to the first
JOINT_SMALL_R = (1, 2, 5)


def joint_seed_means(df: pd.DataFrame, pre: str, ep: str) -> pd.DataFrame:
    """Seed means (mean over keys within a seed, then over seeds) per (method, AUX, p, r) of the LRP-full rows of one
    pre-conditioner: relative success, absolute acceptance (`<ep>_proxy`) and pre-key cosine. proj has no AUX."""
    sub = df[(df.pre == pre) & (df.scheme == SCHEME)].assign(aux=lambda x: x["aux"].fillna(""))
    cols = [f"{ep}_rel", f"{ep}_proxy", "rec_cos_z"]
    return (sub.groupby(["method", "aux", "p", "r", "seed"])[cols].mean()
            .groupby(["method", "aux", "p", "r"])[cols].mean().reset_index())


def joint_envelope(per: pd.DataFrame, ep: str, methods: tuple = JOINT_METHODS) -> pd.DataFrame:
    """Per r, the entry with the largest seed-mean relative success among `methods` and their (AUX, p) settings: an
    upper envelope selected with ground truth (the true-sign reference is never a candidate). Ties go to the method
    listed first, so J1 and J2 are credited only where they are strictly better than the anchor-only decoder."""
    order = {m: i for i, m in enumerate(methods)}
    sub = per[per.method.isin(methods)].assign(_o=lambda x: x.method.map(order)).sort_values(["_o", "aux", "p", "r"])
    e = sub.loc[sub.groupby("r", sort=True)[f"{ep}_rel"].idxmax()].sort_values("r")
    return pd.DataFrame({"r": e.r.to_numpy(), "rel": e[f"{ep}_rel"].to_numpy(), "abs": e[f"{ep}_proxy"].to_numpy(),
                         "cos": e.rec_cos_z.to_numpy(), "method": e.method.to_numpy(), "aux": e.aux.to_numpy(),
                         "p": e.p.to_numpy()})


def joint_r50(curve: pd.DataFrame) -> tuple:
    """(r50, censored, r_max) of an envelope curve by log-linear interpolation (aggregate.r_at_level); a curve that
    never reaches 0.5 is censored at r_max."""
    rmax = float(curve.r.max())
    v = r_at_level(curve.r, curve.rel)
    return (rmax, True, rmax) if v is None else (float(v), False, rmax)


def joint_small(df: pd.DataFrame, per: pd.DataFrame, pre: str, ep: str, methods: tuple = JOINT_METHODS) -> dict:
    """Best evaluated attack at the small budgets r = 1, 2, 5: the envelope entry with the largest seed-mean relative
    success (rel, abs, cos, r, method); when no row of these budgets has a success, `zero` and the largest relative
    one-sided 95% Clopper-Pearson bound over the rows (`cp_rel`)."""
    env = joint_envelope(per, ep, methods)
    env = env[env.r.isin(JOINT_SMALL_R)]
    best = env.loc[env.rel.idxmax()]
    raw = df[(df.pre == pre) & (df.scheme == SCHEME) & df.method.isin(methods) & df.r.isin(JOINT_SMALL_R)]
    zero = bool((raw[f"{ep}_successes"] == 0).all())
    return {"rel": float(best.rel), "abs": float(best["abs"]), "cos": float(best.cos), "r": int(best.r),
            "method": best.method, "zero": zero,
            "cp_rel": float((raw[f"{ep}_cp_upper"] / raw[f"{ep}_genuine"]).max()) if zero else None}


def joint_seed_r50_range(df: pd.DataFrame, pre: str, ep: str, methods: tuple = JOINT_METHODS) -> dict:
    """Per-seed r50 range of the best evaluated attack: for each seed, relative success averaged over keys per
    (method, AUX, p, r), r50 per setting by log-linear interpolation, the smallest per seed (a censored seed counts
    as r_max, the rule of `seed_r50_range`). Returns dict(lo, hi, rmax, n, cens_any, cens_all)."""
    sub = df[(df.pre == pre) & (df.scheme == SCHEME) & df.method.isin(methods)].assign(aux=lambda x: x["aux"].fillna(""))
    rmax = float(sub.r.max())
    curves = sub.groupby(["seed", "method", "aux", "p", "r"])[f"{ep}_rel"].mean().reset_index()
    vals, cens = [], []
    for _, s in curves.groupby("seed", sort=True):
        reached = [v for v in (r_at_level(c.r, c[f"{ep}_rel"]) for _, c in s.groupby(["method", "aux", "p"]))
                   if v is not None]
        cens.append(not reached)
        vals.append(min(reached) if reached else rmax)
    return {"lo": min(vals), "hi": max(vals), "rmax": rmax, "n": len(vals), "cens_any": any(cens),
            "cens_all": all(cens)}


def joint_seed_r50_range_fixed(df: pd.DataFrame, pre: str, ep: str, env: pd.DataFrame) -> dict:
    """Per-seed r50 range with the envelope's per-r selections (method, AUX, p) applied to every seed: for each seed, the
    key-mean relative success of the selected setting at each r, r50 by log-linear interpolation (a censored seed counts
    as r_max). Less optimistic than re-selecting per seed (`joint_seed_r50_range`)."""
    sub = df[(df.pre == pre) & (df.scheme == SCHEME)].assign(aux=lambda x: x["aux"].fillna(""))
    curves = sub.groupby(["seed", "method", "aux", "p", "r"])[f"{ep}_rel"].mean()
    rmax = float(env.r.max())
    vals, cens = [], []
    for seed in sorted(sub.seed.unique()):
        v = [curves.loc[(seed, e.method, e.aux, e.p, e.r)] for e in env.itertuples()]
        x = r_at_level(env.r.to_numpy(), v)
        cens.append(x is None)
        vals.append(rmax if x is None else x)
    return {"lo": min(vals), "hi": max(vals), "rmax": rmax, "n": len(vals), "cens_any": any(cens), "cens_all": all(cens)}


JOINT_PRES = (("none", "none"), ("pca128", "pca128_a1"), ("pca256", "pca256_a1"), ("fi64", "fi64"), ("fi128", "fi128"),
              ("fi256", "fi256"))   # (label, prefix): the pre-conditioners of the joint stage


def load_joint(root) -> dict:
    """Joint-stage runs by configuration name (`d1_base`, ..., `d1_base_fmr1`, ...): the frame of `joint.csv`."""
    out = {}
    for d in sorted(Path(root).glob("joint_d*-*")):
        out[json.loads((d / "metadata.json").read_text())["config"]["name"]] = pd.read_csv(d / "joint.csv")
    return out


def joint_pre(df: pd.DataFrame, prefix: str) -> str:
    names = sorted(df.pre.unique())
    return next(n for n in names if n == prefix or n.startswith(prefix + "_"))


def joint_cell(df: pd.DataFrame, prefix: str, ep: str) -> dict:
    """One (run, pre-conditioner, endpoint) cell: r50 of proj, J1, J2 and of the best evaluated attack (each a
    (value, censored, r_max) tuple), the per-seed range of the best attack, the small-budget acceptance and m."""
    pre = joint_pre(df, prefix)
    per = joint_seed_means(df, pre, ep)
    return {"proj": joint_r50(joint_envelope(per, ep, ("proj",))),
            "j1": joint_r50(joint_envelope(per, ep, ("align_anchor",))),
            "j2": joint_r50(joint_envelope(per, ep, ("joint",))),
            "best": joint_r50(joint_envelope(per, ep)), "seeds": joint_seed_r50_range(df, pre, ep),
            "small": joint_small(df, per, pre, ep), "m": int(df[df.pre == pre].m.iloc[0]),
            "env": joint_envelope(per, ep)}


def joint_small_cell(df: pd.DataFrame, env: pd.DataFrame, pre: str, ep: str, r: int) -> str:
    """One small-budget cell of tables 13/13b: the best attack's relative / absolute acceptance at r with its cosine and
    method, or, when no row at r has a success, the zero count with the largest relative one-sided 95% Clopper-Pearson
    bound over the rows of that r."""
    rows = df[(df.pre == pre) & (df.scheme == SCHEME) & df.method.isin(JOINT_METHODS) & (df.r == r)]
    if (rows[f"{ep}_successes"] == 0).all():
        return f"0 successes; CP95 upper <= {(rows[f'{ep}_cp_upper'] / rows[f'{ep}_genuine']).max():.3f}"
    e = env.set_index("r").loc[r]
    return f"{f3_rate(e.rel)} / {f3_rate(e['abs'])} (cos {e.cos:.3f}, {e.method})"


def _joint_r50_txt(x: tuple, m: int) -> str:
    v, cens, rmax = x
    if cens:
        return f">= {rmax:g}"
    return f"{v:.1f}" + (" (> m: upper estimate)" if v > m else "")


def t_joint(joint: dict, fmr1: bool) -> str:
    """Tables 13 and 13b: r50 at E2 and E1 of the anchor-only projection decoder, of J1 (align_anchor), J2 (joint) and
    of the best evaluated attack, and the best attack's acceptance at r = 1, 2, 5, per encoder and pre-conditioner."""
    rows = []
    for name in sorted(joint, key=lambda n: enc_rank(base_name(n))):
        if name.endswith("_fmr1") != fmr1:
            continue
        df = joint[name]
        for label, prefix in JOINT_PRES:
            c2, c1 = joint_cell(df, prefix, "e2"), joint_cell(df, prefix, "e1")
            g, m = c2["seeds"], c2["m"]
            rng = (f"all {g['n']} seeds >= {g['rmax']:g}" if g["cens_all"] else
                   f"{g['lo']:.1f}-" + (f">= {g['rmax']:g}" if g["cens_any"] else f"{g['hi']:.1f}"))
            delta = c2["best"][0] - c2["proj"][0]
            change = "n/a" if c2["proj"][1] or c2["best"][1] else ("0.0" if abs(delta) < 0.05 else f"{delta:.1f}")
            small = [joint_small_cell(df, c2["env"], joint_pre(df, prefix), "e2", r) for r in JOINT_SMALL_R]
            rows.append([name, label, str(m), _joint_r50_txt(c2["proj"], m), f"{_joint_r50_txt(c2['best'], m)} [{rng}]",
                         change, _joint_r50_txt(c2["j1"], m), _joint_r50_txt(c2["j2"], m),
                         _joint_r50_txt(c1["proj"], m), _joint_r50_txt(c1["best"], m), *small])
    return md_table(["encoder", "pre", "m", "E2 r50 proj", "E2 r50 best attack [per-seed range]", "change (best - proj)",
                     "E2 r50 J1 (align_anchor)", "E2 r50 J2 (joint)", "E1 r50 proj", "E1 r50 best of proj/J1/J2",
                     "best E2 at r = 1: relative / absolute (cosine, method)", "at r = 2", "at r = 5"], rows)


JOINT_NOTE = ("Joint alignment-plus-anchor stage (results/joint/, real embeddings; same seeds, splits, keys, thresholds, "
              "colluders and victims as the anchor stage; parity of proj with anchor.csv is exact, results/joint/PARITY.md). "
              "proj is the anchor-only projection decoder; J1 (align_anchor) aligns the stolen templates with AUX and "
              "takes the axis signs from the anchors; J2 (joint) adds J1 outside the anchors' span to the projection "
              "decoder's exact component. The best evaluated attack is, per r, the largest seed-mean relative success "
              "(mean over keys within a seed, then over seeds) over proj, J1 and J2 and their AUX and p settings: an "
              "upper envelope selected with ground truth; the true-sign reference (eigen_oracle) is not a candidate, and "
              "ties go to proj. r50 is interpolated log-linearly on that curve; a value above m is an upper estimate "
              "(with m exact anchors the projection decoder recovers z). The per-seed range takes, per seed, the "
              "smallest r50 over the settings, so the seed-mean envelope's r50 can lie above that range. E1 compares "
              "proj, J1 and J2 only (the anchor stage's ridge and mlp are not candidates, so section 12 can show a "
              "smaller E1 r50). A zero count at r = 1, 2 or 5 gives the largest relative one-sided 95% Clopper-Pearson "
              "bound over the rows of that r (a descriptive binomial reference limit). Provenance of the ten joint runs: "
              "results/joint/*/metadata.json (seed, configuration, data source, environment, git commit) and "
              "results/joint/PARITY.md.")


def f3_rate(x) -> str:
    """A non-negative rate with three decimals; an observed, non-zero value that rounds to 0.000 prints <0.001 (the
    paper's convention), so a bare 0.000 never stands for an observed acceptance."""
    s = f3(x)
    return "<0.001" if s == "0.000" and x > 0 else s


def pp(x: float) -> str:
    """A difference of two rates in percentage points with two decimals; -0.00 reads 0.00."""
    s = f"{100 * x:.2f}"
    return "0.00" if s == "-0.00" else s


SEED_NOTE = ("Intervals on every metric other than E1/E2 relative success are Student-t intervals over the 5 seed "
             "means (4 df), mean +/- t(0.975, 4) * sd / sqrt(5): between-split variability over 5 overlapping "
             "splits; not an identity-level CI. Lower bounds of nonnegative quantities are truncated at 0; signed "
             "cosines are not truncated.")
CONS_NOTE = ("E1/E2 relative success carries the wider of two 95% intervals, elementwise: the identity-level interval "
             "(the runner's per-row paired-bootstrap standard error, resampling targets, see "
             "`src/ijiss_btp/eval/stats.py:63-72`, averaged over keys and seeds; mean +/- 1.96 * SE_bar, which treats "
             "the overlapping splits as fully correlated) and the seed-level Student-t interval over the 5 seed means "
             "(4 df). The reported interval is their union: lower = min of the two lower bounds, upper = max of the two "
             "upper bounds; lower bounds are truncated at 0.")
CI_NOTE = SEED_NOTE + " " + CONS_NOTE


def ci(m, lo, hi, scale: float = 1.0, nonneg: bool = False) -> str:
    if m is None or not np.isfinite(m):
        return "n/a"
    if nonneg and lo is not None and np.isfinite(lo):
        lo = max(lo, 0.0)
    if lo is None or not np.isfinite(lo):
        return f"{m * scale:.3f} [n/a]"
    return f"{m * scale:.3f} [{lo * scale:.3f}, {hi * scale:.3f}]"


def cons(x, col: str) -> str:
    """Mean with the reported interval of an E1/E2-type column: wider of identity-level and seed-level, lower bound at 0."""
    lo, hi = wider_interval(x, col)
    return ci(x[col], lo, hi, nonneg=True)


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_run(d: Path) -> dict:
    if not (d / "stdout.log").is_file():
        raise FileNotFoundError(f"run {d.name}: stdout.log is missing (an empty log is valid, a missing one is not)")
    meta = json.loads((d / "metadata.json").read_text())
    spec = json.loads((d / "spectra.json").read_text()) if (d / "spectra.json").exists() else None
    return {"dir": d, "meta": meta, "spec": spec}


class Study:
    """Tables of one operating point: runs plus the aggregate summary computed from exactly those runs."""

    def __init__(self, summary: Path, runs: list[dict]):
        self.summary = summary
        self.runs = {r["meta"]["config"]["name"]: r for r in runs}
        self.encs = [e for e in ENCODER_ORDER if e in self.runs]
        if not self.encs:
            raise ValueError(f"no runs for summary {summary}")
        self.anchor = pd.read_csv(summary / "summary_anchor.csv")
        self.r50 = pd.read_csv(summary / "r50.csv")
        self.util = pd.read_csv(summary / "summary_utility.csv")
        self.zsum = pd.read_csv(summary / "summary_zeropair.csv")
        self.post = pd.read_csv(summary / "summary_postcompromise.csv")
        self._zraw: dict = {}
        self._praw: dict = {}
        self._araw: dict = {}
        self._uraw: dict = {}
        self._bd: dict = {}
        self._seedcurves: dict = {}

    # ---- helpers -------------------------------------------------------------------------
    def attack_fmr(self, enc: str) -> float:
        return float(self.runs[enc]["meta"]["config"]["fmr"][0])

    def pres(self, enc: str) -> list[str]:
        return list(self.runs[enc]["spec"]["preconditioned"])

    def pre(self, enc: str, prefix: str):
        return find_pre(self.pres(enc), prefix)

    def raw(self, store: dict, enc: str, name: str) -> pd.DataFrame:
        if enc not in store:
            store[enc] = pd.read_csv(self.runs[enc]["dir"] / name)
        return store[enc]

    def zraw(self, enc):
        return self.raw(self._zraw, enc, "zeropair.csv")

    def praw(self, enc):
        return self.raw(self._praw, enc, "postcompromise.csv")

    def araw(self, enc):
        return self.raw(self._araw, enc, "anchor.csv")

    def uraw(self, enc):
        return self.raw(self._uraw, enc, "utility.csv")

    # ---- paired utility, measured zero-effort level, oracle, absolute acceptance ---------------------------------
    def paired_utility(self, enc: str, pre_a: str, pre_b: str, metric: str, scheme: str = SCHEME) -> dict:
        """Paired per-seed difference metric(pre_a) - metric(pre_b) with the Student-t 95% interval over the seeds.

        The runner writes one utility row per seed (key 0) and pre-conditioner; both pre-conditioners of a seed are
        scored on the same split, genuine and impostor pairs and threshold rule (DEV impostor pairs), so the difference
        is paired by seed. The TMR is read at the attack FMR of the run. Like the marginal intervals, the interval
        describes between-split variability over overlapping partitions; it is not identity-level and does not model
        subject clustering. Returns dict(mean, lo, hi, n, diffs).
        """
        u = self.uraw(enc)
        sel = u[(u.scheme == scheme) & np.isclose(u.fmr_target, self.attack_fmr(enc))]
        a = sel[sel.pre == pre_a].groupby("seed")[metric].mean()
        b = sel[sel.pre == pre_b].groupby("seed")[metric].mean()
        if not len(a) or set(a.index) != set(b.index):
            raise ValueError(f"{enc}: the seeds of {pre_a} ({sorted(a.index)}) and {pre_b} ({sorted(b.index)}) "
                             "do not pair")
        d = (a - b.loc[a.index]).to_numpy(dtype=float)
        m, lo, hi = seed_ci(d)
        return {"mean": m, "lo": lo, "hi": hi, "n": len(d), "diffs": d}

    def measured_fmr(self, enc: str, fmr: float | None = None) -> float:
        """Impostor acceptance measured on the target split's impostor pairs at the DEV threshold, mean over seeds.

        Read from the utility rows of LRP-full without pre-conditioning (`fmr_actual`); LRP-full keeps cosines, so this
        is the raw-space rate at the threshold that E1 uses. `fmr` selects the target FMR (default: the attack FMR).
        """
        u = self.uraw(enc)
        f = self.attack_fmr(enc) if fmr is None else fmr
        sel = u[(u.scheme == SCHEME) & (u.pre == self.pre(enc, "none")) & np.isclose(u.fmr_target, f)]
        if not len(sel):
            raise ValueError(f"{enc}: no LRP-full utility row without pre-conditioning at FMR {f:g}")
        return float(sel.groupby("seed")["fmr_actual"].mean().mean())

    def measured_impostor_level(self, enc: str) -> float:
        """Zero-effort level of E1 relative success from the measured impostor acceptance (not the nominal FMR)."""
        return self.measured_fmr(enc) / self.genuine(enc)[0]

    def oracle_max(self, enc: str, pre: str = "none") -> dict:
        """`eigen_oracle` (eigen alignment whose signs come from the key; a diagnostic, never a key-free attack), LRP-full:
        the largest seed-mean E1 relative success and the largest seed-mean pre-key cosine over AUX type and p (mean over
        keys within a seed, then over seeds). Returns dict(e1, e1_at, e1_abs, e1_abs_max, cos, cos_at, zero, cp_rel),
        `*_at` = (AUX, p); `e1_abs` is the absolute proxy acceptance (`e1_proxy`, averaged alike) of the entry `e1_at`,
        `e1_abs_max` the largest such absolute acceptance over AUX type and p (a bound of its own); `zero` is true
        when no row has a success, and `cp_rel` is then the largest relative one-sided 95% Clopper-Pearson bound over the
        rows (absolute bound / genuine acceptance of the run), as in `best_zero_pair`."""
        raw = self.zraw(enc)
        sub = raw[(raw.pre == self.pre(enc, pre)) & (raw.scheme == SCHEME) & (raw.method == "eigen_oracle")]
        if not len(sub):
            raise ValueError(f"{enc}: no eigen_oracle rows for pre {pre}")
        cols = ["e1_rel", "rec_cos_z", "e1_proxy"]
        per = sub.groupby(["aux", "p", "seed"])[cols].mean().groupby(["aux", "p"])[cols].mean()
        i1, ic = per["e1_rel"].idxmax(), per["rec_cos_z"].idxmax()
        zero = bool((sub["e1_successes"] == 0).all())
        return {"e1": float(per["e1_rel"].max()), "e1_at": (i1[0], int(i1[1])), "e1_abs": float(per.loc[i1, "e1_proxy"]),
                "e1_abs_max": float(per["e1_proxy"].max()), "cos": float(per["rec_cos_z"].max()), "cos_at": (ic[0], int(ic[1])), "zero": zero,
                "cp_rel": float((sub["e1_cp_upper"] / sub["e1_genuine"]).max()) if zero else None}

    def oracle_endpoint(self, enc: str, pre: str, ep: str) -> dict:
        """`eigen_oracle` of LRP-full under one pre-conditioner (prefix, resolved by `pre`) at one endpoint (`e1` or
        `e2`): the largest seed-mean relative success over AUX type and p (`rel`, at `at` = (AUX, p)), the largest
        seed-mean absolute acceptance (`abs_max`, a bound of its own), the largest seed-mean pre-key cosine (`cos`), and,
        when no row has a success at that endpoint (`zero`), the largest relative one-sided 95% Clopper-Pearson bound
        over the rows (`cp_rel`; absolute bound / genuine acceptance of the run). Means are over keys within a seed,
        then over seeds, as in `oracle_max`."""
        raw = self.zraw(enc)
        sub = raw[(raw.pre == self.pre(enc, pre)) & (raw.scheme == SCHEME) & (raw.method == "eigen_oracle")]
        if not len(sub):
            raise ValueError(f"{enc}: no eigen_oracle rows for pre {pre}")
        cols = [f"{ep}_rel", f"{ep}_proxy", "rec_cos_z"]
        per = sub.groupby(["aux", "p", "seed"])[cols].mean().groupby(["aux", "p"])[cols].mean()
        i1 = per[f"{ep}_rel"].idxmax()
        zero = bool((sub[f"{ep}_successes"] == 0).all())
        return {"rel": float(per[f"{ep}_rel"].max()), "at": (i1[0], int(i1[1])), "abs_max": float(per[f"{ep}_proxy"].max()),
                "cos": float(per["rec_cos_z"].max()), "zero": zero,
                "cp_rel": float((sub[f"{ep}_cp_upper"] / sub[f"{ep}_genuine"]).max()) if zero else None}

    def ceiling_abs(self, enc: str, pre: str, scheme: str, endpoint: str) -> float:
        """Absolute proxy acceptance behind the key-known (T2) level: mean over seeds of `e<1|2>_proxy` of the T2
        method that `ceiling_row` selects."""
        x = self.ceiling_row(enc, pre, scheme, endpoint)
        raw = self.praw(enc)
        sub = raw[(raw.pre == pre) & (raw.scheme == scheme) & (raw.method == x.method)]
        return float(sub.groupby("seed")[f"{endpoint}_proxy"].mean().mean())

    def victim_genuine(self, enc: str, pre: str, endpoint: str, scheme: str = SCHEME, variant: str = "exact") -> float:
        """Genuine acceptance of the T1 victims at an endpoint (mean over seeds of `e<1|2>_genuine` of the anchor rows);
        at r50 the best decoder's proxies are accepted at about half of it."""
        a = self.araw(enc)
        sub = a[(a.pre == pre) & (a.scheme == scheme) & (a.variant == variant)]
        if not len(sub):
            raise ValueError(f"{enc}: no anchor rows for pre {pre}, scheme {scheme}, variant {variant}")
        return float(sub.groupby("seed")[f"{endpoint}_genuine"].mean().mean())

    def victims(self, enc: str) -> int:
        """Number of T1 victims per seed and key (`n_targets` of the anchor rows; the same for every r)."""
        n = set(self.araw(enc)["n_targets"].astype(int))
        if len(n) != 1:
            raise ValueError(f"{enc}: the anchor rows have several victim counts {sorted(n)}")
        return n.pop()

    def util_row(self, enc, pre, scheme, fmr=0.001):
        d = self.util[(self.util.encoder == enc) & (self.util.pre == pre) & (self.util.scheme == scheme)
                      & np.isclose(self.util.fmr_target, fmr)]
        return d.iloc[0] if len(d) else None

    def anchor_rows(self, enc, pre, variant="exact", decoder="proj", scheme=SCHEME):
        d = self.anchor[(self.anchor.encoder == enc) & (self.anchor.pre == pre) & (self.anchor.scheme == scheme)
                        & (self.anchor.variant == variant) & (self.anchor.decoder == decoder)]
        return d.sort_values("r")

    def bd(self, endpoint: str, scheme: str, variant: str) -> pd.DataFrame:
        key = (endpoint, scheme, variant)
        if key not in self._bd:
            self._bd[key] = best_decoder_status(self.summary, endpoint, scheme, variant).set_index(["encoder", "pre"])
        return self._bd[key]

    def r50_cell(self, enc, pre, endpoint, scheme=SCHEME, variant="exact") -> str:
        t = self.bd(endpoint, scheme, variant)
        if (enc, pre) not in t.index:
            return "n/a"
        x = t.loc[(enc, pre)]
        lvl = self.t2_level(enc, pre, scheme, endpoint)
        if x.status == "backmap_limited":
            return f">= {x.rmax:g} ({x.decoder}); back-map-limited: T2 level {lvl} < 0.5"
        if x.status == "censored":
            return f">= {x.rmax:g} ({x.decoder})" + (f"; T2 level {lvl}" if lvl else "")
        if np.isfinite(x.ceiling) and x.ceiling < 0.5:   # observed r50 despite a T2 level below 0.5: show both
            return (f"{x.r50:.1f} ({x.decoder}; note: T2 level {lvl} < 0.5, evidence that the key-known level is "
                    f"not an upper bound)")
        return f"{x.r50:.1f} ({x.decoder})"

    def seed_range(self, enc, pre, endpoint, scheme=SCHEME, variant="exact"):
        """Per-seed r50 range for one (enc, pre, scheme, variant, endpoint) cell; None if the cell has no rows.

        For each split seed: average the endpoint relative success over keys at each (decoder, r) of that seed's
        rows in anchor.csv, interpolate r50 per decoder with `aggregate.r_at_level` (log-linear), and take the
        best decoder, i.e. the smallest r50 (same rule as `best_decoder_status`). A seed where no decoder reaches
        0.5 is censored and counts as r_max. Returns dict(lo, hi, rmax, n, n_cens, cens_any, cens_all).
        """
        if enc not in self._seedcurves:
            self._seedcurves[enc] = seed_curves(self.araw(enc))
        return seed_r50_range(self._seedcurves[enc], pre, endpoint, scheme, variant)

    def seed_range_cell(self, enc, pre, endpoint, scheme=SCHEME, variant="exact") -> str:
        """`[min-max]` per-seed r50 range (1 decimal); `>=` marks a censored upper end; all-censored is stated."""
        g = self.seed_range(enc, pre, endpoint, scheme, variant)
        if g is None:
            return "n/a"
        if g["cens_all"]:
            return f"all {g['n']} seeds >= {g['rmax']:g}"
        if g["cens_any"]:
            return f"[{g['lo']:.1f}-(>= {g['hi']:g})] ({g['n_cens']} of {g['n']} seeds censored)"
        return f"[{g['lo']:.1f}-{g['hi']:.1f}]"

    def zero_t2(self, enc, pre, scheme, method, endpoint):
        """CP95 relative upper bound if every raw T2 row of (enc, pre, scheme, method) has zero successes, else None."""
        raw = self.praw(enc)
        sub = raw[(raw.pre == pre) & (raw.scheme == scheme) & (raw.method == method)]
        if len(sub) and (sub[f"{endpoint}_successes"] == 0).all():
            return float((sub[f"{endpoint}_cp_upper"] / sub[f"{endpoint}_genuine"]).max())
        return None

    def t2_level(self, enc, pre, scheme, endpoint) -> str:
        """Key-known (T2) level with its reported (wider) interval; the CP95 bound if every selected row has 0 successes."""
        x = self.ceiling_row(enc, pre, scheme, endpoint)
        if x is None:
            return ""
        cp = self.zero_t2(enc, pre, scheme, x.method, endpoint)
        if cp is not None:
            return f"0 successes; CP95 upper <= {cp:.3f} (rel.)"
        return cons(x, f"{endpoint}_rel")

    def ceiling_row(self, enc, pre, scheme, endpoint):
        """Key-known (T2) level row: the one with the highest mean endpoint relative success (max over T2 methods)."""
        d = self.post[(self.post.encoder == enc) & (self.post.pre == pre) & (self.post.scheme == scheme)]
        if not len(d):
            return None
        return d.sort_values([f"{endpoint}_rel", "method"], ascending=[False, True], kind="stable").iloc[0]

    def ceiling_cell(self, enc, pre, scheme, endpoint) -> str:
        x = self.ceiling_row(enc, pre, scheme, endpoint)
        return "n/a" if x is None else f"{self.t2_level(enc, pre, scheme, endpoint)} ({x.method})"

    def genuine(self, enc: str) -> tuple[float, float]:
        """Mean and sd of the raw-space genuine acceptance at the DEV threshold over the zero-pair run rows."""
        g = self.zraw(enc)["e1_genuine"]
        return float(g.mean()), float(g.std(ddof=1))

    def impostor_level(self, enc: str) -> float:
        return self.attack_fmr(enc) / self.genuine(enc)[0]

    # ---- zero-pair selection ------------------------------------------------------------
    def best_zero_pair(self, enc: str, pre: str, methods=KEYFREE, scheme: str = SCHEME) -> dict | None:
        """Best key-free configuration (max over method, AUX, p), chosen with ground truth."""
        s = self.zsum[(self.zsum.encoder == enc) & (self.zsum.pre == pre) & (self.zsum.scheme == scheme)
                      & self.zsum.method.isin(methods)]
        if not len(s):
            return None
        s = s.sort_values(["best_e1_rel", "best_rec_cos_z"], ascending=False, kind="stable")
        top = s.iloc[0]
        raw = self.zraw(enc)
        sub = raw[(raw.pre == pre) & (raw.scheme == scheme) & (raw.method == top.method) & (raw.aux == top.aux)]
        per = (sub.groupby(["p", "seed"], as_index=False)[["e1_rel", "rec_cos_z"]].mean()
                  .groupby("p", as_index=False)[["e1_rel", "rec_cos_z"]].mean()
                  .sort_values(["e1_rel", "rec_cos_z"], ascending=False, kind="stable"))
        p_best = int(per.iloc[0]["p"])
        at_p = sub[sub.p == p_best]
        return {"row": top, "p": p_best, "zero": bool(top.best_e1_rel == 0.0),
                "cp_abs": float(sub["e1_cp_upper"].max()),
                "cp_rel": float((sub["e1_cp_upper"] / sub["e1_genuine"]).max()),
                "proxy": float(at_p["e1_proxy"].mean()), "trials": int(sub["e1_trials"].iloc[0]), "n_rows": len(sub)}

    def oracle_gap(self, enc: str, pre: str, p: int, aux: str):
        raw = self.zraw(enc)
        sel = raw[(raw.pre == pre) & (raw.scheme == SCHEME) & (raw.p == p) & (raw.aux == aux)]
        vals = {}
        for meth in ("eigen_oracle", "eigen"):
            part = sel[sel.method == meth]
            vals[meth] = part.groupby("seed")["e1_rel"].mean().mean() if len(part) else float("nan")
        return vals["eigen_oracle"] - vals["eigen"]

    @staticmethod
    def zp_cell(b: dict | None, rel: bool = True) -> str:
        if b is None:
            return "n/a"
        if b["zero"]:
            cp = b["cp_rel"] if rel else b["cp_abs"]
            return f"0 successes; CP95 upper <= {cp:.3f}"
        return cons(b["row"], "best_e1_rel")

    # ---- tables --------------------------------------------------------------------------
    def t_spectra(self) -> str:
        rows = []
        for enc in self.encs:
            sp = self.runs[enc]["spec"]
            raw = sp["raw_centered_normalized"]
            for pref in ("none", "pca128_a1", "fi128", "firbig128"):
                pre = self.pre(enc, pref)
                if pre is None:
                    continue
                pc = sp["preconditioned"][pre]
                rows.append([enc, int(raw["dim"]), f3(raw["participation_ratio"]), pre, f3(pc["participation_ratio"]),
                             f3(pc["anisotropy_index"])])
        return md_table(["encoder", "dim", "raw PR (centered, normalized)", "pre", "PR after pre", "anisotropy_index"],
                        rows)

    def t_utility(self) -> tuple[str, str]:
        rows = []
        for enc in self.encs:
            for pre in self.pres(enc):
                u = self.util_row(enc, pre, SCHEME)
                if u is None:
                    continue
                rows.append([enc, pre, int(u.m), int(u.k), ci(u.eer, u.eer_lo, u.eer_hi, 100, nonneg=True),
                             ci(u.tmr, u.tmr_lo, u.tmr_hi, nonneg=True)])
        main = md_table(["encoder", "pre", "m", "k", "EER % [95% CI]", "TMR at FMR 0.1% [95% CI]"], rows)
        rows = []
        for enc in self.encs:
            for pref in ("none", "fi128"):
                pre = self.pre(enc, pref)
                for sch in sorted(self.util[self.util.encoder == enc].scheme.unique()):
                    if sch == SCHEME:
                        continue
                    u = self.util_row(enc, pre, sch)
                    if u is None:
                        continue
                    rows.append([enc, sch, pre, int(u.m), int(u.k), ci(u.eer, u.eer_lo, u.eer_hi, 100, nonneg=True),
                                 ci(u.tmr, u.tmr_lo, u.tmr_hi, nonneg=True)])
        other = md_table(["encoder", "scheme", "pre", "m", "k", "EER % [95% CI]", "TMR at FMR 0.1% [95% CI]"], rows)
        return main, other

    def t_chance(self) -> str:
        """Table 3c: genuine acceptance, trials and zero-effort impostor level, all read from the run rows; the
        measured columns use the impostor acceptance of the target split at the DEV threshold (`measured_fmr`)."""
        rows = []
        for enc in self.encs:
            z, p, a = self.zraw(enc), self.praw(enc), self.araw(enc)
            g, sd = self.genuine(enc)
            trials = sorted(set(z.e1_trials.astype(int)) | set(p.e1_trials.astype(int)))
            rows.append([enc, f"{self.attack_fmr(enc):g}", ", ".join(map(str, trials)),
                         f"{int(a.e1_trials.min())}-{int(a.e1_trials.max())}", f"{g:.3f} (sd {sd:.3f}, n = {len(z)} rows)",
                         f"{self.impostor_level(enc):.4f}", f"{100 * self.measured_fmr(enc):.3f}",
                         f"{self.measured_impostor_level(enc):.4f}"])
        return md_table(["encoder", "target FMR (attack threshold)", "E1 trials per zero-pair/T2 row",
                         "E1 trials per anchor row (min-max)",
                         "e1_genuine: raw-space genuine acceptance at the DEV threshold, mean over zero-pair rows",
                         "zero-effort impostor level (target FMR / e1_genuine)",
                         "measured FMR (%): target-split impostor pairs at the DEV threshold (fmr_actual, lrp_full, "
                         "pre none, mean over seeds)",
                         "measured zero-effort level (measured FMR / e1_genuine)"], rows)

    def t_paired(self) -> str:
        """Table 3d: paired per-seed utility differences FI128 minus PCA128 and FI128 minus none (lrp_full)."""
        rows = []
        for enc in self.encs:
            fi = self.pre(enc, "fi128")
            for ref in (self.pre(enc, "pca128_a1"), self.pre(enc, "none")):
                for metric in ("eer", "tmr"):
                    g = self.paired_utility(enc, fi, ref, metric)
                    rows.append([enc, f"{fi} - {ref}", metric, f"{pp(g['mean'])} [{pp(g['lo'])}, {pp(g['hi'])}]",
                                 ", ".join(pp(v) for v in g["diffs"]), g["n"]])
        return md_table(["encoder", "comparison", "metric", "paired difference (percentage points) [95% interval]",
                         "per-seed differences (seed order)", "seeds"], rows)

    def t_victims(self) -> str:
        """Table 4c: number and genuine acceptance of the T1 victims per pre-conditioner (lrp_full, exact anchors)."""
        rows = []
        for enc in self.encs:
            n = self.victims(enc)
            for pre in self.pres(enc):
                rows.append([enc, pre, n, f3(self.victim_genuine(enc, pre, "e1")), f3(self.victim_genuine(enc, pre, "e2"))])
        return md_table(["encoder", "pre", "victims per seed and key (n_targets)",
                         "E1 genuine acceptance (raw space, DEV raw threshold)",
                         "E2 genuine acceptance (renewed key, DEV threshold under that key)"], rows)

    def eigen_best(self, enc: str, pre: str) -> tuple[int, str]:
        """(p, AUX) of the key-free eigen alignment with the highest mean E1 relative success (mean over keys within a
        seed, then over seeds), chosen with ground truth; the configuration of the oracle gap."""
        raw = self.zraw(enc)
        sub = raw[(raw.pre == pre) & (raw.scheme == SCHEME) & (raw.method == "eigen")]
        per = (sub.groupby(["aux", "p", "seed"], as_index=False)["e1_rel"].mean()
               .groupby(["aux", "p"], as_index=False)["e1_rel"].mean()
               .sort_values("e1_rel", ascending=False, kind="stable").iloc[0])
        return int(per.p), per.aux

    def t_oracle(self) -> str:
        """Tables 5b and 12c: eigen_oracle in the raw space (lrp_full), the largest seed-mean E1 relative success with the
        absolute acceptance of that entry, the largest pre-key cosine over AUX and p, and the oracle gap at the
        key-free eigen selection."""
        rows = []
        for enc in self.encs:
            o = self.oracle_max(enc)
            e1 = f"0 successes; CP95 upper <= {o['cp_rel']:.3f}" if o["zero"] else f3_rate(o["e1"])
            none = self.pre(enc, "none")
            p, aux = self.eigen_best(enc, none)
            at = "n/a (no success)" if o["zero"] else f"{o['e1_at'][1]}, {o['e1_at'][0]}"
            rows.append([enc, e1, at, "n/a" if o["zero"] else f3_rate(o["e1_abs"]), f3(o["cos"]),
                         f"{o['cos_at'][1]}, {o['cos_at'][0]}", f3(self.oracle_gap(enc, none, p, aux)), f"{p}, {aux}"])
        return md_table(["encoder", "max E1 relative of eigen_oracle over AUX and p (pre none)", "at p, AUX",
                         "absolute E1 acceptance at that entry (mean e1_proxy)",
                         "max rec_cos_z (pre-key) of eigen_oracle", "at p, AUX",
                         "oracle gap (eigen_oracle - eigen E1) at the key-free eigen selection", "selected p, AUX"], rows)

    def t_oracle_whitened(self) -> str:
        """Tables 5c and 12d: eigen_oracle after whitening (PCA128, PCA256, FI64, FI128, FI256), LRP-full: per endpoint the largest
        seed-mean relative success over AUX and p (or the zero count with its relative CP95 bound) and the largest
        seed-mean absolute acceptance, with the largest pre-key cosine."""
        rows = []
        for enc in self.encs:
            for label, prefix in WHITENED_PRES:
                cells = []
                for ep in ("e1", "e2"):
                    o = self.oracle_endpoint(enc, prefix, ep)
                    cells += [f"0 successes; CP95 upper <= {o['cp_rel']:.3f}" if o["zero"] else f3_rate(o["rel"]),
                              f3_rate(o["abs_max"])]
                rows.append([enc, label, *cells, f3(o["cos"])])
        return md_table(["encoder", "pre", "max E1 relative of eigen_oracle over AUX and p", "max E1 absolute (mean e1_proxy)",
                         "max E2 relative", "max E2 absolute (mean e2_proxy)", "max rec_cos_z (pre-key)"], rows)

    def t_keyknown(self) -> str:
        """Table 6b: key-known (T2) level of lrp_full (pres none and fi128) with the absolute acceptance behind it."""
        rows = []
        for enc in self.encs:
            raw = self.praw(enc)
            for pref in ("none", "fi128"):
                pre = self.pre(enc, pref)
                for ep in ("e1", "e2"):
                    x = self.ceiling_row(enc, pre, SCHEME, ep)
                    sub = raw[(raw.pre == pre) & (raw.scheme == SCHEME) & (raw.method == x.method)]
                    gen = float(sub.groupby("seed")[f"{ep}_genuine"].mean().mean())
                    rows.append([enc, pre, ep.upper(), self.t2_level(enc, pre, SCHEME, ep),
                                 f3(self.ceiling_abs(enc, pre, SCHEME, ep)), f3(gen), x.method])
        return md_table(["encoder", "pre", "endpoint", "key-known (T2) level, relative [95% interval]",
                         "absolute proxy acceptance (mean e*_proxy)", "genuine acceptance of the same targets "
                         "(mean e*_genuine)", "T2 method"], rows)

    def t_collusion(self) -> tuple[str, str]:
        rows = []
        for enc in self.encs:
            for pref in ("none", "pca128_a1", "fi128"):
                pre = self.pre(enc, pref)
                a = self.anchor_rows(enc, pre)
                for r in R_COLLUSION:
                    x = a[a.r == r]
                    if not len(x):
                        continue
                    x = x.iloc[0]
                    rows.append([enc, pre, int(x.m), r, ci(x.rec_cos_z, x.rec_cos_z_lo, x.rec_cos_z_hi),
                                 f3(x.cstar), f3(x.sqrt_r_m) if r <= x.m else "n/a"])
        coll = md_table(["encoder", "pre", "m", "r", "rec_cos_z (pre-key space, proj decoder) [95% CI]", "cstar",
                         "sqrt_r_m"], rows)
        rows = []
        for enc in self.encs:
            for pre in self.pres(enc):
                m = int(self.anchor_rows(enc, pre).iloc[0].m)
                for sch in ALL_SCHEMES:
                    for var in ("exact", "fresh"):
                        rows.append([enc, pre, m, sch, var,
                                     self.r50_cell(enc, pre, "e1", sch, var), self.seed_range_cell(enc, pre, "e1", sch, var),
                                     self.ceiling_cell(enc, pre, sch, "e1"),
                                     self.r50_cell(enc, pre, "e2", sch, var), self.seed_range_cell(enc, pre, "e2", sch, var),
                                     self.ceiling_cell(enc, pre, sch, "e2")])
        r50 = md_table(["encoder", "pre", "m", "scheme", "anchors", "r50 E1 (best decoder)",
                        "r50 E1 per-seed range [min-max]",
                        "key-known (T2) level E1 [95% interval] (T2 method)", "r50 E2 (best decoder)",
                        "r50 E2 per-seed range [min-max]",
                        "key-known (T2) level E2 [95% interval] (T2 method)"], rows)
        return coll, r50

    def t_zeropair(self) -> str:
        rows = []
        for enc in self.encs:
            g = self.genuine(enc)[0]
            for pref in PRE_PREFIXES:
                pre = self.pre(enc, pref)
                if pre is None:
                    continue
                b = self.best_zero_pair(enc, pre)
                if b is None:
                    continue
                r = b["row"]
                gap = ""
                if pref == "none":
                    p_, aux_ = self.eigen_best(enc, pre)
                    gap = f"{f3(self.oracle_gap(enc, pre, p_, aux_))} (p {p_}, {aux_})"
                bc = self.best_zero_pair(enc, pre, COMMON)  # the common-method row that the manuscript quotes
                if bc is None:
                    abscos = ""
                else:
                    m_ = int(self.util_row(enc, pre, SCHEME).m)
                    abscos = f"{f3(bc['row'].best_abs_cos_z)} / {f3(float(uniform_abs_cos_mean(m_)))}"
                rows.append([enc, pre, self.zp_cell(b), ci(r.best_rec_cos_z, r.best_rec_cos_z_lo, r.best_rec_cos_z_hi),
                             "n/a" if b["zero"] else r.method, "n/a" if b["zero"] else b["p"],
                             "n/a" if b["zero"] else r.aux, f3(b["cp_abs"]) if b["zero"] else "",
                             f"{b['proxy']:.4f}", f"{g:.3f}", gap, abscos])
        return md_table(["encoder", "pre", "best E1 relative [95% CI] (CP95 upper if 0 successes)",
                         "rec_cos_z (pre-key space) [95% CI]", "method", "p (argmax of mean E1)", "AUX",
                         "CP95 upper, absolute proxy acceptance",
                         "absolute proxy acceptance (mean e1_proxy at the selected method, AUX and p)",
                         "e1_genuine (raw-space genuine acceptance)",
                         "oracle gap (eigen_oracle - eigen E1 at the best eigen p and AUX, pre none)",
                         "E|cos| pre-key at the best common-method configuration (eigen, wp) / uniform reference"], rows)

    def t_keycompromise(self) -> str:
        rows = []
        for enc in self.encs:
            raw = self.praw(enc)
            for pref in ("none", "fi128"):
                pre = self.pre(enc, pref)
                d = self.post[(self.post.encoder == enc) & (self.post.pre == pre)]
                for _, x in d.sort_values(["scheme", "method"]).iterrows():
                    sub = raw[(raw.pre == pre) & (raw.scheme == x.scheme) & (raw.method == x.method)]
                    if len(sub) and (sub["e2_successes"] == 0).all():
                        e2 = f"0 successes; CP95 upper <= {(sub['e2_cp_upper'] / sub['e2_genuine']).max():.3f} (rel.)"
                    else:
                        e2 = cons(x, "e2_rel")
                    rows.append([enc, x.scheme, pre, x.method, ci(x.in_system_accept, x.in_system_accept_lo,
                                                                  x.in_system_accept_hi, nonneg=True), e2])
        return md_table(["encoder", "scheme", "pre", "attack", "in-system acceptance [95% CI]",
                         "E2 relative, post-renewal [95% interval (wider of the identity-level and seed-level intervals)]"], rows)

    def t_prop2(self) -> tuple[str, list[str]]:
        rows, lines = [], []
        for enc in self.encs:
            devs = []
            for pref in ("pca128_a1", "fi128"):
                pre = self.pre(enc, pref)
                a, ar = self.anchor_rows(enc, pre), self.anchor_rows(enc, pre, decoder="ridge")
                for r in (20, 40, 80):
                    x, xr = a[a.r == r].iloc[0], ar[ar.r == r].iloc[0]
                    dev = abs(x.rec_cos_z - x.cstar)
                    devs.append((dev, pre, r, xr.rec_cos_z))
                    rows.append([enc, pre, int(x.m), r, f3(x.rec_cos_z), f3(x.cstar), f3(dev), "", f3(xr.rec_cos_z)])
            none = self.pre(enc, "none")
            a, ar = self.anchor_rows(enc, none), self.anchor_rows(enc, none, decoder="ridge")
            exc = []
            for r in (20, 40, 80):
                x, xr = a[a.r == r].iloc[0], ar[ar.r == r].iloc[0]
                exc.append((r, x.rec_cos_z - x.cstar, int(x.m)))
                rows.append([enc, none, int(x.m), r, f3(x.rec_cos_z), f3(x.cstar), "", f3(x.rec_cos_z - x.cstar),
                             f3(xr.rec_cos_z)])
            d = max(devs)
            lines.append(f"- {enc}: largest |rec_cos_z - cstar| (proj decoder) over r in {{20, 40, 80}} for pca128_a1 "
                         f"and fi128 = {d[0]:.3f} (pre {d[1]}, r = {d[2]}; ridge rec_cos_z there = {d[3]:.3f}). "
                         f"Excess of rec_cos_z over cstar for pre none (m = {exc[0][2]}): "
                         + ", ".join(f"r = {r}: {e:.3f}" for r, e, _ in exc) + ".")
        t = md_table(["encoder", "pre", "m", "r", "rec_cos_z (proj)", "cstar", "abs(rec_cos_z - cstar)",
                      "excess rec_cos_z - cstar (pre none)", "rec_cos_z (ridge, shows the intercept gap)"], rows)
        return t, lines

    def t_prop1(self) -> tuple[str, list[str]]:
        rows, lines = [], []
        for enc in self.encs:
            floor = self.impostor_level(enc)
            for pref in ("fi64", "fi128", "fi256", "rbig128", "firbig128"):
                pre = self.pre(enc, pref)
                if pre is None:
                    continue
                b = self.best_zero_pair(enc, pre)
                if b is None:
                    continue
                m = int(self.util_row(enc, pre, SCHEME).m)
                r = b["row"]
                ua, uc = float(uniform_abs_cos_mean(m)), 1.0 / m
                meth = "n/a" if b["zero"] else f"{r.method}, p {b['p']}, AUX {r.aux}"
                rows.append([enc, pre, m, ci(r.best_abs_cos_z, r.best_abs_cos_z_lo, r.best_abs_cos_z_hi), f3(ua),
                             f3(r.best_abs_cos_z / ua), ci(r.best_cos2_z, r.best_cos2_z_lo, r.best_cos2_z_hi),
                             f"{uc:.4f}", f3(r.best_cos2_z / uc), self.zp_cell(b), f"{floor:.4f}",
                             "n/a" if b["zero"] else r.method, "n/a" if b["zero"] else b["p"],
                             "n/a" if b["zero"] else r.aux])
                lines.append(f"- {enc}, {pre}: best_abs_cos_z = {r.best_abs_cos_z:.3f} vs uniform {ua:.3f} "
                             f"(ratio {r.best_abs_cos_z / ua:.2f}); best_cos2_z = {r.best_cos2_z:.4f} vs 1/m = {uc:.4f} "
                             f"(ratio {r.best_cos2_z / uc:.2f}); best E1 relative = {self.zp_cell(b)} next to the zero-effort "
                             f"impostor level {floor:.4f} (selected: {meth}).")
        t = md_table(["encoder", "pre", "m", "best_abs_cos_z [95% CI]", "uniform E[abs cos] (m)", "ratio measured/uniform",
                      "best_cos2_z [95% CI]", "1/m", "ratio measured/(1/m)", "best zero-pair E1 relative",
                      "zero-effort impostor level (target FMR / e1_genuine)", "method", "p (argmax of mean E1)", "AUX"], rows)
        return t, lines

    def t_aniso(self) -> str:
        rows = []
        for enc in self.encs:
            none = self.pre(enc, "none")
            a = self.anchor_rows(enc, none)
            cum = self.runs[enc]["spec"]["preconditioned"][none]["cumvar_first256"]
            for r in R_ANISO:
                x = a[a.r == r]
                if not len(x) or r > len(cum):
                    continue
                x = x.iloc[0]
                rows.append([enc, int(x.m), r, ci(x.rec_cos2_z, x.rec_cos2_z_lo, x.rec_cos2_z_hi), f3(r / x.m),
                             f3(cum[r - 1]), f3(x.rec_cos2_z / (r / x.m)), f3(x.rec_cos2_z / cum[r - 1])])
        return md_table(["encoder", "m", "r", "rec_cos2_z (proj) [95% CI]", "r/m", "top-r explained-variance fraction",
                         "rec_cos2_z / (r/m)", "rec_cos2_z / top-r variance fraction"], rows)

    def t_headline(self) -> str:
        rows = []
        for enc in self.encs:
            sp = self.runs[enc]["spec"]["raw_centered_normalized"]
            none, fi = self.pre(enc, "none"), self.pre(enc, "fi128")
            un, uf = self.util_row(enc, none, SCHEME), self.util_row(enc, fi, SCHEME)
            bn, bf = (self.best_zero_pair(enc, p, COMMON) for p in (none, fi))
            ica = self.best_zero_pair(enc, fi, ("ica",))
            rows.append([enc, f3(sp["participation_ratio"]), ci(un.eer, un.eer_lo, un.eer_hi, 100, nonneg=True) + " %",
                         ci(uf.eer, uf.eer_lo, uf.eer_hi, 100, nonneg=True) + f" % ({fi})",
                         self.r50_cell(enc, none, "e1"), self.r50_cell(enc, fi, "e1"),
                         self.r50_cell(enc, none, "e2"), self.r50_cell(enc, fi, "e2"),
                         self.ceiling_cell(enc, fi, SCHEME, "e1"), self.ceiling_cell(enc, fi, SCHEME, "e2"),
                         self.zp_cell(bn), self.zp_cell(bf), self.zp_cell(ica), f"{self.impostor_level(enc):.4f}"])
        return md_table(["Encoder", "PR", "EER none", "EER fi128", "r50 E1 none", "r50 E1 fi128", "r50 E2 none",
                         "r50 E2 fi128", "key-known (T2) level E1, fi128 lrp_full", "key-known (T2) level E2, fi128 lrp_full",
                         "zero-pair best E1, none (common methods eigen, wp)",
                         "zero-pair best E1, fi128 (common methods eigen, wp)", "zero-pair ICA best E1, fi128",
                         "zero-effort impostor level (target FMR / e1_genuine)"], rows)

    def t_fmr1(self) -> str:
        rows = []
        for enc in self.encs:
            # pca128_a1 (full PCA whitening) by exact name: the prefix pca128 would also match pca128_a0.5
            for pref in ("none", "pca128_a1", "fi128"):
                pre = self.pre(enc, pref)
                b = self.best_zero_pair(enc, pre, COMMON)
                # per-seed range next to each collusion size, as in table 4b, so that the per-seed rule applies
                r50s = [f"{self.r50_cell(enc, pre, ep)} {self.seed_range_cell(enc, pre, ep)}" for ep in ("e1", "e2")]
                rows.append([enc, pre, *r50s, self.zp_cell(b),
                             self.ceiling_cell(enc, pre, "lrp_full", "e1"), self.ceiling_cell(enc, pre, "lrp_full", "e2"),
                             self.ceiling_cell(enc, pre, "biohash_half", "e1"),
                             self.ceiling_cell(enc, pre, "biohash_half", "e2"), f"{self.impostor_level(enc):.4f}"])
        return md_table(["encoder", "pre", "r50 E1 (lrp_full, exact, best decoder) [per-seed range]",
                         "r50 E2 (lrp_full, exact, best decoder) [per-seed range]",
                         "zero-pair best E1 (common methods eigen, wp)",
                         "key-known T2 E1, lrp_full", "key-known T2 E2, lrp_full",
                         "key-known T2 E1, biohash_half", "key-known T2 E2, biohash_half",
                         "zero-effort impostor level (target FMR / e1_genuine)"], rows)


# ---- run-level (both operating points) -------------------------------------------------------
def t_provenance(runs: list[dict], scale_runs: list[dict]) -> str:
    rows = []
    for r in runs:
        m, enc = r["meta"], r["meta"]["config"]["name"]
        rows.append([r["dir"].name, enc, DATASET_LABEL[base_name(enc)], f"{m['config']['fmr'][0]:g}", short_sha(m["git_sha"]),
                     m["started_utc"], f"{m['runtime_s']:.0f}", "yes" if m.get("data_files") else "no"])
    for r in scale_runs:
        m = r["meta"]
        rows.append([r["dir"].name, m["config"]["name"], scale_label(r), "n/a", short_sha(m["git_sha"]),
                     m["started_utc"], f"{m['runtime_s']:.0f}", "yes" if m.get("data_files") else "no"])
    return md_table(["run", "encoder", "dataset / data source", "attack target FMR", "git_sha", "started_utc", "runtime_s",
                     "data-file hash"], rows)


def size_map(r: dict, df: pd.DataFrame) -> dict[int, int]:
    """Requested -> used distractor sizes of a scale run.

    The runner runs each size used once and labels its scale.csv rows with the largest request that maps to it, so
    capped requests can collapse onto one size used, and only metadata.json (`distractor_sizes_requested_to_used`)
    lists every request. When the field is there, every scale.csv pair must agree with it and its sizes used must be
    those of scale.csv; the full mapping is returned. Runs that predate the field give the scale.csv pairs.
    """
    pairs = df[["n_distractors_requested", "n_distractors"]].drop_duplicates()
    out = {int(q): int(u) for q, u in zip(pairs.n_distractors_requested, pairs.n_distractors)}
    if len(out) != len(pairs):
        raise ValueError(f"scale run {r['dir'].name}: one requested size maps to several sizes used")
    rec = r["meta"].get("distractor_sizes_requested_to_used")
    if rec is None:
        return dict(sorted(out.items()))
    rec = {int(k): int(v) for k, v in rec.items()}
    if any(rec.get(q) != u for q, u in out.items()) or set(rec.values()) != set(out.values()):
        raise ValueError(f"scale run {r['dir'].name}: requested -> used sizes in metadata.json {rec} disagree with "
                         f"scale.csv {out}")
    return dict(sorted(rec.items()))


def t_scale(scale_runs: list[dict]) -> tuple[str, str]:
    """Tables 7a (distractor rows: loaded, both removals, sizes; environment) and 7b (one row per scale.csv row)."""
    rows, meta_rows = [], []
    for r in scale_runs:
        m = r["meta"]
        df = pd.read_csv(r["dir"] / "scale.csv")
        shared = m["secondary_removed_shared_identity"]
        sizes = ", ".join(f"{q} -> {u}" for q, u in size_map(r, df).items())
        meta_rows.append([r["dir"].name, scale_label(r), m.get("distractor_source", "n/a"), m["distractors_loaded"],
                          "n/a (no exclusion)" if shared is None else shared, m["distractors_available_before"],
                          m["distractors_removed"], m["distractors_available_after"], sizes, m.get("cpu_model", "n/a"),
                          m.get("omp_num_threads", "n/a")])
        for _, x in df.iterrows():
            rows.append([r["dir"].name, scale_label(r), x.scheme, int(x.n_distractors_requested), int(x.n_distractors), int(x.ef_search),
                         f3(x.rank1_exact), f3(x.rank1_hnsw), f3(x.recall10_tie_aware), f3(x.latency_ms_per_query),
                         f3(x.index_serialized_mb)])
    scale = md_table(["run", "distractor data source", "scheme", "n_distractors requested", "n_distractors used", "ef_search", "rank1_exact",
                      "rank1_hnsw", "recall10_tie_aware", "latency_ms_per_query", "serialized index size (MB)"], rows)
    meta = md_table(["run", "distractor data source", "distractor source", "rows loaded (after the relabel step, if any)",
                     "removed: fingers shared with D1", "drawn for the largest size",
                     "removed: near-duplicates of a D1 row (cos >= 0.95)", "available after the near-duplicate filter",
                     "requested -> used", "CPU model", "OMP threads"], meta_rows)
    return meta, scale


def _span(values) -> str:
    lo, hi = min(values), max(values)
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def t_corpus(runs: list[dict]) -> str:
    """Table 1b: corpus, split and AUX sizes per isotropy run (corpus_counts, split_counts, aux_sizes in metadata.json).

    Split fingers must be equal over the seeds of a run; split images are given as their range over the seeds. For the
    shifted AUX (secondary corpus) the rows removed for a finger shared with D1 stand next to the near-duplicates removed.
    """
    rows = []
    for r in runs:
        m = r["meta"]
        cc, seeds = m["corpus_counts"], list(m["split_counts"].values())
        fingers = []
        for part in ("dev", "tgt", "aux"):
            vals = {s[f"{part}_identities"] for s in seeds}
            if len(vals) != 1:
                raise ValueError(f"run {r['dir'].name}: {part} fingers differ over the seeds: {sorted(vals)}")
            fingers.append(str(vals.pop()))
        images = " / ".join(_span([s[f"{part}_images"] for s in seeds]) for part in ("dev", "tgt", "aux"))
        shifted = m["aux_sizes"].get("shifted")
        sh = ("n/a", "n/a", "n/a", "n/a") if shifted is None else (
            shifted, m["secondary_removed_shared_identity"], m["shifted_aux_removed"], m["shifted_aux_source"])
        rows.append([r["dir"].name, m["config"]["name"], m["data_source"], cc["images"],
                     f"{cc['identities']} ({cc['identities_ge_min_tgt_images']})", "/".join(fingers), images, *sh])
    return md_table(["run", "encoder", "data source (metadata.json)", "images",
                     "fingers (with >= min_tgt_images images)", "split fingers DEV/TGT/AUX",
                     "split images DEV/TGT/AUX (min-max over seeds)", "shifted AUX rows",
                     "shifted AUX removed: fingers shared with D1", "shifted AUX removed: near-duplicates (cos >= 0.95)",
                     "shifted AUX source"], rows)


def capped_sizes(scale_runs: list[dict]) -> list[str]:
    """One deviation line per scale run whose requested distractor sizes were capped (requested != used), listing
    every capped request, also those that collapsed onto one size used."""
    out = []
    for r in scale_runs:
        capped = {q: u for q, u in size_map(r, pd.read_csv(r["dir"] / "scale.csv")).items() if q != u}
        if capped:
            collapsed = len(set(capped.values())) < len(capped)
            out.append(f"- {r['meta']['config']['name']}: requested distractor sizes above the rows available after "
                       "the near-duplicate filter were capped (" + ", ".join(f"{q} -> {u}" for q, u in capped.items())
                       + ("; requests capped onto one size used ran once, under the largest request" if collapsed
                          else "") + "); see table 7a.")
    return out


def results_date(runs: list[dict]) -> str:
    """Date (UTC) of the latest run start; the results record carries it in its title."""
    return max(r["meta"]["started_utc"] for r in runs)[:10]


def t_spectrum_gate(scale_runs: list[dict]) -> str:
    """Table 7c: participation ratio of the distractors vs the primary dev population, with the 30% flag."""
    rows, flags = [], []
    for r in scale_runs:
        m = r["meta"]
        pr_t = m["spectrum_primary_dev"]["participation_ratio"]
        pr_d = m["spectrum_distractors"]["participation_ratio"]
        rel = abs(pr_d - pr_t) / pr_t
        flagged = rel > 0.30
        rows.append([r["dir"].name, scale_label(r), f"{pr_t:.1f}", f"{pr_d:.1f}", f"{100 * rel:.1f}%",
                     "FLAG" if flagged else "no"])
        if flagged:
            flags.append(f"Flag ({r['dir'].name}): {SPECTRUM_FLAG}.")
    out = md_table(["run", "distractor data source", "participation ratio, spectrum_primary_dev",
                    "participation ratio, spectrum_distractors", "relative difference |d - p| / p",
                    "differs by more than 30%"], rows)
    return out + ("\n\n" + "\n".join(flags) if flags else "")


def run_commits(runs: list[dict], scale_runs: list[dict]) -> str:
    iso = sorted({short_sha(r["meta"]["git_sha"]) for r in runs})
    sc = sorted({short_sha(r["meta"]["git_sha"]) for r in scale_runs})
    return "run code commit " + ", ".join(iso) + (f" (scale run: {', '.join(sc)})" if sc else "")


def sinkhorn_counts(runs: list[dict]) -> str:
    parts = []
    for r in runs:
        text = (r["dir"] / "stdout.log").read_text(errors="replace")
        if "Traceback" in text:
            raise RuntimeError(f"Traceback in {r['dir'] / 'stdout.log'}")
        parts.append(f"{r['meta']['config']['name']}: {text.count('Sinkhorn did not converge')}")
    return ", ".join(parts)


def t_skips(runs: list[dict]) -> str:
    rows = []
    for r in runs:
        for line in (r["dir"] / "stdout.log").read_text(errors="replace").splitlines():
            if re.search(r"skip pre=.*scheme=", line):
                rows.append([r["dir"].name, line.strip()])
    return md_table(["run", "line"], rows) if rows else "none"


ORACLE_NOTE = ("`eigen_oracle` is eigen alignment of the stolen templates with AUX (leading p principal axes matched "
               "by rank, variances rescaled) whose per-component signs come from the key: alignment followed by exact "
               "sign resolution, which an attacker could at best obtain from anchors. It is a diagnostic, never a "
               "key-free attack. Values are means over keys and seeds; the maximum over AUX type and p (grid 8 to 128) "
               "is selected with ground truth and has no interval. A maximum with zero successes in every row gives the "
               "largest relative one-sided 95% Clopper-Pearson bound. The joint alignment-plus-anchor attack, with "
               "anchor-voted signs, is in sections 13 and 13b and in the paper's Table S5.")
WHITENED_PRES = (("pca128", "pca128_a1"), ("pca256", "pca256_a1"), ("fi64", "fi64"), ("fi128", "fi128"),
                 ("fi256", "fi256"))   # every whitened pre-conditioner of the paper (label, prefix)
ORACLE_W_NOTE = ("After whitening, E1 maps the proxy back with the public back-map and E2 re-protects it with the "
                 "renewed key (the round trip of the protocol); each maximum (relative, absolute, cosine) is taken on "
                 "its own and can come from a different entry.")
COLLUSION_SCOPE = ("Scope: hashing-scheme collusion numbers (lrp_half, biohash_half, iom_256) hold only for the "
                   "ridge/proj/mlp decoders evaluated; a stronger IoM decoder exists in the synthetic proof check "
                   "(docs/theory/proofcheck/check4), so hashing collusion resistance is not claimed.")


def tables_md(base: Study, fmr1: Study, runs, scale_runs, joint: dict | None = None) -> tuple[str, list[str], list[str]]:
    util, util_other = base.t_utility()
    coll, r50 = base.t_collusion()
    p2, p2_lines = base.t_prop2()
    p1, p1_lines = base.t_prop1()
    scale_meta, scale = t_scale(scale_runs)
    n_note = ("n (E1 trials) is read from the run rows (`e1_trials`) and differs per dataset: see table 3c. "
              "Zero observed successes are reported with the largest runner-reported one-sided 95% "
              "Clopper-Pearson upper bound, never as zero risk.")
    parts = [
        (f"1. Provenance (all {len(runs) + len(scale_runs)} runs)", t_provenance(runs, scale_runs),
         "The ten joint-stage runs of sections 13 and 13b record their provenance in results/joint/*/metadata.json; "
         "results/joint/PARITY.md holds their parity check with the anchor stage."),
        ("1b. Corpora, splits and AUX per isotropy run", t_corpus(runs),
         "Images and fingers are those of the loaded corpus (after the relabel step for D1); in parentheses, the fingers "
         "with at least `min_tgt_images` images, from which TGT is drawn. The shifted AUX of the D1 runs comes from the "
         f"{SECONDARY_CORPUS}: the rows of fingers shared with D1 are removed first, then the "
         "near-duplicates of any D1 row (uncentered cosine >= 0.95)."),
        ("2. Spectra (raw PR is of the centered, normalized embeddings)", base.t_spectra(), None),
        ("3a. Utility, scheme lrp_full, every pre (FMR 0.1% operating point; threshold from DEV)", util, SEED_NOTE),
        ("3b. Utility, other schemes, pres none and fi128", util_other, None),
        ("3c. Raw genuine acceptance, trials and zero-effort impostor level per encoder (FMR 0.1% attack threshold)", base.t_chance(),
         "e1_genuine is the mean over the zero-pair run rows of the raw-space genuine acceptance at the DEV "
         "threshold (the proxy acceptance of a perfect attacker would equal it). An attacker that submits a "
         "random template passes at about the target FMR, so E1 relative = proxy acceptance / e1_genuine has a "
         "zero-effort impostor level of target FMR / e1_genuine. ""Observed values can fall below it, because a reconstructed proxy is not a random impostor capture. Absolute proxy acceptance is taken from `e1_proxy` in the "
         "data (table 5), not converted through TMR. The measured columns replace the target FMR by the impostor "
         "acceptance that the target split shows at the DEV threshold (`fmr_actual` of the utility rows of lrp_full "
         "without pre-conditioning; lrp_full keeps cosines, so this is the raw-space rate at the threshold of E1), "
         "mean over the seeds. " + n_note),
        ("3d. Paired utility differences (lrp_full, FMR 0.1% operating point): FI128 minus PCA128 and minus none",
         base.t_paired(),
         "Per seed, the difference of the utility rows of the two pre-conditioners (key 0; both scored on the same "
         "split, genuine and impostor pairs and DEV threshold rule), in percentage points; the interval is the "
         "Student-t 95% interval over the per-seed differences (4 df). Like the other seed intervals it describes "
         "between-split variability over overlapping splits of one corpus; it is not identity-level and does not model "
         "subject clustering. The mean equals the difference of the seed means of table 3a."),
        ("4a. Collusion (T1, exact anchors, proj decoder, lrp_full): rec_cos_z vs r", coll,
         "`rec_cos_z` is a pre-key-space cosine (Proposition 2 and `cstar` refer to it); `rec_cos` (raw space, "
         "after the pre-conditioner's inverse map) is in `results/isotropy/summary/summary_anchor.csv`. "
         "`sqrt_r_m` is n/a for r > m. " + SEED_NOTE),
        ("4b. Collusion (T1): r50 of E1 and E2 relative success per pre, scheme and anchor variant", r50,
         "r50 is the smallest r at which the mean relative success reaches 0.5 (log-linear interpolation); the "
         "decoder is the best (smallest r50) over ridge, proj and mlp, named in parentheses; `>= r_max` means 0.5 "
         "was not reached up to r_max (censored). The per-seed range is computed separately for each of the 5 split "
         "seeds: the endpoint relative success is averaged over keys within the seed, r50 is interpolated per decoder "
         "(same log-linear rule), the best decoder is the smallest r50 of that seed, and a seed in which no decoder "
         "reaches 0.5 is censored (counted as r_max, shown with `>=`; the range is marked censored if any seed is). "
         "The headline r50 is computed on the seed-mean curve, so it need not equal the midpoint of the range. The mlp decoder ran only on a subset (pres none and fi128; schemes "
         "lrp_full and iom_256; r in {80, 320, 640} for D1 and {80, 320} for D2, from the configs' `mlp_subset`). "
         "The key-known (T2) level is the mean E1 (E2) relative success with the key known, of the T2 method "
         "(analytic, learned) with the highest mean, shown as mean [95% interval (wider of the identity-level and seed-level intervals)]; if every raw T2 row "
         "of that method has zero successes, the cell reads `0 successes; CP95 upper <= x (rel.)`. "
         "`back-map-limited` (lrp_full, exact anchors) means 0.5 was not reached up to r_max and the key-known "
         "level (exact pre-key vector, public LMMSE back-map) has its reported upper bound (wider interval) below 0.5; this "
         "bounds only attacks that reconstruct the pre-key vector and use that back-map. For other schemes and for "
         "fresh anchors the key-known level is a reference, not a bound: evaluated collusion decoders exceed it "
         "(summary_fmr1). If an r50 was observed although the T2 level is below 0.5 (evidence that the key-known "
         "level is not an upper bound), the r50 is shown "
         "with a note. Relative success is not capped at 1. " + CONS_NOTE + " " + COLLUSION_SCOPE),
        ("4c. Genuine acceptance of the T1 victims (lrp_full, exact anchors)", base.t_victims(),
         "Per seed and key, the targets after the largest r of the grid are the victims (fixed as r grows; the "
         "colluders are the first r targets of the same permutation, enrolled users of the stolen database). The "
         "genuine acceptance is the mean over seeds of `e1_genuine` and `e2_genuine` of the anchor rows (enrolment "
         "capture against the two probe captures of each victim); at r50 the best decoder's proxies are accepted at "
         "about half of it."),
        ("5. Zero-pair attacks (T0, lrp_full): best key-free configuration", base.t_zeropair(),
         "Notes. Key-free methods are eigen, ica and wp over both AUX types. `eigen_oracle` uses the key to "
         "resolve signs; it is a diagnostic, never a key-free attack, and appears only in the oracle-gap column. "
         "\"Best\" is the maximum over p, method and AUX, chosen with ground truth; it is an optimistic upper "
         "envelope from the attacker's side and therefore errs towards overstating risk. The E1 value and "
         "`rec_cos_z` come from `summary_zeropair.csv`. E1 and rec_cos_z come from the same argmax-E1 row per seed and key. The last column is the mean absolute pre-key cosine (`best_abs_cos_z`) at the best common-method (eigen, wp) configuration, next to the uniform-direction reference; it is the value that the manuscript prints as E|cos| (`zp.<enc>.<pre>.abscos`). "
         "The p column is the p "
         "with the highest mean E1 for the selected method and AUX. FastICA alignment ran only where m <= 128 "
         "(`ica_max_dim`), and Wasserstein-Procrustes only for lrp_full with k = m, so pre `none` rows have no "
         "ICA result and the none and fi rows compare different attack sets; the headline table therefore uses "
         "the common methods eigen and wp for the none-vs-fi128 comparison. CP95 upper is the largest "
         "runner-reported one-sided 95% Clopper-Pearson upper bound over seed/key runs of the selected method and "
         "AUX (n = `e1_trials` per row, table 3c); the relative form divides by the run's genuine acceptance. "
         + n_note + " " + CI_NOTE),
        ("5b. Sign-resolution reference (eigen_oracle, pre none, lrp_full)", base.t_oracle(), ORACLE_NOTE),
        ("5c. Sign-resolution reference after whitening (eigen_oracle, pca128, pca256, fi64, fi128, fi256, lrp_full)",
         base.t_oracle_whitened(), ORACLE_NOTE + " " + ORACLE_W_NOTE),
        ("6. Key compromise (T2): in-system acceptance and E2 after renewal", base.t_keycompromise(), CI_NOTE),
        ("6b. Key compromise (T2), lrp_full: key-known level with absolute acceptance", base.t_keyknown(),
         "The relative level is the key-known level of table 4b (T2 method with the highest mean); the absolute "
         "acceptance is the mean over seeds of `e1_proxy` (`e2_proxy`) of that method, next to the genuine acceptance "
         "of the same targets. Renewal re-enrols every target from the same enrolment capture under the second key (no "
         "fresh capture). E2 back-maps, normalises and pre-conditions the proxy before the second key, so a pre-key "
         "vector recovered exactly with the old key returns as a positive multiple of itself plus a small offset "
         "(G(B(z)/q) = z/q + (1/q - 1) W^T x_bar with q = ||B(z)||); acceptance compares cosines, so the E2 level checks "
         "that this round trip preserves acceptance to the reported precision, single borderline decisions can flip, "
         "and it is no equality of templates. " + CI_NOTE),
        ("7a. Scale runs: distractor rows and environment", scale_meta,
         "Order of the steps: rows loaded (after the relabel step for the secondary corpus), rows of fingers shared "
         "with D1 removed, seeded draw of the largest requested size, near-duplicates of any D1 row removed "
         "(uncentered cosine >= 0.95). A requested size above the rows available after the filter is capped "
         "(requested -> used); table 7b lists both sizes, and the keys of scripts/manuscript/make_numbers.py follow "
         "the size used."),
        ("7b. Scale (HNSW)", scale, None),
        ("7c. Scale distractor spectrum check: participation ratio of distractors vs primary dev population",
         t_spectrum_gate(scale_runs), None),
        ("8. Skips", t_skips([*runs, *scale_runs]), None),
        ("9. Proposition 2 check: rec_cos_z vs cstar (exact anchors, proj decoder, lrp_full)", p2,
         "The ridge column is shown to expose the intercept gap of the ridge-GCV decoder relative to the projection "
         "decoder; the proj decoder is the one compared with cstar."),
        ("10. Proposition 1 check: zero-pair on fi, rbig and firbig configurations", p1,
         "best_abs_cos_z and best_cos2_z come from the argmax-E1 row (selected with ground truth), so ratios "
         "above 1 include a selection effect. The chance level for abs cos is E[abs cos] of a uniform direction on the "
         "m-sphere and for cos^2 it is 1/m. " + SEED_NOTE),
        ("11. Anisotropy (pre none, proj decoder, exact anchors, lrp_full)", base.t_aniso(),
         "rec_cos2_z against r/m (the isotropic value) and against the top-r explained-variance fraction of the "
         "centered, normalized embeddings (`spectra.json`, preconditioned.none.cumvar_first256)."),
        ("12. FMR 1% operating point (attack threshold at FMR 1%; separate runs, summary_fmr1)", fmr1.t_fmr1(),
         "Encoders carry the suffix `_fmr1`. The E1 relative zero-effort impostor level (target FMR / e1_genuine) at FMR 1% "
         "(" + ", ".join(f"{e}: {fmr1.impostor_level(e):.4f}" for e in fmr1.encs) + "). " + "Observed values can fall below it, because a reconstructed proxy is not a random impostor capture. " + CI_NOTE),
        ("12b. FMR 1% operating point: trials, genuine acceptance and zero-effort impostor level", fmr1.t_chance(), None),
        ("12c. FMR 1% operating point: sign-resolution reference (eigen_oracle, pre none, lrp_full)", fmr1.t_oracle(),
         ORACLE_NOTE),
        ("12d. FMR 1% operating point: sign-resolution reference after whitening (eigen_oracle, pca128, pca256, fi64, "
         "fi128, fi256, lrp_full)", fmr1.t_oracle_whitened(), ORACLE_NOTE + " " + ORACLE_W_NOTE),
    ]
    if joint:
        parts += [("13. Joint alignment-plus-anchor attack (T1, lrp_full, exact anchors; FMR 0.1%)", t_joint(joint, False),
                   JOINT_NOTE),
                  ("13b. Joint alignment-plus-anchor attack at the FMR 1% operating point", t_joint(joint, True),
                   JOINT_NOTE)]
    out = []
    for title, body, note in parts:
        out += [f"### {title}", "", body, ""]
        if note:
            out += [note, ""]
    return "\n".join(out), p2_lines, p1_lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/isotropy/summary")
    ap.add_argument("--summary-fmr1", default="results/isotropy/summary_fmr1")
    ap.add_argument("--isotropy", default="results/isotropy")
    ap.add_argument("--scale", default="results/scale")
    ap.add_argument("--joint", default="results/joint")
    ap.add_argument("--md", default=None, help="write the full RESULTS.md here instead of printing tables")
    args = ap.parse_args(argv)
    runs = [load_run(d) for d in sorted(Path(args.isotropy).glob("isotropy_d*-*"))]
    scale_runs = [load_run(p.parent) for p in sorted(Path(args.scale).glob("*/scale.csv"))]
    base_runs = [r for r in runs if not r["meta"]["config"]["name"].endswith("_fmr1")]
    fmr1_runs = [r for r in runs if r["meta"]["config"]["name"].endswith("_fmr1")]
    base = Study(Path(args.summary), base_runs)
    fmr1 = Study(Path(args.summary_fmr1), fmr1_runs)
    tables, p2_lines, p1_lines = tables_md(base, fmr1, runs, scale_runs, load_joint(args.joint))
    if args.md is None:
        print(tables)
        return
    joint_meta = [{"meta": json.loads(p.read_text())} for p in sorted(Path(args.joint).glob("joint_d*-*/metadata.json"))]
    date = results_date([*runs, *scale_runs, *joint_meta])
    md = [f"# Isotropy study results ({date})", "",
          f"Data sources: all results are on real embeddings (D1: {D1_SOURCE}; D2: {D2_SOURCE}; the shifted AUX of the D1 "
          f"runs and the real scale-run distractors: the {SECONDARY_SOURCE}), except the table 7 rows "
          "of the synthetic-distractor scale run, whose HNSW distractors are synthetic. "
          "The main tables use the FMR 0.1% operating point; table 12 uses separate runs at FMR 1%. "
          "Values are mean [95% CI]. " + CI_NOTE, "", base.t_headline(), "",
          "Headline notes. r50 is the number of malicious enrollees (exact anchors, lrp_full) at which the mean E1 "
          "(E2) relative success reaches 0.5, for the best decoder (named). E1 is raw-space acceptance: the proxy, "
          "mapped back with the public pre-conditioner inverse, is scored against the victim's fresh captures at "
          "the DEV raw threshold (an unprotected or cross-system endpoint). E2 is the in-system endpoint after key "
          "renewal: the proxy is re-protected with the renewed key and scored against the victim's fresh captures "
          "under that key. `back-map-limited` (lrp_full, exact anchors) means 0.5 was not reached up to r_max and "
          "the key-known level (exact pre-key vector, public LMMSE back-map) has its reported upper bound (wider interval) below "
          "0.5; this bounds only attacks that reconstruct the pre-key vector and use that back-map. For other "
          "schemes and for fresh anchors the key-known level is a reference, not a bound: evaluated collusion "
          "decoders exceed it (summary_fmr1). `>= r_max` means 0.5 was not reached up to r_max. Zero-pair columns use the key-free "
          "methods that run for every pre-conditioner (eigen, wp); the ICA column is separate because FastICA "
          "ran only for m <= 128. The zero-effort impostor level of E1 relative success is target FMR / e1_genuine (table 3c). "
          "Observed values can fall below it, because a reconstructed proxy is not a random impostor capture. "
          "Zero observed successes are reported with the one-sided 95% Clopper-Pearson upper bound, never as "
          "zero risk. " + COLLUSION_SCOPE, "",
          "## Generated tables (scripts/figures/results_tables.py)", "", tables,
          "## Theory checks", "",
          "Proposition 2 (rec_cos_z vs cstar, exact anchors, proj decoder, lrp_full):", *p2_lines, "",
          "Proposition 1 (zero-pair, key-free; abs cos and cos^2 against the uniform-direction chance level, and best "
          "E1 relative against the zero-effort impostor level, on fi, rbig and firbig configurations):",
          *p1_lines, "",
          "Key compromise (in-system acceptance and E2 per scheme, none vs fi128): see table 6.", "",
          "## Deviations from the spec, failed runs and reruns", "",
          "- The code changed after the first runs (projection decoder, abs-cos and cos^2 columns, identity-level "
          "intervals, deduplicated scale distractors with full metadata, FMR 1% configs, `rec_cos_z` and the exact "
          "collusion optimum), and the corpora were corrected afterwards: D1 identity labels re-derived from each "
          "source's native naming by content matching, D2 rebuilt from the development folds of an internal "
          "compilation, finger-disjoint from D1, "
          "and every finger shared with D1 removed from the secondary corpus (shifted AUX, real scale-run distractors). "
          f"All {len(runs) + len(scale_runs)} runs listed in table 1 use the final corpora. The runs of d1_base, "
          "d1_large, d2_fingervit and d2_deepprint (and the scale runs) were rerun on the final code; the d2_ctrlvit "
          "runs were added afterwards on the same corpora, with the same seeds, splits and pre-conditioners as "
          "d2_fingervit. Earlier runs are superseded and are not included.",
          *capped_sizes(scale_runs),
          "- Hashing-scheme collusion numbers hold only for the ridge/proj/mlp decoders evaluated. A stronger IoM "
          "decoder exists in the synthetic proof check (+0.08 cos), so hashing collusion resistance is not claimed.",
          "- FMR 1% runs (`*_fmr1`) are separate runs with the attack threshold at FMR 1%; they are aggregated "
          "separately (`results/isotropy/summary_fmr1`) and are not mixed with the FMR 0.1% runs.",
          "- Skips: see table 8.",
          "- Warnings in stdout logs (POT Sinkhorn did not converge, Wasserstein-Procrustes): "
          + sinkhorn_counts(runs) + ". No Traceback in any log.",
          f"- Failed runs: none among the {len(runs) + len(scale_runs)} runs listed in table 1.",
          "- `configs/scale_d1_base_synthetic.yaml`: provenance: features extracted with gazi_base_extended; the "
          "extraction tool's default model is a different one and was not used.", "",
          "## Data sources", "",
          f"- d1_base, d1_large (and their `_fmr1` runs): {DATASET_LABEL['d1_base']}; their shifted AUX: the "
          f"{SECONDARY_SOURCE} (real).",
          f"- d2_fingervit, d2_deepprint (and their `_fmr1` runs): {DATASET_LABEL['d2_fingervit']}.",
          f"- d2_ctrlvit (and its `_fmr1` run): {DATASET_LABEL['d2_ctrlvit']}.",
          *[f"- {r['meta']['config']['name']}: {scale_label(r)}." for r in scale_runs],
          "- Synthetic data enter only as HNSW distractors in the scale run scale_d1_base_synthetic (table 7); "
          "every other result is on real embeddings.", ""]
    Path(args.md).write_text("\n".join(md))
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
