"""Generate manuscript/generated/numbers.tex: one \\R{key} value per reported result.

Status rules, Clopper-Pearson cells, the wider E1/E2 interval (union of identity-level and seed-level) and the zero-pair selection come from
scripts/figures/results_tables.py (the generator of results/isotropy/RESULTS.md) and
scripts/figures/isotropy_figures.py, loaded with importlib and reused, never re-implemented. The manuscript therefore
agrees with RESULTS.md cell for cell, except where the key scheme asks for another rounding (see FORMATS below).

Key scheme (one family per line):
    util.<enc>.<pre>.<scheme>.<eer|tmr|rank1|m>[.lo|.hi]         base runs, FMR 0.1% (percent, one decimal)
    r50.<enc>.<pre>.<scheme>.<exact|fresh>.<e1|e2>[.dec]         integer, $\\geq{}rmax$ or back-map-limited
    t2.<enc>.<pre>.<scheme>.<e1|e2|insys>[.lo|.hi]               key-known (T2) level, CP95 string if 0 successes
    ptwo.<enc>.<pre>.r<r>.<cos|cstar|ridge>, ptwo.<enc>.maxdev, ptwo.<enc>.none.r<r>.excess   check of the collusion
        optimum (prop:collusion); .cos.lo/.cos.hi and .excess.lo/.excess.hi: Student-t interval over the seed means
    zp.<enc>.<pre>.<common|ica>[.lo|.hi|.lo4 (lower end, four decimals)], zp.<enc>.<pre>.<abscos|uniformabscos|cossq|invm>,
        zp.<enc>.oraclegap, zp.<enc>.impostor
    pr.<enc>.raw, pr.<enc>.<pre>, aniso.<enc>.<pre>, dim.<enc>
    scale.<secondary|synthetic>.<scheme>.n<used>.ef<ef>.<rankoneexact|rankonehnsw|recallten|latency|mb|used|requested>
        rows keyed on the distractor size used (scale.csv n_distractors); .requested holds the size requested
    scale.<run>.<prdiff|prprimary|prdist>, scale.<run>.<loaded|sharedremoved|drawn|removed|available>   distractor rows:
        loaded after the relabel step, removed for a finger shared with D1 (secondary run only), drawn for the largest
        size, removed as near-duplicates of a D1 row, available after that filter
    corpus.<d1|d2>.<images|fingers|tgteligible>, split.<d1|d2>.<dev|tgt|aux>   corpus and split sizes (fingers) from the
        runs' corpus_counts and split_counts; split.<enc>.<dev|tgt|aux>.images.<min|max> over the seeds
    aux.<enc>.samedist   same-distribution AUX rows of the first split (aux_sizes; checked against split_counts)
    aux.<enc>.shifted[.sharedremoved|.dupremoved]   shifted AUX of the D1 runs (secondary corpus): rows, rows of fingers
        shared with D1 removed, near-duplicates removed
    sec.rows.<shared|kept>   secondary-corpus rows of fingers shared with D1, and rows kept after removing them; every
        run that uses the secondary corpus must give the same values
    relabel.*, d2.*, sec.* (other than sec.rows), ovl.*, cs.*, control.*   data-provenance keys from the aggregates in
        results/data/, documented in scripts/manuscript/data_numbers.py; left out only in a copy without results/data/
        (the repository and the public code artifact both hold it); CROSS_CHECKS ties them to the run metadata
        (relabel.d1.* = corpus.d1.*, d2.<images|fingers> = corpus.d2.*, sec.comp.rows.<shared|kept> = sec.rows.*,
        control.audit.nd.<images|fingers>.of = corpus.d2.<images|fingers>: the D2 that the ctrl-vit audit compared)
    fmrone.<enc>.<pre>.<scheme>.exact.<e1|e2>[.dec|.seedmin|.seedmax|.seedcens|.rr], fmrone.<enc>.<pre>.zp,
        fmrone.<enc>.impostor
    gen.<enc>[.sd], fmrone.gen.<enc>[.sd]   raw-space genuine acceptance at the attack threshold (e1_genuine, RESULTS 3c)
    zp.<enc>.<pre>.<common|ica>.abs         absolute proxy acceptance of the selected row (e1_proxy, RESULTS table 5)
    r50.<enc>.<pre>.<scheme>.<exact|fresh>.<e1|e2>.<seedmin|seedmax|seedcens>   per-seed r50 range
    n.<enc>.targets, n.seeds, n.keys
    syn.cross.m<m>, syn.mdiv4.m<m>, syn.ecos.m<m>.q<q>, syn.aux.q<q>   (synthetic simulation, check5b/check5c)
    syn.c4.<m|r|cstar|lrp|biohash|iom.min|iom.max>   synthetic IoM-decoder check (check4, Gaussian isotropic law)
    pr.<enc>.<pre>.int, scale.<run>.<scheme>.n<used>.ef<ef>.mbone   rounded keys (integer PR, MB with one decimal)
    paired utility, measured zero-effort level, sign-resolution reference, absolute acceptance (collect_review):
    pair.<enc>.fi128.<pca128|none>.lrp_full.<eer|tmr>[.lo|.hi]   paired per-seed difference FI128 minus the reference
        (Study.paired_utility): percentage points with two decimals and a typeset minus ($-$), Student-t 95% interval
        over the seeds
    zp.<enc>.fmr.meas, zp.<enc>.impostor.meas, fmrone.<enc>.fmr.meas, fmrone.<enc>.impostor.meas   impostor acceptance
        measured on the target split's impostor pairs at the DEV threshold (percent, three decimals) and the zero-effort
        level computed from it (four decimals), next to the nominal zp.<enc>.impostor
    oracle.<enc>.<e1max|cosmax>, fmrone.oracle.<enc>.e1max, oracle.max.e1, fmrone.oracle.max.e1   eigen_oracle (eigen
        alignment with the true signs from the key; a diagnostic, never key-free), raw space, LRP-full: the largest
        seed-mean E1 relative success and pre-key cosine over AUX and p, and the largest over the encoders; with
        `.abs` the absolute acceptance of that largest entry and with `.cp95` the largest relative CP95 bound of the
        zero-success encoders
    t2.<enc>.<pre>.lrp_full.<e1|e2>.abs   absolute proxy acceptance behind the key-known level (three decimals)
    gen.<enc>.<pre>.<e1|e2>, n.<enc>.victims   genuine acceptance of the T1 victims (r50 is reached at about half of
        it) and their number per seed and key

FORMATS: EER/TMR/Rank-1 in percent with one decimal; rates and cosines with three decimals; the zero-effort impostor
level with four decimals, as in RESULTS.md; r50 is rounded to an integer (RESULTS.md prints one decimal); the counts
n.<enc>.targets, the distractor-row counts scale.<run>.<loaded|sharedremoved|drawn|removed|available>,
scale.<run>.<scheme>.n<used>.ef<ef>.<used|requested> and the corpus, split, aux and sec.rows counts carry a thousands
separator ({,}), like the hand-typed counts of the text.

STATUS RULE: a value that has no point estimate never renders as a bare number, so a sentence cannot print it without
its bound.
    zero successes   a zero-pair, T2 or FMR 1% entry with zero observed successes renders as
                     `0 successes (CP95 $\\leq$ x)`, x the relative one-sided 95% Clopper-Pearson bound; its .lo child
                     renders the same string, its .hi child is the bound x itself (the tables print `0 ($\\leq$x)` from
                     it), and its .abs child renders `0 successes (CP95 $\\leq$ y)` with the absolute bound y.
    not reached      a collusion size not reached up to r_max renders as `$\\geq{}r_max$` or `back-map-limited`; a
                     per-seed range end that is a censored seed (counted at r_max) renders as `$\\geq{}r_max$`, so
                     .seedmin is censored when every seed is and .seedmax when any seed is.
This is the value-level counterpart of the guard in manuscript/main.tex, where \\rr refuses a key without .rr (a
censored or partly censored headline), and of scripts/manuscript/check_manuscript.py, which refuses hand-typed numbers
in the numeric sections. tests/test_manuscript.py checks both parts of the rule on numbers.tex. data_numbers.py
applies the same rule to its zero counts.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/generated/numbers.tex"
PROOFCHECK = ROOT / "docs/theory/proofcheck"
ALL_SCHEMES = ("lrp_full", "lrp_half", "biohash_half", "iom_256")
PRE_CANON = ["none", "pca128", "pca256", "pca128h", "fi64", "fi128", "fi256", "rbig128", "firbig128"]
R_PTWO = (20, 40, 80)

HEADER = (
    "% Generated by scripts/manuscript/make_numbers.py. Do not edit by hand.\n"
    "\\makeatletter\n"
    "\\newcommand{\\R}[1]{\\@ifundefined{res@#1}{\\PackageError{results}{Undefined result key #1}"
    "{Regenerate manuscript/generated/numbers.tex}}{\\@nameuse{res@#1}}}\n"
)


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def canon_pre(pre: str) -> str:
    """Canonical key name of an actual pre-conditioner name."""
    table = [(r"none", "none"), (r"pca(\d+)_a1", r"pca\1"), (r"pca128_a0\.5", "pca128h"), (r"fi(\d+)_p\d+", r"fi\1"),
             (r"rbig128_i2", "rbig128"), (r"firbig128_p\d+_i2", "firbig128")]
    for pat, rep in table:
        if re.fullmatch(pat, pre):
            return re.sub(pat, rep, pre) if "\\" in rep else rep
    raise ValueError(f"unknown pre-conditioner name {pre!r}")


# ---- value formatting ------------------------------------------------------------------------
def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def count(n) -> str:
    """Integer count with the thousands separator of the hand-typed counts in the text, e.g. 11{,}147."""
    return f"{int(n):,}".replace(",", "{,}")


def cp_str(x: float, digits: int = 3) -> str:
    return f"0 successes (CP95 $\\leq$ {x:.{digits}f})"


def pp2(x: float) -> str:
    """A difference of two rates in percentage points, two decimals, with a typeset minus sign; -0.00 reads 0.00."""
    s = f"{100 * x:.2f}"
    return "0.00" if s == "-0.00" else s.replace("-", "$-$")


def seed_ends(g: dict) -> tuple[str, str]:
    """Rendered ends of a per-seed r50 range (Study.seed_range): an integer, or $\\geq{}r_max$ for a censored seed.

    A censored seed counts as r_max, so the minimum is censored only when every seed is and the maximum when any is.
    """
    def end(value: float, censored: bool) -> str:
        return f"$\\geq{{}}{g['rmax']:g}$" if censored else str(int(np.floor(value + 0.5)))
    return end(g["lo"], g["cens_all"]), end(g["hi"], g["cens_any"])


def abs_value(b: dict) -> str:
    """Absolute proxy acceptance of a zero-pair entry (four decimals); with zero successes, its absolute CP95 bound."""
    return cp_str(b["cp_abs"], 4) if b["zero"] else f"{b['proxy']:.4f}"


def parse_ci(s: str) -> tuple[str, str, str]:
    """Split a results_tables.ci() string 'm [lo, hi]' into its three numbers."""
    m = re.fullmatch(r"(-?[\d.]+) \[(-?[\d.]+), (-?[\d.]+)\]", s)
    if not m:
        raise ValueError(f"unexpected interval string {s!r}")
    return m.group(1), m.group(2), m.group(3)


def r50_value(x) -> str:
    if x.status == "reached":
        return str(int(np.floor(x.r50 + 0.5)))
    if x.status == "backmap_limited":
        return "back-map-limited"
    return f"$\\geq{{}}{x.rmax:g}$"


class Collector:
    def __init__(self) -> None:
        self.d: dict[str, str] = {}

    def put(self, key: str, value: str) -> None:
        if re.search(r"\d\.\d", key) or not re.fullmatch(r"[A-Za-z0-9_.]+", key):
            raise ValueError(f"illegal key {key!r}")
        if key in self.d and self.d[key] != value:
            raise ValueError(f"duplicate key {key!r} with different value")
        self.d[key] = value


def collect_isotropy(rt, c: Collector) -> None:
    iso = ROOT / "results/isotropy"
    runs = [rt.load_run(d) for d in sorted(iso.glob("isotropy_d*-*"))]
    base_runs = [r for r in runs if not r["meta"]["config"]["name"].endswith("_fmr1")]
    fmr1_runs = [r for r in runs if r["meta"]["config"]["name"].endswith("_fmr1")]
    base = rt.Study(iso / "summary", base_runs)
    fmr1 = rt.Study(iso / "summary_fmr1", fmr1_runs)

    cfg0 = base_runs[0]["meta"]["config"]
    c.put("n.seeds", str(len(cfg0["split"]["seeds"])))
    c.put("n.keys", str(cfg0["keys_per_seed"]))
    for r in runs:
        cf = r["meta"]["config"]
        assert len(cf["split"]["seeds"]) == len(cfg0["split"]["seeds"]) and cf["keys_per_seed"] == cfg0["keys_per_seed"]
    collect_corpus(c, runs)

    for enc in base.encs:
        sp = base.runs[enc]["spec"]
        raw = sp["raw_centered_normalized"]
        c.put(f"dim.{enc}", str(int(raw["dim"])))
        c.put(f"pr.{enc}.raw", f3(raw["participation_ratio"]))
        c.put(f"pr.{enc}.raw.int", f"{raw['participation_ratio']:.0f}")   # rounded key for prose
        c.put(f"aniso.{enc}.raw", f3(raw["anisotropy_index"]))
        trials = sorted(set(base.zraw(enc).e1_trials.astype(int)))
        assert len(trials) == 1, (enc, trials)
        c.put(f"n.{enc}.targets", count(trials[0]))
        c.put(f"zp.{enc}.impostor", f"{base.impostor_level(enc):.4f}")
        g, gsd = base.genuine(enc)
        c.put(f"gen.{enc}", f3(g))
        c.put(f"gen.{enc}.sd", f3(gsd))
        pres = base.pres(enc)
        assert sorted(canon_pre(p) for p in pres) == sorted(PRE_CANON), (enc, pres)
        for pre in pres:
            cp_ = canon_pre(pre)
            pc = sp["preconditioned"][pre]
            c.put(f"pr.{enc}.{cp_}", f3(pc["participation_ratio"]))
            c.put(f"pr.{enc}.{cp_}.int", f"{pc['participation_ratio']:.0f}")
            c.put(f"aniso.{enc}.{cp_}", f3(pc["anisotropy_index"]))
            for sch in sorted(base.util[base.util.encoder == enc].scheme.unique()):
                u = base.util_row(enc, pre, sch)
                if u is None:
                    continue
                k = f"util.{enc}.{cp_}.{sch}"
                c.put(f"{k}.m", str(int(u.m)))
                for col in ("eer", "tmr", "rank1"):
                    c.put(f"{k}.{col}", pct(u[col]))
                    c.put(f"{k}.{col}.lo", pct(max(u[f"{col}_lo"], 0.0)))
                    c.put(f"{k}.{col}.hi", pct(u[f"{col}_hi"]))
            collect_attacks(base, c, enc, pre, cp_)
            collect_zero_pair(base, c, "zp", enc, pre, cp_)
        collect_ptwo(base, c, enc)
        collect_oracle_gap(base, c, enc)
    collect_review(rt, base, fmr1, c)
    collect_joint(rt, c)
    collect_t2_round_trip(rt, c)

    # FMR 1% operating point, keys under the base encoder name
    for enc1 in fmr1.encs:
        enc = rt.base_name(enc1)
        c.put(f"fmrone.{enc}.impostor", f"{fmr1.impostor_level(enc1):.4f}")
        g, gsd = fmr1.genuine(enc1)
        c.put(f"fmrone.gen.{enc}", f3(g))
        c.put(f"fmrone.gen.{enc}.sd", f3(gsd))
        for pre in fmr1.pres(enc1):
            cp_ = canon_pre(pre)
            for sch in ALL_SCHEMES:
                for ep in ("e1", "e2"):
                    t = fmr1.bd(ep, sch, "exact")
                    if (enc1, pre) in t.index:
                        x = t.loc[(enc1, pre)]
                        c.put(f"fmrone.{enc}.{cp_}.{sch}.exact.{ep}", r50_value(x))
                        c.put(f"fmrone.{enc}.{cp_}.{sch}.exact.{ep}.dec", str(x.decoder))
                        g = fmr1.seed_range(enc1, pre, ep, sch, "exact")   # per-seed range, as for FMR 0.1%
                        seedmin, seedmax = seed_ends(g)
                        c.put(f"fmrone.{enc}.{cp_}.{sch}.exact.{ep}.seedmin", seedmin)
                        c.put(f"fmrone.{enc}.{cp_}.{sch}.exact.{ep}.seedmax", seedmax)
                        c.put(f"fmrone.{enc}.{cp_}.{sch}.exact.{ep}.seedcens",
                              "all" if g["cens_all"] else "some" if g["cens_any"] else "none")
                        if r50_value(x)[0].isdigit() and not g["cens_any"]:   # the \rr guard, as at FMR 0.1%
                            c.put(f"fmrone.{enc}.{cp_}.{sch}.exact.{ep}.rr", "ok")
                        put_t2(fmr1, c, f"fmrone.{enc}.{cp_}.{sch}.t2", enc1, pre, sch, ep)
            b = fmr1.best_zero_pair(enc1, pre, rt.COMMON)
            if b is not None:
                put_zp_value(rt, c, f"fmrone.{enc}.{cp_}.zp", b)

    # scale runs
    scale_runs = [rt.load_run(p.parent) for p in sorted((ROOT / "results/scale").glob("*/scale.csv"))]
    collect_scale(rt, c, scale_runs)


def put_t2(study, c: Collector, key: str, enc: str, pre: str, sch: str, ep: str) -> None:
    """Key-known (T2) level: mean with the wider (identity-level or seed-level) 95% interval, or the CP95 string for zero successes."""
    x = study.ceiling_row(enc, pre, sch, ep)
    if x is None:
        return
    cp = study.zero_t2(enc, pre, sch, x.method, ep)
    if cp is not None:   # status rule: no bare zero under a zero-success entry; .hi is the bound
        c.put(key + f".{ep}", cp_str(cp))
        c.put(key + f".{ep}.lo", cp_str(cp))
        c.put(key + f".{ep}.hi", f3(cp))
        return
    m, lo, hi = parse_ci(study.cons_str(x, f"{ep}_rel") if hasattr(study, "cons_str") else _cons(x, f"{ep}_rel"))
    c.put(key + f".{ep}", m)
    c.put(key + f".{ep}.lo", lo)
    c.put(key + f".{ep}.hi", hi)


_CONS = None


def _cons(x, col: str) -> str:
    return _CONS(x, col)


def collect_attacks(base, c: Collector, enc: str, pre: str, cp_: str) -> None:
    for sch in ALL_SCHEMES:
        for var in ("exact", "fresh"):
            for ep in ("e1", "e2"):
                t = base.bd(ep, sch, var)
                if (enc, pre) not in t.index:
                    continue
                x = t.loc[(enc, pre)]
                c.put(f"r50.{enc}.{cp_}.{sch}.{var}.{ep}", r50_value(x))
                c.put(f"r50.{enc}.{cp_}.{sch}.{var}.{ep}.dec", str(x.decoder))
                g = base.seed_range(enc, pre, ep, sch, var)
                kk = f"r50.{enc}.{cp_}.{sch}.{var}.{ep}"
                seedmin, seedmax = seed_ends(g)   # status rule: a censored end reads $\geq{}r_max$
                c.put(f"{kk}.seedmin", seedmin)
                c.put(f"{kk}.seedmax", seedmax)
                if r50_value(x)[0].isdigit() and not g["cens_any"]:  # \rr (main.tex) is guarded on this key: censored, back-map-limited or partly censored headlines get none
                    c.put(f"{kk}.rr", "ok")
                c.put(f"{kk}.seedcens", "all" if g["cens_all"] else "some" if g["cens_any"] else "none")
                if x.status == "reached" and np.isfinite(x.ceiling) and x.ceiling < 0.5:
                    # observed r50 although the T2 level is below 0.5: the T2 level is not an upper bound here
                    c.put(f"r50.{enc}.{cp_}.{sch}.{var}.{ep}.t2below", "yes")
        for ep in ("e1", "e2"):
            put_t2(base, c, f"t2.{enc}.{cp_}.{sch}", enc, pre, sch, ep)
        d = base.post[(base.post.encoder == enc) & (base.post.pre == pre) & (base.post.scheme == sch)]
        if len(d):
            x = d.sort_values(["in_system_accept", "method"], ascending=[False, True], kind="stable").iloc[0]
            c.put(f"t2.{enc}.{cp_}.{sch}.insys", f3(x.in_system_accept))
            c.put(f"t2.{enc}.{cp_}.{sch}.insys.lo", f3(max(x.in_system_accept_lo, 0.0)))
            c.put(f"t2.{enc}.{cp_}.{sch}.insys.hi", f3(x.in_system_accept_hi))


def put_zp_value(rt, c: Collector, key: str, b: dict) -> None:
    """Zero-pair best E1 relative: CP95 string if the best cell has zero successes, else mean with the wider 95% interval."""
    if b["zero"]:   # status rule: no bare zero under a zero-success entry; .hi is the bound
        c.put(key, cp_str(b["cp_rel"]))
        c.put(key + ".lo", cp_str(b["cp_rel"]))
        c.put(key + ".hi", f3(b["cp_rel"]))
        return
    m, lo, hi = parse_ci(rt.cons(b["row"], "best_e1_rel"))
    if float(m) == 0 and float(b["row"]["best_e1_rel"]) > 0:   # observed successes that round to 0.000
        m = "$<0.001$"
    c.put(key, m)
    c.put(key + ".lo", lo)
    c.put(key + ".hi", hi)
    lo_raw, _ = rt.wider_interval(b["row"], "best_e1_rel")
    c.put(key + ".lo4", f"{max(lo_raw, 0.0):.4f}")   # four decimals, comparable with the zero-effort level


def collect_zero_pair(base, c: Collector, fam: str, enc: str, pre: str, cp_: str) -> None:
    rt = _RT
    b = base.best_zero_pair(enc, pre, rt.COMMON)
    if b is not None:
        put_zp_value(rt, c, f"{fam}.{enc}.{cp_}.common", b)
        c.put(f"{fam}.{enc}.{cp_}.common.abs", abs_value(b))
        r = b["row"]
        m = int(base.util_row(enc, pre, rt.SCHEME).m)
        ua = float(rt.uniform_abs_cos_mean(m))
        c.put(f"{fam}.{enc}.{cp_}.abscos", f3(r.best_abs_cos_z))
        c.put(f"{fam}.{enc}.{cp_}.uniformabscos", f3(ua))
        c.put(f"{fam}.{enc}.{cp_}.cossq", f"{r.best_cos2_z:.4f}")
        c.put(f"{fam}.{enc}.{cp_}.invm", f"{1.0 / m:.4f}")
    ica = base.best_zero_pair(enc, pre, ("ica",))
    if ica is not None:
        put_zp_value(rt, c, f"{fam}.{enc}.{cp_}.ica", ica)
        c.put(f"{fam}.{enc}.{cp_}.ica.abs", abs_value(ica))


def collect_oracle_gap(base, c: Collector, enc: str) -> None:
    """eigen_oracle - eigen E1 at the best eigen (AUX, p), pre none; selection Study.eigen_best, as in Study.t_zeropair."""
    none = base.pre(enc, "none")
    p, aux = base.eigen_best(enc, none)
    c.put(f"zp.{enc}.oraclegap", f3(base.oracle_gap(enc, none, p, aux)))


def collect_review(rt, base, fmr1, c: Collector) -> None:
    """Paired utility, measured zero-effort level, oracle and absolute-acceptance keys (families in the module docstring).
    Every value comes from a Study method of results_tables.py, which also prints them in RESULTS.md."""
    for enc in base.encs:
        fi, pca, none = base.pre(enc, "fi128"), base.pre(enc, "pca128_a1"), base.pre(enc, "none")
        for ref_key, ref in (("pca128", pca), ("none", none)):
            for col in ("eer", "tmr"):
                g = base.paired_utility(enc, fi, ref, col)
                k = f"pair.{enc}.fi128.{ref_key}.lrp_full.{col}"
                c.put(k, pp2(g["mean"]))
                c.put(k + ".lo", pp2(g["lo"]))
                c.put(k + ".hi", pp2(g["hi"]))
        c.put(f"n.{enc}.victims", count(base.victims(enc)))
        for pre in base.pres(enc):
            cp_ = canon_pre(pre)
            for ep in ("e1", "e2"):
                c.put(f"gen.{enc}.{cp_}.{ep}", f3(base.victim_genuine(enc, pre, ep)))
                put_t2_abs(base, c, f"t2.{enc}.{cp_}.lrp_full.{ep}.abs", enc, pre, ep)
    for fam, study in (("zp", base), ("fmrone", fmr1)):
        for enc1 in study.encs:
            enc = rt.base_name(enc1)
            c.put(f"{fam}.{enc}.fmr.meas", f"{100 * study.measured_fmr(enc1):.3f}")
            c.put(f"{fam}.{enc}.impostor.meas", f"{study.measured_impostor_level(enc1):.4f}")
    for fam, study in (("oracle", base), ("fmrone.oracle", fmr1)):
        maxima, bounds = [], []
        for enc1 in study.encs:
            enc = rt.base_name(enc1)
            o = study.oracle_max(enc1)
            maxima.append((o["e1"], o["e1_abs_max"]))
            if o["zero"]:   # status rule: no bare zero; .hi is the bound
                c.put(f"{fam}.{enc}.e1max", cp_str(o["cp_rel"]))
                c.put(f"{fam}.{enc}.e1max.hi", f3(o["cp_rel"]))
                bounds.append(o["cp_rel"])
            else:
                c.put(f"{fam}.{enc}.e1max", f3(o["e1"]) if round(o["e1"], 3) > 0 else "$<0.001$")
            if fam == "oracle":
                c.put(f"oracle.{enc}.cosmax", f3(o["cos"]))
        # each maximum on its own: the largest relative success and the largest absolute acceptance can come from
        # different encoders (relative = absolute / genuine acceptance, which differs between encoders)
        c.put(f"{fam}.max.e1", f3(max(rel for rel, _ in maxima)))
        c.put(f"{fam}.max.e1.abs", f3(max(ab for _, ab in maxima)))
        if bounds:   # the zero-success entries' largest relative CP95 bound, next to the point-estimate maximum
            c.put(f"{fam}.max.e1.cp95", f3(max(bounds)))
    collect_oracle_whitened(rt, base, fmr1, c)


def collect_oracle_whitened(rt, base, fmr1, c: Collector) -> None:
    """Sign-resolution reference after whitening: eigen_oracle of LRP-full under every whitened
    pre-conditioner of the paper (PCA128, PCA256, FI64, FI128, FI256) at E1 and E2, and the raw E2 reference (equal to the raw E1 one, since without a pre-conditioner
    the E2 proxy is the normalised estimate). Each maximum over encoders, whitenings and (for .abs) endpoints stands on
    its own; zero successes render with the relative CP95 bound (status rule), and .hi holds the bound."""
    for fam, study in (("oracle", base), ("fmrone.oracle", fmr1)):
        rel, absolute, bounds = {"e1": [], "e2": []}, [], []
        for enc1 in study.encs:
            enc = rt.base_name(enc1)
            for label, prefix in (("none", "none"), *rt.WHITENED_PRES):
                for ep in ("e1", "e2"):
                    if label == "none" and ep == "e1":
                        continue   # the raw E1 reference keeps its key <fam>.<enc>.e1max
                    o = study.oracle_endpoint(enc1, prefix, ep)
                    key = f"{fam}.{enc}.{label}.{ep}max"
                    if o["zero"]:
                        c.put(key, cp_str(o["cp_rel"]))
                        c.put(key + ".hi", f3(o["cp_rel"]))
                    else:
                        c.put(key, f3(o["rel"]) if round(o["rel"], 3) > 0 else "$<0.001$")
                    if label == "none":
                        continue
                    absolute.append(o["abs_max"])
                    if o["zero"]:
                        bounds.append(o["cp_rel"])
                    else:
                        rel[ep].append(o["rel"])
                    if fam == "oracle":
                        c.put(f"oracle.{enc}.{label}.cosmax", f3(o["cos"]))
        for ep in ("e1", "e2"):
            c.put(f"{fam}.w.max.{ep}", f3(max(rel[ep])))
        c.put(f"{fam}.w.max.abs", f3(max(absolute)))
        if bounds:
            c.put(f"{fam}.w.max.cp95", f3(max(bounds)))


def _r50_key(x: tuple) -> str:
    """r50 of the joint stage with one decimal, so that no displayed difference comes from rounding, or
    $\\geq{}r_max$ when the curve never reaches 0.5."""
    v, cens, rmax = x
    return f"$\\geq{{}}{rmax:g}$" if cens else f"{v:.1f}"


JOINT_METHOD_LAB = {"proj": "proj", "align_anchor": "J1", "joint": "J2"}


def put_joint_small_r(rt, c: Collector, df, pre: str, key: str) -> None:
    """Per r = 1, 2, 5: E2 relative and absolute success and pre-key cosine of the projection decoder (`.r<r>.proj`) and of
    the best evaluated attack (`.r<r>.best`, with `.method`: proj, J1 or J2), the seed-mean entries of the envelope; when
    no row at r has a success, the relative CP95 bound of those rows (status rule)."""
    per = rt.joint_seed_means(df, pre, "e2")
    for who, methods in (("proj", ("proj",)), ("best", rt.JOINT_METHODS)):
        env = rt.joint_envelope(per, "e2", methods).set_index("r")
        for r in rt.JOINT_SMALL_R:
            e = env.loc[r]
            rows = df[(df.pre == pre) & (df.scheme == "lrp_full") & df.method.isin(methods) & (df.r == r)]
            k = f"{key}.r{r}.{who}"
            if (rows["e2_successes"] == 0).all():   # status rule: the relative and the absolute bound, no bare zero
                c.put(k, cp_str(float((rows["e2_cp_upper"] / rows["e2_genuine"]).max())))
                c.put(k + ".abs", cp_str(float(rows["e2_cp_upper"].max()), 4))
            else:
                c.put(k, f3(e.rel) if round(e.rel, 3) > 0 else "$<0.001$")
                c.put(k + ".abs", "$<0.001$" if f"{e['abs']:.3f}" == "0.000" and e["abs"] > 0 else f"{e['abs']:.3f}")
            # cosine and method are siblings, not children, of a success entry that may be a zero count
            c.put(f"{key}.r{r}.cos.{who}", f3(e.cos))
            if who == "best":
                c.put(f"{key}.r{r}.method", JOINT_METHOD_LAB[e.method])


def collect_t2_round_trip(rt, c: Collector) -> None:
    """The T2 method behind the FI128 E2 levels (one for all five encoders), the smallest analytic (exact-inverse) E2
    level over the encoders, and the largest net decision difference of a run (accepted proxy minus accepted genuine
    pairs) of that method: the empirical effect of the E2 round trip's offset on acceptance."""
    iso = ROOT / "results/isotropy"
    runs = [rt.load_run(d) for d in sorted(iso.glob("isotropy_d*-*"))]
    base = rt.Study(iso / "summary", [r for r in runs if not r["meta"]["config"]["name"].endswith("_fmr1")])
    methods, levels, nets = set(), [], []
    for enc in base.encs:
        pre = base.pre(enc, "fi128")
        methods.add(str(base.ceiling_row(enc, pre, "lrp_full", "e2").method))
        raw = base.praw(enc)
        a = raw[(raw.pre == pre) & (raw.scheme == "lrp_full") & (raw.method == "analytic")]
        levels.append(a.groupby("seed").e2_rel.mean().mean())
        nets.append(int(((a.e2_proxy - a.e2_genuine) * a.e2_pair_trials).round().abs().max()))
    if len(methods) != 1:
        raise ValueError(f"FI128 E2 levels come from several T2 methods: {methods}")
    c.put("t2.fi128.e2.method", methods.pop())
    c.put("t2.fi128.e2.analytic.min", f3(min(levels)))
    c.put("t2.fi128.e2.analytic.netmax", str(max(nets)))


def _range_end(v: float, cens: bool, rmax: float) -> str:
    return f"$\\geq{{}}{rmax:g}$" if cens else f"{v:.1f}"


def collect_joint(rt, c: Collector, root=None) -> None:
    """Joint alignment-plus-anchor stage (results/joint/): per encoder and pre-conditioner, r50 at E2 of the anchor-only
    projection decoder (`.e2.proj`) and of the best evaluated attack (`.e2.best`, the envelope over proj, J1 and J2
    selected with ground truth), and the best attack's acceptance at r = 1, 2, 5 (`.e2.small` with `.small.cos`; a zero
    count renders with its relative CP95 bound); summaries for the text: the largest r50 reduction at FMR 0.1%
    (`joint.drop.max.from/.to`), the largest reduction after whitening (`<fam>.whitened.maxdrop`, one decimal), the best
    acceptance at r = 1 in the raw D1 spaces at FMR 1% with the largest cosine of those entries, and the largest
    small-budget acceptance at FMR 0.1% (`joint.small.max`, `.abs`)."""
    joint = rt.load_joint(root or ROOT / "results/joint")
    drops, d1r1, small = {}, [], []
    for name in sorted(joint):
        fam = "fmrone.joint" if name.endswith("_fmr1") else "joint"
        enc = rt.base_name(name)
        for label, prefix in rt.JOINT_PRES:
            cell = rt.joint_cell(joint[name], prefix, "e2")
            c.put(f"{fam}.{enc}.{label}.e2.proj", _r50_key(cell["proj"]))
            c.put(f"{fam}.{enc}.{label}.e2.best", _r50_key(cell["best"]))
            pre = rt.joint_pre(joint[name], prefix)
            ranges = {"proj": rt.joint_seed_r50_range(joint[name], pre, "e2", ("proj",)), "best": cell["seeds"]}
            for who, g in ranges.items():
                c.put(f"{fam}.{enc}.{label}.e2.{who}.seedmin", _range_end(g["lo"], g["cens_all"], g["rmax"]))
                c.put(f"{fam}.{enc}.{label}.e2.{who}.seedmax", _range_end(g["hi"], g["cens_any"], g["rmax"]))
            # a reduction is resolved only when the per-seed ranges do not overlap (the paper's rule)
            resolved = (not ranges["best"]["cens_any"] and not ranges["proj"]["cens_all"]
                        and ranges["best"]["hi"] < ranges["proj"]["lo"])
            c.put(f"{fam}.{enc}.{label}.e2.resolved", "yes" if resolved else "no")
            # robust: resolved also with the envelope's per-r selections applied to every seed (no per-seed re-selection)
            fixed = rt.joint_seed_r50_range_fixed(joint[name], pre, "e2", cell["env"])
            robust = resolved and not fixed["cens_any"] and fixed["hi"] < ranges["proj"]["lo"]
            c.put(f"{fam}.{enc}.{label}.e2.robust", "yes" if robust else "no")
            put_joint_small_r(rt, c, joint[name], pre, f"{fam}.{enc}.{label}.e2")
            s = cell["small"]
            if s["zero"]:
                c.put(f"{fam}.{enc}.{label}.e2.small", cp_str(s["cp_rel"]))
            else:
                c.put(f"{fam}.{enc}.{label}.e2.small", f3(s["rel"]) if round(s["rel"], 3) > 0 else "$<0.001$")
            c.put(f"{fam}.{enc}.{label}.e2.small.cos", f3(s["cos"]))
            if not (cell["proj"][1] or cell["best"][1]):
                drops[(fam, enc, label)] = cell["proj"][0] - cell["best"][0]
            env = cell["env"]
            if fam == "joint":
                small.append(env[env.r.isin(rt.JOINT_SMALL_R)])
            if fam == "fmrone.joint" and label == "none" and enc in ("d1_base", "d1_large"):
                d1r1.append(env[env.r == 1].iloc[0])
    top = max((k for k in drops if k[0] == "joint"), key=drops.get)
    c.put("joint.drop.max.from", c.d[f"joint.{top[1]}.{top[2]}.e2.proj"])
    c.put("joint.drop.max.to", c.d[f"joint.{top[1]}.{top[2]}.e2.best"])
    for fam in ("joint", "fmrone.joint"):
        c.put(f"{fam}.whitened.maxdrop", f"{max(v for k, v in drops.items() if k[0] == fam and k[2] != 'none'):.1f}")
    c.put("fmrone.joint.d1fi.maxdrop", f"{max(v for k, v in drops.items() if k[0] == 'fmrone.joint' and k[1] in ('d1_base', 'd1_large') and k[2] in ('fi64', 'fi128')):.1f}")
    c.put("fmrone.joint.d1raw.r1.max", f3(max(x.rel for x in d1r1)))
    c.put("fmrone.joint.d1raw.r1.cosmax", f3(max(x.cos for x in d1r1)))
    s = pd.concat(small)
    c.put("joint.small.max", f3(s.rel.max()))
    c.put("joint.small.max.abs", f3(s["abs"].max()))


def put_t2_abs(study, c: Collector, key: str, enc: str, pre: str, ep: str) -> None:
    """Absolute proxy acceptance behind the key-known level of LRP-full; zero successes render with the absolute
    CP95 bound (the status rule of the zero-pair .abs keys)."""
    x = study.ceiling_row(enc, pre, "lrp_full", ep)
    if x is None:
        return
    if study.zero_t2(enc, pre, "lrp_full", x.method, ep) is not None:
        raw = study.praw(enc)
        sub = raw[(raw.pre == pre) & (raw.scheme == "lrp_full") & (raw.method == x.method)]
        c.put(key, cp_str(float(sub[f"{ep}_cp_upper"].max()), 4))
        return
    c.put(key, f3(study.ceiling_abs(enc, pre, "lrp_full", ep)))


def collect_ptwo(base, c: Collector, enc: str) -> None:
    """Check of Proposition 4 (exact collusion optimum), exact anchors, proj decoder, lrp_full (RESULTS table 9)."""
    devs = []
    for pref in ("pca128_a1", "fi128", "none"):
        pre = base.pre(enc, pref)
        cp_ = canon_pre(pre)
        a, ar = base.anchor_rows(enc, pre), base.anchor_rows(enc, pre, decoder="ridge")
        for r in R_PTWO:
            x, xr = a[a.r == r].iloc[0], ar[ar.r == r].iloc[0]
            k = f"ptwo.{enc}.{cp_}.r{r}"
            c.put(f"{k}.cos", f3(x.rec_cos_z))
            # Student-t interval over the seed means (RESULTS.md table 4a, same summary row)
            c.put(f"{k}.cos.lo", f3(x.rec_cos_z_lo))
            c.put(f"{k}.cos.hi", f3(x.rec_cos_z_hi))
            c.put(f"{k}.cstar", f3(x.cstar))
            c.put(f"{k}.ridge", f3(xr.rec_cos_z))
            if pref == "none":
                c.put(f"{k}.excess", f3(x.rec_cos_z - x.cstar))
                c.put(f"{k}.excess.lo", f3(x.rec_cos_z_lo - x.cstar))   # c*(r,d) is a constant
                c.put(f"{k}.excess.hi", f3(x.rec_cos_z_hi - x.cstar))
            else:
                dev = abs(x.rec_cos_z - x.cstar)   # unrounded, as in RESULTS.md table 9
                c.put(f"{k}.dev", f3(dev))
                devs.append(dev)
    c.put(f"ptwo.{enc}.maxdev", f3(max(devs)))


def collect_scale(rt, c: Collector, scale_runs: list[dict]) -> None:
    """Scale-run keys. Rows are keyed on the distractor size used (n_distractors), not on the size requested, so a
    capped size never appears under the requested number; `.requested` keeps the requested size."""
    for r in scale_runs:
        name = r["meta"]["config"]["name"]
        run = {"scale_d1_base_secondary": "secondary", "scale_d1_base_synthetic": "synthetic"}[name]
        rt.scale_label(r)   # raises on an unlabeled scale run
        m = r["meta"]
        pr_t = m["spectrum_primary_dev"]["participation_ratio"]
        pr_d = m["spectrum_distractors"]["participation_ratio"]
        c.put(f"scale.{run}.prprimary", f"{pr_t:.1f}")
        c.put(f"scale.{run}.prdist", f"{pr_d:.1f}")
        c.put(f"scale.{run}.prdiff", f"{100 * abs(pr_d - pr_t) / pr_t:.1f}")
        c.put(f"scale.{run}.loaded", count(m["distractors_loaded"]))
        c.put(f"scale.{run}.drawn", count(m["distractors_available_before"]))
        c.put(f"scale.{run}.removed", count(m["distractors_removed"]))
        c.put(f"scale.{run}.available", count(m["distractors_available_after"]))
        shared = m["secondary_removed_shared_identity"]
        if shared is not None:   # secondary-corpus distractors: rows of fingers shared with D1 removed before the draw
            c.put(f"scale.{run}.sharedremoved", count(shared))
            c.put("sec.rows.shared", count(shared))
            c.put("sec.rows.kept", count(m["distractors_loaded"] - shared))
        df = pd.read_csv(r["dir"] / "scale.csv")
        dup = df.duplicated(["scheme", "n_distractors", "ef_search"])
        if dup.any():
            raise ValueError(f"scale run {r['dir'].name}: several rows with one size used per scheme and ef: "
                             f"{df[dup][['scheme', 'n_distractors', 'ef_search']].values.tolist()}")
        for _, x in df.iterrows():
            k = f"scale.{run}.{x.scheme}.n{int(x.n_distractors)}.ef{int(x.ef_search)}"
            c.put(f"{k}.rankoneexact", f3(x.rank1_exact))
            c.put(f"{k}.rankonehnsw", f3(x.rank1_hnsw))
            c.put(f"{k}.recallten", f3(x.recall10_tie_aware))
            c.put(f"{k}.latency", f3(x.latency_ms_per_query))
            c.put(f"{k}.mb", f3(x.index_serialized_mb))
            c.put(f"{k}.mbone", f"{x.index_serialized_mb:.1f}")   # rounded key for prose
            c.put(f"{k}.used", count(x.n_distractors))
            c.put(f"{k}.requested", count(x.n_distractors_requested))


def collect_corpus(c: Collector, runs: list[dict]) -> None:
    """Corpus, split and AUX sizes from the isotropy runs' metadata (corpus_counts, split_counts, aux_sizes).

    Corpus and split-finger keys are per corpus (d1, d2) and must agree over every run and seed of the corpus (the
    Collector refuses a second, different value); split-image ranges over the seeds are per encoder (the FMR 1% runs
    use the same seeds as their base runs, so they must give the same ranges). The shifted AUX (secondary corpus, D1 runs
    only) gives its rows, the rows of fingers shared with D1 removed and the near-duplicates removed; the rows kept
    after the shared-finger removal are the shifted rows plus the near-duplicates.
    """
    for r in runs:
        m = r["meta"]
        enc = m["config"]["name"].removesuffix("_fmr1")
        corpus = enc.split("_")[0]
        cc = m["corpus_counts"]
        c.put(f"corpus.{corpus}.images", count(cc["images"]))
        c.put(f"corpus.{corpus}.fingers", count(cc["identities"]))
        c.put(f"corpus.{corpus}.tgteligible", count(cc["identities_ge_min_tgt_images"]))
        seeds = list(m["split_counts"].values())
        for part in ("dev", "tgt", "aux"):
            for s in seeds:
                c.put(f"split.{corpus}.{part}", count(s[f"{part}_identities"]))
            images = [s[f"{part}_images"] for s in seeds]
            c.put(f"split.{enc}.{part}.images.min", count(min(images)))
            c.put(f"split.{enc}.{part}.images.max", count(max(images)))
        same = m["aux_sizes"]["same_dist"]   # the runner records the AUX rows of the first split
        if same != seeds[0]["aux_images"]:
            raise ValueError(f"{m['config']['name']}: aux_sizes.same_dist {same} is not the first split's "
                             f"{seeds[0]['aux_images']} AUX images")
        c.put(f"aux.{enc}.samedist", count(same))
        shifted = m["aux_sizes"].get("shifted")
        if shifted is not None:
            shared, dup = m["secondary_removed_shared_identity"], m["shifted_aux_removed"]
            c.put(f"aux.{enc}.shifted", count(shifted))
            c.put(f"aux.{enc}.shifted.sharedremoved", count(shared))
            c.put(f"aux.{enc}.shifted.dupremoved", count(dup))
            c.put("sec.rows.shared", count(shared))
            c.put("sec.rows.kept", count(shifted + dup))


# ---- synthetic simulation outputs (docs/theory/proofcheck) -------------------------------------
LINE_5B = re.compile(
    r"m=\s*(\d+) n_dev=\s*(\d+)\s+Ecos at N/n_dev=\[((?:np\.int64\(\d+\),? ?)+)\]: \[([\d.\s]+)\]\s+"
    r"crossing N/n_dev ~ ([\d.]+)\s+\(m/4 = (\d+)\)")
LINE_5C = re.compile(r"n_aux/n_dev=\s*(\d+): E cos \(oracle sign, N=(\d+) n_dev\) = ([\d.]+)")


def collect_synthetic(c: Collector) -> None:
    """Parse check5b/check5c strictly: any unexpected non-empty line is an error."""
    for line in (PROOFCHECK / "check5b_crossing_vs_m.out").read_text().splitlines():
        if not line.strip():
            continue
        m = LINE_5B.fullmatch(line.strip())
        if not m:
            raise ValueError(f"unexpected line in check5b_crossing_vs_m.out: {line!r}")
        mm = int(m.group(1))
        qs = [int(q) for q in re.findall(r"np\.int64\((\d+)\)", m.group(3))]
        vals = m.group(4).split()
        assert len(qs) == len(vals), line
        c.put(f"syn.cross.m{mm}", m.group(5))
        c.put(f"syn.mdiv4.m{mm}", m.group(6))
        c.put(f"syn.ndev.m{mm}", m.group(2))
        for q, v in zip(qs, vals):
            c.put(f"syn.ecos.m{mm}.q{q}", f3(float(v)))
    n_multiples: set[int] = set()
    for line in (PROOFCHECK / "check5c_aux_size.out").read_text().splitlines():
        if not line.strip():
            continue
        m = LINE_5C.fullmatch(line.strip())
        if not m:
            raise ValueError(f"unexpected line in check5c_aux_size.out: {line!r}")
        c.put(f"syn.aux.q{int(m.group(1))}", f3(float(m.group(3))))
        n_multiples.add(int(m.group(2)))
    assert len(n_multiples) == 1, n_multiples  # one N/n_dev for the whole auxiliary-size block
    c.put("syn.aux.N", str(n_multiples.pop()))


LINE_4_HEAD = re.compile(r"c\*\(r=(\d+), m=(\d+)\) = ([\d.]+)\s+sqrt\(r/m\) = [\d.]+\s+trials=\d+, victims/trial=\d+")
LINE_4_ROW = re.compile(r"gauss_iso\s+(.+?)\s{2,}(.+?)\s{2,}([\d.]+)\s+[\d.]+\s+-?[\d.]+")


def collect_check4(c: Collector) -> None:
    """Synthetic stronger-decoder check (Gaussian isotropic law only): LRP projection, BioHashing and GRP-IoM decoders."""
    lines = (PROOFCHECK / "check4_prop2_mc.out").read_text().splitlines()
    head = LINE_4_HEAD.fullmatch(lines[0].strip())
    if not head:
        raise ValueError(f"unexpected header in check4_prop2_mc.out: {lines[0]!r}")
    c.put("syn.c4.r", head.group(1))
    c.put("syn.c4.m", head.group(2))
    c.put("syn.c4.cstar", f"{float(head.group(3)):.4f}")
    rows = {}
    for line in lines[2:]:
        m = LINE_4_ROW.match(line.strip())
        if m:
            rows[(m.group(1).strip(), m.group(2).strip())] = float(m.group(3))
    lrp = rows[("LRP-full", "projection")]
    bio = [v for (s, _), v in rows.items() if s.startswith("BioHash")]
    iom = [v for (s, _), v in rows.items() if s.startswith("GRP-IoM")]
    assert len(bio) == 2 and len(iom) == 2, rows
    assert f"{min(bio):.2f}" == f"{max(bio):.2f}", bio    # one rounded value stands for both BioHashing decoders
    c.put("syn.c4.lrp", f"{lrp:.4f}")
    c.put("syn.c4.biohash", f"{max(bio):.2f}")
    c.put("syn.c4.iom.min", f"{min(iom):.2f}")
    c.put("syn.c4.iom.max", f"{max(iom):.2f}")


# Data keys that must equal the run-metadata keys written before them by collect_isotropy: the D1 relabel against the
# D1 corpus that the runs loaded, the D2 extraction against the D2 corpus, and the secondary-corpus composition
# against the rows that the runs removed as shared with D1 and kept.
CROSS_CHECKS = (("relabel.d1.placed", "corpus.d1.images"), ("relabel.d1.fingers", "corpus.d1.fingers"),
                ("d2.images", "corpus.d2.images"), ("d2.fingers", "corpus.d2.fingers"),
                ("control.audit.nd.images.of", "corpus.d2.images"), ("control.audit.nd.fingers.of", "corpus.d2.fingers"),
                ("sec.comp.rows.shared", "sec.rows.shared"), ("sec.comp.rows.kept", "sec.rows.kept"))


def cross_check(d: dict[str, str]) -> None:
    """Refuse data keys that disagree with the run metadata (CROSS_CHECKS); a missing key is refused as well."""
    for data_key, run_key in CROSS_CHECKS:
        missing = [k for k in (data_key, run_key) if k not in d]
        if missing:
            raise ValueError(f"cross-check {data_key} = {run_key}: {', '.join(missing)} missing")
        if d[data_key] != d[run_key]:
            raise ValueError(f"cross-check failed: {data_key} = {d[data_key]}, but {run_key} = {d[run_key]}")


def collect_data(c: Collector, root: Path = ROOT) -> bool:
    """Data-provenance keys from <root>/results/data/ (scripts/manuscript/data_numbers.py documents them).

    The repository and the public code artifact both hold results/data/; in a copy without that directory the keys are
    left out and the function returns False. With it, every aggregate that data_numbers reads must exist, and the
    keys must pass cross_check against the run-metadata keys already in `c`.
    """
    if not (Path(root) / "results/data").is_dir():
        return False
    load_module("data_numbers", "scripts/manuscript/data_numbers.py").collect(c.put, root)
    cross_check(c.d)
    return True


_RT = None


def build() -> dict[str, str]:
    global _RT, _CONS
    _RT = load_module("results_tables", "scripts/figures/results_tables.py")
    _CONS = _RT.cons
    c = Collector()
    collect_isotropy(_RT, c)
    collect_synthetic(c)
    collect_check4(c)
    if not collect_data(c):
        print("results/data/ not found: the data-provenance keys are left out")
    return c.d


def render(d: dict[str, str]) -> str:
    body = "".join(f"\\@namedef{{res@{k}}}{{{d[k]}}}\n" for k in sorted(d))
    return HEADER + body + "\\makeatother\n"


def main() -> int:
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(d), encoding="utf-8")
    fam: dict[str, int] = {}
    for k in d:
        fam[k.split(".")[0]] = fam.get(k.split(".")[0], 0) + 1
    print(f"wrote {OUT} with {len(d)} keys: " + ", ".join(f"{f}={n}" for f, n in sorted(fam.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
