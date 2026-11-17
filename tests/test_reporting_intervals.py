"""Reporting rules of the results record: the wider E1/E2 interval and the per-seed r50 range."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/figures"))
_spec = importlib.util.spec_from_file_location("results_tables_t", ROOT / "scripts/figures/results_tables.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


def _frame():
    return pd.DataFrame({
        "x_rel": [0.30, 0.50, 0.20],
        "x_rel_lo": [0.25, 0.40, 0.18],        # seed-level interval
        "x_rel_hi": [0.35, 0.60, 0.22],
        "x_rel_cons_lo": [0.28, 0.45, 0.05],   # identity-level interval
        "x_rel_cons_hi": [0.40, 0.55, 0.21],
    })


def test_wider_interval_is_elementwise_union():
    lo, hi = rt.wider_interval(_frame(), "x_rel")
    assert list(lo) == pytest.approx([0.25, 0.40, 0.05])
    assert list(hi) == pytest.approx([0.40, 0.60, 0.22])


def test_wider_interval_on_row_and_nonfinite_component():
    df = _frame()
    df.loc[0, "x_rel_lo"] = np.nan
    df.loc[0, "x_rel_hi"] = np.nan
    lo, hi = rt.wider_interval(df.iloc[0], "x_rel")
    assert (lo, hi) == pytest.approx((0.28, 0.40))


def test_cons_formats_wider_interval_and_truncates_at_zero():
    df = pd.DataFrame({"x_rel": [0.02], "x_rel_lo": [-0.01], "x_rel_hi": [0.05],
                       "x_rel_cons_lo": [0.00], "x_rel_cons_hi": [0.09]})
    assert rt.cons(df.iloc[0], "x_rel") == "0.020 [0.000, 0.090]"


def test_seed_range_per_seed_r50_and_censoring():
    rows = []
    # seed 1 reaches 0.5 between r=10 and r=20, seed 2 between 20 and 40, seed 3 never (censored)
    curves = {1: [0.1, 0.4, 0.6, 0.9], 2: [0.0, 0.2, 0.4, 0.8], 3: [0.0, 0.1, 0.2, 0.3]}
    for seed, vals in curves.items():
        for key in (0, 1):
            for r, v in zip((5, 10, 20, 40), vals):
                for dec in ("proj", "ridge"):
                    rows.append({"pre": "none", "scheme": "lrp_full", "variant": "exact", "decoder": dec,
                                 "seed": seed, "key": key, "r": r, "e1_rel": v if dec == "proj" else v / 2,
                                 "e2_rel": v})
    df = pd.DataFrame(rows)
    st = object.__new__(rt.Study)
    st._seedcurves = {}
    st.araw = lambda enc: df
    g = st.seed_range("e", "none", "e1")
    # seed 1: proj crosses 0.5 at r = exp(ln10 + 0.5*(ln20-ln10)); seed 2 at r = 20 * 2**(1/2); ridge never reaches
    assert g["lo"] == pytest.approx(10 * 2 ** 0.5) and g["hi"] == 40.0   # censored seed counts as r_max
    assert g["n"] == 3 and g["n_cens"] == 1 and g["cens_any"] and not g["cens_all"]
    assert "censored" in st.seed_range_cell("e", "none", "e1")


# ---- paired utility differences, measured zero-effort level, eigen_oracle reference,
# absolute acceptance -------------------------------------------------------------------------------------------
def _study(**raw):
    """A Study without files: the raw per-run tables are given as frames, the attack FMR is 0.1%."""
    st = object.__new__(rt.Study)
    st._seedcurves = {}
    for name, df in raw.items():
        setattr(st, name, (lambda d: (lambda enc: d))(df))
    st.attack_fmr = lambda enc: 0.001
    st.pre = lambda enc, prefix: prefix
    return st


def _utility():
    rows = []
    # per seed: EER and TMR of three pre-conditioners at both FMR targets; key 0 only (the runner's utility rows)
    eer = {"none": [0.30, 0.32, 0.31], "pca128": [0.36, 0.35, 0.37], "fi128": [0.29, 0.30, 0.30]}
    tmr = {"none": [0.10, 0.12, 0.11], "pca128": [0.12, 0.13, 0.12], "fi128": [0.15, 0.16, 0.18]}
    fmr_actual = [0.0006, 0.0008, 0.0007]
    for i, seed in enumerate((1, 2, 3)):
        for pre in eer:
            for f in (0.001, 0.01):
                rows.append({"seed": seed, "key": 0, "pre": pre, "scheme": "lrp_full", "fmr_target": f,
                             "eer": eer[pre][i], "tmr": tmr[pre][i] + (0.2 if f == 0.01 else 0.0),
                             "fmr_actual": fmr_actual[i] if f == 0.001 else 10 * fmr_actual[i]})
        rows.append({"seed": seed, "key": 0, "pre": "fi128", "scheme": "lrp_half", "fmr_target": 0.001, "eer": 0.9,
                     "tmr": 0.9, "fmr_actual": 0.9})
    return pd.DataFrame(rows)


def test_paired_utility_is_the_student_t_interval_of_the_per_seed_differences():
    """FI128 - PCA128 per seed (same split, pairs and key), at the attack FMR for the TMR; Student-t over the seeds."""
    from scipy import stats
    st = _study(uraw=_utility())
    for metric, d in (("eer", np.array([-0.07, -0.05, -0.07])), ("tmr", np.array([0.03, 0.03, 0.06]))):
        g = st.paired_utility("e", "fi128", "pca128", metric)
        h = stats.t.ppf(0.975, 2) * d.std(ddof=1) / np.sqrt(3)
        assert g["n"] == 3 and list(g["diffs"]) == pytest.approx(list(d))
        assert (g["mean"], g["lo"], g["hi"]) == pytest.approx((d.mean(), d.mean() - h, d.mean() + h))
    # the mean of the paired differences is the difference of the seed means (the marginal values of the tables)
    g = st.paired_utility("e", "fi128", "none", "eer")
    assert g["mean"] == pytest.approx(np.mean([0.29, 0.30, 0.30]) - np.mean([0.30, 0.32, 0.31]))


def test_paired_utility_refuses_unpaired_seeds():
    u = _utility()
    u = u[~((u.pre == "pca128") & (u.seed == 3))]
    with pytest.raises(ValueError):
        _study(uraw=u).paired_utility("e", "fi128", "pca128", "eer")


def test_measured_zero_effort_level_uses_the_target_split_impostor_acceptance():
    """The impostor acceptance measured on the target impostor pairs at the DEV threshold (utility rows of LRP-full
    without pre-conditioning, at the attack FMR), averaged over the seeds, divided by the genuine acceptance."""
    st = _study(uraw=_utility(), zraw=pd.DataFrame({"e1_genuine": [0.2, 0.2]}))
    assert st.measured_fmr("e") == pytest.approx(0.0007)
    assert st.measured_fmr("e", 0.01) == pytest.approx(0.007)
    assert st.measured_impostor_level("e") == pytest.approx(0.0007 / 0.2)


def test_oracle_max_takes_the_largest_seed_mean_over_aux_and_p():
    """eigen_oracle (eigen alignment with the true signs from the key) only, LRP-full: mean over keys within a seed,
    then over seeds, then the largest value over (AUX, p), separately for E1 relative success and the cosine."""
    rows = []
    for seed in (1, 2):
        for key in (0, 1):
            for aux in ("same_dist", "shifted"):
                for p in (8, 16):
                    e1 = {("same_dist", 8): 0.02, ("same_dist", 16): 0.01, ("shifted", 8): 0.0,
                          ("shifted", 16): 0.03 * seed}[(aux, p)]
                    cos = {8: 0.3, 16: 0.4}[p] if aux == "same_dist" else 0.1
                    for meth, scale in (("eigen_oracle", 1.0), ("eigen", 5.0)):
                        rows.append({"seed": seed, "key": key, "pre": "none", "scheme": "lrp_full", "aux": aux, "p": p,
                                     "method": meth, "e1_rel": scale * e1, "rec_cos_z": scale * cos,
                                     "e1_proxy": 0.1 * scale * e1})
    rows.append({"seed": 1, "key": 0, "pre": "none", "scheme": "lrp_half", "aux": "same_dist", "p": 8,
                 "method": "eigen_oracle", "e1_rel": 0.9, "rec_cos_z": 0.9, "e1_proxy": 0.9})
    g = _study(zraw=pd.DataFrame(rows).assign(e1_successes=1, e1_cp_upper=0.01, e1_genuine=0.1)).oracle_max("e")
    assert g["e1"] == pytest.approx(0.045) and g["e1_at"] == ("shifted", 16)
    assert g["cos"] == pytest.approx(0.4) and g["cos_at"] == ("same_dist", 16)
    assert not g["zero"]


def test_oracle_max_gives_the_absolute_acceptance_of_the_selected_entry():
    """Next to the largest relative E1 success, the absolute proxy acceptance of the same
    (AUX, p) entry, averaged like the relative value (over keys within a seed, then over seeds)."""
    rows = []
    for seed, (rel8, abs8) in {1: (0.1, 0.02), 2: (0.3, 0.06)}.items():
        for key in (0, 1):
            rows.append({"seed": seed, "key": key, "pre": "none", "scheme": "lrp_full", "aux": "same_dist", "p": 8,
                         "method": "eigen_oracle", "e1_rel": rel8, "e1_proxy": abs8 + 0.01 * key, "rec_cos_z": 0.1,
                         "e1_successes": 1, "e1_cp_upper": 0.01, "e1_genuine": 0.2})
            rows.append({"seed": seed, "key": key, "pre": "none", "scheme": "lrp_full", "aux": "same_dist", "p": 16,
                         "method": "eigen_oracle", "e1_rel": 0.05, "e1_proxy": 0.9, "rec_cos_z": 0.2,
                         "e1_successes": 1, "e1_cp_upper": 0.01, "e1_genuine": 0.2})
    g = _study(zraw=pd.DataFrame(rows)).oracle_max("e")
    assert g["e1_at"] == ("same_dist", 8) and g["e1"] == pytest.approx(0.2)
    assert g["e1_abs"] == pytest.approx(((0.02 + 0.03) / 2 + (0.06 + 0.07) / 2) / 2)
    # the largest absolute acceptance over (AUX, p), averaged alike, is an upper bound in its own right: here it sits at
    # p = 16, where the relative success is not the largest
    assert g["e1_abs_max"] == pytest.approx(0.9)


def test_oracle_max_with_zero_successes_carries_the_clopper_pearson_bound():
    """Status rule: when no eigen_oracle row has a success, the maximum is no bare zero; the bound is the largest
    relative one-sided 95% Clopper-Pearson bound over the rows (absolute bound / genuine acceptance of the run)."""
    rows = [{"seed": s, "key": 0, "pre": "none", "scheme": "lrp_full", "aux": "same_dist", "p": p,
             "method": "eigen_oracle", "e1_rel": 0.0, "rec_cos_z": 0.1, "e1_proxy": 0.0, "e1_successes": 0, "e1_cp_upper": cp,
             "e1_genuine": g} for s, p, cp, g in ((1, 8, 0.0025, 0.10), (2, 8, 0.0025, 0.125), (1, 16, 0.0025, 0.1),
                                                  (2, 16, 0.0025, 0.125))]
    g = _study(zraw=pd.DataFrame(rows)).oracle_max("e")
    assert g["zero"] and g["e1"] == 0.0 and g["cp_rel"] == pytest.approx(0.025)


def test_oracle_endpoint_gives_each_endpoint_and_pre_conditioner_its_own_maximum():
    """eigen_oracle per pre-conditioner and endpoint: the largest seed-mean relative success over (AUX, p) (mean over
    keys within a seed, then over seeds), the largest seed-mean absolute acceptance, and, when no row of that endpoint
    has a success, the largest relative one-sided 95% Clopper-Pearson bound; rows of other pre-conditioners, methods
    and schemes are ignored."""
    rows = []
    for seed in (1, 2):
        for aux, p, e1, e2 in (("same_dist", 8, 0.0, 0.02 * seed), ("shifted", 16, 0.0, 0.01)):
            for key in (0, 1):
                rows.append({"seed": seed, "key": key, "pre": "fi128", "scheme": "lrp_full", "aux": aux, "p": p,
                             "method": "eigen_oracle", "e1_rel": e1, "e2_rel": e2, "e1_proxy": 0.0,
                             "e2_proxy": 0.5 * e2 + 0.001 * key, "rec_cos_z": 0.1 * p / 8, "e1_successes": 0,
                             "e2_successes": 1, "e1_cp_upper": 0.003, "e2_cp_upper": 0.01, "e1_genuine": 0.1 * seed,
                             "e2_genuine": 0.2})
    for meth, pre, scheme in (("eigen", "fi128", "lrp_full"), ("eigen_oracle", "none", "lrp_full"),
                              ("eigen_oracle", "fi128", "lrp_half")):
        rows.append({"seed": 1, "key": 0, "pre": pre, "scheme": scheme, "aux": "same_dist", "p": 8, "method": meth,
                     "e1_rel": 0.9, "e2_rel": 0.9, "e1_proxy": 0.9, "e2_proxy": 0.9, "rec_cos_z": 0.9,
                     "e1_successes": 5, "e2_successes": 5, "e1_cp_upper": 0.5, "e2_cp_upper": 0.5,
                     "e1_genuine": 0.5, "e2_genuine": 0.5})
    st = _study(zraw=pd.DataFrame(rows))
    e2 = st.oracle_endpoint("e", "fi128", "e2")
    assert e2["rel"] == pytest.approx(0.03) and e2["at"] == ("same_dist", 8) and not e2["zero"]
    assert e2["abs_max"] == pytest.approx(0.5 * 0.03 + 0.0005) and e2["cp_rel"] is None
    assert e2["cos"] == pytest.approx(0.2)
    e1 = st.oracle_endpoint("e", "fi128", "e1")
    assert e1["zero"] and e1["rel"] == 0.0 and e1["cp_rel"] == pytest.approx(0.03)   # 0.003 / genuine 0.1 of seed 1
    with pytest.raises(ValueError):
        st.oracle_endpoint("e", "pca128", "e1")


def test_rate_cell_prints_a_nonzero_value_that_rounds_to_zero_as_below_the_last_digit():
    """The paper's convention for a rate with three decimals: an observed, non-zero value that rounds to 0.000 prints
    <0.001, never a bare 0.000; an exact zero stays 0.000; missing or non-finite values print n/a."""
    assert rt.f3_rate(0.00024) == "<0.001" and rt.f3_rate(0.0004999) == "<0.001"
    assert rt.f3_rate(0.0) == "0.000"
    assert rt.f3_rate(0.0006) == "0.001" and rt.f3_rate(0.0164) == "0.016" and rt.f3_rate(0.203) == "0.203"
    assert rt.f3_rate(None) == "n/a" and rt.f3_rate(float("nan")) == "n/a"


def test_ceiling_abs_is_the_absolute_acceptance_of_the_selected_t2_method():
    post = pd.DataFrame({"encoder": ["e", "e"], "pre": ["fi128", "fi128"], "scheme": ["lrp_full"] * 2,
                         "method": ["analytic", "learned"], "e1_rel": [0.4, 0.3], "e2_rel": [1.0, 0.9]})
    raw = pd.DataFrame({"seed": [1, 1, 2, 1, 2], "key": [0, 1, 0, 0, 0], "pre": ["fi128"] * 5,
                        "scheme": ["lrp_full"] * 5, "method": ["analytic", "analytic", "analytic", "learned", "learned"],
                        "e1_proxy": [0.02, 0.04, 0.05, 0.9, 0.9], "e2_proxy": [0.2, 0.2, 0.3, 0.9, 0.9],
                        "e1_successes": [1] * 5, "e2_successes": [1] * 5})
    st = _study(praw=raw)
    st.post = post
    assert st.ceiling_abs("e", "fi128", "lrp_full", "e1") == pytest.approx((0.03 + 0.05) / 2)
    assert st.ceiling_abs("e", "fi128", "lrp_full", "e2") == pytest.approx((0.2 + 0.3) / 2)


def test_victim_genuine_and_count_come_from_the_exact_anchor_rows():
    a = pd.DataFrame({"seed": [1, 1, 2, 2, 1], "key": [0, 1, 0, 1, 0], "pre": ["fi128"] * 4 + ["none"],
                      "scheme": ["lrp_full"] * 5, "variant": ["exact"] * 4 + ["exact"], "r": [5, 5, 5, 5, 5],
                      "n_targets": [280] * 5, "e1_genuine": [0.1, 0.1, 0.2, 0.2, 0.7],
                      "e2_genuine": [0.4, 0.6, 0.5, 0.5, 0.7]})
    st = _study(araw=a)
    assert st.victim_genuine("e", "fi128", "e2") == pytest.approx(0.5)
    assert st.victim_genuine("e", "fi128", "e1") == pytest.approx(0.15)
    assert st.victims("e") == 280
    st2 = _study(araw=a.assign(n_targets=[280, 280, 280, 279, 280]))
    with pytest.raises(ValueError):
        st2.victims("e")
