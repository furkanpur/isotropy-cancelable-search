import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ijiss_btp.theory import collusion_cos_optimum
from ijiss_btp.experiments.aggregate import load_runs, r_at_level, seed_ci, summarize


def test_r_at_level():
    assert r_at_level([10, 100], [0.0, 1.0], 0.5) == pytest.approx(np.sqrt(1000.0))
    assert r_at_level([10, 100], [0.6, 0.9], 0.5) == 10.0
    assert r_at_level([10, 100], [0.1, 0.2], 0.5) is None


def test_r_at_level_nan_before_crossing():
    assert r_at_level([10, 100], [float("nan"), 0.9], 0.5) == 100.0


def test_seed_ci_contains_mean():
    m, lo, hi = seed_ci([0.1, 0.2, 0.3])
    assert lo < m < hi and m == pytest.approx(0.2)


def test_seed_ci_ignores_nan():
    m, lo, hi = seed_ci([0.1, float("nan"), 0.3])
    assert m == pytest.approx(0.2)
    assert np.isfinite(lo) and np.isfinite(hi) and lo < m < hi
    assert all(np.isnan(x) for x in seed_ci([float("nan")]))


def _fake_run(d, name, shift, pr):
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"config": {"name": name}}))
    cum = list(np.linspace(0.1, 1.0, 64))
    (d / "spectra.json").write_text(json.dumps({"raw_centered_normalized": {"cumvar_first256": cum, "participation_ratio": pr}}))
    rows = []
    for seed in (1, 2):
        for pre, m in (("none", 64), ("fi128_p512", 128)):
            # "fi128_p512" never reaches 0.5, so its r50 is censored (NaN in r50.csv)
            curve = ((10, 0.2 + shift), (100, 0.8 + shift)) if pre == "none" else ((10, 0.05), (100, 0.3))
            for r, v in curve:
                rows.append({"seed": seed, "key": 0, "pre": pre, "scheme": "lrp_full", "m": m, "k": m,
                             "variant": "exact", "decoder": "ridge", "r": r, "rec_cos": v, "rec_cos_z": v / 2, "e1_rel": v, "e2_rel": v,
                             "e1_rel_lo": v - 0.1, "e1_rel_hi": v + 0.1, "e2_rel_lo": v - 0.2, "e2_rel_hi": v + 0.2,
                             "rec_abs_cos_z": v / 3, "rec_cos2_z": v / 4})
    pd.DataFrame(rows).to_csv(d / "anchor.csv", index=False)
    pd.DataFrame([{"seed": s, "key": 0, "pre": p, "scheme": "lrp_full", "m": 8, "k": 8, "fmr_target": 0.001,
                   "eer": 0.1, "tmr": 0.5, "fmr_actual": 0.001, "rank1": 0.4}
                  for s in (1, 2) for p in ("none", "fi128_p512")]).to_csv(d / "utility.csv", index=False)
    pd.DataFrame([{"seed": s, "key": 0, "pre": "none", "scheme": "lrp_full", "m": 8, "k": 8, "aux": "same_dist",
                   "method": "eigen", "p": 8, "rec_cos_z": 0.3 + shift, "e1_rel": 0.1 + shift,
                   "e1_rel_lo": 0.0, "e1_rel_hi": 0.2, "rec_abs_cos_z": 0.2, "rec_cos2_z": 0.1} for s in (1, 2)]).to_csv(d / "zeropair.csv", index=False)


def test_summarize_and_figures(tmp_path):
    _fake_run(tmp_path / "a", "enc_a", 0.0, 300.0)
    _fake_run(tmp_path / "b", "enc_b", 0.05, 20.0)
    summarize(load_runs([tmp_path / "a", tmp_path / "b"]), tmp_path / "out")
    r50 = pd.read_csv(tmp_path / "out" / "r50.csv")
    assert set(r50.encoder) == {"enc_a", "enc_b"}
    assert (tmp_path / "out" / "summary_utility.csv").exists()
    assert (tmp_path / "out" / "summary_zeropair.csv").exists()
    sa = pd.read_csv(tmp_path / "out" / "summary_anchor.csv")
    assert {"rec_cos_z", "rec_cos_z_lo", "rec_cos_z_hi", "cstar", "sqrt_r_m"} <= set(sa.columns)
    row = sa[(sa.r == 10) & (sa.m == 64)]
    assert len(row) > 0
    assert row.cstar.to_numpy() == pytest.approx(collusion_cos_optimum(10, 64))
    assert row.sqrt_r_m.to_numpy() == pytest.approx(np.sqrt(10 / 64))
    sz = pd.read_csv(tmp_path / "out" / "summary_zeropair.csv")
    assert {"best_rec_cos_z", "best_rec_cos_z_lo", "best_rec_cos_z_hi"} <= set(sz.columns)
    proc = subprocess.run([sys.executable, "scripts/figures/isotropy_figures.py", "--summary", str(tmp_path / "out"),
                           "--runs", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "figs")],
                          capture_output=True, text=True, check=True)
    for name in ("fig_spectra.pdf", "fig_anchor_theory.pdf", "fig_zeropair.pdf", "fig_tradeoff.pdf",
                 "fig_tradeoff_points.csv"):
        assert (tmp_path / "figs" / name).exists()
    pts = pd.read_csv(tmp_path / "figs" / "fig_tradeoff_points.csv")
    for enc in ("enc_a", "enc_b"):
        cen = pts[(pts.encoder == enc) & (pts.pre == "fi128_p512")]
        assert len(cen) == 1 and bool(cen.censored.iloc[0]) and cen.y_plotted.iloc[0] == 100
        reached = pts[(pts.encoder == enc) & (pts.pre == "none")]
        assert len(reached) == 1 and not bool(reached.censored.iloc[0]) and np.isfinite(reached.y_plotted.iloc[0])


def test_conservative_ci_formula(tmp_path):
    _fake_run(tmp_path / "a", "enc_a", 0.0, 300.0)
    summarize(load_runs([tmp_path / "a"]), tmp_path / "out")
    sa = pd.read_csv(tmp_path / "out" / "summary_anchor.csv")
    row = sa[(sa.pre == "none") & (sa.r == 10)].iloc[0]
    # fake rows: e1 half-width 0.1 -> se = 0.2/(2*1.959964) in every row and seed; mean e1_rel = 0.2
    h = 1.959964 * (0.2 / (2 * 1.959964))
    assert row.e1_rel == pytest.approx(0.2)
    assert row.e1_rel_cons_lo == pytest.approx(0.2 - h) and row.e1_rel_cons_hi == pytest.approx(0.2 + h)
    assert row.e2_rel_cons_lo == pytest.approx(0.2 - 0.2) and row.e2_rel_cons_hi == pytest.approx(0.2 + 0.2)
    assert {"rec_abs_cos_z", "rec_cos2_z"} <= set(sa.columns)
    assert row.rec_abs_cos_z == pytest.approx(0.2 / 3) and row.rec_cos2_z == pytest.approx(0.2 / 4)


def test_conservative_ci_varies_se_across_keys_and_seeds(tmp_path):
    d = tmp_path / "a"
    _fake_run(d, "enc_a", 0.0, 300.0)
    a = pd.read_csv(d / "anchor.csv")
    a = a[(a.pre == "none") & (a.r == 10)].copy()
    # seed 1: keys with half-widths 0.1 and 0.3 -> mean se 0.2/1.959964; seed 2: half-width 0.5
    a = pd.concat([a[a.seed == 1].assign(key=0), a[a.seed == 1].assign(key=1), a[a.seed == 2]])
    hw = {(1, 0): 0.1, (1, 1): 0.3, (2, 0): 0.5}
    a["e1_rel_lo"] = a.e1_rel - [hw[(s, k)] for s, k in zip(a.seed, a.key)]
    a["e1_rel_hi"] = a.e1_rel + [hw[(s, k)] for s, k in zip(a.seed, a.key)]
    a.to_csv(d / "anchor.csv", index=False)
    summarize(load_runs([d]), tmp_path / "out")
    row = pd.read_csv(tmp_path / "out" / "summary_anchor.csv").iloc[0]
    se_bar = np.mean([np.mean([0.1, 0.3]), 0.5]) / 1.959964
    assert row.e1_rel_cons_hi - row.e1_rel == pytest.approx(1.959964 * se_bar)


def test_zeropair_argmax_row_carries_se_and_cosines(tmp_path):
    d = tmp_path / "a"
    _fake_run(d, "enc_a", 0.0, 300.0)
    rows = []
    for s in (1, 2):
        # p=4 attains the max e1 (0.5) with its own SE and cosines; p=8 has larger cosines but smaller e1
        rows.append({"seed": s, "key": 0, "pre": "none", "scheme": "lrp_full", "m": 8, "k": 8, "aux": "same_dist",
                     "method": "eigen", "p": 4, "rec_cos_z": 0.1, "e1_rel": 0.5, "e1_rel_lo": 0.4, "e1_rel_hi": 0.6,
                     "rec_abs_cos_z": 0.11, "rec_cos2_z": 0.01})
        rows.append({"seed": s, "key": 0, "pre": "none", "scheme": "lrp_full", "m": 8, "k": 8, "aux": "same_dist",
                     "method": "eigen", "p": 8, "rec_cos_z": 0.9, "e1_rel": 0.2, "e1_rel_lo": 0.0, "e1_rel_hi": 1.0,
                     "rec_abs_cos_z": 0.91, "rec_cos2_z": 0.81})
    pd.DataFrame(rows).to_csv(d / "zeropair.csv", index=False)
    summarize(load_runs([d]), tmp_path / "out")
    sz = pd.read_csv(tmp_path / "out" / "summary_zeropair.csv").iloc[0]
    assert sz.best_e1_rel == pytest.approx(0.5)
    h = 1.959964 * (0.2 / (2 * 1.959964))
    assert sz.best_e1_rel_cons_lo == pytest.approx(0.5 - h) and sz.best_e1_rel_cons_hi == pytest.approx(0.5 + h)
    assert sz.best_rec_cos_z == pytest.approx(0.1)
    assert sz.best_abs_cos_z == pytest.approx(0.11) and sz.best_cos2_z == pytest.approx(0.01)


def _add_postcompromise(d, ceilings):
    """Write a fake postcompromise.csv; `ceilings` maps pre -> key-known E1/E2 relative success."""
    rows = []
    for seed in (1, 2):
        for pre, c in ceilings.items():
            for method in ("analytic", "learned"):
                rows.append({"seed": seed, "key": 0, "pre": pre, "scheme": "lrp_full", "m": 64, "k": 64, "method": method,
                             "in_system_accept": 1.0, "e1_rel": c, "e1_rel_lo": c - 0.05, "e1_rel_hi": c + 0.05,
                             "e2_rel": c, "e2_rel_lo": c - 0.05, "e2_rel_hi": c + 0.05})
    pd.DataFrame(rows).to_csv(d / "postcompromise.csv", index=False)


def _add_proj_decoder(d):
    a = pd.read_csv(d / "anchor.csv")
    p = a[a.decoder == "ridge"].copy()
    p["decoder"] = "proj"
    good = p.pre == "none"
    for c in ("e1_rel", "e2_rel"):
        p.loc[good, c] = np.where(p.loc[good, "r"] == 10, 0.6, 0.9)
    pd.concat([a, p]).to_csv(d / "anchor.csv", index=False)


def _load_figures_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("isotropy_figures", "scripts/figures/isotropy_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tradeoff_status_best_decoder_and_e2_figure(tmp_path):
    _fake_run(tmp_path / "a", "enc_a", 0.0, 300.0)
    _fake_run(tmp_path / "b", "enc_b", 0.05, 20.0)
    for d in ("a", "b"):
        _add_proj_decoder(tmp_path / d)
    _add_postcompromise(tmp_path / "a", {"none": 0.9, "fi128_p512": 0.3})   # T2 level < 0.5 with conservative upper < 0.5: backmap_limited
    _add_postcompromise(tmp_path / "b", {"none": 0.9, "fi128_p512": 0.8})   # attainable but never reached: censored
    summarize(load_runs([tmp_path / "a", tmp_path / "b"]), tmp_path / "out")
    subprocess.run([sys.executable, "scripts/figures/isotropy_figures.py", "--summary", str(tmp_path / "out"),
                    "--runs", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "figs")],
                   capture_output=True, text=True, check=True)
    assert (tmp_path / "figs" / "fig_tradeoff_e2.pdf").exists()
    for fname, col in (("fig_tradeoff_points.csv", "r50_e1"), ("fig_tradeoff_e2_points.csv", "r50_e2")):
        pts = pd.read_csv(tmp_path / "figs" / fname)
        assert {"encoder", "pre", "eer", col, "censored", "y_plotted", "decoder", "status"} <= set(pts.columns)
        st = pts.set_index(["encoder", "pre"]).status
        assert st[("enc_a", "fi128_p512")] == "backmap_limited"
        assert st[("enc_b", "fi128_p512")] == "censored"
        assert st[("enc_a", "none")] == "reached" and st[("enc_b", "none")] == "reached"
        none = pts[(pts.encoder == "enc_a") & (pts.pre == "none")].iloc[0]
        assert none.decoder == "proj" and none[col] == pytest.approx(10.0)   # best decoder = smallest r50
        un = pts[(pts.encoder == "enc_a") & (pts.pre == "fi128_p512")].iloc[0]
        assert un.y_plotted == 100 and bool(un.censored)


def test_observed_r50_is_not_overridden_by_ceiling(tmp_path):
    _fake_run(tmp_path / "a", "enc_a", 0.0, 300.0)
    _add_postcompromise(tmp_path / "a", {"none": 0.3, "fi128_p512": 0.3})   # pre none reaches 0.5 at r ~ 50 anyway
    summarize(load_runs([tmp_path / "a"]), tmp_path / "out")
    t = _load_figures_module().best_decoder_status(tmp_path / "out", "e1").set_index("pre")
    assert t.loc["none", "status"] == "reached" and t.loc["none", "ceiling"] == pytest.approx(0.3)
    assert t.loc["fi128_p512", "status"] == "backmap_limited"
    assert t.loc["fi128_p512", "ceiling_hi"] == pytest.approx(0.35)


def test_non_lrp_full_scheme_below_half_is_censored_not_backmap_limited(tmp_path):
    _fake_run(tmp_path / "a", "enc_a", 0.0, 300.0)
    _add_postcompromise(tmp_path / "a", {"none": 0.9, "fi128_p512": 0.3})
    for f in ("anchor.csv", "postcompromise.csv"):
        df = pd.read_csv(tmp_path / "a" / f)
        df["scheme"] = "iom_256"
        df.to_csv(tmp_path / "a" / f, index=False)
    summarize(load_runs([tmp_path / "a"]), tmp_path / "out")
    t = _load_figures_module().best_decoder_status(tmp_path / "out", "e1", "iom_256").set_index("pre")
    assert t.loc["fi128_p512", "ceiling"] == pytest.approx(0.3)
    assert t.loc["fi128_p512", "status"] == "censored"


def test_lrp_full_exact_with_conservative_upper_at_half_is_censored(tmp_path):
    _fake_run(tmp_path / "a", "enc_a", 0.0, 300.0)
    _add_postcompromise(tmp_path / "a", {"none": 0.9, "fi128_p512": 0.46})   # point 0.46 < 0.5, cons_hi 0.51 >= 0.5
    summarize(load_runs([tmp_path / "a"]), tmp_path / "out")
    t = _load_figures_module().best_decoder_status(tmp_path / "out", "e1").set_index("pre")
    assert t.loc["fi128_p512", "ceiling"] == pytest.approx(0.46)
    assert t.loc["fi128_p512", "ceiling_hi"] >= 0.5
    assert t.loc["fi128_p512", "status"] == "censored"


def test_anchor_decoder_prefers_proj():
    mod = _load_figures_module()
    assert mod.anchor_decoder(pd.DataFrame({"decoder": ["ridge", "proj", "mlp"]})) == "proj"
    assert mod.anchor_decoder(pd.DataFrame({"decoder": ["ridge", "mlp"]})) == "ridge"
