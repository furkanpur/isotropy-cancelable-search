"""Results record of the joint alignment-plus-anchor stage (scripts/figures/results_tables.py): seed means, the
ground-truth-selected envelope over the evaluated attacks, r50 from its curve and the small-budget acceptance."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/figures"))
_spec = importlib.util.spec_from_file_location("results_tables_j", ROOT / "scripts/figures/results_tables.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


def _rows():
    """Two seeds, two keys; proj (no AUX, p 0), align_anchor and joint at two (AUX, p), eigen_oracle at r 0."""
    out = []
    curves = {("proj", "", 0): {1: 0.0, 2: 0.1, 5: 0.2, 10: 0.4, 20: 0.8},
              ("align_anchor", "same_dist", 8): {1: 0.05, 2: 0.05, 5: 0.05, 10: 0.05, 20: 0.05},
              ("joint", "same_dist", 8): {1: 0.0, 2: 0.1, 5: 0.3, 10: 0.6, 20: 0.9},
              ("joint", "shifted", 16): {1: 0.0, 2: 0.0, 5: 0.1, 10: 0.2, 20: 0.4}}
    for seed in (1, 2):
        for key in (0, 1):
            for (meth, aux, p), c in curves.items():
                for r, v in c.items():
                    val = v + (0.02 if seed == 2 else -0.02) * (v > 0)
                    out.append({"seed": seed, "key": key, "pre": "fi128", "scheme": "lrp_full", "method": meth,
                                "aux": aux or None, "p": p, "r": r, "e2_rel": val, "e2_proxy": 0.5 * val,
                                "rec_cos_z": 0.01 * r, "e2_successes": int(val > 0), "e2_cp_upper": 0.003,
                                "e2_genuine": 0.5})
    out.append({"seed": 1, "key": 0, "pre": "fi128", "scheme": "lrp_full", "method": "eigen_oracle", "aux": "same_dist",
                "p": 8, "r": 0, "e2_rel": 0.99, "e2_proxy": 0.99, "rec_cos_z": 0.9, "e2_successes": 9, "e2_cp_upper": 0.5,
                "e2_genuine": 0.5})
    return pd.DataFrame(out)


def test_envelope_takes_the_best_evaluated_attack_per_r_and_leaves_the_oracle_out():
    per = rt.joint_seed_means(_rows(), "fi128", "e2")
    env = rt.joint_envelope(per, "e2")
    assert list(env.r) == [1, 2, 5, 10, 20]
    assert list(env.rel) == pytest.approx([0.05, 0.1, 0.3, 0.6, 0.9])   # seed means; the oracle's 0.99 is excluded
    assert list(env.method) == ["align_anchor", "proj", "joint", "joint", "joint"]   # ties go to the first method
    proj = rt.joint_envelope(per, "e2", ("proj",))
    assert list(proj.rel) == pytest.approx([0.0, 0.1, 0.2, 0.4, 0.8])


def test_r50_of_the_envelope_and_of_proj_by_log_linear_interpolation():
    per = rt.joint_seed_means(_rows(), "fi128", "e2")
    best, cens, rmax = rt.joint_r50(rt.joint_envelope(per, "e2"))
    assert not cens and rmax == 20 and best == pytest.approx(5 * 2 ** ((0.5 - 0.3) / 0.3))
    proj, cens, _ = rt.joint_r50(rt.joint_envelope(per, "e2", ("proj",)))
    assert not cens and proj == pytest.approx(10 * 2 ** ((0.5 - 0.4) / 0.4))
    j1, cens, rmax = rt.joint_r50(rt.joint_envelope(per, "e2", ("align_anchor",)))
    assert cens and j1 == rmax == 20


def test_small_budget_acceptance_reports_the_best_entry_or_the_zero_count_bound():
    df = _rows()
    per = rt.joint_seed_means(df, "fi128", "e2")
    s = rt.joint_small(df, per, "fi128", "e2")
    assert s["rel"] == pytest.approx(0.3) and s["r"] == 5 and s["method"] == "joint" and not s["zero"]
    assert s["abs"] == pytest.approx(0.15) and s["cos"] == pytest.approx(0.05)
    zero = df.assign(e2_rel=0.0, e2_proxy=0.0, e2_successes=0)
    z = rt.joint_small(zero, rt.joint_seed_means(zero, "fi128", "e2"), "fi128", "e2")
    assert z["zero"] and z["cp_rel"] == pytest.approx(0.003 / 0.5)


def test_per_seed_r50_range_takes_the_best_configuration_of_each_seed():
    g = rt.joint_seed_r50_range(_rows(), "fi128", "e2")
    # seed 1 runs 0.02 lower than seed 2 wherever the value is positive; joint at (same_dist, 8) is best in both
    s1 = 5 * 2 ** ((0.5 - 0.28) / (0.58 - 0.28))
    s2 = 5 * 2 ** ((0.5 - 0.32) / (0.62 - 0.32))
    assert g["lo"] == pytest.approx(s2) and g["hi"] == pytest.approx(s1) and not g["cens_any"]


def test_small_cell_renders_a_zero_count_at_one_r_with_its_bound():
    """Each small budget r = 1, 2, 5 is rendered on its own: a zero count at r = 1 carries the largest relative
    Clopper-Pearson bound of the rows of r = 1, while r = 5 keeps its acceptance."""
    df = _rows()
    df.loc[df.r == 1, ["e2_rel", "e2_proxy", "e2_successes"]] = 0
    per = rt.joint_seed_means(df, "fi128", "e2")
    env = rt.joint_envelope(per, "e2")
    assert rt.joint_small_cell(df, env, "fi128", "e2", 1) == "0 successes; CP95 upper <= 0.006"
    assert rt.joint_small_cell(df, env, "fi128", "e2", 5).startswith("0.300 / 0.150 (cos 0.050, joint)")


def test_fixed_selection_range_applies_the_envelope_choice_to_every_seed():
    """Per seed, the envelope's per-r selection (method, AUX, p) is applied to that seed's curve; with one setting best
    at every r, the range equals that setting's per-seed r50 range."""
    df = _rows()
    per = rt.joint_seed_means(df, "fi128", "e2")
    env = rt.joint_envelope(per, "e2")
    g = rt.joint_seed_r50_range_fixed(df, "fi128", "e2", env)
    s1 = 5 * 2 ** ((0.5 - 0.28) / (0.58 - 0.28))
    s2 = 5 * 2 ** ((0.5 - 0.32) / (0.62 - 0.32))
    assert g["lo"] == pytest.approx(s2) and g["hi"] == pytest.approx(s1) and not g["cens_any"]
