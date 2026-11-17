import numpy as np
import pytest

from ijiss_btp.eval.endpoints import e1, e2
from ijiss_btp.eval.stats import (
    bootstrap_mean_ci,
    clopper_pearson_upper,
    make_pairs,
    rank1,
    rates_at_threshold,
    threshold_at_fmr,
)
from ijiss_btp.precondition import Identity
from ijiss_btp.protect.keyed import parse_scheme


def test_make_pairs_genuine_and_impostor():
    ids = np.array(["a", "a", "a", "b", "b", "c"])
    gen, imp = make_pairs(np.arange(6), ids, np.random.default_rng(0), n_imp=50)
    assert len(gen) == 4
    assert len(imp) == 50 and np.all(ids[imp[:, 0]] != ids[imp[:, 1]])


def test_make_pairs_rejects_single_identity():
    with pytest.raises(ValueError):
        make_pairs(np.arange(3), np.array(["a", "a", "a"]), np.random.default_rng(0), n_imp=5)


def test_threshold_at_fmr_never_exceeds_target_with_ties():
    imp = np.array([0.1] * 90 + [0.5] * 10)
    assert np.mean(imp >= threshold_at_fmr(imp, 0.05)) <= 0.05
    grid = np.linspace(0, 1, 1001)
    assert np.mean(grid >= threshold_at_fmr(grid, 0.01)) <= 0.01


def test_rates_and_rank1():
    assert rates_at_threshold(np.array([0.9, 0.2]), np.array([0.1, 0.95]), 0.5) == {"tmr": 0.5, "fmr": 0.5}
    assert rank1(np.array([[0.9, 0.1], [0.2, 0.3]]), np.array([0, 0]), np.array([0, 1])) == 0.5


def test_clopper_pearson_upper_bound():
    assert abs(clopper_pearson_upper(0, 500) - 0.005974) < 1e-5
    assert clopper_pearson_upper(5, 5) == 1.0


def test_bootstrap_ci_brackets_mean():
    v = np.random.default_rng(0).binomial(1, 0.3, 400).astype(float)
    lo, hi = bootstrap_mean_ci(v, n_boot=500, seed=1)
    assert lo < v.mean() < hi


def _space(n_ids=40):
    rng = np.random.default_rng(0)
    x = np.repeat(rng.standard_normal((n_ids, 16)), 3, 0) + 0.2 * rng.standard_normal((n_ids * 3, 16))
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    enr = np.arange(n_ids) * 3
    return xn, enr, np.stack([enr + 1, enr + 2], 1)


def test_e1_perfect_and_random_proxies():
    xn, enr, probes = _space()
    perfect = e1(xn, xn[enr], enr, probes, 0.5)
    assert perfect.rel == 1.0 and perfect.rec_cos > 0.999 and perfect.n_trials == 2 * len(enr)
    assert e1(xn, np.random.default_rng(1).standard_normal((len(enr), 16)), enr, probes, 0.5).rel < 0.2


def test_e2_after_renewal_with_isometric_scheme():
    xn, enr, probes = _space()
    s = parse_scheme("lrp:full")
    key2 = s.keygen(16, 5)
    res = e2(s, Identity(), key2, s.protect(xn, key2), xn[enr], enr, probes, 0.5)
    assert res.rel == 1.0


def test_bootstrap_ratio_ci_brackets_ratio():
    from ijiss_btp.eval.stats import bootstrap_ratio_ci

    rng = np.random.default_rng(0)
    num = (rng.random(400) < 0.3).astype(float)
    den = (rng.random(400) < 0.6).astype(float)
    lo, hi = bootstrap_ratio_ci(num, den, n_boot=500, seed=1)
    assert lo <= num.mean() / den.mean() <= hi
    assert np.isnan(bootstrap_ratio_ci(num, np.zeros(400))[0])
