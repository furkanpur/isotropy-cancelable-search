import numpy as np
import pytest

from ijiss_btp.attacks.anchor import anchor_decode
from ijiss_btp.protect.keyed import parse_scheme
from ijiss_btp.theory import collusion_cos_optimum


def test_known_values():
    assert collusion_cos_optimum(20, 128) == pytest.approx(0.3911, abs=1e-3)
    assert collusion_cos_optimum(40, 128) == pytest.approx(0.5566, abs=1e-3)


@pytest.mark.parametrize("m", [16, 64, 128, 768])
def test_bounded_and_monotone(m):
    r = np.arange(1, m)
    v = collusion_cos_optimum(r, m)
    assert np.all(v <= np.sqrt(r / m) + 1e-12)
    assert np.all(np.diff(v) > 0)
    assert collusion_cos_optimum(m, m) == 1.0
    assert isinstance(collusion_cos_optimum(3, m), float)


def test_monte_carlo_matches_closed_form():
    rng = np.random.default_rng(0)
    m, r, trials, nv = 32, 8, 400, 50
    vals = []
    for _ in range(trials):
        q, _ = np.linalg.qr(rng.standard_normal((m, r)))
        z = rng.standard_normal((nv, m))
        vals.append(np.mean(np.linalg.norm(z @ q, axis=1) / np.linalg.norm(z, axis=1)))
    assert np.mean(vals) == pytest.approx(collusion_cos_optimum(r, m), abs=0.005)


def _mean_cos(zhat, z):
    a = zhat / np.linalg.norm(zhat, axis=1, keepdims=True)
    b = z / np.linalg.norm(z, axis=1, keepdims=True)
    return float(np.mean(np.sum(a * b, 1)))


def _decode_cos(z, r=16, n_vict=2000):
    scheme = parse_scheme("lrp:full")
    key = scheme.keygen(64, seed=3)
    feats = scheme.features(scheme.protect(z, key))
    anchors, victims = np.arange(r), np.arange(r, r + n_vict)
    zhat = anchor_decode(feats[anchors], z[anchors], feats[victims], "ridge", seed=0)
    return _mean_cos(zhat, z[victims])


def test_proposition2_with_real_decoder():
    rng = np.random.default_rng(1)
    n = 16 + 2000
    cstar = collusion_cos_optimum(16, 64)
    assert cstar - 0.05 <= _decode_cos(rng.standard_normal((n, 64))) <= cstar + 0.02
    za = rng.standard_normal((n, 64)) * (1.0 / np.arange(1, 65))
    assert _decode_cos(za) >= cstar + 0.15


def test_proj_decoder_attains_cstar_where_ridge_sits_below():
    # Ridge-GCV fits an intercept (centres features and targets), so it sits below c* at small r;
    # the intercept-free minimum-norm projection decoder attains the optimum.
    rng = np.random.default_rng(1)
    r, m, n_vict = 4, 64, 4000          # c* = 0.2359, sqrt(r/m) = 0.25
    cstar = collusion_cos_optimum(r, m)
    z = rng.standard_normal((r + n_vict, m))
    scheme = parse_scheme("lrp:full")
    feats = scheme.features(scheme.protect(z, scheme.keygen(m, seed=3)))
    zhat = anchor_decode(feats[:r], z[:r], feats[r:], "proj")
    val = _mean_cos(zhat, z[r:])
    assert abs(val - cstar) <= 0.005
    assert np.sqrt(r / m) - val >= 0.008   # distinguishes c* from the sqrt(r/m) bound


def test_uniform_abs_cos_mean_values():
    from ijiss_btp.theory import uniform_abs_cos_mean

    for m, v in ((32, 0.142), (64, 0.100), (128, 0.0707), (256, 0.0499)):
        assert uniform_abs_cos_mean(m) == pytest.approx(v, abs=5e-4)
    rng = np.random.default_rng(0)
    u = rng.standard_normal((200000, 64))
    mc = np.mean(np.abs(u[:, 0]) / np.linalg.norm(u, axis=1))
    assert mc == pytest.approx(uniform_abs_cos_mean(64), abs=0.002)
