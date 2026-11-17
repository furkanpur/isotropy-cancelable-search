# tests/test_keyed.py
import numpy as np
import pytest

from ijiss_btp.protect.keyed import parse_scheme


def _z(n=200, m=32, seed=0):
    return np.random.default_rng(seed).standard_normal((n, m))


def _unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_lrp_full_is_an_isometry():
    z = _z()
    s = parse_scheme("lrp:full")
    t = s.protect(z, s.keygen(32, 1))
    assert np.allclose(np.linalg.norm(t, axis=1), np.linalg.norm(z, axis=1))
    assert np.allclose(s.sim_matrix(t[:10], t[10:20]), _unit(z[:10]) @ _unit(z[10:20]).T)


def test_biohash_outputs_signs_and_k_rule():
    s = parse_scheme("biohash:half")
    b = s.protect(_z(), s.keygen(32, 2))
    assert b.shape == (200, 16) and set(np.unique(b)) <= {-1.0, 1.0}


def test_iom_codes_features_and_similarity():
    s = parse_scheme("iom:64")
    c = s.protect(_z(), s.keygen(32, 3))
    assert c.shape == (200, 8) and c.min() >= 0 and c.max() < 8
    f = s.features(c)
    assert f.shape == (200, 64) and np.all(f.sum(1) == 8)
    assert np.allclose(np.diag(s.sim_matrix(c[:5], c[:5])), 1.0)
    assert np.allclose(s.sim_pairs(c[:5], c[:5]), 1.0)


def test_index_codes_types():
    z = _z()
    for spec, binary, width in (("lrp:full", False, 32), ("biohash:half", True, 2), ("iom:64", True, 8)):
        s = parse_scheme(spec)
        arr, is_bin = s.index_codes(s.protect(z, s.keygen(32, 4)))
        assert is_bin == binary and arr.shape == (200, width)


def test_invalid_specs():
    with pytest.raises(ValueError):
        parse_scheme("rot:full")
    with pytest.raises(ValueError):
        parse_scheme("iom:60").k_for(32)
    with pytest.raises(ValueError):
        parse_scheme("lrp:64").k_for(32)


def test_proposition1_isotropic_templates_hide_the_key_at_second_order():
    rng = np.random.default_rng(9)
    s = parse_scheme("lrp:half")
    iso = rng.standard_normal((40000, 16))
    aniso = iso / np.arange(1, 17)
    k1, k2 = s.keygen(16, 10), s.keygen(16, 11)

    def gap(z):
        return np.linalg.norm(np.cov(s.protect(z, k1).T) - np.cov(s.protect(z, k2).T))

    assert gap(iso) < 0.2
    assert gap(aniso) > 5 * gap(iso)
