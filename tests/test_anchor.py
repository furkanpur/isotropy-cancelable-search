import numpy as np

from ijiss_btp.attacks.anchor import anchor_decode
from ijiss_btp.protect.keyed import parse_scheme


def _cos(a, b):
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def test_ridge_recovers_linear_map_with_enough_exact_anchors():
    rng = np.random.default_rng(0)
    s = parse_scheme("lrp:full")
    key = s.keygen(16, 1)
    z = rng.standard_normal((300, 16))
    t = s.protect(z, key)
    assert _cos(anchor_decode(t[:40], z[:40], t[40:], "ridge"), z[40:]).mean() > 0.99


def test_proposition2_isotropic_bound_is_tight():
    rng = np.random.default_rng(1)
    m, r = 64, 16
    s = parse_scheme("lrp:full")
    vals = []
    for rep in range(10):
        key = s.keygen(m, rep)
        z = rng.standard_normal((r + 400, m))
        t = s.protect(z, key)
        vals.append(_cos(anchor_decode(t[:r], z[:r], t[r:], "ridge"), z[r:]).mean())
    bound = np.sqrt(r / m)
    assert np.mean(vals) <= bound + 0.03
    assert np.mean(vals) >= bound - 0.12


def test_anisotropic_inputs_need_far_fewer_anchors():
    rng = np.random.default_rng(2)
    m, r = 64, 16
    s = parse_scheme("lrp:full")
    z = rng.standard_normal((r + 400, m)) / np.arange(1, m + 1)
    t = s.protect(z, s.keygen(m, 3))
    assert _cos(anchor_decode(t[:r], z[:r], t[r:], "ridge"), z[r:]).mean() > 0.9


def test_mlp_decoder_shape():
    rng = np.random.default_rng(3)
    f, y = rng.standard_normal((60, 8)), rng.standard_normal((60, 4))
    assert anchor_decode(f, y, f[:5], "mlp", seed=0).shape == (5, 4)


def test_mlp_decoder_is_scale_invariant():
    rng = np.random.default_rng(5)
    f, y = rng.standard_normal((80, 8)), rng.standard_normal((80, 4))
    a = anchor_decode(f, y, f[:5], "mlp", seed=0)
    b = anchor_decode(0.01 * f, y, 0.01 * f[:5], "mlp", seed=0)
    assert np.allclose(a, b, atol=1e-8)
    assert np.allclose(anchor_decode(f, 100 * y, f[:5], "mlp", seed=0), 100 * a, atol=1e-6)


def test_projection_decoder_equals_span_projection_cosine():
    from ijiss_btp.theory import collusion_cos_optimum

    rng = np.random.default_rng(5)
    m, r, nv = 64, 16, 4000
    s = parse_scheme("lrp:full")
    key = s.keygen(m, 3)
    z = rng.standard_normal((r + nv, m))
    t = s.protect(z, key)
    zhat = anchor_decode(t[:r], z[:r], t[r:], "proj")
    q, _ = np.linalg.qr(z[:r].T)  # orthonormal basis of the anchors' span in z-space
    v = z[r:]
    expected = np.linalg.norm(v @ q, axis=1) / np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(_cos(zhat, v), expected, atol=1e-8)
    assert abs(expected.mean() - collusion_cos_optimum(r, m)) < 0.01
