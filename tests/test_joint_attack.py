"""Anchor-resolved eigen alignment (align_anchor) and the joint decoder, checked on synthetic Gaussian data only."""
import numpy as np
import pytest

from ijiss_btp.attacks.anchor import fit_projection
from ijiss_btp.attacks.joint import AnchorSpan, joint_decoder
from ijiss_btp.attacks.keyfree import EigenAligner, EigenBasis
from ijiss_btp.precondition import haar_orthogonal
from ijiss_btp.protect.keyed import parse_scheme
from ijiss_btp.theory import collusion_cos_optimum, uniform_abs_cos_mean

LRP = parse_scheme("lrp:full")


def _cos(a, b):
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def _gaussian(n, lam, rot, rng):
    """Synthetic zero-mean Gaussian rows with covariance rot^T diag(lam) rot (a symmetric law)."""
    return (rng.standard_normal((n, len(lam))) * np.sqrt(lam)) @ rot


def _separated(seed=0, n=20000):
    """Synthetic anisotropic Gaussian with well-separated eigenvalues 2^-i (m = 8), a Haar key and a large AUX."""
    rng = np.random.default_rng(seed)
    lam = 2.0 ** -np.arange(8)
    rot = haar_orthogonal(8, 100 + seed)
    key = LRP.keygen(8, 200 + seed)
    z, aux = _gaussian(n, lam, rot, rng), _gaussian(n, lam, rot, rng)
    return z, LRP.protect(z, key), aux, key, lam, rot, rng


def test_one_anchor_with_a_projection_on_every_axis_gives_the_true_signs():
    z, f, aux, key, lam, rot, rng = _separated()
    al = EigenAligner(f, aux)
    z_a = ((rng.choice([-1.0, 1.0], len(lam)) * np.sqrt(lam)) @ rot)[None]   # one unit-variance step on every axis
    anchors = (LRP.protect(z_a, key), z_a)
    np.testing.assert_array_equal(al.signs(8, anchors=anchors), al.signs(8, oracle_key=key))
    v = slice(0, 2000)
    np.testing.assert_allclose(al.decoder(8, anchors=anchors)(f[v]), al.decoder(8, oracle_key=key)(f[v]), atol=1e-12)


def test_align_anchor_approaches_the_oracle_with_drawn_colluders():
    z, f, aux, key, _, _, _ = _separated(seed=1)
    al = EigenAligner(f, aux)
    v = np.arange(100, 4100)
    oracle = _cos(al.decoder(8, oracle_key=key)(f[v]), z[v]).mean()
    assert oracle > 0.99
    for r in (1, 5):
        a = np.arange(r)   # colluders drawn from the enrolled population, disjoint from the victims
        got = _cos(al.decoder(8, anchors=(f[a], z[a]))(f[v]), z[v]).mean()
        assert got >= oracle - 0.02, (r, got, oracle)
    np.testing.assert_array_equal(al.signs(8, anchors=(f[:5], z[:5])), al.signs(8, oracle_key=key))


def test_skewness_rule_fails_on_a_symmetric_law_where_anchor_signs_succeed():
    m = 16
    lam = 1.25 ** -np.arange(m)   # distinct but closely spaced eigenvalues, no dominant axis
    agree_skew, agree_one, agree_three, cos_skew, cos_three, cos_oracle = [], [], [], [], [], []
    for rep in range(10):
        rng = np.random.default_rng(300 + rep)
        rot = haar_orthogonal(m, 400 + rep)
        key = LRP.keygen(m, 500 + rep)
        z, aux = _gaussian(20000, lam, rot, rng), _gaussian(20000, lam, rot, rng)
        f = LRP.protect(z, key)
        al = EigenAligner(f, aux)
        truth = al.signs(m, oracle_key=key)
        agree_skew.append(np.mean(al.signs(m) == truth))
        agree_one.append(np.mean(al.signs(m, anchors=(f[:1], z[:1])) == truth))
        agree_three.append(np.mean(al.signs(m, anchors=(f[:3], z[:3])) == truth))
        v = slice(3, 3003)
        cos_skew.append(_cos(al.decoder(m)(f[v]), z[v]).mean())
        cos_three.append(_cos(al.decoder(m, anchors=(f[:3], z[:3]))(f[v]), z[v]).mean())
        cos_oracle.append(_cos(al.decoder(m, oracle_key=key)(f[v]), z[v]).mean())
    assert 0.2 <= np.mean(agree_skew) <= 0.75   # chance level: a symmetric law carries no sign evidence
    assert abs(np.mean(cos_skew)) < 0.3
    # one drawn colluder fixes most signs; it misses an axis on which its own coordinate is close to zero
    assert np.mean(agree_one) >= 0.9
    assert min(agree_three) == 1.0
    np.testing.assert_allclose(cos_three, cos_oracle, atol=1e-12)
    assert np.mean(cos_oracle) > 0.99


def test_isotropic_law_align_anchor_is_at_chance_and_joint_stays_at_or_below_projection():
    m, r = 32, 16
    rng = np.random.default_rng(6)
    key = LRP.keygen(m, 7)
    z, aux = rng.standard_normal((4000, m)), rng.standard_normal((4000, m))
    f = LRP.protect(z, key)
    al = EigenAligner(f, aux)
    a, v = np.arange(r), np.arange(r, 4000)
    span = AnchorSpan(f[a], z[a])
    proj = span.project(f[v])
    cos_proj = _cos(proj, z[v]).mean()
    assert abs(cos_proj - collusion_cos_optimum(r, m)) < 0.02
    # no key information in an isotropic covariance: at most the sign-agnostic chance level
    assert abs(_cos(al.decoder(m, anchors=(f[a], z[a]))(f[v]), z[v]).mean()) < uniform_abs_cos_mean(m) + 0.05
    out_of_span = span.complement(z[v])
    for p in (2, 8, m):
        aligned = al.decoder(p, anchors=(f[a], z[a]))(f[v])
        added = span.complement(aligned)
        assert abs(_cos(added, out_of_span).mean()) < 0.1   # the added part is uncorrelated with the victim
        assert _cos(span.joint(proj, aligned), z[v]).mean() <= cos_proj + 0.01
    assert _cos(joint_decoder(f[a], z[a], al, 2)(f[v]), z[v]).mean() >= cos_proj - 0.05


def _anisotropic(seed=8, m=16, n=6000):
    """Synthetic anisotropic Gaussian (eigenvalues 1.25^-i) with a non-zero mean, a Haar key and a separate AUX."""
    rng = np.random.default_rng(seed)
    lam = 1.25 ** -np.arange(m)
    rot = haar_orthogonal(m, seed + 1)
    mu = rng.standard_normal(m) * 0.3
    z, aux = mu + _gaussian(n, lam, rot, rng), mu + _gaussian(n, lam, rot, rng)
    key = LRP.keygen(m, seed + 2)
    return z, LRP.protect(z, key), aux, key


def test_joint_restricted_to_the_anchor_span_is_the_projection_decoder():
    z, f, aux, _ = _anisotropic()
    a, v = np.arange(5), np.arange(5, 3005)
    al = EigenAligner(f, aux)
    got = joint_decoder(f[a], z[a], al, 8)(f[v])
    proj = fit_projection(f[a], z[a])(f[v])
    q, _ = np.linalg.qr(z[a].T)   # an orthonormal basis of the anchors' pre-key span, computed independently
    np.testing.assert_allclose(got @ q @ q.T, proj, atol=1e-10)
    np.testing.assert_allclose((got - proj) @ z[a].T, 0.0, atol=1e-10)
    aligned = al.decoder(8, anchors=(f[a], z[a]))(f[v])
    np.testing.assert_allclose(got - proj, aligned - aligned @ q @ q.T, atol=1e-10)


def test_p0_adds_only_the_aux_mean_outside_the_span_and_reduces_joint_to_projection_for_a_zero_mean_aux():
    z, f, aux, _ = _anisotropic(seed=9)
    a, v = np.arange(4), np.arange(4, 1004)
    proj = fit_projection(f[a], z[a])(f[v])
    span = AnchorSpan(f[a], z[a])
    mu_a = aux.mean(0)
    np.testing.assert_allclose(joint_decoder(f[a], z[a], EigenAligner(f, aux), 0)(f[v]),
                               proj + span.complement(mu_a[None]), atol=1e-12)
    centred = EigenAligner(f, aux - mu_a)
    np.testing.assert_allclose(joint_decoder(f[a], z[a], centred, 0)(f[v]), proj, atol=1e-12)


def test_one_colluder_suffices_for_the_joint_decoder():
    z, f, aux, key, _, _, _ = _separated(seed=2)
    a, v = np.arange(1), np.arange(1, 2001)
    al = EigenAligner(f, aux)
    span = AnchorSpan(f[a], z[a])
    assert span.basis.shape == (8, 1)
    aligned = al.decoder(8, anchors=(f[a], z[a]))(f[v])
    got = joint_decoder(f[a], z[a], al, 8)(f[v])
    assert got.shape == z[v].shape and np.isfinite(got).all()
    np.testing.assert_allclose(got, span.joint(span.project(f[v]), aligned), atol=1e-12)
    cos_proj, cos_joint = _cos(span.project(f[v]), z[v]).mean(), _cos(got, z[v]).mean()
    assert cos_joint > 0.95 > cos_proj + 0.3


def test_eigen_bases_computed_once_give_the_same_aligner():
    z, f, aux, key = _anisotropic(seed=10, n=3000)
    stolen = EigenBasis.of(f)
    for other in (aux, aux[::2]):
        cached, fresh = EigenAligner(stolen, EigenBasis.of(other)), EigenAligner(f, other)
        for name in ("mu_c", "lc", "vc", "mu_a", "la", "va"):
            np.testing.assert_array_equal(getattr(cached, name), getattr(fresh, name))
        np.testing.assert_array_equal(cached.decoder(6)(f[:50]), fresh.decoder(6)(f[:50]))
        np.testing.assert_array_equal(cached.decoder(6, oracle_key=key)(f[:50]), fresh.decoder(6, oracle_key=key)(f[:50]))


def test_anchor_span_rank_uses_the_projection_decoders_cutoff():
    rng = np.random.default_rng(13)
    m = 128
    u, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    v, _ = np.linalg.qr(rng.standard_normal((m, 3)))
    # third singular value 1e-14: above pinv's default cut-off (1e-15), below matrix_rank's (m * eps = 2.8e-14)
    z_a = (u * np.array([1.0, 0.5, 1e-14])) @ v.T
    f_a = LRP.protect(z_a, LRP.keygen(m, 14))
    sv = np.linalg.svd(f_a, compute_uv=False)
    assert AnchorSpan(f_a, z_a).basis.shape[1] == int(np.sum(sv > 1e-15 * sv[0])) == 3


def test_duplicated_anchors_keep_the_joint_decomposition():
    z, f, aux, _ = _anisotropic(seed=15, n=600)
    a, v = np.array([0, 0, 1, 2]), np.arange(3, 503)
    span = AnchorSpan(f[a], z[a])
    assert span.basis.shape[1] == 3
    got = joint_decoder(f[a], z[a], EigenAligner(f, aux), 8)(f[v])
    proj = fit_projection(f[a], z[a])(f[v])
    np.testing.assert_allclose(got @ span.basis @ span.basis.T, proj, atol=1e-10)
    np.testing.assert_allclose((got - proj) @ z[a].T, 0.0, atol=1e-10)


def test_eigen_bases_compare_and_hash_by_identity():
    x = np.random.default_rng(12).standard_normal((50, 4))
    a, b = EigenBasis.of(x), EigenBasis.of(x)
    assert a == a and a != b and len({a, b, a}) == 2


def test_degenerate_inputs_raise():
    z, f, aux, key = _anisotropic(seed=11, n=500)
    al = EigenAligner(f, aux)
    with pytest.raises(ValueError, match="at least one"):
        al.decoder(4, anchors=(f[:0], z[:0]))
    with pytest.raises(ValueError, match="same number"):
        al.decoder(4, anchors=(f[:2], z[:3]))
    with pytest.raises(ValueError, match="dimension"):
        al.decoder(4, anchors=(f[:2, :-1], z[:2]))
    with pytest.raises(ValueError, match="dimension"):
        al.decoder(4, anchors=(f[:2], z[:2, :-1]))
    with pytest.raises(ValueError, match="2-D"):
        al.decoder(4, anchors=(f[0], z[0]))
    bad = z[:2].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        al.decoder(4, anchors=(f[:2], bad))
    with pytest.raises(ValueError, match="either"):
        al.decoder(4, oracle_key=key, anchors=(f[:2], z[:2]))
    with pytest.raises(ValueError, match="p must be"):
        al.decoder(-1, anchors=(f[:2], z[:2]))
    with pytest.raises(ValueError, match="at least one"):
        AnchorSpan(f[:0], z[:0])
    with pytest.raises(ValueError, match="same number"):
        AnchorSpan(f[:3], z[:2])
    with pytest.raises(ValueError, match="at least one"):
        joint_decoder(f[:0], z[:0], al, 4)
