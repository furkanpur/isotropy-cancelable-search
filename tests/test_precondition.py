import numpy as np
import pandas as pd
import pytest

from ijiss_btp.precondition import (
    FisherIsotropic,
    Identity,
    MarginalGauss,
    PCAWhiten,
    fisher_traces,
    make_preconditioner,
)


def _labelled(seed=0, n_ids=200, per=6, d=24):
    rng = np.random.default_rng(seed)
    spec = 1.0 / np.arange(1, d + 1)
    centers = rng.standard_normal((n_ids, d)) * np.sqrt(spec) * 2.0
    x = np.repeat(centers, per, 0) + rng.standard_normal((n_ids * per, d)) * np.sqrt(spec[::-1])
    return x, np.repeat(np.arange(n_ids), per)


def test_pca_whiten_gives_identity_covariance():
    x, _ = _labelled()
    z = PCAWhiten(10, 1.0).fit(x).transform(x)
    assert np.allclose(np.cov(z.T), np.eye(10), atol=1e-8)


def test_pca_full_rank_inverse_round_trip():
    x, _ = _labelled()
    pre = PCAWhiten(x.shape[1], 1.0).fit(x)
    assert np.allclose(pre.inverse(pre.transform(x)), x, atol=1e-8)


def test_fisher_isotropic_identity_total_covariance_and_sorted_within():
    x, y = _labelled()
    fi = FisherIsotropic(8, 20).fit(x, y)
    assert np.allclose(np.cov(fi.transform(x).T), np.eye(8), atol=1e-8)
    assert np.all(np.diff(fi.within_var) >= -1e-12)


def test_fisher_isotropic_needs_labels():
    x, _ = _labelled()
    with pytest.raises(ValueError):
        FisherIsotropic(4, 8).fit(x)


def test_fisher_isotropic_back_map_is_right_inverse():
    x, y = _labelled()
    fi = FisherIsotropic(8, 20).fit(x, y)
    z = np.random.default_rng(3).standard_normal((30, 8))
    assert np.allclose(fi.transform(fi.inverse(z)), z, atol=1e-8)


def test_proposition3_fisher_is_optimal_among_key_hiding_maps():
    x, y = _labelled(n_ids=300, per=8)
    d, m = x.shape[1], 6
    fi = FisherIsotropic(m, d).fit(x, y)
    b_fi, t_fi = fisher_traces(fi.proj, x, y)
    xc = x - x.mean(0)
    w, v = np.linalg.eigh(xc.T @ xc / len(x))
    st_inv_half = v @ np.diag(w ** -0.5) @ v.T
    means = pd.DataFrame(xc).groupby(y).transform("mean").to_numpy()
    sb = means.T @ means / len(x)
    optimum = np.linalg.eigvalsh(st_inv_half @ sb @ st_inv_half)[-m:].sum()
    assert np.isclose(b_fi / t_fi * m, optimum, rtol=1e-9)
    rng = np.random.default_rng(5)
    for _ in range(20):
        q, _ = np.linalg.qr(rng.standard_normal((d, m)))
        b_r, t_r = fisher_traces(st_inv_half @ q, x, y)
        assert np.isclose(t_r, m, atol=1e-6)
        assert b_fi / t_fi >= b_r / t_r - 1e-9


def test_fisher_isotropic_drops_numerically_null_directions():
    x, y = _labelled()
    x2 = np.hstack([x, 2.0 * x[:, :1]])
    fi = FisherIsotropic(4, 25).fit(x2, y)
    assert fi.p0_effective == 24
    z = fi.transform(x2)
    assert np.allclose(np.cov(z.T), np.eye(4), atol=1e-8)


def test_fisher_isotropic_rejects_fewer_than_m_directions_above_the_rank_floor():
    """Algorithm 1, steps 3 and 6: when fewer than m eigenvalues of the DEV covariance lie
    above the floor (1e-8 times the largest), the fit is rejected instead of returning m directions. The guard predates
    the experiments (commit 9bcd3d8); this test pins it, so no committed result depends on a change."""
    x, y = _labelled()
    x3 = np.hstack([x[:, :3], x[:, :3] @ np.ones((3, 5))])   # 8 dimensions of rank 3
    with pytest.raises(ValueError, match="numerical rank 3 < m=4"):
        FisherIsotropic(4, 8).fit(x3, y)
    assert FisherIsotropic(3, 8).fit(x3, y).p0_effective == 3


def test_fisher_isotropic_back_map_is_lmmse():
    x, y = _labelled()
    fi = FisherIsotropic(8, 20).fit(x, y)
    z = fi.transform(x)
    ls = np.linalg.lstsq(z, x - x.mean(0), rcond=None)[0]
    assert np.allclose(fi.back, ls, atol=1e-8)


def test_within_var_is_within_class_variance_of_outputs():
    x, y = _labelled()
    fi = FisherIsotropic(8, 20).fit(x, y)
    z = fi.transform(x)
    zw = z - pd.DataFrame(z).groupby(y).transform("mean").to_numpy()
    assert np.allclose(zw.var(0, ddof=1), fi.within_var, atol=1e-8)
    assert np.all((fi.within_var >= 0) & (fi.within_var <= 1))


def test_invalid_dimensions_raise():
    x, y = _labelled()
    with pytest.raises(ValueError):
        FisherIsotropic(0, 8)
    with pytest.raises(ValueError):
        FisherIsotropic(9, 8)
    with pytest.raises(ValueError):
        FisherIsotropic(4, 30).fit(x, y)
    with pytest.raises(ValueError):
        PCAWhiten(30, 1.0).fit(x)
    with pytest.raises(ValueError):
        make_preconditioner("rbig:8:0")


def test_marginal_gauss_standardizes_skewed_marginals():
    z = np.random.default_rng(2).exponential(size=(5000, 4))
    g = MarginalGauss().fit(z).transform(z)
    assert np.allclose(g.mean(0), 0.0, atol=0.05) and np.allclose(g.std(0), 1.0, atol=0.05)
    assert np.all(np.abs(np.mean(((g - g.mean(0)) / g.std(0)) ** 3, 0)) < 0.1)


def test_gaussianized_chain_is_consistent():
    x, y = _labelled()
    pre = make_preconditioner("firbig:8:20:2").fit(x, y)
    z = pre.transform(x[:50])
    assert z.shape == (50, 8)
    assert np.allclose(pre.transform(pre.inverse(z)), z, atol=1e-8)


def test_make_preconditioner_specs():
    assert isinstance(make_preconditioner("none"), Identity)
    assert make_preconditioner("pca:16:1").name == "pca16_a1"
    assert make_preconditioner("fi:8:16").name == "fi8_p16"
    assert make_preconditioner("rbig:8:2").name == "rbig8_i2"
    assert make_preconditioner("firbig:8:16:2").name == "firbig8_p16_i2"
    with pytest.raises(ValueError):
        make_preconditioner("lda:8")
