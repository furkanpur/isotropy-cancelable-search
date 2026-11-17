"""Unit tests for ijiss_btp transformation, security, and attack modules."""
import numpy as np
import pytest

from ijiss_btp.protect.linear import projection_matrix, rp_protect, biohash_protect
from ijiss_btp.protect.non_linear import nl_prp_protect
from ijiss_btp.protect.iom import ps_iom_protect
from ijiss_btp.attacks.analytical import invert_linear_template, evaluate_proxy_match_rate
from ijiss_btp.security.unlinkability import compute_unlinkability_metric
from ijiss_btp.eval.metrics import compute_eer, compute_identification_metrics
from ijiss_btp.attacks.coordinate_wise import (
    coordinate_wise_inversion_attack,
    ps_iom_known_key_inversion,
)


def test_projection_matrix_orthonormality():
    d_in = 768
    d_out = 512
    q = projection_matrix(d_in, d_out, seed=42)
    assert q.shape == (d_in, d_out)
    prod = q.T @ q
    identity = np.eye(d_out, dtype=np.float32)
    assert np.allclose(prod, identity, atol=1e-5)


def test_linear_and_biohash_transforms():
    n, d = 100, 768
    k = 256
    emb = np.random.randn(n, d).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    t_rp = rp_protect(emb, k, seed=42)
    assert t_rp.shape == (n, k)
    norms = np.linalg.norm(t_rp, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    t_bio = biohash_protect(emb, k, seed=42)
    assert t_bio.shape == (n, k // 8)
    assert t_bio.dtype == np.uint8


def test_non_linear_polyphase_rp():
    n, d = 50, 768
    k = 512
    emb = np.random.randn(n, d).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    t_nl = nl_prp_protect(emb, k, seed=42, gamma=0.75, beta=0.25)
    assert t_nl.shape == (n, k)
    norms = np.linalg.norm(t_nl, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_ps_iom_hashing():
    n, d = 50, 768
    k = 256
    emb = np.random.randn(n, d).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    t_iom = ps_iom_protect(emb, k, seed=42, block_size=8)
    assert t_iom.shape == (n, k // 8)
    assert t_iom.dtype == np.uint8


def test_pseudo_inversion_contrast():
    n, d, k = 100, 768, 512
    emb = np.random.randn(n, d).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    # Linear RP
    t_lin = rp_protect(emb, k, seed=42)
    e_hat_lin = invert_linear_template(t_lin, d, seed=42)
    cos_sim_lin = np.mean(np.sum(e_hat_lin * emb, axis=1))
    assert cos_sim_lin > 0.80  # Linear leaks high correlation

    # Non-linear PRP
    t_nl = nl_prp_protect(emb, k, seed=42)
    e_hat_nl = invert_linear_template(t_nl, d, seed=42)
    cos_sim_nl = np.mean(np.sum(e_hat_nl * emb, axis=1))
    assert cos_sim_nl < 0.40  # NL-PRP slashes correlation


def test_unlinkability_metric():
    rng = np.random.default_rng(42)
    mated = rng.normal(0.0, 0.05, size=20000)
    non_mated = rng.normal(0.0, 0.05, size=20000)
    d_sys, m_mean, nm_mean = compute_unlinkability_metric(mated, non_mated)
    assert d_sys < 0.08  # Empirical finite-sample histogram variation is well bounded


def test_eer_computation():
    genuine = np.array([0.8, 0.85, 0.9, 0.75, 0.82])
    impostor = np.array([0.1, 0.15, 0.2, 0.05, 0.25])
    eer, th = compute_eer(genuine, impostor)
    assert eer == 0.0

    # Test discrete tied-score ROC crossing benchmark
    gen_tied = np.array([0.0, 1.0])
    imp_tied = np.array([0.0, 0.0, 1.0])
    eer_tied, th_tied = compute_eer(gen_tied, imp_tied)
    assert np.isclose(eer_tied, 3.0 / 7.0, atol=1e-5)

    # Test non-strict boundary tie handling
    assert np.isclose(compute_eer([1, 1], [1, 1])[0], 0.50, atol=1e-5)
    assert np.isclose(compute_eer([1, 2], [0, 1])[0], 0.25, atol=1e-5)


def test_informed_coordinate_and_ps_iom_attacks():
    rng = np.random.default_rng(42)
    d, k = 128, 64
    n_train, n_test = 200, 50

    train_e = rng.standard_normal((n_train, d)).astype(np.float32)
    train_e /= np.linalg.norm(train_e, axis=1, keepdims=True)
    test_e = rng.standard_normal((n_test, d)).astype(np.float32)
    test_e /= np.linalg.norm(test_e, axis=1, keepdims=True)

    # Test NL-PRP coordinate-wise attack runs without error
    train_t = nl_prp_protect(train_e, k, seed=42)
    test_t = nl_prp_protect(test_e, k, seed=42)
    recon_nl = coordinate_wise_inversion_attack(train_t, train_e, test_t, d, seed=42, n_bins=20)
    assert recon_nl.shape == (n_test, d)
    assert np.allclose(np.linalg.norm(recon_nl, axis=1), 1.0, atol=1e-5)

    # Test PS-IoM known-key inversion attack holds reconstruction to near chance
    train_iom = ps_iom_protect(train_e, k, seed=42, block_size=8)
    test_iom = ps_iom_protect(test_e, k, seed=42, block_size=8)
    recon_iom = ps_iom_known_key_inversion(train_iom, train_e, test_iom, d, k, seed=42, block_size=8)
    assert recon_iom.shape == (n_test, d)
    cos_iom = np.mean(np.sum(recon_iom * test_e, axis=1))
    assert cos_iom < 0.50  # PS-IoM holds reconstruction strictly below decision threshold (0.58)


def test_neural_inversion_deterministic_repeatability():
    from ijiss_btp.attacks.neural_inversion import train_neural_inverter
    import torch

    rng = np.random.default_rng(42)
    n_train, n_val, d, k = 64, 32, 128, 64
    train_e = rng.standard_normal((n_train, d)).astype(np.float32)
    train_e /= np.linalg.norm(train_e, axis=1, keepdims=True)
    val_e = rng.standard_normal((n_val, d)).astype(np.float32)
    val_e /= np.linalg.norm(val_e, axis=1, keepdims=True)

    train_t = nl_prp_protect(train_e, k, seed=42)
    val_t = nl_prp_protect(val_e, k, seed=42)

    # Train run 1
    m1, val_cos1 = train_neural_inverter(
        train_t, train_e, val_t, val_e, epochs=3, batch_size=32, seed=42
    )

    # Train run 2 with same seed
    m2, val_cos2 = train_neural_inverter(
        train_t, train_e, val_t, val_e, epochs=3, batch_size=32, seed=42
    )

    assert np.isclose(val_cos1, val_cos2, atol=1e-6)

    # Evaluate on val_t
    with torch.no_grad():
        out1 = m1(torch.from_numpy(val_t)).cpu().numpy()
        out2 = m2(torch.from_numpy(val_t)).cpu().numpy()

    assert np.allclose(out1, out2, atol=1e-5)

