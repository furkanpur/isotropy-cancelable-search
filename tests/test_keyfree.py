import numpy as np

from ijiss_btp.attacks.keyfree import EigenAligner, ica_align_decoder, wasserstein_procrustes_decoder
from ijiss_btp.protect.keyed import parse_scheme


def _cos(a, b):
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def _skewed(n, m, rng):
    return (rng.exponential(size=(n, m)) - 1.0) / np.arange(1, m + 1) ** 0.7


def test_eigen_alignment_recovers_anisotropic_skewed_key():
    rng = np.random.default_rng(0)
    m = 8
    s = parse_scheme("lrp:full")
    key = s.keygen(m, 1)
    victims, aux = _skewed(6000, m, rng), _skewed(6000, m, rng)
    stolen = s.protect(victims, key)
    assert _cos(EigenAligner(stolen, aux).decoder(m)(stolen), victims).mean() > 0.9


def test_proposition1_isotropic_gaussian_defeats_zero_pair_attacks():
    rng = np.random.default_rng(1)
    m = 32
    s = parse_scheme("lrp:full")
    victims, aux = rng.standard_normal((3000, m)), rng.standard_normal((3000, m))
    stolen = s.protect(victims, s.keygen(m, 2))
    decoders = [
        EigenAligner(stolen, aux).decoder(m),
        ica_align_decoder(stolen, aux, seed=0),
        wasserstein_procrustes_decoder(stolen, aux, seed=0, n_iter=10, n_sub=500),
    ]
    for dec in decoders:
        assert abs(_cos(dec(stolen), victims).mean()) < 0.15


def test_oracle_signs_are_at_least_as_good():
    rng = np.random.default_rng(2)
    m = 8
    s = parse_scheme("lrp:full")
    key = s.keygen(m, 3)
    victims, aux = _skewed(4000, m, rng), _skewed(4000, m, rng)
    stolen = s.protect(victims, key)
    al = EigenAligner(stolen, aux)
    attack = _cos(al.decoder(m)(stolen), victims).mean()
    oracle = _cos(al.decoder(m, oracle_key=key)(stolen), victims).mean()
    assert oracle >= attack - 1e-9


def test_wasserstein_procrustes_matches_or_improves_eigen_on_skewed_data():
    rng = np.random.default_rng(3)
    m = 8
    s = parse_scheme("lrp:full")
    victims, aux = _skewed(3000, m, rng), _skewed(3000, m, rng)
    stolen = s.protect(victims, s.keygen(m, 4))
    eig = _cos(EigenAligner(stolen, aux).decoder(m)(stolen), victims).mean()
    wp = _cos(wasserstein_procrustes_decoder(stolen, aux, seed=0, n_iter=15, n_sub=1000)(stolen), victims).mean()
    assert wp >= eig - 0.05


def _identifiable(n, rng):
    cols = [
        rng.gamma(0.5, size=n),
        rng.exponential(size=n),
        rng.gamma(2.0, size=n),
        rng.gamma(6.0, size=n),
        rng.beta(2.0, 5.0, size=n),
    ]
    z = np.stack(cols, 1)
    return (z - z.mean(0)) / z.std(0)


def test_ica_alignment_recovers_key_when_components_are_identifiable():
    rng = np.random.default_rng(4)
    s = parse_scheme("lrp:full")
    key = s.keygen(5, 5)
    victims, aux = _identifiable(6000, rng), _identifiable(6000, rng)
    stolen = s.protect(victims, key)
    assert _cos(ica_align_decoder(stolen, aux, seed=0)(stolen), victims).mean() > 0.9
