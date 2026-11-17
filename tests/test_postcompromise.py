import numpy as np

from ijiss_btp.attacks.postcompromise import biohash_inverse, iom_preimage, lrp_lmmse_inverse
from ijiss_btp.protect.keyed import parse_scheme


def _mcos(a, b):
    return float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))


def test_lrp_square_key_inverts_exactly():
    s = parse_scheme("lrp:full")
    key = s.keygen(16, 1)
    z = np.random.default_rng(0).standard_normal((50, 16))
    assert np.allclose(lrp_lmmse_inverse(s.protect(z, key), key), z)


def test_lrp_lmmse_beats_min_norm_on_anisotropic_data():
    s = parse_scheme("lrp:half")
    key = s.keygen(32, 2)
    scale = 1.0 / np.arange(1, 33)
    z = np.random.default_rng(1).standard_normal((2000, 32)) * scale
    t = s.protect(z, key)
    assert _mcos(lrp_lmmse_inverse(t, key, np.diag(scale ** 2)), z) > _mcos(lrp_lmmse_inverse(t, key), z)


def test_biohash_inverse_reproduces_bits():
    s = parse_scheme("biohash:half")
    key = s.keygen(32, 3)
    bits = s.protect(np.random.default_rng(2).standard_normal((100, 32)), key)
    assert np.array_equal(s.protect(biohash_inverse(bits, key, 1.0), key), bits)


def test_iom_preimage_reproduces_codes():
    s = parse_scheme("iom:128")
    key = s.keygen(32, 4)
    codes = s.protect(np.random.default_rng(3).standard_normal((60, 32)), key)
    assert np.mean(s.protect(iom_preimage(codes, key, s.block), key) == codes) > 0.99
