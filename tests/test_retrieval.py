import numpy as np
import pytest

from ijiss_btp.eval.retrieval import tie_aware_recall
from ijiss_btp.protect.keyed import parse_scheme


def _identical_gallery():
    scheme = parse_scheme("biohash:full")
    code = np.array([[1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]])
    gallery = np.tile(code, (20, 1))
    probes = code.copy()
    return scheme, gallery, probes


def test_tie_aware_recall_counts_equal_scores_as_hits():
    scheme, gallery, probes = _identical_gallery()
    exact = [list(range(10))]
    found = [list(range(10, 20))]  # disjoint ids, but every score ties the exact k-th score
    assert tie_aware_recall(scheme, gallery, probes, exact, found, k=10) == pytest.approx(1.0)


def test_tie_aware_recall_zero_when_all_results_are_far():
    scheme, gallery, probes = _identical_gallery()
    gallery = np.vstack([gallery[:10], -gallery[:10]])  # 10 identical codes, then 10 opposite codes
    exact = [list(range(10))]
    found = [list(range(10, 20))]
    assert tie_aware_recall(scheme, gallery, probes, exact, found, k=10) == pytest.approx(0.0)


def test_tie_aware_recall_ignores_padding_entries():
    scheme, gallery, probes = _identical_gallery()
    exact = [list(range(10))]
    found = [list(range(10, 15)) + [-1] * 5]
    assert tie_aware_recall(scheme, gallery, probes, exact, found, k=10) == pytest.approx(0.5)
