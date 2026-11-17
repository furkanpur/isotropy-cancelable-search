# tests/test_label_check.py
"""Histogram AUC and device-pair grouping of scripts/data/label_check.py (synthetic data only)."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/data"))
import label_check as lc  # noqa: E402


def _exact_auc(g, i) -> float:
    g = np.asarray(g, dtype=np.float64)[:, None]
    i = np.asarray(i, dtype=np.float64)[None, :]
    return float(((g > i).sum() + 0.5 * (g == i).sum()) / (g.size * i.size))


def _hist(s, bins):
    return np.bincount(lc.bin_index(np.asarray(s), bins), minlength=bins)


def test_hist_auc_matches_brute_force_auc():
    rng = np.random.default_rng(0)
    for _ in range(5):
        g = np.clip(rng.normal(0.4, 0.3, 300), -1, 1)
        i = np.clip(rng.normal(0.1, 0.3, 700), -1, 1)
        assert lc.hist_auc(_hist(g, 4000), _hist(i, 4000)) == pytest.approx(_exact_auc(g, i), abs=1e-3)
    centres = (np.arange(400) + 0.5) / 400 * 2 - 1           # scores on bin centres: ties are exact ties
    g, i = rng.choice(centres, 250), rng.choice(centres, 250)
    assert lc.hist_auc(_hist(g, 400), _hist(i, 400)) == pytest.approx(_exact_auc(g, i), abs=1e-12)
    assert lc.hist_auc(_hist(g, 400), np.zeros(400, np.int64)) is None


def _synthetic():
    rows = [("A", "s1", "d1"), ("A", "s1", "d2"), ("A", "s1", "d2"), ("B", "s1", "d1"), ("B", "s1", "d1"),
            ("C", "s1", "d2"), ("C", "s1", "d3"), ("D", "s1", "d3"), ("E", "s2", ""), ("E", "s2", ""),
            ("F", "s2", "")]
    man = pd.DataFrame(rows, columns=["identity", "source", "device"])
    x = np.random.default_rng(3).normal(size=(len(man), 8))
    return man, x / np.linalg.norm(x, axis=1, keepdims=True)


def test_device_pair_grouping_matches_brute_force():
    man, x = _synthetic()
    res = lc.label_check(man, {"m": x}, bins=200_000, min_pairs=1, block=4)
    s = x @ x.T
    expect: dict = {}
    for a, b in itertools.combinations(range(len(man)), 2):
        da = (man.source[a], man.device[a])
        db = (man.source[b], man.device[b])
        key = tuple(sorted([da, db]))
        e = expect.setdefault(key, {"g": [], "i": []})
        e["g" if man.identity[a] == man.identity[b] else "i"].append(s[a, b])
    got = {(r["source"], r["devices"]): r for r in res["models"]["m"]["device_pairs"]}
    want = {k: v for k, v in expect.items() if v["g"]}
    assert len(got) == len(want)
    for (da, db), v in want.items():
        src = da[0] if da[0] == db[0] else f"{da[0]}/{db[0]}"
        dev = da[1] if da == db else f"{da[1]} x {db[1]}"
        r = got[(src, dev)]
        assert r["genuine_pairs"] == len(v["g"]) and r["impostor_pairs"] == len(v["i"])
        assert r["same_device"] == (da == db)
        if v["i"]:
            assert r["auc"] == pytest.approx(_exact_auc(v["g"], v["i"]), abs=1e-4)


def test_per_source_and_identity_structure():
    man, x = _synthetic()
    res = lc.label_check(man, {"m": x}, bins=200_000, min_pairs=1, block=5)
    s1 = res["models"]["m"]["per_source"]["s1"]
    # s1 genuine pairs: A 3 (1 same-device), B 1 (same), C 1 (cross) -> 5 = 2 same + 3 cross
    assert s1["genuine_pairs"] == 5 and s1["same_device"]["pairs"] == 2 and s1["cross_device"]["pairs"] == 3
    assert s1["impostor_pairs"] == 28 - 5                         # 8 images in s1: 28 pairs
    s2 = res["models"]["m"]["per_source"]["s2"]
    assert s2["genuine_pairs"] == 1 and s2["cross_device"] is None and s2["impostor_pairs"] == 2
    assert res["identity_structure"] == {
        "s1": {"images": 8, "identities": 4, "multi_device_identities": 2, "max_devices_per_identity": 2},
        "s2": {"images": 3, "identities": 2, "multi_device_identities": 0, "max_devices_per_identity": 1}}
    allp = res["models"]["m"]["all"]
    assert allp["genuine_pairs"] == 6 and allp["impostor_pairs"] == 55 - 6
    text = json.dumps(res)
    for ident in man["identity"].unique():                         # no identity label anywhere in the output
        assert f'"{ident}"' not in text


def test_main_writes_aggregates_and_provenance(tmp_path, monkeypatch):
    man, x = _synthetic()
    root = tmp_path / "corpus"
    root.mkdir()
    man.assign(path=[f"p/{i}.png" for i in range(len(man))]).to_parquet(root / "manifest.parquet")
    np.save(root / "emb_m.npy", x.astype(np.float32))
    out = tmp_path / "lc.json"
    monkeypatch.setenv("CANVEC_GIT_SHA", "abc123-dirty")
    lc.main(["--root", str(root), "--out", str(out), "--models", "m", "--min-pairs", "1", "--label", "synthetic test"])
    res = json.loads(out.read_text())
    assert res["data_source"] == "synthetic test" and res["n_images"] == 11 and res["n_identities"] == 6
    assert set(res["inputs_sha256"]) == {"manifest.parquet", "emb_m.npy"}
    prov = res["provenance"]
    assert prov["git_sha"] == "abc123-dirty" and len(prov["script_sha256"]) == 64
    assert prov["numpy"] == np.__version__ and prov["pandas"] == pd.__version__
    assert "p/0.png" not in out.read_text()


def test_git_sha_falls_back_to_the_repository_file(tmp_path, monkeypatch):
    monkeypatch.delenv("CANVEC_GIT_SHA", raising=False)
    (tmp_path / "GIT_SHA").write_text("deadbeef\n")
    assert lc.git_sha(tmp_path) == "deadbeef"
    assert lc.git_sha(tmp_path / "missing") is None
