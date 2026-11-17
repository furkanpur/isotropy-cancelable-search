import hashlib

import numpy as np
import pandas as pd
import pytest

from ijiss_btp.data.loaders import (
    Embeddings,
    center_normalize,
    drop_near_duplicates,
    load_features_by_id,
    load_keyed_npz,
    load_sin_corpus,
    sha256_file,
    synthetic_clusters,
)
from ijiss_btp.data.splits import corpus_counts, group_rows, make_split, split_counts


def test_load_sin_corpus_reads_embeddings_and_identities(tmp_path):
    np.save(tmp_path / "emb_base.npy", np.arange(12, dtype=np.float32).reshape(4, 3))
    pd.DataFrame({"identity": ["a", "a", "b", "b"]}).to_parquet(tmp_path / "manifest.parquet")
    emb = load_sin_corpus(tmp_path, "base")
    assert emb.x.dtype == np.float64 and emb.x.shape == (4, 3)
    assert list(emb.ids) == ["a", "a", "b", "b"]


def test_load_keyed_npz_groups_by_identity_prefix(tmp_path):
    np.savez(tmp_path / "f.npz", **{"id1_02": np.ones(3), "id1_01": np.zeros(3), "id2_01": np.full(3, 2.0)})
    emb = load_keyed_npz(tmp_path / "f.npz")
    assert list(emb.ids) == ["id1", "id1", "id2"]
    assert np.allclose(emb.x[0], 0.0)


def test_load_features_by_id_samples_identities(tmp_path):
    obj = {f"ID_{i}": [np.full(4, i, dtype=np.float32), np.full(4, -i, dtype=np.float32)] for i in range(10)}
    np.savez(tmp_path / "f.npz", features_by_id=np.array(obj, dtype=object))
    emb = load_features_by_id(tmp_path / "f.npz", n_ids=3, per_id=1, seed=0)
    assert emb.x.shape == (3, 4) and len(np.unique(emb.ids)) == 3


def test_embeddings_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        Embeddings(np.zeros((3, 2)), np.array(["a", "b"]), "bad")


def test_center_normalize_gives_unit_rows():
    x = np.random.default_rng(0).standard_normal((50, 8)) + 5.0
    assert np.allclose(np.linalg.norm(center_normalize(x, x.mean(0)), axis=1), 1.0)


def test_drop_near_duplicates_removes_copies():
    rng = np.random.default_rng(1)
    prim = Embeddings(rng.standard_normal((20, 16)), np.array([f"p{i}" for i in range(20)]), "prim")
    sec_x = np.vstack([prim.x[:5] * 1.0001, rng.standard_normal((10, 16))])
    sec = Embeddings(sec_x, np.array([f"s{i}" for i in range(15)]), "sec")
    assert len(drop_near_duplicates(sec, prim, 0.99).x) == 10


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"canvec")
    assert sha256_file(p) == hashlib.sha256(b"canvec").hexdigest()


def test_synthetic_clusters_shapes():
    emb = synthetic_clusters(n_ids=50, per_id=3, dim=16, decay=0.0, seed=0)
    assert emb.x.shape == (150, 16) and len(np.unique(emb.ids)) == 50


def test_group_rows_returns_sorted_rows():
    g = group_rows(np.array(["b", "a", "b", "a", "c"]))
    assert {k: list(v) for k, v in g.items()} == {"a": [1, 3], "b": [0, 2], "c": [4]}


def test_make_split_is_disjoint_and_deterministic():
    ids = np.concatenate([np.repeat(np.array([f"i{k:03d}" for k in range(100)]), 3), np.array(["solo1", "solo2"])])
    s1 = make_split(ids, n_dev=30, n_tgt=40, seed=7)
    s2 = make_split(ids, n_dev=30, n_tgt=40, seed=7)
    assert list(s1.tgt_ids) == list(s2.tgt_ids)
    dev, tgt, aux = set(s1.dev_ids), set(s1.tgt_ids), set(s1.aux_ids)
    assert not (dev & tgt) and not (dev & aux) and not (tgt & aux)
    assert len(tgt) == 40 and len(dev) == 30
    assert all(len(s1.rows_by_id[t]) >= 3 for t in s1.tgt_ids)
    t = s1.tgt_ids[0]
    assert s1.enrolled_row(t) == s1.rows_by_id[t][0]
    assert s1.known_row(t) == s1.rows_by_id[t][1]
    assert list(s1.probe_rows(t)) == list(s1.rows_by_id[t][1:3])


def test_make_split_raises_when_not_enough_targets():
    with pytest.raises(ValueError):
        make_split(np.repeat(np.array([f"i{k}" for k in range(10)]), 2), n_dev=2, n_tgt=5, seed=0)


def test_corpus_and_split_counts():
    ids = np.concatenate([np.repeat(np.array([f"i{k:03d}" for k in range(100)]), 3), np.array(["solo1", "solo2"])])
    assert corpus_counts(ids, 3) == {"images": 302, "identities": 102, "identities_ge_min_tgt_images": 100}
    s = make_split(ids, n_dev=30, n_tgt=40, seed=7)
    c = split_counts(s)
    assert (c["dev_identities"], c["tgt_identities"], c["aux_identities"]) == (30, 40, 32)
    assert c["tgt_images"] == 120 and c["dev_images"] + c["tgt_images"] + c["aux_images"] == 302
