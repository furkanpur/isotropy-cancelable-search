# tests/test_shared_identity.py
"""A secondary corpus (shifted AUX, distractors) made identity-disjoint from the primary corpus.

`exclude_primary_identities: true` drops every secondary row whose identity occurs in the primary corpus that was
loaded, after relabelling and before the near-duplicate filter. Both corpora must carry a relabel map (one identity
namespace); otherwise the runners refuse instead of comparing strings from two namespaces.
"""
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from ijiss_btp.data.loaders import (
    Embeddings,
    drop_shared_identities,
    exclude_primary,
    load_sin_corpus,
    sha256_file,
    synthetic_clusters,
)
from ijiss_btp.experiments.isotropy import load_data, run

needs_faiss = pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
PRIM_SPEC = {"kind": "sin", "relabel": "/maps/primary.csv"}
SEC_SPEC_ON = {"kind": "sin", "relabel": "/maps/secondary.csv", "relabel_key": "path",
               "exclude_primary_identities": True}


# ---- unit level ----
def test_drop_shared_identities_removes_the_rows_of_primary_identities():
    prim = Embeddings(np.zeros((3, 2)), np.array(["a", "b", "c"]), "p")
    sec = Embeddings(np.arange(8.0).reshape(4, 2), np.array(["a", "x", "c", "y"]), "s")
    out = drop_shared_identities(sec, prim)
    assert list(out.ids) == ["x", "y"]
    np.testing.assert_array_equal(out.x, [[2, 3], [6, 7]])
    assert out.source == "s:shared_ids_removed=2"


def test_drop_shared_identities_keeps_everything_without_overlap():
    prim = Embeddings(np.zeros((2, 2)), np.array(["a", "b"]), "p")
    sec = Embeddings(np.ones((3, 2)), np.array(["x", "y", "x"]), "s")
    out = drop_shared_identities(sec, prim)
    assert list(out.ids) == ["x", "y", "x"] and out.source == "s:shared_ids_removed=0"


def test_exclude_primary_follows_the_spec_flag():
    prim = Embeddings(np.zeros((2, 2)), np.array(["a", "b"]), "p")
    sec = Embeddings(np.ones((3, 2)), np.array(["a", "y", "b"]), "s")
    same, n = exclude_primary({"kind": "sin"}, sec, prim, PRIM_SPEC)
    assert same is sec and n is None
    same, n = exclude_primary({**SEC_SPEC_ON, "exclude_primary_identities": False}, sec, prim, PRIM_SPEC)
    assert same is sec and n is None
    out, n = exclude_primary(SEC_SPEC_ON, sec, prim, PRIM_SPEC)
    assert n == 2 and list(out.ids) == ["y"]
    with pytest.raises(ValueError, match="primary"):
        exclude_primary(SEC_SPEC_ON, sec, None, PRIM_SPEC)


def test_exclude_primary_refuses_a_primary_spec_without_relabel():
    prim = Embeddings(np.zeros((2, 2)), np.array(["a", "b"]), "p")
    sec = Embeddings(np.ones((3, 2)), np.array(["a", "y", "b"]), "s")
    for prim_spec in ({"kind": "sin"}, {"kind": "sin", "relabel": None}, None):
        with pytest.raises(ValueError, match="primary corpus spec has no relabel"):
            exclude_primary(SEC_SPEC_ON, sec, prim, prim_spec)


def test_exclude_primary_refuses_a_secondary_spec_without_relabel():
    prim = Embeddings(np.zeros((2, 2)), np.array(["a", "b"]), "p")
    sec = Embeddings(np.ones((3, 2)), np.array(["a", "y", "b"]), "s")
    with pytest.raises(ValueError, match="secondary corpus spec has no relabel"):
        exclude_primary({"kind": "sin", "exclude_primary_identities": True}, sec, prim, PRIM_SPEC)


def test_relabel_key_without_relabel_raises(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    np.save(root / "emb_m.npy", np.zeros((2, 2), dtype=np.float32))
    pd.DataFrame({"identity": ["a", "b"], "path": ["/hostdata/1.png", "/hostdata/2.png"]}).to_parquet(
        root / "manifest.parquet")
    for key in ("path", "basename"):
        with pytest.raises(ValueError, match="relabel_key without relabel"):
            load_sin_corpus(root, "m", relabel=None, relabel_key=key)
        with pytest.raises(ValueError, match="relabel_key without relabel"):
            load_data({"kind": "sin", "root": str(root), "model": "m", "relabel_key": key})
    assert list(load_sin_corpus(root, "m").ids) == ["a", "b"]   # neither given: unchanged


# ---- runner level: primary and secondary corpora in the SIN layout ----
N_SHARED_DUP, N_DUP_ONLY, N_SHARED_ONLY, N_NEITHER = 4, 3, 5, 40


def _save_sin(root, x, ids, paths):
    root.mkdir(parents=True)
    np.save(root / "emb_m.npy", np.asarray(x, dtype=np.float32))
    pd.DataFrame({"identity": ids, "path": paths}).to_parquet(root / "manifest.parquet")
    return root


def _fixture(tmp_path, n_ids, per_id, dim):
    """Primary: synthetic clusters with an identity-preserving basename relabel map. Secondary rows: shared identity
    and an exact copy of a primary row; a copy under a new identity; a fresh vector under a primary identity; fresh
    vectors under new identities; one unmapped row. Returns (primary root, secondary root, secondary map, primary map)."""
    prim = synthetic_clusters(n_ids, per_id, dim, 1.0, 0)
    xp = prim.x.astype(np.float32).astype(np.float64)
    p_root = _save_sin(tmp_path / "prim", xp, prim.ids, [f"/hostdata/p/{i}.png" for i in range(len(xp))])
    pmap = tmp_path / "prim_map.csv"
    pd.DataFrame({"file": [f"{i}.png" for i in range(len(xp))], "identity": prim.ids}).to_csv(pmap, index=False)
    rng = np.random.default_rng(5)
    # energy in the trailing coordinates, where the power-law primary spectrum is weak
    fresh = rng.standard_normal((N_SHARED_ONLY + N_NEITHER + 1, dim)) * np.sqrt(np.arange(1, dim + 1))
    cos = (fresh / np.linalg.norm(fresh, axis=1, keepdims=True)) @ (xp / np.linalg.norm(xp, axis=1, keepdims=True)).T
    assert cos.max() < 0.9   # fixture sanity: fresh rows are no near-duplicates of the primary
    rows_shared_dup = [per_id * i for i in range(N_SHARED_DUP)]                 # identities s00000..
    rows_dup_only = [per_id * (10 + i) for i in range(N_DUP_ONLY)]              # copies of s00010..
    x = np.vstack([xp[rows_shared_dup], xp[rows_dup_only], fresh])
    labels = ([prim.ids[r] for r in rows_shared_dup] + [f"n{i}" for i in range(N_DUP_ONLY)]
              + [f"s{20 + i:05d}" for i in range(N_SHARED_ONLY)] + [f"m{i:02d}" for i in range(N_NEITHER)])
    paths = [f"/hostdata/sec/{i % 7}/{i}.png" for i in range(len(x))]
    s_root = _save_sin(tmp_path / "sec", x, ["orig"] * len(x), paths)
    rmap = tmp_path / "sec_map.csv"
    pd.DataFrame({"file": paths[:len(labels)], "identity": labels}).to_csv(rmap, index=False)   # last row unmapped
    return p_root, s_root, rmap, pmap


ISO_CFG = {
    "name": "excl",
    "split": {"n_dev": 80, "n_tgt": 120, "min_tgt_images": 3, "seeds": [1]},
    "keys_per_seed": 1,
    "fmr": [0.01],
    "preconditioners": ["none"],
    "schemes": ["lrp:full"],
    "attacks": {"anchor": {"r_grid": [5], "variants": ["exact"], "decoders": ["ridge"], "min_targets": 50},
                "zeropair": {"p_grid": [4], "methods": ["eigen"], "expensive_keys": 0},
                "postcompromise": {"enabled": False}},
    "bootstrap": {"n_boot": 20},
}


def _iso_cfg(p_root, s_root, rmap, exclude, pmap):
    data = {"kind": "sin", "root": str(p_root), "model": "m",
            "shifted_aux": {"kind": "sin", "root": str(s_root), "model": "m", "near_dup_cos": 0.95,
                            "relabel": str(rmap), "relabel_key": "path", "exclude_primary_identities": exclude}}
    if pmap is not None:
        data["relabel"] = str(pmap)
    return {**ISO_CFG, "data": data}


def test_isotropy_excludes_shared_identities_before_the_near_duplicate_filter(tmp_path):
    p_root, s_root, rmap, pmap = _fixture(tmp_path, 260, 3, 24)
    run(_iso_cfg(p_root, s_root, rmap, True, pmap), tmp_path / "on")
    meta = json.loads((tmp_path / "on" / "metadata.json").read_text())
    assert meta["secondary_removed_shared_identity"] == N_SHARED_DUP + N_SHARED_ONLY
    assert meta["shifted_aux_removed"] == N_DUP_ONLY   # the shared copies are gone before the filter
    assert meta["aux_sizes"]["shifted"] == N_NEITHER
    keys = list(meta)
    assert keys.index("secondary_removed_shared_identity") == keys.index("shifted_aux_removed") + 1
    src = meta["shifted_aux_source"]
    assert ":relabel=sec_map.csv:key=path:dropped=1:shared_ids_removed=9:dedup<0.95" in src
    assert meta["data_files"]["shifted:relabel:sec_map.csv"] == sha256_file(rmap)
    assert meta["data_files"]["relabel:prim_map.csv"] == sha256_file(pmap)
    assert meta["corpus_counts"] == {"images": 780, "identities": 260, "identities_ge_min_tgt_images": 260}
    assert meta["split_counts"]["1"] == {"dev_identities": 80, "tgt_identities": 120, "aux_identities": 60,
                                         "dev_images": 240, "tgt_images": 360, "aux_images": 180}


def test_isotropy_exclusion_without_overlap_removes_nothing(tmp_path):
    p_root, s_root, rmap, pmap = _fixture(tmp_path, 260, 3, 24)
    disjoint = pd.read_csv(rmap)
    disjoint["identity"] = [f"z{i}" for i in range(len(disjoint))]   # no identity shared with the primary
    disjoint.to_csv(tmp_path / "disjoint.csv", index=False)
    run(_iso_cfg(p_root, s_root, tmp_path / "disjoint.csv", True, pmap), tmp_path / "dis")
    meta = json.loads((tmp_path / "dis" / "metadata.json").read_text())
    assert meta["secondary_removed_shared_identity"] == 0
    assert meta["shifted_aux_removed"] == N_SHARED_DUP + N_DUP_ONLY
    assert meta["aux_sizes"]["shifted"] == N_SHARED_ONLY + N_NEITHER


def test_isotropy_refuses_exclusion_when_the_primary_has_no_relabel(tmp_path):
    p_root, s_root, rmap, _ = _fixture(tmp_path, 260, 3, 24)
    with pytest.raises(ValueError, match="primary corpus spec has no relabel"):
        run(_iso_cfg(p_root, s_root, rmap, True, None), tmp_path / "noprim")


def test_isotropy_refuses_exclusion_when_the_secondary_has_no_relabel(tmp_path):
    p_root, s_root, rmap, pmap = _fixture(tmp_path, 260, 3, 24)
    cfg = _iso_cfg(p_root, s_root, rmap, True, pmap)
    del cfg["data"]["shifted_aux"]["relabel"], cfg["data"]["shifted_aux"]["relabel_key"]
    with pytest.raises(ValueError, match="secondary corpus spec has no relabel"):
        run(cfg, tmp_path / "nosec")


def test_isotropy_without_the_flag_keeps_shared_identities(tmp_path):
    p_root, s_root, rmap, pmap = _fixture(tmp_path, 260, 3, 24)
    run(_iso_cfg(p_root, s_root, rmap, False, pmap), tmp_path / "off")
    meta = json.loads((tmp_path / "off" / "metadata.json").read_text())
    assert meta["secondary_removed_shared_identity"] is None
    assert meta["shifted_aux_removed"] == N_SHARED_DUP + N_DUP_ONLY
    assert meta["aux_sizes"]["shifted"] == N_SHARED_ONLY + N_NEITHER
    assert "shared_ids_removed" not in meta["shifted_aux_source"]


def test_isotropy_without_shifted_aux_records_none(tmp_path):
    p_root, _, _, _ = _fixture(tmp_path, 260, 3, 24)
    run({**ISO_CFG, "data": {"kind": "sin", "root": str(p_root), "model": "m"}}, tmp_path / "plain")
    meta = json.loads((tmp_path / "plain" / "metadata.json").read_text())
    assert meta["secondary_removed_shared_identity"] is None and meta["shifted_aux_source"] is None


SCALE_CFG = {
    "split": {"n_dev": 60, "n_tgt": 100, "seed": 1},
    "preconditioner": "fi:16:24",
    "schemes": ["lrp:full"],
    "hnsw": {"m": 16, "ef_construction": 100, "ef_search": [64]},
}


def _scale_raw(tmp_path, name, distractors, p_root, pmap):
    data = {"kind": "sin", "root": str(p_root), "model": "m"}
    if pmap is not None:
        data["relabel"] = str(pmap)
    cfg = {**SCALE_CFG, "data": data, "distractors": distractors}
    cfg_path = tmp_path / f"{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / name
    # faiss runs in a subprocess (see tests/test_hnsw_scale.py)
    r = subprocess.run([sys.executable, "-m", "ijiss_btp.experiments.hnsw_scale", "--config", str(cfg_path),
                        "--out", str(out)], capture_output=True, text=True)
    return r, out


def _scale(tmp_path, name, distractors, p_root, pmap):
    r, out = _scale_raw(tmp_path, name, distractors, p_root, pmap)
    assert (out / "metadata.json").exists(), r.stderr[-2000:]
    return json.loads((out / "metadata.json").read_text()), pd.read_csv(out / "scale.csv")


def _dis(s_root, rmap, sizes, exclude=True):
    d = {"kind": "sin", "root": str(s_root), "model": "m", "sizes": sizes, "seed": 2, "near_dup_cos": 0.95,
         "relabel": str(rmap), "relabel_key": "path"}
    if exclude:
        d["exclude_primary_identities"] = True
    return d


@needs_faiss
def test_scale_excludes_shared_identities_before_the_draw_and_the_filter(tmp_path):
    p_root, s_root, rmap, pmap = _fixture(tmp_path, 200, 3, 32)
    pool = N_SHARED_DUP + N_DUP_ONLY + N_SHARED_ONLY + N_NEITHER   # rows the relabel map assigns
    eligible = pool - N_SHARED_DUP - N_SHARED_ONLY
    meta, df = _scale(tmp_path, "on", _dis(s_root, rmap, [10, eligible]), p_root, pmap)
    assert meta["distractors_loaded"] == pool
    assert meta["secondary_removed_shared_identity"] == N_SHARED_DUP + N_SHARED_ONLY
    assert meta["distractors_available_before"] == eligible   # the draw comes after the exclusion
    assert meta["distractors_removed"] == N_DUP_ONLY
    assert meta["distractors_available_after"] == N_NEITHER
    keys = list(meta)
    assert keys.index("secondary_removed_shared_identity") == keys.index("distractors_removed") - 1
    assert ":key=path:dropped=1:shared_ids_removed=9:dedup<0.95" in meta["distractor_source"]
    assert meta["data_files"]["distractors:relabel:sec_map.csv"] == sha256_file(rmap)
    assert sorted(df.n_distractors) == [10, N_NEITHER]
    assert meta["corpus_counts"] == {"images": 600, "identities": 200, "identities_ge_min_tgt_images": 200}
    assert meta["split_counts"]["1"]["aux_identities"] == 40


@needs_faiss
def test_scale_runs_each_used_distractor_size_once(tmp_path):
    p_root, s_root, rmap, pmap = _fixture(tmp_path, 200, 3, 32)
    meta, df = _scale(tmp_path, "cap", _dis(s_root, rmap, [10, 43, 60]), p_root, pmap)
    assert meta["distractors_available_after"] == N_NEITHER == 40
    assert len(df) == 2   # one scheme, one ef: the capped 43 and 60 share the 40-distractor gallery
    assert sorted(zip(df.n_distractors, df.n_distractors_requested)) == [(10, 10), (40, 60)]
    assert meta["distractor_sizes_requested_to_used"] == {"10": 10, "43": 40, "60": 40}


@needs_faiss
def test_scale_without_the_flag_keeps_shared_identities(tmp_path):
    p_root, s_root, rmap, pmap = _fixture(tmp_path, 200, 3, 32)
    pool = N_SHARED_DUP + N_DUP_ONLY + N_SHARED_ONLY + N_NEITHER
    meta, _ = _scale(tmp_path, "off", _dis(s_root, rmap, [10, pool], exclude=False), p_root, pmap)
    assert meta["secondary_removed_shared_identity"] is None
    assert meta["distractors_available_before"] == pool
    assert meta["distractors_removed"] == N_SHARED_DUP + N_DUP_ONLY


@needs_faiss
def test_scale_refuses_exclusion_without_both_relabel_maps(tmp_path):
    p_root, s_root, rmap, pmap = _fixture(tmp_path, 200, 3, 32)
    r, out = _scale_raw(tmp_path, "noprim", _dis(s_root, rmap, [10, 40]), p_root, None)
    assert r.returncode != 0 and "primary corpus spec has no relabel" in r.stderr
    assert not (out / "metadata.json").exists()
    dis = _dis(s_root, rmap, [10, 40])
    del dis["relabel"], dis["relabel_key"]
    r, out = _scale_raw(tmp_path, "nosec", dis, p_root, pmap)
    assert r.returncode != 0 and "secondary corpus spec has no relabel" in r.stderr
    assert not (out / "metadata.json").exists()
