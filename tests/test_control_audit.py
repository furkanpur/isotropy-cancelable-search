# tests/test_control_audit.py
"""Overlap audit of the control encoder (scripts/data/control_audit.py).
Every manifest, key table and embedding here is a synthetic fixture; no real biometric data is used."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
spec = importlib.util.spec_from_file_location("control_audit", ROOT / "scripts" / "data" / "control_audit.py")
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

DIGEST = "d" * 64


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fixture(seed=0, planted=True):
    """Synthetic: 3 train fingers (source a), 4 D2 fingers (sources a, b); D2 image 0 is a planted near-duplicate of
    train image 0.  Returns (train_man, train_emb, d2_man, d2_emb, keys)."""
    rng = np.random.default_rng(seed)
    dim = 16
    tr = pd.DataFrame({"path": [f"t{i}" for i in range(9)], "identity": [f"a:s{i // 3}:f" for i in range(9)],
                       "source": "a", "subject": [f"s{i // 3}" for i in range(9)], "finger": "f",
                       "key": [f"kt{i}" for i in range(9)]})
    d2 = pd.DataFrame({"path": [f"d{i}" for i in range(12)],
                       "identity": ["a:s0:g"] * 3 + [f"{'a' if i < 6 else 'b'}:u{i // 3}:f" for i in range(3, 12)],
                       "source": ["a"] * 6 + ["b"] * 6, "subject": ["s0"] * 3 + [f"u{i // 3}" for i in range(3, 12)],
                       "finger": "f"})
    te = unit(rng.normal(size=(9, dim)))
    centres = unit(rng.normal(size=(4, dim)))
    de = unit(np.repeat(centres, 3, axis=0) + 0.05 * rng.normal(size=(12, dim)))
    if planted:
        de[0] = te[0]
    keys = pd.DataFrame({"path": list(tr["path"]) + list(d2["path"]),
                         "key": list(tr["key"]) + [f"kd{i}" for i in range(12)]})
    return tr, te, d2, de, keys


def test_blockwise_max_cosine_equals_dense_computation_for_any_block_size():
    _, te, _, de, _ = fixture()
    dense = (de @ te.T).max(axis=1)
    for block in (1, 5, 12, 100):
        np.testing.assert_allclose(ca.max_cosine(de, te, block=block), dense, rtol=0, atol=1e-6)


def test_genuine_percentile_uses_all_genuine_pairs():
    _, _, d2, de, _ = fixture()
    ident = d2["identity"].to_numpy()
    g = [float(de[i] @ de[j]) for i in range(12) for j in range(i + 1, 12) if ident[i] == ident[j]]
    assert len(g) == 12   # 4 fingers x 3 pairs
    assert ca.genuine_scores(de, ident).shape == (12,)
    assert ca.genuine_percentile(de, ident, 99.9) == pytest.approx(float(np.quantile(g, 0.999)), abs=1e-6)


def test_audit_counts_overlaps_and_planted_near_duplicate_overall_and_by_source():
    tr, te, d2, de, keys = fixture()
    res, mx, _ = ca.audit(tr, te, d2, de, keys, block=4)
    np.testing.assert_allclose(mx, (de @ te.T).max(axis=1), atol=1e-6)
    assert res["near_duplicates"]["threshold"] == ca.genuine_percentile(de, d2["identity"].to_numpy(), 99.9)
    assert res["identity_overlap"] == 0 and res["content_overlap"] == 0
    assert res["d2_rows"] == 12 and res["d2_keys_checked"] == 12
    nd = res["near_duplicates"]
    assert nd["threshold_rule"].startswith("D2 genuine")
    assert nd["genuine_pairs"] == 12 and nd["d2_images"] == 12 and nd["train_images"] == 9
    assert nd["images_at_or_above"]["total"] == 1 and nd["fingers_at_or_above"]["total"] == 1
    assert nd["images_at_or_above"]["by_source"] == {"a": 1, "b": 0}
    assert nd["fingers_at_or_above"]["by_source"] == {"a": 1, "b": 0}
    assert nd["max_cosine"]["overall"]["max"] == pytest.approx(1.0, abs=1e-6) and nd["threshold"] < 1.0
    assert set(nd["max_cosine"]["by_source"]) == {"a", "b"}
    q = nd["max_cosine"]["overall"]["quantiles"]
    assert q["0.5"] <= q["0.99"] <= nd["max_cosine"]["overall"]["max"]


def test_without_a_planted_duplicate_nothing_reaches_the_threshold():
    tr, te, d2, de, keys = fixture(planted=False)
    res, _, _ = ca.audit(tr, te, d2, de, keys)
    assert res["near_duplicates"]["images_at_or_above"]["total"] == 0
    assert res["near_duplicates"]["fingers_at_or_above"]["by_source"] == {"a": 0, "b": 0}


def test_identity_and_content_overlap_are_counted_not_assumed():
    tr, te, d2, de, keys = fixture()
    tr.loc[0, "identity"] = d2.loc[7, "identity"]           # a shared finger
    keys.loc[keys["path"] == "d3", "key"] = "kt4"           # a shared exact image
    res, _, _ = ca.audit(tr, te, d2, de, keys)
    assert res["identity_overlap"] == 1 and res["content_overlap"] == 1


def test_a_d2_row_without_a_key_fails_closed():
    tr, te, d2, de, keys = fixture()
    with pytest.raises(ValueError, match="no content key"):
        ca.audit(tr, te, d2, de, keys[keys["path"] != "d5"])


def test_embedding_row_count_must_match_manifest():
    tr, te, d2, de, keys = fixture()
    with pytest.raises(ValueError, match="rows"):
        ca.audit(tr, te[:-1], d2, de, keys)


def _write_run(d, man, emb, weights=DIGEST, status="ok", smoke=False):
    d.mkdir(parents=True)
    man.to_parquet(d / "manifest.parquet")
    np.save(d / "emb_ctrlvit.npy", emb)
    m = ca.sha256_file(d / "manifest.parquet")
    (d / "embed_summary.json").write_text(json.dumps({
        "status": status, "weights_sha256": weights, "manifest_sha256": m, "input_manifest_sha256": m,
        "emb_sha256": ca.sha256_file(d / "emb_ctrlvit.npy"), "data_source": "synthetic fixture", "smoke": smoke, "complete": True,
        "self_consistency": {"rows": 3, "min_cosine": 1.0},
        "input_images": {"combined_sha256": "c" * 64, "images": len(man)}}))


def _setup(tmp_path, weights_d2=DIGEST, c1_subjects=None):
    tr, te, d2, de, keys = fixture()
    _write_run(tmp_path / "train", tr, te)
    _write_run(tmp_path / "d2", d2, de, weights=weights_d2)
    keys.to_parquet(tmp_path / "keys.parquet")
    ts = tmp_path / "trainset"
    ts.mkdir()
    flags = d2.drop_duplicates("identity")[["identity", "source"]].assign(subject_shared=[True, False, False, False])
    flags.to_parquet(ts / "d2_subject_flags.parquet")
    c1 = ca.disjointness(tr, d2, keys)["subjects"] if c1_subjects is None else c1_subjects
    (ts / "training_set.json").write_text(json.dumps({"disjointness": {"subjects": c1}}))
    return ["--train", str(tmp_path / "train"), "--d2", str(tmp_path / "d2"), "--compilation-keys", str(tmp_path / "keys.parquet"),
            "--trainset", str(ts), "--json", str(tmp_path / "out" / "overlap_audit.json"), "--label", "synthetic fixture"]


def test_main_writes_aggregate_json_with_provenance_and_no_paths_or_identities(tmp_path, monkeypatch):
    monkeypatch.setenv("CANVEC_IMAGE", "canvec/exp:core")
    monkeypatch.setenv("CANVEC_IMAGE_ID", "sha256:test")
    args = _setup(tmp_path)
    assert ca.main(args) == 0
    text = (tmp_path / "out" / "overlap_audit.json").read_text()
    res = json.loads(text)
    assert res["data_source"] == "synthetic fixture"
    assert res["weights_sha256"] == DIGEST
    assert res["train_embed"]["weights_sha256"] == DIGEST and len(res["train_embed"]["manifest_sha256"]) == 64
    assert res["d2_embed"]["input_images_sha256"] == "c" * 64 and res["train_embed"]["input_images_sha256"] == "c" * 64
    assert res["d2_embed"]["manifest_sha256"] == ca.sha256_file(tmp_path / "d2" / "manifest.parquet")
    assert res["subjects"]["d2_fingers_with_shared_subject"] == {"total": 1, "by_source": {"a": 1}}
    assert res["subjects"]["c1_subjects"]["by_subject_key"]["shared"] == 1
    assert res["subjects"]["agreement"] == {"flags_vs_recomputed": True, "c1_json_vs_recomputed": True}
    pv = res["provenance"]
    assert {"script_sha256", "numpy", "pandas", "python", "env"} <= set(pv)
    assert pv["env"]["image"] == "canvec/exp:core" and pv["env"]["image_id"] == "sha256:test"
    for blk in (res["train_embed"], res["d2_embed"]):
        assert blk["status"] == "ok" and blk["smoke"] is False and blk["weights_sha256"] == DIGEST
        assert blk["self_consistency"] == {"rows": 3, "min_cosine": 1.0} and blk["complete"] is True
    assert "finger_level" in res
    assert "a:u0:f" not in text and "a:s0:g" not in text
    assert (tmp_path / "out" / "max_cosine_d2.npy").exists()   # per-image values stay beside the JSON, on the host


def test_main_refuses_runs_embedded_with_different_weights(tmp_path):
    args = _setup(tmp_path, weights_d2="e" * 64)
    assert ca.main(args) == 1
    assert not (tmp_path / "out" / "overlap_audit.json").exists()


def test_main_exits_nonzero_after_writing_when_overlap_is_not_zero(tmp_path):
    args = _setup(tmp_path)
    keys = pd.read_parquet(tmp_path / "keys.parquet")
    keys.loc[keys["path"] == "d3", "key"] = "kt4"
    keys.to_parquet(tmp_path / "keys.parquet")
    assert ca.main(args) == 1
    assert json.loads((tmp_path / "out" / "overlap_audit.json").read_text())["content_overlap"] == 1


def test_main_refuses_to_overwrite(tmp_path):
    args = _setup(tmp_path)
    assert ca.main(args) == 0
    assert ca.main(args) == 2


def test_main_requires_a_label(tmp_path):
    args = _setup(tmp_path)
    i = args.index("--label")
    with pytest.raises(SystemExit):
        ca.main(args[:i] + args[i + 2:])


def test_main_exits_nonzero_when_subject_counts_disagree_with_c1(tmp_path):
    args = _setup(tmp_path, c1_subjects={"by_subject_key": {"train": 3, "d2": 9, "shared": 0}})
    assert ca.main(args) == 1
    res = json.loads((tmp_path / "out" / "overlap_audit.json").read_text())
    assert res["subjects"]["agreement"]["c1_json_vs_recomputed"] is False


@pytest.mark.parametrize("kw", [{"status": "failed"}, {"smoke": True}])
def test_read_embed_run_refuses_runs_that_are_not_complete_ok_runs(tmp_path, kw):
    tr, te, *_ = fixture()
    _write_run(tmp_path / "r", tr, te, **kw)
    with pytest.raises(ValueError, match="not a complete ok run"):
        ca._read_embed_run(tmp_path / "r", "train")


def test_read_embed_run_refuses_a_manifest_or_embeddings_that_differ_from_the_summary(tmp_path):
    tr, te, *_ = fixture()
    _write_run(tmp_path / "r", tr, te)
    np.save(tmp_path / "r" / "emb_ctrlvit.npy", te[::-1].copy())
    with pytest.raises(ValueError, match="differ from the hashes"):
        ca._read_embed_run(tmp_path / "r", "train")
    _write_run(tmp_path / "q", tr, te)
    tr.iloc[::-1].reset_index(drop=True).to_parquet(tmp_path / "q" / "manifest.parquet")
    with pytest.raises(ValueError, match="differ from the hashes"):
        ca._read_embed_run(tmp_path / "q", "train")


# ---- finger-level statistic ---------------------------------------------------------------------------------
def _loose(rng, centre, n, dim):
    """n images of one finger: centre plus large noise (cosine to the centre about 0.6)."""
    return unit(centre + rng.normal(size=(n, dim)) * (1.3 / np.sqrt(dim)))


def _tight(rng, centre, n, dim):
    return unit(centre + 0.03 * rng.normal(size=(n, dim)))


def finger_fixture():
    """Synthetic.  Training fingers T0..T3 (loose).  D2: G0 is the same physical finger as T0 under another identity key
    (loose, independent noise), G1..G3 are tight clusters (so the D2 genuine 99.9th percentile is high), G4 is a loose
    finger absent from training."""
    rng = np.random.default_rng(7)
    dim, n = 128, 10
    cs = unit(rng.normal(size=(6, dim)))
    te = np.vstack([_loose(rng, cs[i], n, dim) for i in range(4)])
    de = np.vstack([_loose(rng, cs[0], n, dim), _tight(rng, cs[4], n, dim), _tight(rng, cs[5], n, dim),
                    _tight(rng, unit(rng.normal(size=(1, dim)))[0], n, dim),
                    _loose(rng, unit(rng.normal(size=(1, dim)))[0], n, dim)])
    tr = pd.DataFrame({"path": [f"t{i}" for i in range(4 * n)], "identity": [f"a:s{i // n}:f" for i in range(4 * n)],
                       "source": "a", "subject": [f"s{i // n}" for i in range(4 * n)], "finger": "f",
                       "key": [f"kt{i}" for i in range(4 * n)]})
    d2 = pd.DataFrame({"path": [f"d{i}" for i in range(5 * n)],
                       "identity": [f"{'a' if i // n < 3 else 'b'}:u{i // n}:f" for i in range(5 * n)],
                       "source": ["a" if i // n < 3 else "b" for i in range(5 * n)],
                       "subject": [f"u{i // n}" for i in range(5 * n)], "finger": "f"})
    keys = pd.DataFrame({"path": list(tr["path"]) + list(d2["path"]),
                         "key": list(tr["key"]) + [f"kd{i}" for i in range(5 * n)]})
    return tr, te, d2, de, keys


def test_finger_mean_cross_cosine_equals_dense_mean_over_all_image_pairs():
    tr, te, d2, de, _ = finger_fixture()
    r = ca.finger_overlap(tr, te, d2, de)
    n = 10
    own = [float(np.mean([de[i] @ de[j] for i in range(f * n, f * n + n) for j in range(i + 1, f * n + n)]))
           for f in range(5)]
    cross = [max(float((de[f * n:f * n + n] @ te[t * n:t * n + n].T).mean()) for t in range(4)) for f in range(5)]
    assert r["overall"]["fingers"] == 5 and r["train_fingers"] == 4 and r["d2_fingers_single_image_excluded"] == 0
    assert r["overall"]["own_genuine_mean"]["mean"] == pytest.approx(np.mean(own), abs=1e-6)
    assert r["overall"]["max_cross_mean_cosine"]["mean"] == pytest.approx(np.mean(cross), abs=1e-6)
    assert r["overall"]["max_cross_mean_cosine"]["max"] == pytest.approx(max(cross), abs=1e-6)


def test_finger_statistic_flags_a_planted_same_finger_in_the_fixture_that_the_image_scan_misses():
    tr, te, d2, de, keys = finger_fixture()
    res, _, _ = ca.audit(tr, te, d2, de, keys)
    assert res["identity_overlap"] == 0 and res["content_overlap"] == 0
    assert res["near_duplicates"]["images_at_or_above"]["total"] == 0          # the image-level scan sees nothing
    fl = res["finger_level"]
    assert fl["overall"]["reaching_own_genuine_mean"] >= 1                     # G0 reaches its own genuine mean
    assert fl["by_source"]["a"]["reaching_own_genuine_mean"] >= 1
    assert fl["by_source"]["a"]["max_minus_own_genuine_mean"]["max"] > -0.1
    assert fl["by_source"]["b"]["reaching_own_genuine_mean"] == 0              # unplanted fingers, incl. loose G4
    assert fl["by_source"]["b"]["max_minus_own_genuine_mean"]["max"] < -0.2


def test_finger_statistic_excludes_single_image_fingers_and_names_polyu_cross_as_null_reference():
    tr, te, d2, de, _ = finger_fixture()
    d2 = pd.concat([d2, d2.iloc[:1].assign(identity="z:single", source="polyu_cross")], ignore_index=True)
    d2.loc[40:49, "source"] = "polyu_cross"
    de = np.vstack([de, de[:1]])
    r = ca.finger_overlap(tr, te, d2, de)
    assert r["d2_fingers_single_image_excluded"] == 1 and r["null_reference_source"] == "polyu_cross"
    assert r["by_source"]["polyu_cross"]["fingers"] == 1


# ---- characterisation of flagged fingers, label-structure scan, sensitivity (Task C3) -----------------
def meta_fixture():
    """Synthetic.  Training fingers T0 (casia_v5 007 L1), T1 (casia_v5 008 L2) tight; T2, T3 (sd301) loose.  D2 fingers:
    G0 casia_ageing 0007 L_index, same physical finger as T0 (planted, loose: flagged; linked family, same numeric subject
    and position); G1 sd302 00002223 03 (tight, random: structural candidate, not flagged); G2 sd302 00002224 07 (tight:
    same numeric subject as T3, other position); G3 casia_v5 008 L3 (same centre as T1, other finger of the same subject:
    flagged); G4 uareu (loose, unrelated); G5 casia_v5 008 L0 (loose, unrelated, same subject as G3)."""
    rng = np.random.default_rng(11)
    dim, n = 64, 20
    c = unit(rng.normal(size=(12, dim)))

    def mk(centre, noise):
        return unit(centre + rng.normal(size=(n, dim)) * (noise / np.sqrt(dim)))
    tspec = [("casia_v5", "007", "L1", "casia_v5:007:L1", c[0], 0.6), ("casia_v5", "008", "L2", "casia_v5:008:L2", c[1], 0.6),
             ("sd301", "00002223", "03", "sd301:00002223:03", c[2], 1.3),
             ("sd301", "00002224", "04", "sd301:00002224:04", c[3], 1.3)]
    dspec = [("casia_ageing", "0007", "L_index", "casia_ageing:0007:L_index", c[0], 1.3),
             ("sd302", "00002223", "03", "sd302:00002223:03", c[4], 0.03),
             ("sd302", "00002224", "07", "sd302:00002224:07", c[5], 0.03),
             ("casia_v5", "008", "L3", "casia_v5:008:L3", c[1], 1.3),
             ("uareu", "012", "2", "uareu:012:2", c[6], 1.3),
             ("casia_v5", "008", "L0", "casia_v5:008:L0", c[7], 1.3)]

    def frame(spec, tag):
        rows = [(f"{tag}{i}_{j}", ident, src, sub, fin, "dev")
                for i, (src, sub, fin, ident, _, _) in enumerate(spec) for j in range(n)]
        return pd.DataFrame(rows, columns=["path", "identity", "source", "subject", "finger", "device"])
    tr, d2 = frame(tspec, "t"), frame(dspec, "d")
    tr["key"] = tr["path"].map(lambda p: "k" + p)
    te = np.vstack([mk(s[4], s[5]) for s in tspec])
    de = np.vstack([mk(s[4], s[5]) for s in dspec])
    keys = pd.DataFrame({"path": list(tr["path"]) + list(d2["path"]),
                         "key": list(tr["key"]) + ["k" + p for p in d2["path"]]})
    return tr, te, d2, de, keys


def test_characterisation_counts_the_categories_of_flagged_fingers_and_of_all_fingers():
    tr, te, d2, de, _ = meta_fixture()
    tab = ca.finger_table(tr, te, d2, de)
    assert set(tab["dmeta"]["identity"][tab["reach"]]) == {"casia_ageing:0007:L_index", "casia_v5:008:L3"}
    ch = ca.characterise(tab, te, de)
    fl = ch["flagged"]["overall"]
    assert fl["n"] == 2 and fl["same_subject_key"] == 1 and fl["same_source"] == 1
    assert fl["source_pairs"] == {"casia_ageing->casia_v5": 1, "casia_v5->casia_v5": 1}
    assert fl["position_comparable"] == 2 and fl["same_finger_position"] == 1       # L_index is read as casia_v5 L1
    assert fl["linked_family_same_numeric_subject"] == 1
    assert fl["best_cross_image_cos"]["n"] == 2 and fl["own_genuine_mean"]["n"] == 2 and fl["images"]["min"] == 20
    assert set(ch["flagged"]["by_source"]) == {"casia_ageing", "casia_v5"}
    allb = ch["all"]["overall"]
    assert allb["n"] == 6 and set(ch["all"]["by_source"]) == {"casia_ageing", "casia_v5", "sd302", "uareu"}
    assert sum(allb["source_pairs"].values()) == 6
    assert ch["position_normalisation"]["map"]["casia_ageing"]["casia_ageing:L_index"] == "left_index"
    assert ch["position_vocabulary"]["d2"]["casia_ageing"]["values"] == ["casia_ageing:L_index"]


def test_best_image_pair_cosines_equal_a_dense_computation():
    tr, te, d2, de, _ = meta_fixture()
    tab = ca.finger_table(tr, te, d2, de)
    ch = ca.characterise(tab, te, de)
    n = 20
    g0 = de[:n]
    best_t = max((g0 @ te[t * n:(t + 1) * n].T).max() for t in range(4))
    b = ch["flagged"]["by_source"]["casia_ageing"]
    assert b["best_cross_image_cos"]["max"] == pytest.approx(best_t, abs=1e-6)
    assert b["best_genuine_image_cos"]["max"] == pytest.approx((g0 @ g0.T)[np.triu_indices(n, 1)].max(), abs=1e-6)


def test_label_structure_scan_finds_unlinked_cross_source_candidates_without_embeddings():
    tr, te, d2, de, _ = meta_fixture()
    tab = ca.finger_table(tr, te, d2, de)
    ls, cand = ca.label_structure(tab)
    assert cand == {"casia_ageing:0007:L_index", "sd302:00002223:03"}
    assert ls["candidates"] == {"total": 2, "by_source_pair": {"casia_ageing->casia_v5": 1, "sd302->sd301": 1}}
    assert ls["candidates_among_flagged"] == 1
    assert ls["candidates_without_position_requirement"]["total"] == 3          # G2 shares only the numeric subject
    assert ls["candidates_best_match_is_candidate"] >= 1


def test_sensitivity_subsets_and_aucs_match_the_label_check():
    tr, te, d2, de, _ = meta_fixture()
    tab = ca.finger_table(tr, te, d2, de)
    _, cand = ca.label_structure(tab)
    ex = ca.exclusion_sets(tab, cand)
    assert ex["a_as_is"] == set()
    assert ex["b_without_flagged"] == {"casia_ageing:0007:L_index", "casia_v5:008:L3"}
    assert ex["c_without_flagged_subjects"] == ex["b_without_flagged"] | {"casia_v5:008:L0"}
    assert ex["d_without_structural_candidates"] == cand
    rng = np.random.default_rng(3)
    other = unit(de + 0.3 * rng.normal(size=de.shape))
    res = ca.sensitivity(d2, {"ctrlvit": de, "fingervit": other}, ex, bins=200, block=64)
    ref = ca.label_check.label_check(d2, {"ctrlvit": de}, bins=200, block=64)["models"]["ctrlvit"]["all"]
    a = res["subsets"]["a_as_is"]
    assert a["images"] == len(d2) and a["fingers"] == 6 and a["encoders"]["ctrlvit"]["auc"] == ref["auc"]
    assert a["encoders"]["ctrlvit"]["same_device"]["auc"] == ref["same_device"]["auc"]
    assert res["subsets"]["b_without_flagged"]["fingers"] == 4 and res["subsets"]["b_without_flagged"]["images"] == 80
    assert res["subsets"]["c_without_flagged_subjects"]["fingers"] == 3
    assert set(a["encoders"]) == {"ctrlvit", "fingervit"}


def test_audit_includes_characterisation_label_structure_and_main_adds_sensitivity(tmp_path):
    tr, te, d2, de, keys = meta_fixture()
    res, _, _ = ca.audit(tr, te, d2, de, keys)
    assert "characterisation" in res and res["label_structure"]["candidates"]["total"] == 2
    _write_run(tmp_path / "train", tr, te)
    _write_run(tmp_path / "d2", d2, de)
    keys.to_parquet(tmp_path / "keys.parquet")
    ts = tmp_path / "trainset"
    ts.mkdir()
    dj = ca.disjointness(tr, d2, keys)
    f = d2.drop_duplicates("identity")[["identity", "source"]].reset_index(drop=True)
    f["subject_shared"] = ca.subject_key(f["identity"]).isin(set(ca.subject_key(tr["identity"]))).to_numpy()
    f.to_parquet(ts / "d2_subject_flags.parquet")
    (ts / "training_set.json").write_text(json.dumps({"disjointness": {"subjects": dj["subjects"]}}))
    base = tmp_path / "base"
    base.mkdir()
    d2.to_parquet(base / "manifest.parquet")
    np.save(base / "emb_fingervit.npy", de)
    np.save(base / "emb_deepprint.npy", de)
    args = ["--train", str(tmp_path / "train"), "--d2", str(tmp_path / "d2"), "--compilation-keys", str(tmp_path / "keys.parquet"),
            "--trainset", str(ts), "--json", str(tmp_path / "out" / "o.json"), "--label", "synthetic fixture",
            "--d2-baselines", str(base)]
    assert ca.main(args) == 0
    out = json.loads((tmp_path / "out" / "o.json").read_text())
    assert set(out["sensitivity"]["subsets"]["a_as_is"]["encoders"]) == {"ctrlvit", "fingervit", "deepprint"}
    assert "emb_fingervit.npy" in out["inputs_sha256"]["d2_baselines"]
    # a baseline manifest that differs from the D2 manifest is refused
    d2.iloc[::-1].reset_index(drop=True).to_parquet(base / "manifest.parquet")
    args[args.index("--json") + 1] = str(tmp_path / "out2" / "o.json")
    assert ca.main(args) == 1
    assert not (tmp_path / "out2" / "o.json").exists()


def test_finger_codes_are_written_with_a_source_prefix_never_bare():
    import re
    man = pd.DataFrame({"source": ["casia_v5"] * 8 + ["sd301"] * 2,
                        "finger": [f"{h}{i}" for h in "LR" for i in range(4)] + ["01", "02"]})
    text = json.dumps({"map": ca.position_map_json(), "vocabulary": ca._vocabulary(man, cap=20)})
    # the export scan's rule: a bare review-step-like code (a capital R and a digit) after a space, bracket, slash or quote
    assert not re.search(r"(?:^|(?<=[\s(\[/\"']))R[1-9]", text)
    assert '"casia_v5:R1": "right_index"' in text and "casia_v5:R3" in text and "sd301:01" in text
    assert ca.POSITION_NOTE.count("<source>:<code>") == 1
