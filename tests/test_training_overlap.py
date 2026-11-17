# tests/test_training_overlap.py
"""Worst-case training-overlap measurement of scripts/data/training_overlap.py (tiny synthetic fixtures, no GPU)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/data"))  # importable by name so spawned pool workers can unpickle
import extract_d2 as eg  # noqa: E402
import training_overlap as to  # noqa: E402

# compilation-like identities (source:subject:finger) and a tiny labelled corpus of flat images: a flat image keeps its
# coarse key under any resize and container (the 4-bit value on a 16 x 16 grid), so packaged copies are easy to make.
F1, F2, F3, F4, X, Y, Z = "s1:7:R1", "s1:8:L2", "s2:3:R1", "s2:4:R2", "s2:9:L1", "s1:5:R3", "s2:6:L3"
COMPILATION_ROWS = [("images/s1/g1.png", F1, "s1", 24), ("images/s1/g2.png", F1, "s1", 40),
             ("images/s1/g3.bmp", F2, "s1", 56), ("images/s2/g4.png", F3, "s2", 72),
             ("images/s2/g5.png", X, "s2", 72),                      # same image as g4 under another identity
             ("images/s2/g6.png", F4, "s2", 88), ("images/s1/g7.png", Y, "s1", 104),
             ("images/s2/g8.png", "", "s2", 120),                     # a compilation image without an identity
             ("images/s2/missing.png", Z, "s2", None)]                # a manifest row without a file
IDENTITIES = (F1, F2, F3, F4, X, Y, Z)


def _flat(path: Path, value: int, size=(40, 30), mode="L") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = size if mode == "L" else size + (3,)
    Image.fromarray(np.full(shape, value, np.uint8), mode).save(path)
    return path


def _junk(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not an image")
    return path


def _copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.save(dst)
    return dst


def _compilation(root: Path) -> Path:
    rows = []
    for path, ident, src, value in COMPILATION_ROWS:
        if value is not None:
            _flat(root / path, value)
        rows.append(dict(path=path, identity=ident, source=src, subject="1", finger="1", device="", capture="contact",
                         dpi="", subset="", exclusion="none", nfiq2="", quality_status=""))
    pd.DataFrame(rows).to_csv(root / "MANIFEST.csv", index=False)
    return root


def _key_table(compilation: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    lab, meta = eg.build_compilation_keys(compilation, workers=1)
    lab.to_parquet(out / "compilation_keys.parquet", index=False)
    (out / "compilation_keys.json").write_text(json.dumps(meta))
    return out / "compilation_keys.parquet"


def _d1(tmp: Path, compilation: Path) -> tuple[Path, Path, Path]:
    """First corpus: copies of compilation images in another container, plus one image the relabel map does not assign."""
    imgs = tmp / "d1_images"
    for name, src in (("d1_a.bmp", "images/s1/g1.png"), ("d1_b.bmp", "images/s1/g3.bmp"),
                      ("d1_c.bmp", "images/s2/g4.png"), ("d1_d.bmp", "images/s2/g6.png")):
        _copy(compilation / src, imgs / name)
    _flat(imgs / "d1_u.bmp", 136)
    names = ["d1_a.bmp", "d1_b.bmp", "d1_c.bmp", "d1_d.bmp", "d1_u.bmp"]
    pd.DataFrame({"path": [f"/elsewhere/test/{n}" for n in names], "identity": "old"}).to_parquet(
        tmp / "d1_manifest.parquet")
    pd.DataFrame({"file": names[:4], "identity": [F1, F2, F3, F4]}).to_csv(tmp / "d1_relabel.csv", index=False)
    return imgs, tmp / "d1_manifest.parquet", tmp / "d1_relabel.csv"


def _assert_no_leak(text: str, words=()) -> None:
    for w in tuple(words) + IDENTITIES:
        assert w not in text, w
    for ext in (".png", ".bmp", ".jpg", ".tif"):
        assert ext not in text, ext


# ---- keys and file lists ----
def test_coarse_key_known_answer(tmp_path):
    arr = (np.arange(256).reshape(16, 16) * 7 % 256).astype(np.uint8)   # 16 x 16: the resize is the identity
    Image.fromarray(arr, "L").save(tmp_path / "k.png")
    assert to.coarse_key(tmp_path / "k.png") == hashlib.sha1((arr >> 4).tobytes()).hexdigest()


def test_coarse_key_matches_a_packaged_copy_of_a_flat_image(tmp_path):
    orig = _flat(tmp_path / "orig.bmp", 200, size=(300, 200))
    packaged = _flat(tmp_path / "packaged.png", 200, size=(224, 224), mode="RGB")
    want = hashlib.sha1(np.full((16, 16), 200 >> 4, np.uint8).tobytes()).hexdigest()
    assert to.coarse_key(orig) == to.coarse_key(packaged) == want


def test_coarse_key_resizes_with_bilinear(tmp_path):
    """Known answer on an image that is neither flat nor 16 x 16; BICUBIC (or NEAREST) would give another key."""
    arr = ((np.arange(37 * 53) * 97) % 251).astype(np.uint8).reshape(37, 53)
    Image.fromarray(arr, "L").save(tmp_path / "pattern.png")
    assert to.coarse_key(tmp_path / "pattern.png") == "cbb1eb7840e696786b1afcf915275377a56f3456"
    with Image.open(tmp_path / "pattern.png") as im:
        for other in (Image.BICUBIC, Image.NEAREST):
            thumb = np.asarray(im.convert("L").resize((16, 16), other), dtype=np.uint8) >> 4
            assert hashlib.sha1(thumb.tobytes()).hexdigest() != "cbb1eb7840e696786b1afcf915275377a56f3456"


def test_coarse_key_tells_content_apart_and_is_none_for_unreadable(tmp_path):
    arr = np.full((16, 16), 100, np.uint8)
    Image.fromarray(arr, "L").save(tmp_path / "a.png")
    arr[3, 4] = 116                                                        # one thumbnail cell moves up one level
    Image.fromarray(arr, "L").save(tmp_path / "b.png")
    assert to.coarse_key(tmp_path / "a.png") != to.coarse_key(tmp_path / "b.png")
    assert to.coarse_key(_junk(tmp_path / "c.png")) is None
    assert to.coarse_key(tmp_path / "absent.png") is None


def test_list_files_digest_and_expected_count(tmp_path):
    for rel in ("b.png", "a.bmp", "sub/c.tif"):
        _junk(tmp_path / "p" / rel)
    names = to.list_files(tmp_path / "p")
    assert names == ["a.bmp", "b.png", "sub/c.tif"]
    assert to.list_digest(names) == hashlib.sha256(b"a.bmp\nb.png\nsub/c.tif\n").hexdigest()
    assert to.write_list(tmp_path / "list.txt", names) == to.list_digest(names)
    assert hashlib.sha256((tmp_path / "list.txt").read_bytes()).hexdigest() == to.list_digest(names)
    assert to.pool_listing(tmp_path / "p", expect=3) == names
    assert to.pool_listing(tmp_path / "p", expect=None) == names
    with pytest.raises(SystemExit, match="expected 4"):
        to.pool_listing(tmp_path / "p", expect=4)


# ---- (a) exact overlap ----
def _table():
    rows = [(F1, "s1", "k1a"), (F1, "s1", "k1b"), (F2, "s1", "k2a"), (F3, "s2", "k_amb"), (F4, "s2", "k4a"),
            (X, "s2", "k_amb"), (Y, "s1", "ky"), (Z, "s2", None)]
    t = pd.DataFrame(rows, columns=["identity", "source", "key"])
    return t[["identity", "key"]], dict(zip(t["identity"], t["source"]))


def test_exact_overlap_counts_images_fingers_and_pool_matches():
    table, src = _table()
    d1 = pd.DataFrame({"identity": [F1, F1, F2, F3, F4, ""], "key": ["k1a", "k1a", "k2a", "k_amb", None, "k_u"]})
    pool = ["k1b", "k2a", "k_amb", "ky", "k_other", None, "k2a"]
    res = to.exact_overlap(d1, {F1, F2, F3, F4}, table, src, pool)
    img = res["d1_images"]
    # the D1 images of F2 (k2a) and F3 (k_amb) occur in the pool; F1's two images (k1a) and F4's unreadable one do not
    assert (img["in_pool"], img["of"], img["fraction"]) == (2, 5, 0.4)
    assert img["per_source"] == {"s1": {"in_pool": 1, "of": 3, "fraction": pytest.approx(1 / 3, abs=1e-6)},
                                 "s2": {"in_pool": 1, "of": 2, "fraction": 0.5}}
    assert img["unassigned"] == {"images": 1, "in_pool": 0} and img["unreadable"] == 1
    fin = res["d1_fingers"]
    # F1 (through its other image k1b) and F2 count; F3 only through a key shared with X; F4 not at all
    assert (fin["in_pool"], fin["of"], fin["fraction"]) == (2, 4, 0.5)
    assert fin["per_source"] == {"s1": {"in_pool": 2, "of": 2, "fraction": 1.0},
                                 "s2": {"in_pool": 0, "of": 2, "fraction": 0.0}}
    assert fin["only_via_ambiguous_keys"] == 1 and fin["worst_case"]["in_pool"] == 3
    assert fin["compilation_images"] == 5 and fin["compilation_images_in_pool"] == 3
    assert res["pool_match"] == {"files": 7, "exact": {"images": 4, "per_source": {"s1": 4}},
                                 "exact_ambiguous": {"images": 1, "keys": 1}, "none": 1, "unreadable": 1}


def test_exact_overlap_refuses_identities_missing_from_the_key_table():
    table, src = _table()
    d1 = pd.DataFrame({"identity": [F1], "key": ["k1a"]})
    with pytest.raises(ValueError, match="not in the compilation key table"):
        to.exact_overlap(d1, {F1, "s9:1:R1"}, table, src, ["k1a"])


def test_exact_command_end_to_end(tmp_path):
    compilation = _compilation(tmp_path / "compilation")
    keys = _key_table(compilation, tmp_path / "keys")
    imgs, manifest, relabel = _d1(tmp_path, compilation)
    pool = tmp_path / "pool_a"
    for name, src in (("p1.png", "images/s1/g2.png"), ("p2.png", "images/s1/g3.bmp"),
                      ("p3.tif", "images/s2/g5.png"), ("p4.png", "images/s1/g7.png")):
        _copy(compilation / src, pool / name)
    _flat(pool / "p5.png", 152)
    _junk(pool / "p6.png")
    out = tmp_path / "out"
    to.main(["exact", "--pool", str(pool), "--expect-files", "6", "--d1-manifest", str(manifest),
             "--d1-images", str(imgs), "--d1-map", str(relabel), "--compilation", str(compilation), "--compilation-keys", str(keys),
             "--out", str(out), "--json", str(out / "a.json"), "--workers", "1"])
    text = (out / "a.json").read_text()
    res = json.loads(text)
    assert res["pool"]["files"] == 6 and res["pool"]["label"].endswith("6 files")
    assert res["pool"]["file_list_sha256"] == hashlib.sha256((out / "pool_a_files.txt").read_bytes()).hexdigest()
    img, fin, pm = res["d1_images"], res["d1_fingers"], res["pool_match"]
    # D1 images in the pool: d1_b (F2, s1) and d1_c (F3, s2: the pool holds X's identical image)
    assert (img["in_pool"], img["of"]) == (2, 4) and img["unassigned"] == {"images": 1, "in_pool": 0}
    assert img["per_source"]["s1"]["in_pool"] == 1 and img["per_source"]["s2"]["in_pool"] == 1
    # fingers: F1 (other impression g2) and F2 through unique keys; F3 only through the key it shares with X
    assert (fin["in_pool"], fin["of"], fin["only_via_ambiguous_keys"]) == (2, 4, 1)
    assert pm == {"files": 6, "exact": {"images": 3, "per_source": {"s1": 3}},
                  "exact_ambiguous": {"images": 1, "keys": 1}, "none": 1, "unreadable": 1}
    assert res["inputs_sha256"]["d1_relabel_map"] == eg.sha256_file(relabel)
    assert res["provenance"]["script_sha256"] == eg.sha256_file(ROOT / "scripts/data/training_overlap.py")
    assert res["thresholds"] and "data_source" in res
    _assert_no_leak(text, ("d1_a", "p1", "elsewhere", str(tmp_path)))
    assert res["outputs_sha256"] == {"pool_a_keys": eg.sha256_file(out / "pool_a_keys.parquet"),
                                     "d1_exact_matches": eg.sha256_file(out / "d1_exact_matches.parquet")}


# ---- (b) coarse matching ----
def test_unique_matches_keeps_only_single_reference_hits():
    idx, status = to.unique_matches(["c1", "c2", "c9", None, "c3", "c1"], ["c1", "c2", "c2", None, "c3"])
    assert idx.tolist() == [0, -1, -1, -1, 4, 0]
    assert status == ["unique", "ambiguous", "unmatched", "unreadable", "unique", "unique"]


def test_pool_b_match_categories():
    ref = pd.DataFrame({"coarse": ["c1", "c2", "c2", "c3"], "source": ["s1", "s1", "s2", "s2"]})
    res = to.pool_b_match(["e1", "eX", "eY", "eZ", None, "eW"], ["c1", "c3", "c2", "c9", None, "c3"], {"e1"}, ref)
    assert res == {"files": 6, "exact": 1, "coarse_unique": {"images": 2, "per_source": {"s2": 2}},
                   "coarse_ambiguous": 1, "none": 1, "unreadable": 1}


def _old_and_pool_b(tmp: Path, compilation: Path) -> tuple[Path, Path]:
    old = tmp / "old"
    for name, value in (("o1.png", 24), ("o2.png", 72), ("o3.png", 200), ("o4.png", 120)):
        _flat(old / name, value, size=(224, 224), mode="RGB")              # packaged 224 x 224 RGB copies
    _junk(old / "o5.png")
    pool = tmp / "pool_b"
    for name, value in (("q1.png", 56), ("q2.png", 72), ("q3.png", 216)):
        _flat(pool / name, value, size=(224, 224), mode="RGB")
    _copy(compilation / "images/s1/g7.png", pool / "q4.png")                      # an exact copy of a labelled image
    _junk(pool / "q5.png")
    return old, pool


def test_coarse_command_pairs_and_pool_match(tmp_path):
    compilation = _compilation(tmp_path / "compilation")
    keys = _key_table(compilation, tmp_path / "keys")
    old, pool = _old_and_pool_b(tmp_path, compilation)
    out = tmp_path / "out"
    to.main(["coarse", "--pool", str(pool), "--expect-files", "5", "--old-images", str(old), "--compilation", str(compilation),
             "--compilation-keys", str(keys), "--out", str(out), "--workers", "1"])
    pairs = pd.read_csv(out / "calibration_pairs.csv")
    # o1 matches g1 only; o4 matches the unlabelled g8 only; o2 matches g4 and g5 (excluded); o3 matches nothing
    assert pairs.values.tolist() == [["o1.png", "images/s1/g1.png"], ["o4.png", "images/s2/g8.png"]]
    s = json.loads((out / "coarse_summary.json").read_text())
    assert s["calibration_candidates"] == {"old_test_images": 5, "unique": 2, "ambiguous_excluded": 1,
                                           "unmatched": 1, "unreadable": 1, "pairs": 2, "originals_labelled": 1}
    assert s["pool_match_to_compilation"] == {"files": 5, "exact": 1, "coarse_unique": {"images": 1, "per_source": {"s1": 1}},
                                       "coarse_ambiguous": 1, "none": 1, "unreadable": 1}
    assert s["compilation_reference"] == {"manifest_rows": 9, "files": 8, "unreadable": 0, "distinct_coarse_keys": 7,
                                   "coarse_keys_on_several_files": 1}
    assert s["pool"]["files"] == 5
    assert s["pool"]["file_list_sha256"] == hashlib.sha256((out / "pool_b_files.txt").read_bytes()).hexdigest()
    keys_b = pd.read_parquet(out / "pool_b_keys.parquet")
    assert keys_b["name"].tolist() == ["q1.png", "q2.png", "q3.png", "q4.png", "q5.png"]
    _assert_no_leak((out / "coarse_summary.json").read_text(), ("o1", "q1", str(tmp_path)))


# ---- (b) embeddings ----
def _models(root: Path) -> Path:
    for rel in ("finger-vit_giant/config.json", "finger-vit_giant/model.safetensors",
                "fixed-length-fingerprint-extractors/best_model.pyt"):
        _junk(root / rel)
    return root


def test_embed_command_writes_aligned_normalised_rows(tmp_path, monkeypatch):
    compilation = _compilation(tmp_path / "compilation")
    keys = _key_table(compilation, tmp_path / "keys")
    old, pool = _old_and_pool_b(tmp_path, compilation)
    imgs, manifest, _ = _d1(tmp_path, compilation)
    out = tmp_path / "out"
    to.main(["coarse", "--pool", str(pool), "--old-images", str(old), "--compilation", str(compilation), "--compilation-keys", str(keys),
             "--out", str(out), "--workers", "1"])
    calls = {}

    def fake(models, paths, batch_size=1, workers=8):                     # a vector from the image's mean value
        calls.update(models=str(models), n=len(paths), batch_size=batch_size, workers=workers)
        return np.array([[float(np.asarray(Image.open(p).convert("L")).mean()), 1.0, 0.0] for p in paths])

    monkeypatch.setattr(to, "encode_fingervit", fake)
    to.main(["embed", "--pool", str(pool), "--old-images", str(old), "--compilation", str(compilation),
             "--models", str(_models(tmp_path / "models")), "--d1-manifest", str(manifest), "--d1-images", str(imgs),
             "--out", str(out), "--fv-batch", "4", "--workers", "2"])
    assert (out / "emb_pool_names.txt").read_text().split() == ["q1.png", "q2.png", "q3.png", "q4.png"]
    pool_x = np.load(out / "emb_pool.npy")
    assert pool_x.dtype == np.float32 and pool_x.shape == (4, 3)
    np.testing.assert_allclose(pool_x, eg.l2_normalize([[56, 1, 0], [72, 1, 0], [216, 1, 0], [104, 1, 0]]), rtol=1e-6)
    packaged, original = np.load(out / "emb_pair_packaged.npy"), np.load(out / "emb_pair_original.npy")
    np.testing.assert_allclose(packaged, eg.l2_normalize([[24, 1, 0], [120, 1, 0]]), rtol=1e-6)
    np.testing.assert_allclose(original, packaged, rtol=1e-6)
    d1 = np.load(out / "emb_d1.npy")
    np.testing.assert_allclose(d1, eg.l2_normalize([[24, 1, 0], [56, 1, 0], [72, 1, 0], [88, 1, 0], [136, 1, 0]]),
                               rtol=1e-6)
    s = json.loads((out / "embed_summary.json").read_text())
    assert s["batch_size"] == 4 and s["loader_workers"] == 2
    assert s["rows"] == {"pool": 4, "pair_packaged": 2, "pair_original": 2, "d1": 5}
    assert set(s["weights_sha256"]) == {"finger-vit_giant/config.json", "finger-vit_giant/model.safetensors"}
    assert s["inputs_sha256"] == {"calibration_pairs": eg.sha256_file(out / "calibration_pairs.csv"),
                                  "pool_b_keys": eg.sha256_file(out / "pool_b_keys.parquet")}
    assert calls == {"models": str(tmp_path / "models"), "n": 13, "batch_size": 4, "workers": 2}
    _assert_no_leak((out / "embed_summary.json").read_text(), ("q1", "o1", "d1_a", str(tmp_path)))


# ---- (b) calibration statistics ----
def test_quantiles_report_the_fixed_levels():
    q = to.quantiles(np.arange(1001, dtype=float))
    assert list(q) == ["n", "min", "p0.1", "p1", "p5", "p50", "p95", "p99", "p99.9", "max"]
    assert (q["n"], q["min"], q["max"]) == (1001, 0.0, 1000.0)
    assert q["p0.1"] == pytest.approx(1.0) and q["p1"] == pytest.approx(10.0) and q["p99.9"] == pytest.approx(999.0)


def test_genuine_scores_are_all_within_identity_pairs():
    x = np.random.default_rng(0).normal(size=(7, 5))
    ids = np.array(["a", "b", "a", "c", "b", "a", "d"])
    xn = eg.l2_normalize(x)
    want = sorted(xn[i] @ xn[j] for i in range(7) for j in range(i + 1, 7) if ids[i] == ids[j])
    np.testing.assert_allclose(np.sort(to.genuine_scores(x, ids)), want, rtol=1e-9)


def test_impostor_pairs_are_seeded_distinct_and_cross_identity():
    ids = np.repeat(np.arange(30), 3).astype(str)
    a, b = to.impostor_pairs(ids, 2000, seed=7)
    assert len(a) == 2000 and (a < b).all() and (ids[a] != ids[b]).all()
    assert len(set(zip(a.tolist(), b.tolist()))) == 2000
    a2, b2 = to.impostor_pairs(ids, 2000, seed=7)
    assert (a == a2).all() and (b == b2).all()
    a3, b3 = to.impostor_pairs(ids, 2000, seed=8)
    assert not ((a == a3).all() and (b == b3).all())
    with pytest.raises(ValueError, match="impostor pairs"):
        to.impostor_pairs(np.array(["x", "x", "x", "y"]), 4, seed=1)          # only 3 cross-identity pairs exist


def test_pair_scores_match_brute_force():
    x = np.random.default_rng(2).normal(size=(9, 4))
    a, b = np.array([0, 1, 2, 7]), np.array([3, 8, 5, 8])
    xn = eg.l2_normalize(x)
    np.testing.assert_allclose(to.pair_scores(x, a, b, block=3), [xn[i] @ xn[j] for i, j in zip(a, b)], rtol=1e-9)


def test_tau_rule_and_reported_thresholds():
    pos = to.quantiles(np.linspace(0.95, 1.0, 1001))
    rule = to.tau_rule(pos, to.quantiles(np.linspace(0.2, 0.9, 1001)))
    assert rule["tau"] == pytest.approx(pos["p1"]) and rule["separable"] is True
    assert to.tau_rule(pos, to.quantiles(np.linspace(0.2, 0.99, 1001)))["separable"] is False
    th = to.report_thresholds({"p0.1": 0.95, "p1": 0.97, "p5": 0.99}, {"p99.9": 0.9})
    assert [t["name"] for t in th] == ["D2 genuine p99.9", "positives p0.1", "tau = positives p1", "positives p5"]
    assert [t["value"] for t in th] == [0.9, 0.95, 0.97, 0.99]


def test_counts_above_per_source_images_and_fingers():
    maxcos = np.array([0.99, 0.5, 0.97, 0.2, 0.98, 0.1])
    ids = np.array(["a", "a", "b", "b", "c", "d"])
    src = np.array(["s1", "s1", "s1", "s1", "s2", "s2"])
    (r,) = to.counts_above(maxcos, ids, src, [{"name": "t", "value": 0.975}])
    assert (r["name"], r["value"]) == ("t", 0.975)
    assert (r["images"]["above"], r["images"]["of"]) == (2, 6)
    assert r["images"]["per_source"] == {"s1": {"above": 1, "of": 4, "fraction": 0.25},
                                         "s2": {"above": 1, "of": 2, "fraction": 0.5}}
    assert (r["fingers"]["above"], r["fingers"]["of"]) == (2, 4)
    assert r["fingers"]["per_source"] == {"s1": {"above": 1, "of": 2, "fraction": 0.5},
                                          "s2": {"above": 1, "of": 2, "fraction": 0.5}}
    with pytest.raises(ValueError, match="several sources"):
        to.counts_above(maxcos, ids, np.array(["s1", "s2", "s1", "s1", "s2", "s2"]), [{"name": "t", "value": 0.5}])


def test_coarse_copy_check_counts_unique_copies_above_tau_per_source():
    res = to.coarse_copy_check(["c0", "c1", "c2", None, "c4"], {"c0": 1, "c1": 2, "c2": 1, "c4": 1}, {"c0", "c1", "c4"},
                               np.array([0.999, 0.999, 0.999, 0.999, 0.5]), tau=0.99,
                               sources=["s1", "s1", "s2", "s2", "s2"])
    # c0 and c4 are unique in compilation and occur in the pool; c1 is on two compilation images, c2 is not in the pool
    assert (res["d2_images_with_a_coarse_copy_in_pool"], res["at_or_above_tau"], res["fraction"]) == (2, 1, 0.5)
    assert res["min_max_cosine"] == 0.5
    assert res["by_source"] == {"s1": {"copies": 1, "at_or_above_tau": 1, "fraction": 1.0},
                                "s2": {"copies": 1, "at_or_above_tau": 0, "fraction": 0.0}}
    assert "same coarse key" in res["definition"] and "misses" in res["definition"]


def test_pair_count_adds_a_per_pair_bound_only_for_zero():
    scores = np.array([0.1, 0.5, 0.9, 0.95])
    assert to.pair_count(scores, 0.5) == {"at_or_above": 3, "n": 4, "fraction": 0.75}
    zero = to.pair_count(scores, 0.99)
    assert (zero["at_or_above"], zero["n"], zero["fraction"]) == (0, 4, 0.0)
    assert zero["upper_95_per_pair"] == pytest.approx(1 - 0.05 ** (1 / 4), rel=1e-9)


def _calibration_inputs():
    pos = np.linspace(0.99, 1.0, 1000)
    pos_src = np.array(["a", "b"] * 500)
    gen = np.concatenate([np.linspace(0.1, 0.98, 498), [0.9995, 0.9999]])      # two genuine pairs reach tau
    gen_src = np.array(["a"] * 250 + ["b"] * 248 + ["b", "b"])
    imp = np.linspace(-0.3, 0.9, 2000)
    return pos, pos_src, gen, gen_src, imp


def test_calibrate_counts_pairs_and_bounds_at_every_threshold():
    pos, pos_src, gen, gen_src, imp = _calibration_inputs()
    cal = to.calibrate(pos, pos_src, gen, gen_src, imp)
    qp, qg = cal["quantiles"]["positives"], cal["quantiles"]["d2_genuine"]
    assert cal["quantiles"]["d2_impostor"]["n"] == 2000
    tau = cal["tau_rule"]["tau"]
    assert tau == pytest.approx(qp["p1"]) and cal["tau_rule"]["separable"] is (tau > qg["p99.9"])
    assert [t["name"] for t in cal["pairs_at_or_above"]] == [t["name"] for t in cal["thresholds"]]
    at_tau = next(t for t in cal["pairs_at_or_above"] if t["name"] == "tau = positives p1")
    assert at_tau["genuine"] == {"at_or_above": 2, "n": 500, "fraction": 2 / 500}
    assert at_tau["impostor"]["at_or_above"] == 0 and at_tau["impostor"]["n"] == 2000
    assert at_tau["impostor"]["upper_95_per_pair"] == pytest.approx(1 - 0.05 ** (1 / 2000), rel=1e-9)
    assert "per pair" in cal["per_pair_bound"]["note"] and "maximum" in cal["per_pair_bound"]["note"]
    by_pos, by_gen = cal["by_source"]["positives"], cal["by_source"]["d2_genuine"]
    assert {s: v["n"] for s, v in by_pos.items()} == {"a": 500, "b": 500}
    assert sum(v["below_tau"] for v in by_pos.values()) == int((pos < tau).sum())
    assert by_gen["a"]["at_or_above_tau"] == 0 and by_gen["b"]["at_or_above_tau"] == 2
    assert (by_gen["a"]["n"], by_gen["b"]["n"]) == (250, 250)


def test_calibrate_stops_without_positives():
    _, _, gen, gen_src, imp = _calibration_inputs()
    with pytest.raises(SystemExit, match="calibration pairs"):
        to.calibrate(np.empty(0), np.empty(0, dtype=str), gen, gen_src, imp)


def test_chain_problems_compares_recorded_and_recomputed_hashes():
    now = {"calibration_pairs": "p", "compilation_manifest": "m", "pool_file_list": "l", "emb_pool": "e", "pool_b_keys": "k"}
    coarse = {"calibration_pairs": "p", "compilation_manifest": "m", "pool_file_list": "l", "old_test_file_list": "o"}
    assert to.chain_problems(coarse, {"outputs_sha256": {"emb_pool": "e"}}, now) == []
    assert to.chain_problems(coarse, {"inputs_sha256": {"pool_b_keys": "k"}}, now) == []
    bad = to.chain_problems(dict(coarse, calibration_pairs="x"), {"outputs_sha256": {"emb_pool": "y"},
                                                                  "inputs_sha256": {"pool_b_keys": "z"}}, now)
    assert bad == ["the coarse step recorded another calibration_pairs",
                   "the embedding step wrote another emb_pool", "the embedding step read another pool_b_keys"]
    assert to.chain_problems({}, {}, now)[0] == "the coarse step recorded another calibration_pairs"


def test_fvc_databases_lists_the_database_folders(tmp_path):
    for rel in ("fvc/FVC2000/DB1_B/1.tif", "fvc/FVC2000/DB2_B/1.tif", "fvc/FVC2004/DB3_B/1.tif",
                "nist/sd302/x.png", "casia/db_misc.bmp"):
        _junk(tmp_path / "images" / rel)
    assert to.fvc_databases(tmp_path / "images") == {"FVC2000": ["DB1_B", "DB2_B"], "FVC2004": ["DB3_B"]}
    _junk(tmp_path / "images" / "fvc/FVC2002/DB4_B/1.tif")
    assert to.fvc_databases(tmp_path / "images")["FVC2002"] == ["DB4_B"]


def test_markdown_section_returns_the_lines_under_a_heading():
    text = "# T\nintro\n### Databases\nline a\n#### Sub\nline b\n### Next\nline c\n"
    assert to.markdown_section(text, "Databases") == ["line a", "#### Sub", "line b"]
    assert to.markdown_section(text, "Missing") == []


def _measure_fixture(tmp: Path) -> dict:
    """Synthetic embeddings with a known answer: D2 images 0, 4 and 16 have exact copies in the pool."""
    rng = np.random.default_rng(11)
    dim = 32
    compilation = _compilation(tmp / "compilation")
    ids = np.repeat([f"t:{k}:R1" for k in range(10)], 3)
    src = np.repeat(["s1"] * 5 + ["s2"] * 5, 3)
    paths = [f"images/t/{i}.png" for i in range(30)]
    gm = pd.read_csv(compilation / "MANIFEST.csv", dtype=str, keep_default_na=False)   # the D2 images are compilation rows too
    extra = gm.iloc[[0] * 30].assign(path=paths, identity=ids, source=src)
    pd.concat([gm, extra]).to_csv(compilation / "MANIFEST.csv", index=False)
    keys = _key_table(compilation, tmp / "keys")
    imgs, manifest, relabel = _d1(tmp, compilation)
    x2 = eg.l2_normalize(np.repeat(rng.normal(size=(10, dim)), 3, axis=0) + 0.4 * rng.normal(size=(30, dim)))
    d2 = tmp / "d2"
    d2.mkdir()
    pd.DataFrame({"identity": ids, "source": src, "path": paths}).to_parquet(d2 / "manifest.parquet")
    np.save(d2 / "emb_fingervit.npy", x2.astype(np.float32))
    (d2 / "extract_summary.json").write_text(json.dumps({"composition": {"per_source": {
        "s1": {"images": 15, "identities": 5}, "s2": {"images": 15, "identities": 5}}}}))
    out = tmp / "out"
    out.mkdir()
    pool = np.vstack([x2[[0, 4, 16]], eg.l2_normalize(rng.normal(size=(40, dim)))])
    np.save(out / "emb_pool.npy", pool.astype(np.float32))
    names = [f"q{i}.png" for i in range(len(pool))]
    (out / "emb_pool_names.txt").write_text("".join(f"{n}\n" for n in names))
    (out / "pool_b_files.txt").write_text("".join(f"{n}\n" for n in names))
    original = eg.l2_normalize(rng.normal(size=(200, dim)))
    original[0] = x2[7]                                                     # this compilation original lies in D2
    packaged = eg.l2_normalize(original + 0.01 * rng.normal(size=original.shape))
    np.save(out / "emb_pair_packaged.npy", packaged.astype(np.float32))
    np.save(out / "emb_pair_original.npy", original.astype(np.float32))
    # originals: D2 image 7 (s1), then compilation images of s1 (odd rows) and s2 (even rows)
    originals = [paths[7]] + ["images/s1/g1.png" if i % 2 else "images/s2/g6.png" for i in range(1, 200)]
    pd.DataFrame({"old_image": [f"o{i}.png" for i in range(200)], "compilation_path": originals}).to_csv(
        out / "calibration_pairs.csv", index=False)
    np.savez(tmp / "features.npz", **{f"o{i}": 3.0 * packaged[i] for i in range(150)})   # stored for 150 of them
    # coarse keys: D2 image 0's key is on no other compilation image and occurs in the pool; image 1's is on two compilation images
    pd.DataFrame({"path": paths + ["images/x/dup.png"],
                  "coarse": [f"c{i}" for i in range(30)] + ["c1"]}).to_parquet(out / "compilation_coarse.parquet")
    pd.DataFrame({"name": names, "exact": [f"e{i}" for i in range(len(names))],
                  "coarse": ["c0", "c1"] + ["zz"] * (len(names) - 2)}).to_parquet(out / "pool_b_keys.parquet")
    x1 = eg.l2_normalize(rng.normal(size=(5, dim)))
    x1[0] = pool[0]                                                         # D1 image d1_a (F1, s1) is in the pool
    np.save(out / "emb_d1.npy", x1.astype(np.float32))
    cpu = {"image": "canvec/exp:core", "image_id": "sha256:cpu", "git_sha": "abc"}
    (out / "coarse_summary.json").write_text(json.dumps({
        "pool": {"label": "pool label, 43 files", "files": 43, "file_list_sha256": "0" * 64},
        "calibration_candidates": {"old_test_images": 210, "unique": 200, "ambiguous_excluded": 4, "unmatched": 6,
                                   "unreadable": 0, "pairs": 200, "originals_labelled": 190},
        "pool_match_to_compilation": {"files": 43, "exact": 0}, "compilation_reference": {"files": 31},
        "inputs_sha256": {"compilation_manifest": eg.sha256_file(compilation / "MANIFEST.csv"),
                          "calibration_pairs": eg.sha256_file(out / "calibration_pairs.csv"),
                          "pool_file_list": eg.sha256_file(out / "pool_b_files.txt")},
        "provenance": {"env": cpu}, "seconds": 1.0}))
    (out / "embed_summary.json").write_text(json.dumps({
        "rows": {"pool": 43, "pair_packaged": 200, "pair_original": 200, "d1": 5}, "batch_size": 1,
        "loader_workers": 16, "weights_sha256": {"finger-vit_giant/config.json": "2" * 64},
        "preprocessing": "p", "code": {}, "software": {"torch": "x"},
        "outputs_sha256": {f"emb_{k}": eg.sha256_file(out / f"emb_{k}.npy")
                           for k in ("pool", "pair_packaged", "pair_original", "d1")},
        "provenance": {"env": dict(cpu, image="canvec/extract:gpu", image_id="sha256:gpu")}, "seconds": 2.0}))
    return {"compilation": compilation, "keys": keys, "manifest": manifest, "relabel": relabel, "d2": d2, "out": out,
            "features": tmp / "features.npz", "packaged": packaged, "original": original}


def _measure(f: dict) -> dict:
    to.main(["measure", "--d2", str(f["d2"]), "--old-fingervit", str(f["features"]), "--compilation", str(f["compilation"]),
             "--compilation-keys", str(f["keys"]), "--d1-manifest", str(f["manifest"]), "--d1-map", str(f["relabel"]),
             "--out", str(f["out"]), "--json", str(f["out"] / "b.json"), "--impostor-pairs", "300", "--seed", "5"])
    return json.loads((f["out"] / "b.json").read_text())


def test_measure_stops_when_the_chain_of_hashes_breaks(tmp_path):
    f = _measure_fixture(tmp_path)
    summary = json.loads((f["out"] / "coarse_summary.json").read_text())
    summary["inputs_sha256"]["calibration_pairs"] = "0" * 64
    (f["out"] / "coarse_summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit, match="coarse step recorded another calibration_pairs"):
        _measure(f)
    assert not (f["out"] / "b.json").exists()


def test_measure_command_end_to_end(tmp_path):
    f = _measure_fixture(tmp_path)
    res = _measure(f)
    text = (f["out"] / "b.json").read_text()
    cal = res["calibration"]
    pos = eg.cosine_rows(np.load(f["out"] / "emb_pair_packaged.npy"), np.load(f["out"] / "emb_pair_original.npy"))
    tau = res["tau_rule"]["tau"]
    # per-source view of the positives (by the compilation source of the original) and of the genuine pairs
    by_pos = cal["positives"]["by_source"]
    assert {s: v["n"] for s, v in by_pos.items()} == {"s1": 101, "s2": 99}
    assert sum(v["below_tau"] for v in by_pos.values()) == int((pos < tau).sum()) > 0
    by_gen = cal["d2_genuine"]["by_source"]
    assert set(by_gen) == {"s1", "s2"} and all((v["n"], v["at_or_above_tau"]) == (15, 0) for v in by_gen.values())
    # pair counts at every reported threshold, with the per-pair bound where none reaches it
    pairs = {p["name"]: p for p in cal["pairs_at_or_above_thresholds"]}
    assert set(pairs) == {t["name"] for t in res["thresholds"]}
    at = pairs["tau = positives p1"]
    assert (at["genuine"]["at_or_above"], at["genuine"]["n"]) == (0, 30)
    assert at["genuine"]["upper_95_per_pair"] == pytest.approx(1 - 0.05 ** (1 / 30), rel=1e-9)
    assert (at["impostor"]["at_or_above"], at["impostor"]["n"]) == (0, 300)
    assert at["impostor"]["upper_95_per_pair"] == pytest.approx(1 - 0.05 ** (1 / 300), rel=1e-9)
    assert "not the detector's per-image false-positive rate" in cal["per_pair_bound"]["note"]
    assert cal["per_pair_bound"]["method"]
    assert res["provenance_chain"]["problems"] == []
    assert set(res["provenance_chain"]["checked"]) >= {"calibration_pairs", "compilation_manifest", "pool_file_list",
                                                        "emb_pool", "emb_pair_packaged", "emb_pair_original", "emb_d1"}
    assert res["d2_coarse_copy_check"]["by_source"] == {"s1": {"copies": 1, "at_or_above_tau": 1, "fraction": 1.0},
                                                         "s2": {"copies": 0, "at_or_above_tau": 0, "fraction": None}}
    assert cal["positives"]["quantiles"]["n"] == 200 and cal["positives"]["quantiles"]["min"] > 0.99
    assert cal["positives"]["crosscheck_stored_features"]["quantiles"]["n"] == 150
    assert cal["positives"]["crosscheck_stored_features"]["quantiles"]["min"] == pytest.approx(1.0, abs=1e-6)
    assert cal["positives"]["crosscheck_d2_embeddings"]["quantiles"]["n"] == 1
    assert cal["positives"]["crosscheck_d2_embeddings"]["quantiles"]["min"] == pytest.approx(1.0, abs=1e-6)
    assert cal["d2_genuine"]["quantiles"]["n"] == 30
    assert cal["d2_impostor"]["quantiles"]["n"] == 300 and cal["d2_impostor"]["seed"] == 5
    rule = res["tau_rule"]
    assert rule["tau"] == pytest.approx(cal["positives"]["quantiles"]["p1"]) and rule["separable"] is True
    assert {t["name"] for t in res["thresholds"]} == {"D2 genuine p99.9", "positives p0.1", "tau = positives p1",
                                                      "positives p5"}
    at_tau = next(e for e in res["d2"]["by_threshold"] if e["name"] == "tau = positives p1")
    assert (at_tau["images"]["above"], at_tau["images"]["of"]) == (3, 30)
    assert (at_tau["fingers"]["above"], at_tau["fingers"]["of"]) == (3, 10)
    assert at_tau["images"]["per_source"]["s1"]["above"] == 2 and at_tau["fingers"]["per_source"]["s2"]["above"] == 1
    assert res["d2_coarse_copy_check"]["d2_images_with_a_coarse_copy_in_pool"] == 1
    assert res["d2_coarse_copy_check"]["at_or_above_tau"] == 1
    d1_tau = next(e for e in res["d1_vs_pool"]["by_threshold"] if e["name"] == "tau = positives p1")
    assert (d1_tau["images"]["above"], d1_tau["images"]["of"]) == (1, 4)
    assert d1_tau["fingers"]["above"] == 1 and d1_tau["fingers"]["per_source"]["s1"]["above"] == 1
    assert res["d1_vs_pool"]["unassigned"] == {"images": 1, "at_or_above_tau": 0}
    assert res["extraction"]["batch_size"] == 1 and res["extraction"]["env"]["image_id"] == "sha256:gpu"
    assert res["provenance"]["coarse"]["env"]["image_id"] == "sha256:cpu"
    assert len(res["provenance"]["measure"]["script_sha256"]) == 64
    assert res["inputs_sha256"]["d2_manifest"] == eg.sha256_file(f["d2"] / "manifest.parquet")
    assert res["pool"]["files"] == 43 and "data_source" in res
    # the descriptions claim no more than the calibration shows
    assert "at most 0.1%" in res["detector"] and "stay below" not in res["detector"]
    assert "taken as its original" in cal["positives"]["definition"]
    _assert_no_leak(text, ("t:0:R1", "q0", "o0", "images/", ".npy", str(tmp_path)))
    near = pd.read_parquet(f["out"] / "d2_pool_nearest.parquet")
    assert len(near) == 30 and near["pool_name"].iloc[0] == "q0.png"


README = ("# flx\nintro\n### Databases\n\nIn the paper, a synthetic databases created with the `SFinge` synthetic "
          "fingerprint generator were used for training.\n\nIn addition MCYT330 optical and capacitive sensor databases "
          "were used.\n\n### Image processing\nother text\n")


def test_deepprint_command_records_the_sources_and_the_evidence(tmp_path):
    d2 = tmp_path / "d2"
    d2.mkdir()
    (d2 / "extract_summary.json").write_text(json.dumps({"composition": {"per_source": {
        "sd302": {"images": 6, "identities": 2}, "casia_v5": {"images": 10, "identities": 2}}}}))
    pd.DataFrame({"identity": ["a"], "source": ["casia_v5"], "path": ["images/a.png"]}).to_parquet(
        d2 / "manifest.parquet")
    compilation = tmp_path / "compilation"
    for rel in ("fvc/FVC2000/DB1_B/1.tif", "fvc/FVC2000/DB3_B/1.tif", "fvc/FVC2002/DB2_B/1.tif"):
        _junk(compilation / "images" / rel)
    # the MANIFEST may list DB4 rows without a file (compilation excludes the synthetic FVC database)
    pd.DataFrame({"path": ["images/fvc/FVC2000/DB1_B/1.tif", "images/fvc/FVC2000/DB4_B/1.tif"],
                  "identity": ["fvc2000:DB1_B:1", ""], "source": ["fvc2000", "fvc"],
                  "exclusion": ["none", "synthetic_source"]}).to_csv(compilation / "MANIFEST.csv", index=False)
    (tmp_path / "README.md").write_text(README)
    to.main(["deepprint", "--d2", str(d2), "--compilation", str(compilation), "--flx-readme", str(tmp_path / "README.md"),
             "--json", str(tmp_path / "c.json")])
    text = (tmp_path / "c.json").read_text()
    res = json.loads(text)
    assert list(res["d2_sources"]) == ["casia_v5", "sd302"]
    assert res["flx_training_sets"] == ["SFinGe synthetic", "MCYT330"]
    assert res["named_training_sets_among_d2_sources"] == []
    assert res["statement"] == ("The training sets named in the flx documentation (SFinGe synthetic, MCYT330) are not "
                                "among the D2 sources.")
    assert res["compilation_fvc_databases"] == {"FVC2000": ["DB1_B", "DB3_B"], "FVC2002": ["DB2_B"]}
    assert res["compilation_fvc_statement"] == "compilation real part: FVC DB1-DB3 only; no SFinGe-generated DB4"
    assert res["compilation_manifest_db4_rows"] == {"rows": 1, "with_file": 0, "with_identity": 0,
                                             "exclusion": {"synthetic_source": 1}}
    readme = res["flx_readme"]
    assert readme["sha256"] == eg.sha256_file(tmp_path / "README.md") and readme["section"] == "Databases"
    assert readme["training_sets_named_in_section"] == ["SFinGe synthetic", "MCYT330"]
    assert "SFinge" not in text                                            # the section is quoted in the report only
    assert res["pool"] is None and res["thresholds"] is None
    assert set(res["inputs_sha256"]) == {"d2_extract_summary", "d2_manifest", "flx_readme"}
    assert len(res["provenance"]["script_sha256"]) == 64


def test_deepprint_reports_a_db4_folder(tmp_path):
    _junk(tmp_path / "images" / "fvc/FVC2004/DB4_B/1.tif")
    assert to.fvc_statement(to.fvc_databases(tmp_path / "images")).startswith("compilation real part holds FVC DB4")


def test_deepprint_sources_flags_a_named_training_set():
    res = to.deepprint_sources({"composition": {"per_source": {"mcyt330_optical": {"images": 1, "identities": 1}}}})
    assert res["named_training_sets_among_d2_sources"] == ["mcyt330_optical"]
    assert "not among" not in res["statement"]
