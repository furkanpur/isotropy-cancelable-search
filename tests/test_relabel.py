# tests/test_relabel.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ijiss_btp.data.loaders import load_sin_corpus
from ijiss_btp.experiments.isotropy import _data_file_hashes, load_data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/data"))  # importable by name so spawned pool workers can unpickle
import build_relabel_map as brm  # noqa: E402


def _corpus(root: Path) -> Path:
    root.mkdir()
    np.save(root / "emb_m.npy", np.arange(8, dtype=np.float32).reshape(4, 2))
    pd.DataFrame({"identity": ["a", "a", "a", "b"],
                  "path": ["/x/1.png", "/x/2.png", "/y/3.png", "/y/4.png"]}).to_parquet(root / "manifest.parquet")
    return root


def _map(path: Path, rows) -> Path:
    pd.DataFrame(rows, columns=["file", "identity"]).to_csv(path, index=False)
    return path


def test_relabel_changes_ids_and_drops_unmapped(tmp_path):
    root = _corpus(tmp_path / "c")
    m = _map(tmp_path / "m.csv", [("1.png", "f1"), ("2.png", "f2"), ("3.png", "f2")])
    emb = load_sin_corpus(root, "m", relabel=m)
    assert list(emb.ids) == ["f1", "f2", "f2"]
    assert emb.x.shape == (3, 2)
    np.testing.assert_array_equal(emb.x[:, 0], [0, 2, 4])
    assert "relabel" in emb.source and "dropped=1" in emb.source


def test_no_relabel_unchanged(tmp_path):
    emb = load_sin_corpus(_corpus(tmp_path / "c"), "m")
    assert list(emb.ids) == ["a", "a", "a", "b"] and emb.x.shape == (4, 2)
    assert "relabel" not in emb.source


def test_duplicate_basename_in_map_raises(tmp_path):
    root = _corpus(tmp_path / "c")
    m = _map(tmp_path / "m.csv", [("1.png", "f1"), ("1.png", "f2")])
    with pytest.raises(ValueError, match="duplicate"):
        load_sin_corpus(root, "m", relabel=m)


def test_load_data_and_hashes_include_relabel(tmp_path):
    root = _corpus(tmp_path / "c")
    m = _map(tmp_path / "m.csv", [("1.png", "f1"), ("4.png", "f2")])
    spec = {"kind": "sin", "root": str(root), "model": "m", "relabel": str(m)}
    assert list(load_data(spec).ids) == ["f1", "f2"]
    assert any(k.startswith("relabel:") for k in _data_file_hashes(spec))
    plain = {k: v for k, v in spec.items() if k != "relabel"}
    assert not any(k.startswith("relabel:") for k in _data_file_hashes(plain))


# ---- builder ----
def _img(seed: int, size=(12, 9)) -> Image.Image:
    a = np.random.default_rng(seed).integers(0, 256, (size[1], size[0]), dtype=np.uint8)
    return Image.fromarray(a, "L")


def test_builder_matches_reencoded_and_drops_ambiguous_unmatched(tmp_path):
    compilation, d1 = tmp_path / "compilation", tmp_path / "d1"
    compilation.mkdir()
    d1.mkdir()
    _img(1).save(compilation / "g1.bmp")
    _img(2).save(compilation / "g2.bmp")
    _img(3).save(compilation / "g3.bmp")
    _img(3).save(compilation / "g3b.bmp")        # same pixels, other identity -> ambiguous
    _img(4).save(compilation / "g4.bmp")         # empty identity -> ignored
    pd.DataFrame({"path": ["g1.bmp", "g2.bmp", "g3.bmp", "g3b.bmp", "g4.bmp"],
                  "identity": ["F1", "F2", "F3", "F4", ""],
                  "source": ["s1", "s1", "s2", "s2", "s2"],
                  "exclusion": ["", "bad", "", "", ""]}).to_csv(compilation / "MANIFEST.csv", index=False)
    _img(1).convert("RGB").save(d1 / "a.png")   # re-encoded, RGB container
    _img(2).save(d1 / "b.png")                  # matched although flagged by exclusion
    _img(3).save(d1 / "c.png")                  # ambiguous
    _img(4).save(d1 / "d.png")                  # only in empty-identity row
    _img(5).save(d1 / "e.png")                  # unmatched
    _img(1, (9, 12)).save(d1 / "f.png")         # different size, unmatched
    names = ["a", "b", "c", "d", "e", "f"]
    pd.DataFrame({"identity": ["x"] * 6, "path": [f"/orig/{n}.png" for n in names]}).to_parquet(tmp_path / "man.parquet")
    out = tmp_path / "out" / "map.csv"
    summary = brm.build(tmp_path / "man.parquet", d1, compilation, out, workers=2)
    got = pd.read_csv(out)
    assert dict(zip(got["file"], got["identity"])) == {"a.png": "F1", "b.png": "F2"}
    assert summary["n_images"] == 6 and summary["matched"] == 2
    assert summary["ambiguous"] == 1 and summary["unmatched"] == 3
    assert summary["true_fingers"] == 2
    assert summary["images_per_finger_hist"] == {"1": 2}
    assert summary["per_source"] == {"s1": 2}
    sj = json.loads(out.with_suffix(".summary.json").read_text())
    assert sj == summary and "a.png" not in json.dumps(sj)


def test_relabel_empty_identity_raises(tmp_path):
    root = _corpus(tmp_path / "c")
    m = _map(tmp_path / "m.csv", [("1.png", "f1"), ("2.png", "")])
    with pytest.raises(ValueError, match="empty"):
        load_sin_corpus(root, "m", relabel=m)


def test_content_key_known_answer(tmp_path):
    import hashlib

    arr = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    p = tmp_path / "k.png"
    Image.fromarray(arr, "L").save(p)
    assert brm.content_key(p) == hashlib.sha1(repr((4, 3)).encode() + arr.tobytes()).hexdigest()


def _tiny_setup(tmp_path, man_paths):
    compilation, d1 = tmp_path / "compilation", tmp_path / "d1"
    compilation.mkdir()
    d1.mkdir()
    _img(1).save(compilation / "g1.bmp")
    (compilation / "bad.bmp").write_bytes(b"not an image")
    pd.DataFrame({"path": ["g1.bmp", "bad.bmp"], "identity": ["F1", "F2"],
                  "source": ["s", "s"]}).to_csv(compilation / "MANIFEST.csv", index=False)
    _img(1).save(d1 / "a.png")
    (d1 / "corrupt.png").write_bytes(b"junk")
    pd.DataFrame({"identity": ["x"] * len(man_paths), "path": man_paths}).to_parquet(tmp_path / "man.parquet")
    return d1, compilation


def test_builder_counts_unreadable(tmp_path):
    d1, compilation = _tiny_setup(tmp_path, ["/o/a.png", "/o/corrupt.png"])
    s = brm.build(tmp_path / "man.parquet", d1, compilation, tmp_path / "o" / "m.csv", workers=1)
    assert s["compilation_unreadable"] == 1 and s["images_unreadable"] == 1 and s["matched"] == 1


def test_builder_duplicate_basenames_raise(tmp_path):
    d1, compilation = _tiny_setup(tmp_path, ["/o/a.png", "/p/a.png"])
    with pytest.raises(ValueError, match="duplicate"):
        brm.build(tmp_path / "man.parquet", d1, compilation, tmp_path / "o" / "m.csv", workers=1)


# ---- path-keyed relabel (secondary corpora whose basenames repeat across folders) ----
def _corpus_repeated_basenames(root: Path) -> Path:
    root.mkdir()
    np.save(root / "emb_m.npy", np.arange(8, dtype=np.float32).reshape(4, 2))
    pd.DataFrame({"identity": ["a", "a", "b", "b"],
                  "path": ["/hostdata/x/1.png", "/hostdata/y/1.png", "/hostdata/y/2.png",
                           "/hostdata/z/3.png"]}).to_parquet(root / "manifest.parquet")
    return root


def test_relabel_path_key_matches_the_exact_manifest_path(tmp_path):
    root = _corpus_repeated_basenames(tmp_path / "c")
    m = _map(tmp_path / "m.csv", [("/hostdata/x/1.png", "f1"), ("/hostdata/y/1.png", "f2"),
                                  ("/hostdata/z/3.png", "f2"), ("2.png", "f9")])   # a basename never matches here
    emb = load_sin_corpus(root, "m", relabel=m, relabel_key="path")
    assert list(emb.ids) == ["f1", "f2", "f2"]
    np.testing.assert_array_equal(emb.x[:, 0], [0, 2, 6])
    assert ":relabel=m.csv:key=path:dropped=1" in emb.source


def test_relabel_path_key_duplicate_path_raises(tmp_path):
    root = _corpus_repeated_basenames(tmp_path / "c")
    m = _map(tmp_path / "m.csv", [("/hostdata/x/1.png", "f1"), ("/hostdata/x/1.png", "f2")])
    with pytest.raises(ValueError, match="duplicate"):
        load_sin_corpus(root, "m", relabel=m, relabel_key="path")


def test_relabel_basename_key_is_the_default_and_recorded(tmp_path):
    root = _corpus(tmp_path / "c")
    m = _map(tmp_path / "m.csv", [("1.png", "f1"), ("2.png", "f2")])
    a = load_sin_corpus(root, "m", relabel=m)
    b = load_sin_corpus(root, "m", relabel=m, relabel_key="basename")
    assert list(a.ids) == list(b.ids) == ["f1", "f2"] and a.source == b.source
    assert ":relabel=m.csv:key=basename:dropped=2" in a.source


def test_relabel_unknown_key_raises(tmp_path):
    root = _corpus(tmp_path / "c")
    m = _map(tmp_path / "m.csv", [("1.png", "f1")])
    with pytest.raises(ValueError, match="relabel_key"):
        load_sin_corpus(root, "m", relabel=m, relabel_key="stem")


def test_load_data_passes_the_relabel_key(tmp_path):
    root = _corpus_repeated_basenames(tmp_path / "c")
    m = _map(tmp_path / "m.csv", [("/hostdata/y/1.png", "f2")])
    spec = {"kind": "sin", "root": str(root), "model": "m", "relabel": str(m), "relabel_key": "path"}
    emb = load_data(spec)
    assert list(emb.ids) == ["f2"] and emb.x[0, 0] == 2
    assert any(k.startswith("relabel:") for k in _data_file_hashes(spec))


# ---- builder, path mode ----
def test_parse_path_map():
    assert brm.parse_path_map(["/hostdata/a=/c/a", "/hostdata/b/=/c/b/"]) == [("/hostdata/a", "/c/a"),
                                                                             ("/hostdata/b", "/c/b")]
    assert brm.parse_path_map(None) == []
    for bad in ("nomap", "=/c", "/h=", ""):
        with pytest.raises(ValueError, match="path map"):
            brm.parse_path_map([bad])


def test_map_path_uses_the_longest_prefix_on_a_component_boundary():
    maps = brm.parse_path_map(["/hostdata/db=/c/db", "/hostdata/db/DB1=/c/db/DB1_B"])
    assert brm.map_path("/hostdata/db/DB1/1_1.tif", maps) == "/c/db/DB1_B/1_1.tif"
    assert brm.map_path("/hostdata/db/DB2/1_1.tif", maps) == "/c/db/DB2/1_1.tif"
    assert brm.map_path("/hostdata/db", maps) == "/c/db"
    assert brm.map_path("/hostdata/dbx/1.tif", maps) is None   # not a component boundary
    assert brm.map_path("/elsewhere/1.tif", maps) is None


def _path_mode_setup(tmp_path):
    compilation, mnt = tmp_path / "compilation", tmp_path / "mnt"
    (compilation / "images").mkdir(parents=True)
    _img(1).save(compilation / "images" / "g1.bmp")
    _img(2).save(compilation / "images" / "g2.bmp")
    pd.DataFrame({"path": ["images/g1.bmp", "images/g2.bmp"], "identity": ["F1", "F2"],
                  "source": ["s1", "s2"]}).to_csv(compilation / "MANIFEST.csv", index=False)
    (mnt / "A").mkdir(parents=True)
    (mnt / "B_new").mkdir(parents=True)
    _img(1).convert("RGB").save(mnt / "A" / "1.png")   # the same basename as B/1.png, other content
    _img(2).save(mnt / "B_new" / "1.png")               # folder renamed on the data host: B -> B_new
    _img(3).save(mnt / "A" / "2.png")                   # unmatched
    (mnt / "A" / "3.png").write_bytes(b"junk")          # unreadable
    paths = ["/hostdata/A/1.png", "/hostdata/B/1.png", "/hostdata/A/2.png", "/hostdata/A/3.png",
             "/hostdata/C/1.png"]                       # C: no path map entry
    pd.DataFrame({"identity": ["x"] * len(paths), "path": paths}).to_parquet(tmp_path / "man.parquet")
    return compilation, [f"/hostdata/A={mnt / 'A'}", f"/hostdata/B={mnt / 'B_new'}"]


def test_builder_path_mode_keys_by_the_manifest_path(tmp_path):
    compilation, pmap = _path_mode_setup(tmp_path)
    out = tmp_path / "out" / "sec.csv"
    s = brm.build(tmp_path / "man.parquet", None, compilation, out, workers=1, key="path", path_map=pmap)
    got = pd.read_csv(out)
    assert dict(zip(got["file"], got["identity"])) == {"/hostdata/A/1.png": "F1", "/hostdata/B/1.png": "F2"}
    assert s["key"] == "path" and s["n_images"] == 5 and s["matched"] == 2
    assert s["unmatched"] == 3 and s["ambiguous"] == 0
    assert s["paths_unmapped"] == 1 and s["images_unreadable"] == 1 and s["duplicate_basenames"] == 2
    assert s["path_map_entries"] == 2 and s["true_fingers"] == 2 and s["per_source"] == {"s1": 1, "s2": 1}
    sj = json.loads(out.with_suffix(".summary.json").read_text())
    assert sj == s
    text = json.dumps(sj)
    assert "hostdata" not in text and str(tmp_path) not in text and "1.png" not in text


def test_builder_path_mode_cli_without_images(tmp_path):
    compilation, pmap = _path_mode_setup(tmp_path)
    out = tmp_path / "out" / "sec.csv"
    args = ["--manifest", str(tmp_path / "man.parquet"), "--compilation", str(compilation), "--out", str(out),
            "--workers", "1", "--key", "path"]
    for p in pmap:
        args += ["--path-map", p]
    brm.main(args)
    assert len(pd.read_csv(out)) == 2


def test_builder_path_mode_duplicate_paths_raise(tmp_path):
    d1, compilation = _tiny_setup(tmp_path, ["/o/a.png", "/o/a.png"])
    with pytest.raises(ValueError, match="duplicate paths"):
        brm.build(tmp_path / "man.parquet", None, compilation, tmp_path / "o" / "m.csv", workers=1, key="path",
                  path_map=[f"/o={d1}"])


def test_builder_path_mode_allows_repeated_basenames(tmp_path):
    d1, compilation = _tiny_setup(tmp_path, ["/o/a.png", "/p/a.png"])   # basename mode raises on this manifest
    s = brm.build(tmp_path / "man.parquet", None, compilation, tmp_path / "o" / "m.csv", workers=1, key="path",
                  path_map=[f"/o={d1}", f"/p={d1}"])
    assert s["matched"] == 2 and s["true_fingers"] == 1 and s["duplicate_basenames"] == 1


def test_builder_basename_mode_needs_images_and_rejects_bad_key(tmp_path):
    d1, compilation = _tiny_setup(tmp_path, ["/o/a.png"])
    with pytest.raises(ValueError, match="images"):
        brm.build(tmp_path / "man.parquet", None, compilation, tmp_path / "o" / "m.csv", workers=1)
    with pytest.raises(ValueError, match="key"):
        brm.build(tmp_path / "man.parquet", d1, compilation, tmp_path / "o" / "m.csv", workers=1, key="stem")


def test_builder_basename_mode_summary_unchanged(tmp_path):
    d1, compilation = _tiny_setup(tmp_path, ["/o/a.png"])
    s = brm.build(tmp_path / "man.parquet", d1, compilation, tmp_path / "o" / "m.csv", workers=1)
    assert not ({"key", "paths_unmapped", "duplicate_basenames", "path_map_entries"} & set(s))
