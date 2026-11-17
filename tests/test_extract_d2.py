# tests/test_extract_d2.py
"""Selection logic and manifest/embedding alignment of scripts/data/extract_d2.py (tiny fixtures, no GPU)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ijiss_btp.data.loaders import load_sin_corpus

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/data"))
import extract_d2 as eg  # noqa: E402


def _row(path, identity, source="casia_v5", exclusion="none", subject="1", finger="1", device="d1",
         capture="contact", dpi="500", subset="s"):
    return dict(path=path, identity=identity, source=source, subject=subject, finger=finger, device=device,
                capture=capture, dpi=dpi, subset=subset, exclusion=exclusion, nfiq2="", quality_status="")


def _manifest(rows) -> pd.DataFrame:
    return pd.DataFrame(rows).astype(str)


def _identities(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["identity", "source", "impressions", "genuine_pairs", "fold", "partition"]
                        ).astype(str)


def _ident(identity, fold, partition="development", source="casia_v5"):
    return (identity, source, "0", "0", fold, partition)


# ---- fold filter ----
def test_fold_filter_keeps_only_requested_development_folds():
    man = _manifest([_row(f"images/a/{i}.bmp", "A") for i in range(3)]
                    + [_row(f"images/b/{i}.bmp", "B") for i in range(3)]
                    + [_row(f"images/c/{i}.bmp", "C") for i in range(3)]
                    + [_row(f"images/h/{i}.bmp", "H") for i in range(3)])
    ids = _identities([_ident("A", "0"), _ident("B", "1"), _ident("C", "2"), _ident("H", "", "holdout")])
    sel0, c0 = eg.select_compilation(man, ids, folds=(0,))
    assert set(sel0["identity"]) == {"A"}
    sel01, _ = eg.select_compilation(man, ids, folds=(0, 1))
    assert set(sel01["identity"]) == {"A", "B"}
    assert c0["rows_in_folds"] == 3


# ---- exclusion filter ----
def test_exclusion_none_filter_applies_before_counting_images():
    man = _manifest([_row("images/a/1.bmp", "A"), _row("images/a/2.bmp", "A", exclusion="duplicate"),
                     _row("images/a/3.bmp", "A", exclusion="altered"),
                     _row("images/b/1.bmp", "B"), _row("images/b/2.bmp", "B"),
                     _row("images/b/3.bmp", "B", exclusion="label_conflict"),
                     _row("images/x/1.bmp", "", exclusion="spoof"), _row("images/x/2.bmp", "", exclusion="spoof")])
    ids = _identities([_ident("A", "0"), _ident("B", "0")])
    sel, counts = eg.select_compilation(man, ids, folds=(0,), min_images=2)
    assert set(sel["exclusion"]) == {"none"}
    assert list(sel["identity"]) == ["B", "B"]          # A has one usable image left -> dropped
    assert counts["rows_with_identity"] == 6 and counts["rows_exclusion_none"] == 3
    assert counts["identities_below_min_images"] == 1


# ---- LivDet live-only guard ----
def test_livdet_live_only():
    live = "images/livdet/2015/GreenBit/Train/Live/{}.png"
    alive = "images/livdet/2011/ItaldataTrain/Alive/{}.png"
    rows = [_row(live.format(i), "L1", source="livdet", capture="live") for i in range(2)]
    rows += [_row(alive.format(i), "L2", source="livdet", capture="live") for i in range(2)]
    # mislabelled spoofs (exclusion says none) must still be dropped by the guard
    rows += [_row(f"images/livdet/2015/GreenBit/Train/Fake/Latex/{i}.png", "L3", source="livdet") for i in range(2)]
    rows += [_row(f"images/livdet/2013/Swipe/Spoof/Gelatin/{i}.png", "L4", source="livdet") for i in range(2)]
    rows += [_row(f"images/livdet/2009/Identix/{i}.png", "L5", source="livdet") for i in range(2)]  # no live marker
    rows += [_row(f"images/casia/{i}.bmp", "C1") for i in range(2)]  # not LivDet: no marker needed
    man = _manifest(rows)
    mask = eg.livdet_live_mask(man)
    assert mask.tolist() == [True] * 4 + [False] * 6 + [True] * 2
    ids = _identities([_ident(i, "0", source="livdet") for i in ("L1", "L2", "L3", "L4", "L5")] + [_ident("C1", "0")])
    sel, counts = eg.select_compilation(man, ids, folds=(0,), exclude_sources=())   # guard tested without the exclusion
    assert set(sel["identity"]) == {"L1", "L2", "C1"}
    assert counts["livdet_non_live_dropped"] == 6


# ---- source and first-corpus exclusions, before the fold rule ----
def test_livdet_excluded_by_default_with_counts():
    assert eg.EXCLUDED_SOURCES == ("livdet",)
    rows = [_row(f"images/livdet/2015/GreenBit/Train/Live/{i}.png", "L1", source="livdet") for i in range(3)]
    rows += [_row(f"images/casia/{i}.bmp", "C1") for i in range(2)]
    ids = _identities([_ident("L1", "0", source="livdet"), _ident("C1", "0")])
    sel, counts = eg.select_compilation(_manifest(rows), ids, folds=(0,))
    assert set(sel["identity"]) == {"C1"}
    assert counts["rows_excluded_source"] == 3 and counts["identities_excluded_source"] == 1


def test_d1_identities_excluded_before_the_fold_rule():
    man = _manifest([_row(f"images/a/{i}.bmp", "A") for i in range(3)]
                    + [_row(f"images/b/{i}.bmp", "B") for i in range(3)]
                    + [_row(f"images/c/{i}.bmp", "C") for i in range(3)]
                    + [_row(f"images/d/{i}.bmp", "D") for i in range(2)])
    ids = _identities([_ident("A", "0"), _ident("B", "1"), _ident("C", "1"), _ident("D", "3")])
    assert eg.pick_folds(man, ids, max_images=6)[0] == (0,)            # every larger prefix exceeds 6 images
    folds, sizes = eg.pick_folds(man, ids, max_images=6, exclude_identities={"C", "D"})
    assert folds == (0, 1, 2, 3, 4) and sizes == {"0,1,2,3,4": 6}       # C and D removed before the rule
    sel, counts = eg.select_compilation(man, ids, folds=(0, 1), exclude_identities={"C", "D"})
    assert set(sel["identity"]) == {"A", "B"}
    assert counts["rows_excluded_d1"] == 5 and counts["identities_excluded_d1"] == 2
    assert counts["identities_excluded_d1_in_folds"] == 1               # D sits in fold 3


# ---- exact duplicates ----
def _dup_counts(**nonzero):
    base = dict.fromkeys(["keys_shared_with_other_identities", "identities_dropped_d1_map_duplicate",
                          "identities_dropped_d1_map_duplicate_via_unselected_images",
                          "images_dropped_d1_map_duplicate", "images_dropped_duplicate_livdet",
                          "images_dropped_duplicate_other", "identities_with_duplicate_livdet",
                          "identities_with_duplicate_other", "images_dropped_within_identity"], 0)
    base.update(nonzero)
    return base


def test_dedup_exact_rules_within_the_probed_set():
    frame = pd.DataFrame({
        "identity": ["A", "B", "C", "C", "D", "D", "E", "F"],
        "path": ["a/1", "b/1", "c/2", "c/1", "d/1", "d/2", "e/1", "f/1"],
        "key": ["k1", "k1", "k2", "k2", "k3", "k3", "k3", "k4"],
    }, index=[10, 11, 12, 13, 14, 15, 16, 17])
    keep, counts = eg.dedup_exact(frame)           # without a full table, the frame itself is the reference
    assert keep.index.equals(frame.index)
    # k1 spans A and B: both dropped; k2 twice in C: keep the first path (c/1); k3 spans D and E: all dropped
    assert keep.to_dict() == {10: False, 11: False, 12: False, 13: True, 14: False, 15: False, 16: False, 17: True}
    assert counts == _dup_counts(keys_shared_with_other_identities=2, images_dropped_duplicate_other=5,
                                 identities_with_duplicate_other=4, images_dropped_within_identity=1)


def test_dedup_exact_against_the_full_labelled_set():
    d2 = pd.DataFrame({
        "identity": ["A", "A", "B", "B", "B", "C", "C", "C", "E", "E", "F", "F"],
        "path": ["a/1", "a/2", "b/1", "b/2", "b/3", "c/2", "c/1", "c/3", "e/1", "e/2", "f/1", "f/2"],
        "key": ["k1", "k2", "k3", "k4", "k5", "k6", "k6", "k7", "k8", "k9", "k9", "k10"],
    })
    full = pd.concat([d2[["identity", "key"]].assign(kind="other"), pd.DataFrame({
        "identity": ["D1x", "L1", "O1", "A"], "key": ["k1", "k3", "k4", "k2"],
        "kind": ["d1_map", "livdet", "other", "other"]})])   # A's own excluded copy of k2 is not another identity
    keep, counts = eg.dedup_exact(d2, full)
    kept = d2.loc[keep, "path"].tolist()
    # A shares k1 with a first-corpus map identity: the whole identity goes; b/1 (LivDet) and b/2 (other) go;
    # c/2 repeats c/1 within C; e/2 and f/1 share k9 across two identities of the selection
    assert kept == ["b/3", "c/1", "c/3", "e/1", "f/2"]
    assert counts == _dup_counts(keys_shared_with_other_identities=4, identities_dropped_d1_map_duplicate=1,
                                 images_dropped_d1_map_duplicate=2, images_dropped_duplicate_livdet=1,
                                 images_dropped_duplicate_other=3, identities_with_duplicate_livdet=1,
                                 identities_with_duplicate_other=3, images_dropped_within_identity=1)


def test_dedup_exact_drops_an_identity_matched_through_an_unselected_image():
    d2 = pd.DataFrame({"identity": ["G", "G", "H", "H"], "path": ["g/1", "g/2", "h/1", "h/2"],
                       "key": ["g1", "g2", "h1", "h2"]})
    full = pd.concat([d2[["identity", "key"]].assign(kind="other"),
                      pd.DataFrame({"identity": ["G", "D1z"], "key": ["gx", "gx"], "kind": ["other", "d1_map"]})])
    keep, counts = eg.dedup_exact(d2, full)            # G's excluded image gx is identical to a map identity's image
    assert d2.loc[keep, "path"].tolist() == ["h/1", "h/2"]
    assert counts == _dup_counts(identities_dropped_d1_map_duplicate=1, images_dropped_d1_map_duplicate=2,
                                 identities_dropped_d1_map_duplicate_via_unselected_images=1)


def test_select_compilation_dedups_after_unusable_and_before_min_images():
    man = _manifest([_row("images/x/1.bmp", "X"), _row("images/x/2.bmp", "X"),
                     _row("images/y/1.bmp", "Y"), _row("images/y/2.bmp", "Y"), _row("images/y/3.bmp", "Y"),
                     _row("images/z/1.bmp", "Z"), _row("images/z/2.bmp", "Z")])
    ids = _identities([_ident(i, "0") for i in "XYZ"])
    keys = {"images/x/1.bmp": "s", "images/x/2.bmp": "x2", "images/y/1.bmp": "s", "images/y/2.bmp": "y2",
            "images/y/3.bmp": "y3", "images/z/1.bmp": "z1", "images/z/2.bmp": "z1"}
    sel, counts = eg.select_compilation(man, ids, folds=(0,), keys=keys)
    # X loses its shared image and falls below two images; Y keeps two; Z keeps one copy and is dropped
    assert list(sel["path"]) == ["images/y/2.bmp", "images/y/3.bmp"]
    assert counts["images_dropped_duplicate_other"] == 2 and counts["images_dropped_within_identity"] == 1
    assert counts["identities_below_min_images"] == 2
    with pytest.raises(ValueError, match="content key"):
        eg.select_compilation(man, ids, folds=(0,), keys={"images/x/1.bmp": "s"})


def test_select_compilation_checks_duplicates_against_the_full_table():
    man = _manifest([_row(f"images/{k}/{i}.bmp", k.upper()) for k in "xyz" for i in range(3)])
    ids = _identities([_ident(i, "0") for i in "XYZ"])
    keys = {f"images/{k}/{i}.bmp": f"{k}{i}" for k in "xyz" for i in range(3)}
    full = pd.DataFrame({"identity": ["X", "X", "X", "Y", "Y", "Y", "Z", "Z", "Z", "D1y", "LV"],
                         "key": ["x0", "x1", "x2", "y0", "y1", "y2", "z0", "z1", "z2", "y1", "z0"],
                         "kind": ["other"] * 9 + ["d1_map", "livdet"]})
    sel, counts = eg.select_compilation(man, ids, folds=(0,), keys=keys, full_keys=full)
    assert list(sel["path"]) == ["images/x/0.bmp", "images/x/1.bmp", "images/x/2.bmp",
                                 "images/z/1.bmp", "images/z/2.bmp"]          # Y dropped whole, z/0 dropped
    assert counts["identities_dropped_d1_map_duplicate"] == 1 and counts["images_dropped_duplicate_livdet"] == 1


def test_identities_with_an_exact_d1_image_are_dropped():
    man = _manifest([_row(f"images/p/{i}.bmp", "P") for i in range(3)]
                    + [_row(f"images/q/{i}.bmp", "Q") for i in range(3)])
    ids = _identities([_ident("P", "0"), _ident("Q", "0")])
    keys = {f"images/{c}/{i}.bmp": f"{c}{i}" for c in "pq" for i in range(3)}
    sel, counts = eg.select_compilation(man, ids, folds=(0,), keys=keys, d1_keys={"q1", "unrelated"})
    assert set(sel["identity"]) == {"P"}
    assert counts["identities_with_d1_image"] == 1 and counts["rows_dropped_d1_image"] == 3


# ---- identity disjointness ----
def test_selection_is_identity_disjoint_from_other_folds_and_holdout():
    man = _manifest([_row(f"images/a/{i}.bmp", "A") for i in range(2)]
                    + [_row(f"images/b/{i}.bmp", "B") for i in range(2)]
                    + [_row(f"images/z/{i}.bmp", "Z") for i in range(2)])   # Z has no IDENTITIES row
    ids = _identities([_ident("A", "0"), _ident("B", "3")])
    sel, counts = eg.select_compilation(man, ids, folds=(0,))
    assert set(sel["identity"]) == {"A"}
    assert counts["rows_without_identity_row"] == 2
    eg.check_identity_disjoint(sel, ids, folds=(0,))       # passes
    leaked = pd.concat([sel, man[man["identity"] == "B"].assign(fold="3", partition="development")])
    with pytest.raises(ValueError, match="outside"):
        eg.check_identity_disjoint(leaked, ids, folds=(0,))
    holdout = _identities([_ident("A", "0", "holdout")])
    with pytest.raises(ValueError, match="outside"):
        eg.check_identity_disjoint(sel, holdout, folds=(0,))


def test_duplicate_identity_rows_rejected():
    man = _manifest([_row(f"images/a/{i}.bmp", "A") for i in range(2)])
    ids = _identities([_ident("A", "0"), _ident("A", "1")])
    with pytest.raises(ValueError, match="duplicate"):
        eg.select_compilation(man, ids, folds=(0,))


def test_identity_spanning_two_sources_rejected():
    sel = _manifest([_row("images/a/1.bmp", "A"), _row("images/a/2.bmp", "A", source="sd302")])
    sel = sel.assign(fold="0", partition="development")
    with pytest.raises(ValueError, match="source"):
        eg.check_identity_disjoint(sel, _identities([_ident("A", "0")]), folds=(0,))


# ---- min images, unreadable images, ordering ----
def test_min_images_after_dropping_unreadable_and_deterministic_order():
    man = _manifest([_row("images/b/2.bmp", "B"), _row("images/b/1.bmp", "B"),
                     _row("images/a/1.bmp", "A"), _row("images/a/2.bmp", "A"), _row("images/a/3.bmp", "A")])
    ids = _identities([_ident("A", "0"), _ident("B", "0")])
    sel, counts = eg.select_compilation(man, ids, folds=(0,), unreadable={"images/b/1.bmp"})
    assert list(sel["identity"]) == ["A", "A", "A"]
    assert list(sel["path"]) == ["images/a/1.bmp", "images/a/2.bmp", "images/a/3.bmp"]
    assert counts["unreadable_dropped"] == 1
    sel2, _ = eg.select_compilation(man.iloc[::-1], ids, folds=(0,))
    assert list(sel2["path"]) == ["images/a/1.bmp", "images/a/2.bmp", "images/a/3.bmp",
                                  "images/b/1.bmp", "images/b/2.bmp"]


def _five_folds():
    """Identities A..E in development folds 0..4 with 3 images each, plus a holdout identity H."""
    man = _manifest([_row(f"images/{k}/{i}.bmp", k) for k in "ABCDEH" for i in range(3)])
    ids = _identities([_ident(k, str(f)) for f, k in enumerate("ABCDE")] + [_ident("H", "", "holdout")])
    return man, ids


def test_fold_candidates_are_all_development_folds_then_shorter_prefixes():
    assert eg.FOLD_CANDIDATES == ((0, 1, 2, 3, 4), (0, 1, 2, 3), (0, 1, 2), (0, 1), (0,))
    assert eg.MAX_IMAGES == 25_000
    assert eg.FOLD_RULE == "all development folds (0-4), or the largest prefix within 25,000 images"


def test_pick_folds_uses_all_folds_when_they_fit():
    man, ids = _five_folds()
    assert eg.pick_folds(man, ids, max_images=15) == ((0, 1, 2, 3, 4), {"0,1,2,3,4": 15})   # holdout ignored


def test_pick_folds_uses_the_next_prefix_when_all_folds_do_not_fit():
    man, ids = _five_folds()
    assert eg.pick_folds(man, ids, max_images=14) == ((0, 1, 2, 3), {"0,1,2,3,4": 15, "0,1,2,3": 12})
    assert eg.pick_folds(man, ids, max_images=7) == ((0, 1), {"0,1,2,3,4": 15, "0,1,2,3": 12, "0,1,2": 9,
                                                              "0,1": 6})


def test_pick_folds_uses_the_last_candidate_when_none_fits():
    man, ids = _five_folds()
    folds, sizes = eg.pick_folds(man, ids, max_images=2)
    assert folds == (0,) and sizes == {"0,1,2,3,4": 15, "0,1,2,3": 12, "0,1,2": 9, "0,1": 6, "0": 3}


def test_limit_per_source_keeps_whole_identities():
    sel = _manifest([_row(f"i/{k}/{j}.bmp", k, source=src) for k, src in
                     (("A", "s1"), ("B", "s1"), ("C", "s2"), ("D", "s2"), ("E", "s2")) for j in range(2)])
    sub = eg.limit_per_source(sel, 1)
    assert list(sub["identity"]) == ["A", "A", "C", "C"]


def test_corpus_size_requirement():
    sel = _manifest([_row(f"i/{k}/{j}.bmp", k) for k, n in (("A", 3), ("B", 3), ("C", 2), ("D", 2)) for j in range(n)])
    req = eg.check_corpus_size(sel, n_ge3=2, n_further=1)
    assert req == {"n_identities_ge3": 2, "n_identities": 4, "required_ge3": 2, "required_further_gt": 1,
                   "further_identities": 2, "passed": True}
    with pytest.raises(ValueError):
        eg.check_corpus_size(sel, n_ge3=3, n_further=0)
    with pytest.raises(ValueError):
        eg.check_corpus_size(sel, n_ge3=2, n_further=2)    # needs more than 2 further identities


# ---- manifest / embedding alignment ----
def _write_images(root: Path, rows) -> None:
    for r in rows:
        p = root / r["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((40, 30), int(r["subject"]), dtype=np.uint8), "L").save(p)


def _fake_encoder(scale):
    def enc(paths):
        out = []
        for p in paths:
            v = float(np.asarray(Image.open(p)).mean())
            out.append([scale * v, 1.0, 0.0])
        return np.asarray(out)
    return enc


def test_extract_to_sin_rows_align_with_manifest(tmp_path):
    rows = [_row(f"images/s{v}/{v}.png", f"id{v // 10}", subject=str(v)) for v in (11, 12, 13, 21, 22, 31, 32, 33)]
    man = _manifest(rows)
    _write_images(tmp_path / "compilation", rows)
    ids = _identities([_ident(f"id{k}", "0") for k in (1, 2, 3)])
    sel, _ = eg.select_compilation(man.sample(frac=1.0, random_state=3), ids, folds=(0,))
    info = eg.extract_to_sin(sel, tmp_path / "compilation", {"fingervit": _fake_encoder(1.0), "deepprint": _fake_encoder(2.0)},
                             tmp_path / "out")
    got = pd.read_parquet(tmp_path / "out/manifest.parquet")
    assert list(got.columns) == ["identity", "source", "subject", "finger", "device", "capture", "dpi", "path"]
    assert got["path"].tolist() == sel["path"].tolist()
    for name, scale in (("fingervit", 1.0), ("deepprint", 2.0)):
        x = np.load(tmp_path / f"out/emb_{name}.npy")
        assert x.dtype == np.float32 and x.shape == (len(got), 3)
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, rtol=1e-6)
        expect = eg.l2_normalize(np.stack([[scale * float(s), 1.0, 0.0] for s in got["subject"]]))
        np.testing.assert_allclose(x, expect, rtol=1e-6)
        emb = load_sin_corpus(tmp_path / "out", name)
        assert list(emb.ids) == got["identity"].tolist()
    assert info["shapes"] == {"fingervit": [8, 3], "deepprint": [8, 3]}
    assert set(info["files"]) == {"manifest.parquet", "emb_fingervit.npy", "emb_deepprint.npy"}


def test_write_sin_rejects_misaligned_or_bad_arrays(tmp_path):
    sel = _manifest([_row("images/a/1.bmp", "A"), _row("images/a/2.bmp", "A")])
    with pytest.raises(ValueError, match="rows"):
        eg.write_sin(tmp_path / "o1", sel, {"m": np.ones((3, 4))})
    with pytest.raises(ValueError, match="finite"):
        eg.write_sin(tmp_path / "o2", sel, {"m": np.array([[1.0, np.nan], [1.0, 2.0]])})
    with pytest.raises(ValueError, match="zero"):
        eg.write_sin(tmp_path / "o3", sel, {"m": np.array([[0.0, 0.0], [1.0, 2.0]])})


# ---- aggregates and parity helpers ----
def test_composition_is_aggregate_only():
    rows = [_row(f"images/sd302/u{k}_{j}.png", f"sd302:{k}", source="sd302", subject=str(k), device="C",
                 capture="roll", dpi="") for k in range(3) for j in range(2 + k)]
    rows += [_row(f"images/livdet/Live/v{j}.png", "livdet:x", source="livdet", subject="", capture="live")
             for j in range(2)]
    sel = _manifest(rows)
    comp = eg.composition(sel)
    assert comp["n_images"] == 11 and comp["n_identities"] == 4
    assert comp["n_identities_ge3"] == 2 and comp["n_identities_eq2"] == 2
    assert comp["impressions_per_finger_hist"] == {"2": 2, "3": 1, "4": 1}
    assert comp["per_source"]["sd302"] == {"images": 9, "identities": 3}
    assert comp["dpi"] == {"500": 2, "unknown": 9}
    assert comp["capture"] == {"live": 2, "roll": 9}
    assert comp["n_subjects"] == 3
    text = json.dumps(comp)
    for r in rows:
        assert r["path"] not in text and r["identity"] not in text


def test_make_square_compatible_pads_only_odd_differences():
    torch = pytest.importorskip("torch")
    even = torch.rand(1, 192, 224)                       # the old corpus after the 32-row crop
    assert eg.make_square_compatible(even) is even
    tall = torch.rand(1, 11, 8)
    out = eg.make_square_compatible(tall)
    assert tuple(out.shape) == (1, 11, 9) and torch.equal(out[..., :8], tall) and float(out[..., 8].abs().sum()) == 0
    wide = torch.rand(1, 8, 11)
    out = eg.make_square_compatible(wide)
    assert tuple(out.shape) == (1, 9, 11) and torch.equal(out[:, :8], wide) and float(out[:, 8].abs().sum()) == 0
    for t in (tall, wide, torch.rand(1, 5, 5)):          # even difference: flx's floor(d/2) per side gives a square
        h, w = eg.make_square_compatible(t).shape[-2:]
        assert (h - w) % 2 == 0


def test_compilation_deepprint_input_keeps_every_row(monkeypatch):
    torch = pytest.importorskip("torch")
    img = np.random.default_rng(1).integers(0, 256, (70, 50), dtype=np.uint8)
    full = eg.gray_tensor(img)
    assert tuple(full.shape) == (1, 70, 50) and full.dtype == torch.float32
    assert torch.equal(full[0], torch.from_numpy(img).float() / 255)
    monkeypatch.setattr(eg, "read_gray", lambda path: img)                  # compilation path without OpenCV
    assert tuple(eg.deepprint_input("any.png").shape) == (1, 70, 50)
    with pytest.raises(ValueError):
        eg.gray_tensor(img.astype(np.float32))


def test_rgba_report():
    sel = _manifest([_row("i/a.png", "A", source="s1"), _row("i/b.png", "A", source="s1"),
                     _row("i/c.png", "B", source="s2"), _row("i/d.png", "B", source="s2")])
    probe = pd.DataFrame({"path": ["i/a.png", "i/b.png", "i/c.png", "i/d.png", "i/unused.png"],
                          "mode": ["RGBA", "L", "RGBA", "RGBA", "RGBA"],
                          "alpha_nonopaque": [True, False, False, False, True]})
    rep = eg.rgba_report(sel, probe)
    assert rep == {"pil_modes_by_source": {"s1": {"L": 1, "RGBA": 1}, "s2": {"RGBA": 2}},
                   "rgba_images": 3, "rgba_nonopaque_alpha": 1, "rgba_nonopaque_alpha_by_source": {"s1": 1}}


def test_batch_padded_appends_a_row_only_for_a_final_batch_of_one():
    p33 = [f"p{i}" for i in range(33)]                                    # 33 mod 32 == 1
    assert eg.batch_padded(p33, 32) == (p33 + ["p32"], 33)
    p64 = [f"p{i}" for i in range(64)]                                    # 64 mod 32 == 0
    assert eg.batch_padded(p64, 32) == (p64, 64)
    assert eg.batch_padded(["a"], 32) == (["a", "a"], 1)
    assert eg.batch_padded(p33[:31], 32) == (p33[:31], 31)


# ---- parity gate ----
def _parity_and_context():
    sw = {"torch": "2.9.0", "torchvision": "0.24.0", "transformers": "4.57.1", "numpy": "2.2.6", "PIL": "12.0.0",
          "cv2": "4.11.0", "flx_tree_sha256": "f" * 64, "pandas": "2.3.3"}
    parity = {"passed": True, "threshold": eg.PARITY_THRESHOLD, "n": eg.PARITY_N,
              "results": {"fingervit": {"passed": True}, "deepprint": {"passed": True}},
              "code": {"extract_d2.py": "c" * 64}, "weights_sha256": {"w": "1"},
              "settings": {"fingervit_batch_size": 1}, "env": {"image_id": "sha256:img"}, "software": dict(sw)}
    ctx = {"code": {"extract_d2.py": "c" * 64}, "weights": {"w": "1"}, "fv_batch": 1,
           "env": {"image_id": "sha256:img"}, "software": dict(sw, pandas="9.9")}   # pandas does not gate
    return parity, ctx


def test_parity_gate_accepts_a_matching_parity():
    parity, ctx = _parity_and_context()
    assert eg.parity_problems(parity, **ctx) == []


@pytest.mark.parametrize("mutate, reason", [
    (lambda p: p.update(passed=False), "did not pass"),
    (lambda p: p["results"]["deepprint"].update(passed=False), "deepprint did not pass"),
    (lambda p: p["results"].pop("fingervit"), "no fingervit result"),
    (lambda p: p.update(threshold=0.99), "threshold"),
    (lambda p: p.update(n=100), "n=100"),
    (lambda p: p["code"].update({"extract_d2.py": "d" * 64}), "another version of this script"),
    (lambda p: p.update(weights_sha256={"w": "2"}), "other weights"),
    (lambda p: p["settings"].update(fingervit_batch_size=8), "batch size"),
    (lambda p: p["env"].update(image_id="sha256:other"), "another image"),
    (lambda p: p["software"].update(torch="2.8.0"), "another torch"),
    (lambda p: p["software"].update(cv2="4.12.0"), "another cv2"),
    (lambda p: p["software"].update(flx_tree_sha256="0" * 64), "another flx_tree_sha256"),
])
def test_parity_gate_refusals(mutate, reason):
    parity, ctx = _parity_and_context()
    mutate(parity)
    problems = eg.parity_problems(parity, **ctx)
    assert any(reason in p for p in problems), problems


def test_parity_helpers():
    a = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    np.testing.assert_allclose(eg.cosine_rows(a, b), [1.0, 1.0, 0.0], atol=1e-12)
    st = eg.parity_stats(np.array([1.0, 0.9995, 0.9999, 0.998]), threshold=0.999)
    assert st["n"] == 4 and st["passed"] is True and st["min"] == 0.998
    assert eg.parity_stats(np.array([0.99, 0.999, 0.98]), threshold=0.999)["passed"] is False
    k1 = eg.sample_parity_keys({"1_1", "1_2", "2_1", "3_3"}, {"1_1", "1_2", "2_1", "9_9"}, n=2, seed=5)
    k2 = eg.sample_parity_keys({"1_1", "1_2", "2_1", "3_3"}, {"1_1", "1_2", "2_1", "9_9"}, n=2, seed=5)
    assert k1 == k2 and k1 == sorted(k1) and set(k1) <= {"1_1", "1_2", "2_1"} and len(k1) == 2
    with pytest.raises(ValueError):
        eg.sample_parity_keys({"a"}, {"a"}, n=2, seed=0)
