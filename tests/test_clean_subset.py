# tests/test_clean_subset.py
"""Clean-subset sensitivity of scripts/data/clean_subset.py (tiny synthetic fixtures; the data-host run reads real data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/data"))  # the scripts import their neighbours by name
import clean_subset as cs  # noqa: E402
import extract_d2 as eg  # noqa: E402
import training_overlap as to  # noqa: E402
from ijiss_btp.data.loaders import Embeddings, center_normalize, load_sin_corpus  # noqa: E402
from ijiss_btp.data.splits import make_split  # noqa: E402
from ijiss_btp.eval.stats import eer  # noqa: E402
from ijiss_btp.experiments.isotropy import SeedContext, spectrum_stats, stable_seed  # noqa: E402

TAU = 0.997
LABEL = "synthetic: tiny test fixture, not real data"
RAW, PREKEY = "raw (uncentred) cosine space", "experiments' pre-key space (DEV-mean centred, then L2-normalised)"
# split sizes of the fixture's isotropy configs (the real ones are larger than the fixture corpora)
SPLITS = {"d1_base": [11, 12], "d1_large": [13, 14], "d2_fingervit": [21, 22], "d2_deepprint": [31, 32]}
# first corpus: (identity, source, D1 images, route into candidate pool A); None = clean
F1 = ([(f"s1:{k}:R1", "s1", 3, None) for k in range(4)] + [(f"s1:{k}:R1", "s1", 3, "compilation") for k in range(4, 8)]
      + [("s1:8:R1", "s1", 3, "both"), ("s1:9:R1", "s1", 1, None), ("s1:10:R1", "s1", 1, "compilation")]
      + [(f"s2:{k}:L1", "s2", 2, None) for k in range(2)] + [("s2:2:L1", "s2", 2, "ambiguous")]
      + [(f"s2:{k}:L1", "s2", 2, "compilation") for k in range(3, 6)])
OTHER = "s2:99:R1"   # a compilation identity outside the first corpus that shares one key with s2:2:L1
# second corpus: (identity, source, images, flagged)
F2 = ([(f"t1:{k}:R1", "t1", 3, False) for k in range(5)] + [(f"t1:{k}:R1", "t1", 3, True) for k in range(5, 11)]
      + [("t2:0:L1", "t2", 4, False)] + [(f"t2:{k}:L1", "t2", 3, True) for k in range(1, 4)])
IDENTITIES = tuple(i for i, *_ in F1) + tuple(i for i, *_ in F2) + (OTHER,)


def _assert_no_leak(text: str, words=()) -> None:
    for w in tuple(words) + IDENTITIES:
        assert w not in text, w
    for ext in (".png", ".bmp", ".npy", ".parquet", ".csv"):
        assert ext not in text, ext


def _exact_auc(g, i) -> float:
    g, i = np.asarray(g)[:, None], np.asarray(i)[None, :]
    return float(((g > i).sum() + 0.5 * (g == i).sum()) / (g.size * i.size))


# ---- flags ----
def test_d1_flags_unite_the_key_table_and_the_d1_images():
    table = pd.DataFrame({"identity": ["s1:1:R1", "s1:1:R1", "s1:2:R1", "s2:3:R1", "s2:4:R1", OTHER, "s2:5:R1"],
                          "key": ["k1a", "k1b", "k2a", "k_amb", "k4a", "k_amb", "k5a"]})
    src = {"s1:1:R1": "s1", "s1:2:R1": "s1", "s2:3:R1": "s2", "s2:4:R1": "s2", OTHER: "s2", "s2:5:R1": "s2",
           "s1:6:R1": "s1"}
    matches = pd.DataFrame({"identity": ["s1:1:R1", "s1:2:R1", "s2:4:R1", "s2:5:R1", "s1:6:R1", ""],
                            "in_pool": [False, False, False, True, False, True]})
    d1_ids = ["s2:5:R1", "s1:1:R1", "s1:2:R1", "s2:3:R1", "s2:4:R1", "s1:6:R1"]
    f = cs.d1_flags(matches, table, src, ["k1b", "k_amb", None, "k_other"], d1_ids)
    assert f["identity"].tolist() == ["s1:1:R1", "s1:2:R1", "s1:6:R1", "s2:3:R1", "s2:4:R1", "s2:5:R1"]
    assert f["source"].tolist() == ["s1", "s1", "s1", "s2", "s2", "s2"]
    # s1:1 through another labelled impression; s2:3 through a key it shares with a non-D1 identity (worst case);
    # s2:5 through its D1 image only; s1:6 has no readable compilation image; the unassigned D1 image counts for nobody
    assert f["via_compilation"].tolist() == [True, False, False, True, False, False]
    assert f["via_d1"].tolist() == [False, False, False, False, False, True]
    assert f["flagged"].tolist() == [True, False, False, True, False, True]
    with pytest.raises(ValueError, match="not in the compilation key table"):
        cs.d1_flags(matches, table, src, ["k1b"], d1_ids + ["s9:1:R1"])


def test_d2_flags_need_one_image_at_or_above_tau():
    man = pd.DataFrame({"identity": ["b:1", "a:10", "a:9", "a:10", "B:2", "b:1"],
                        "source": ["s2", "s1", "s1", "s1", "s3", "s2"], "path": [f"images/t/{i}.png" for i in range(6)]})
    near = pd.DataFrame({"path": man["path"], "max_cosine": [0.5, 0.9972, 0.99, 0.2, 0.9971, 0.9972]})
    f = cs.d2_flags(near, man, tau=0.9972)
    assert f["identity"].tolist() == ["B:2", "a:10", "a:9", "b:1"]
    assert f["source"].tolist() == ["s3", "s1", "s1", "s2"]
    assert f["flagged"].tolist() == [False, True, False, True]          # the threshold itself counts
    with pytest.raises(ValueError, match="not aligned"):
        cs.d2_flags(near.iloc[::-1].reset_index(drop=True), man, tau=0.9972)


def test_reproduction_problems_compare_the_counts_overall_and_per_source():
    flags = pd.DataFrame({"identity": list("abcde"), "source": ["s1", "s1", "s2", "s2", "s2"],
                          "flagged": [True, False, True, True, False]})
    recorded = json.loads(json.dumps(to.tally(flags["flagged"], flags["source"], "above")))   # as read from JSON
    assert cs.reproduction_problems(flags, recorded, "above") == []
    recorded["above"], recorded["per_source"]["s2"]["above"] = 4, 3
    del recorded["per_source"]["s1"]
    assert cs.reproduction_problems(flags, recorded, "above") == [
        "3 of 5 flagged here, 4 of 5 recorded", "s1: 1 of 2 flagged here, none recorded",
        "s2: 2 of 3 flagged here, 3 of 3 recorded"]


def test_hash_problems_name_the_inputs_that_differ():
    assert cs.hash_problems({"a": "1", "b": "2"}, {"a": "1"}) == []
    assert cs.hash_problems({"a": "1"}, {"a": "9", "c": "3"}) == [
        "the overlap measurement recorded another a", "the overlap measurement recorded no c"]


# ---- subsets and strata ----
def _flags(rows):
    return pd.DataFrame(rows, columns=["identity", "source", "flagged"])


def test_with_images_counts_rows_and_applies_the_image_rule():
    flags = _flags([("a", "s1", True), ("b", "s1", False), ("c", "s1", False), ("d", "s2", True)])
    f = cs.with_images(flags, ["a", "a", "c", "a", "c", "b"], min_images=2)
    assert f["images"].tolist() == [3, 1, 2, 0] and f["eligible"].tolist() == [True, False, True, False]
    assert cs.with_images(flags, ["a", "b"], min_images=1)["eligible"].tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="no flag"):
        cs.with_images(flags, ["a", "z"], min_images=2)


def test_paired_rows_need_the_same_rows_for_every_encoder():
    class E:
        def __init__(self, ids):
            self.ids = np.array(ids)
    assert cs.paired_rows({"m1": E(["a", "b", "a"]), "m2": E(["a", "b", "a"])}).tolist() == ["a", "b", "a"]
    with pytest.raises(ValueError, match="different rows"):
        cs.paired_rows({"m1": E(["a", "b", "a"]), "m2": E(["a", "a", "b"])})


def test_finger_counts_per_source_with_the_dropped_fingers():
    f = cs.with_images(_flags([("a", "s1", True), ("b", "s1", False), ("c", "s1", False), ("d", "s2", True),
                               ("e", "s2", False)]), ["a", "a", "b", "c", "c", "d", "d", "e", "e"], min_images=2)
    counts = cs.finger_counts(f)
    assert counts["per_source"]["s1"] == {"fingers": 3, "clean": 2, "flagged": 1, "images": 5,
                                          "eligible": {"clean": 1, "flagged": 1},
                                          "dropped_by_image_rule": {"clean": 1, "flagged": 0}}
    assert counts["all_sources"] == {"fingers": 5, "clean": 3, "flagged": 2, "images": 9,
                                     "eligible": {"clean": 2, "flagged": 2},
                                     "dropped_by_image_rule": {"clean": 1, "flagged": 0}}
    assert list(counts["per_source"]) == ["s1", "s2"]


def test_strata_pool_all_sources_and_keep_sources_with_enough_clean_fingers():
    rows = ([(f"a{k}", "s1", False) for k in range(4)] + [(f"b{k}", "s1", True) for k in range(3)]
            + [(f"c{k}", "s2", False) for k in range(2)] + [(f"d{k}", "s3", True) for k in range(5)])
    images = [i for i, *_ in rows for _ in range(2) if i != "a3"] + ["a3"]   # a3 has one image only
    f = cs.with_images(_flags(rows), images, min_images=2)
    assert cs.strata(f, 3) == ["all_sources", "s1"]
    assert cs.strata(f, 4) == ["all_sources"]                               # a3 does not count
    clean, flagged = cs.stratum_fingers(f, "s1")
    assert clean.tolist() == ["a0", "a1", "a2"] and flagged.tolist() == ["b0", "b1", "b2"]
    clean, flagged = cs.stratum_fingers(f, "all_sources")
    assert clean.tolist() == ["a0", "a1", "a2", "c0", "c1"] and len(flagged) == 8


def test_stratum_seeds_depend_on_corpus_stratum_and_purpose_only():
    s = cs.stratum_seeds(5, "d1", "all_sources", 3)
    assert s["clean"]["impostors"] == stable_seed(5, "d1", "all_sources", "clean", "impostors")
    assert s["flagged"][2]["subsample"] == stable_seed(5, "d1", "all_sources", "flagged", 2, "subsample")
    values = [*s["clean"].values()] + [v for f in s["flagged"] for v in f.values()]
    assert len(values) == 2 + 3 * 3 and len(set(values)) == len(values)
    assert s == cs.stratum_seeds(5, "d1", "all_sources", 3) and s != cs.stratum_seeds(5, "d2", "all_sources", 3)


def test_matched_subsamples_are_seeded_draws_without_replacement():
    flagged = np.array([f"g{k:02d}" for k in range(12)])
    subs = cs.matched_subsamples(flagged, 5, [11, 12, 13])
    assert len(subs) == 3
    for s in subs:
        assert len(s) == 5 and len(set(s)) == 5 and set(s) <= set(flagged) and (s == np.sort(s)).all()
    again = cs.matched_subsamples(flagged, 5, [11, 12, 13])
    assert all((a == b).all() for a, b in zip(subs, again))
    assert len({tuple(s) for s in subs}) > 1
    with pytest.raises(ValueError, match="fewer flagged"):
        cs.matched_subsamples(flagged, 13, [1])


def test_stratum_problems_need_two_clean_and_enough_flagged_fingers():
    rows = ([("a0", "s1", False), ("a1", "s1", False)] + [(f"b{k}", "s1", True) for k in range(5)]
            + [("c0", "s2", False), ("d0", "s2", True)]
            + [("e0", "s3", False), ("e1", "s3", False), ("e2", "s3", False), ("g0", "s3", True)])
    f = cs.with_images(_flags(rows), [i for i, *_ in rows for _ in range(2)], min_images=2)
    assert cs.stratum_problems(f, ["all_sources", "s1"]) == []              # 6 clean against 7 flagged; 2 against 5
    assert cs.stratum_problems(f, ["all_sources", "s2", "s3"]) == [
        "s2: 1 clean (at least 2 needed)", "s3: 1 flagged for 3 clean (at least as many needed)"]


def test_composition_counts_fingers_and_genuine_pairs_per_source():
    rows = [("a", "s1", False), ("b", "s2", False), ("c", "s1", True), ("d", "s2", True), ("e", "s2", True)]
    images = ["a"] * 3 + ["b"] * 2 + ["c"] * 4 + ["d"] * 2 + ["e"] * 5     # C(k, 2): 3, 1, 6, 1, 10 genuine pairs
    f = cs.with_images(_flags(rows), images, min_images=2)
    comp = cs.composition(f, np.array(["a", "b"]), [np.array(["c", "d"]), np.array(["d", "e"])])
    assert comp["clean"] == {"s1": {"fingers": 1, "genuine_pairs": 3}, "s2": {"fingers": 1, "genuine_pairs": 1}}
    assert comp["flagged_subsamples"] == {
        "s1": {"fingers": {"mean": 0.5, "min": 0, "max": 1}, "genuine_pairs": {"mean": 3.0, "min": 0, "max": 6}},
        "s2": {"fingers": {"mean": 1.5, "min": 1, "max": 2}, "genuine_pairs": {"mean": 6.0, "min": 1, "max": 11}}}


# ---- statistics ----
def test_impostor_sample_draws_distinct_pairs_or_takes_them_all():
    ids = np.array(["a", "a", "b", "c", "c", "c"])
    every = sorted((i, j) for i in range(6) for j in range(i + 1, 6) if ids[i] != ids[j])
    assert len(every) == 11
    for n in (11, 50):                                                      # no more pairs than requested exist
        a, b = cs.impostor_sample(ids, n, seed=1)
        assert sorted(zip(a.tolist(), b.tolist())) == every
    big = np.repeat(np.arange(40).astype(str), 3)
    a, b = cs.impostor_sample(big, 500, seed=3)
    ra, rb = to.impostor_pairs(big, 500, seed=3)
    assert (a == ra).all() and (b == rb).all()
    assert len(set(zip(a.tolist(), b.tolist()))) == 500 and (a < b).all() and (big[a] != big[b]).all()


def test_one_per_finger_picks_one_seeded_row_per_identity():
    ids = np.array(["b", "a", "b", "c", "a", "b"])
    rows = cs.one_per_finger(ids, seed=4)
    assert ids[rows].tolist() == ["a", "b", "c"]
    assert (rows == cs.one_per_finger(ids, seed=4)).all()
    assert any((cs.one_per_finger(ids, seed=s) != rows).any() for s in range(5, 30))


def test_subset_stats_match_brute_force():
    rng = np.random.default_rng(0)
    ids = np.repeat([f"f{k}" for k in range(10)], 3)
    x = 5.0 * (np.repeat(rng.normal(size=(10, 6)), 3, axis=0) + 0.8 * rng.normal(size=(30, 6)))  # not normalised
    fingers = np.array(["f1", "f3", "f4", "f7", "f8"])
    st = cs.subset_stats(x, ids, fingers, n_imp=1000, imp_seed=3, one_seed=5, bins=4000)
    rows = np.flatnonzero(np.isin(ids, fingers))
    xn, sub = eg.l2_normalize(x[rows]), ids[rows]
    pairs = [(i, j) for i in range(15) for j in range(i + 1, 15)]
    gen = np.array([xn[i] @ xn[j] for i, j in pairs if sub[i] == sub[j]])
    imp = np.array([xn[i] @ xn[j] for i, j in pairs if sub[i] != sub[j]])    # all 90 exist: fewer than requested
    assert (st["fingers"], st["images"], st["genuine_pairs"], st["impostor_pairs"]) == (5, 15, 15, 90)
    assert st["mean_genuine"] == pytest.approx(gen.mean()) and st["mean_impostor"] == pytest.approx(imp.mean())
    assert st["eer"] == pytest.approx(eer(gen, imp))
    assert st["auc"] == pytest.approx(_exact_auc(gen, imp), abs=1e-3)
    assert st["pr_all_images"] == pytest.approx(spectrum_stats(xn)["participation_ratio"])
    one = cs.one_per_finger(sub, 5)
    assert st["pr_one_image_per_finger"] == pytest.approx(spectrum_stats(xn[one])["participation_ratio"])
    assert (st["pr_all_images_samples"], st["pr_one_image_per_finger_samples"]) == (15, 5)
    assert cs.subset_stats(3.0 * x, ids, fingers, 1000, 3, 5, 4000) == pytest.approx(st)   # on normalised rows
    with pytest.raises(ValueError, match="no rows"):
        cs.subset_stats(x, ids, np.array(["f1", "zz"]), 1000, 3, 5, 4000)


def _entry(value):
    return {**{k: value for k in cs.STATISTICS}, **{k: 10 for k in cs.COUNTS}}


def test_compare_reports_the_flagged_mean_and_percentile_range():
    v = np.linspace(0.90, 0.99, 20)
    flagged = [dict(_entry(float(x)), images=30 + i) for i, x in enumerate(v)]
    clean = dict(_entry(0.95), auc=0.80, images=28)
    res = cs.compare(clean, flagged)
    assert set(res["statistics"]) == set(cs.STATISTICS) and set(res["counts"]) == set(cs.COUNTS)
    e = res["statistics"]["eer"]
    assert e["clean"] == 0.95 and e["flagged_mean"] == pytest.approx(v.mean())
    assert e["flagged_p2.5"] == pytest.approx(np.percentile(v, 2.5))
    assert e["flagged_p97.5"] == pytest.approx(np.percentile(v, 97.5))
    assert e["clean_outside_flagged_range"] is False
    assert res["statistics"]["auc"]["clean_outside_flagged_range"] is True
    assert res["counts"]["images"] == {"clean": 28, "flagged_min": 30, "flagged_mean": 39.5, "flagged_max": 49}


# ---- the experiments' pre-key space ----
def test_prekey_space_centres_on_the_dev_mean_of_the_configured_first_split():
    rng = np.random.default_rng(3)
    ids = np.repeat([f"f{k:02d}" for k in range(12)], 3)
    x = rng.normal(size=(36, 5)) + 4.0                                      # a large shared component
    split = {"n_dev": 4, "n_tgt": 5, "min_tgt_images": 3, "seeds": [7, 8]}
    xc, info = cs.prekey_space(x, ids, split)
    ctx = SeedContext(Embeddings(x, ids, "synthetic"), {"split": split, "fmr": [0.01]}, 7, None)   # the runner itself
    np.testing.assert_allclose(xc, ctx.xn, rtol=1e-12, atol=1e-15)
    s = make_split(ids, 4, 5, 7, 3)
    np.testing.assert_allclose(xc, center_normalize(x, x[s.rows(s.dev_ids)].mean(0)), rtol=1e-12, atol=1e-15)
    assert info == {"seed": 7, "n_dev": 4, "n_tgt": 5, "min_tgt_images": 3, "dev_fingers": 4, "dev_images": 12}
    second = make_split(ids, 4, 5, 8, 3)                                    # the first seed, not another one
    assert set(second.dev_ids) != set(s.dev_ids)
    assert not np.allclose(xc, center_normalize(x, x[second.rows(second.dev_ids)].mean(0)))


def test_split_config_must_describe_the_loaded_embeddings(tmp_path):
    spec = {"kind": "sin", "root": "/data/d1", "model": "base", "relabel": "/elsewhere/d1_relabel.csv"}
    split = {"n_dev": 4, "n_tgt": 5, "seeds": [11]}
    path = tmp_path / "isotropy_d1_base.yaml"
    path.write_text(yaml.safe_dump({"data": {"kind": "sin", "root": "/data/primary", "model": "base",
                                             "relabel": "/data/relabel/d1_relabel.csv"}, "split": split}))
    assert cs.split_config(path, spec) == (split, _sha(path))
    path.write_text(yaml.safe_dump({"data": {"kind": "sin", "root": "/data/primary", "model": "large",
                                             "relabel": "/data/relabel/d1_relabel.csv"}, "split": split}))
    with pytest.raises(SystemExit, match="other embeddings"):
        cs.split_config(path, spec)
    path.write_text(yaml.safe_dump({"data": {"kind": "sin", "root": "/data/primary", "model": "base"}, "split": split}))
    with pytest.raises(SystemExit, match="other embeddings"):
        cs.split_config(path, spec)                                         # no relabel map: another identity space


def test_evaluate_shares_fingers_pairs_and_seeds_between_the_spaces():
    rng = np.random.default_rng(5)
    ids = np.repeat([f"f{k:02d}" for k in range(14)], 3)
    x = rng.normal(size=(42, 6)) + 2.0
    flags = cs.with_images(_flags([(f"f{k:02d}", "s1", k >= 5) for k in range(14)]), ids, min_images=2)
    seeds = cs.stratum_seeds(9, "d1", "all_sources", 3)
    same = cs.evaluate({"raw": x, "prekey": x.copy()}, ids, flags, "all_sources", seeds, 40)
    assert same["spaces"]["raw"]["statistics"] == same["spaces"]["prekey"]["statistics"]   # same input, same output
    assert (same["spaces"]["raw"]["label"], same["spaces"]["prekey"]["label"]) == (RAW, PREKEY)
    assert set(same["counts"]) == {"fingers", "images", "genuine_pairs", "impostor_pairs"}
    xc = center_normalize(x, x[:9].mean(0))
    res = cs.evaluate({"raw": x, "prekey": xc}, ids, flags, "all_sources", seeds, 40)
    assert res["counts"] == same["counts"] and res["spaces"]["raw"] == same["spaces"]["raw"]
    assert res["fingers"] == {"clean": 5, "flagged": 9, "subsample_size": 5}
    clean, flagged = cs.stratum_fingers(flags, "all_sources")
    subs = cs.matched_subsamples(flagged, 5, [q["subsample"] for q in seeds["flagged"]])
    want = cs.subset_stats(xc, ids, clean, 40, seeds["clean"]["impostors"], seeds["clean"]["one_image_per_finger"],
                           cs.AUC_BINS)
    flag = [cs.subset_stats(xc, ids, sub, 40, q["impostors"], q["one_image_per_finger"], cs.AUC_BINS)
            for sub, q in zip(subs, seeds["flagged"])]
    got = res["spaces"]["prekey"]["statistics"]
    for k in cs.STATISTICS:
        assert got[k]["clean"] == pytest.approx(want[k])
        assert got[k]["flagged_mean"] == pytest.approx(np.mean([q[k] for q in flag]))


# ---- end to end ----
def _clusters(rng, n_per, dim):
    centres = 2.0 * rng.normal(size=(len(n_per), dim))
    return np.vstack([c + rng.normal(size=(n, dim)) for c, n in zip(centres, n_per)])


def _sha(path) -> str:
    return eg.sha256_file(path)


def _fixture(tmp: Path) -> dict:
    """Both corpora in the SIN layout, the key table and the per-image results of an overlap run."""
    rng = np.random.default_rng(7)
    overlap, keys, d1, d2, out = (tmp / n for n in ("overlap", "keys", "d1", "d2", "out"))
    for p in (overlap, keys, d1, d2):
        p.mkdir()
    # first corpus: every D1 image is a labelled compilation image; each finger has one more impression outside D1
    files, idents, d1_keys, table, pool = [], [], [], [], ["k_unrelated", None]
    for ident, src, n_img, route in F1:
        for _ in range(n_img):
            n = len(files)
            files.append(f"img{n}.bmp")
            idents.append(ident)
            d1_keys.append(f"k{n}")
            table.append((f"images/{src}/{n}.png", ident, src, f"k{n}"))
        extra = f"x:{ident}"
        table.append((f"images/{src}/x{len(table)}.png", ident, src, extra))
        if route in ("compilation", "ambiguous"):
            pool.append(extra)
        if route == "ambiguous":
            table.append(("images/s2/other.png", OTHER, "s2", extra))
        if route == "both":
            pool.append(d1_keys[-1])
    all_files, all_keys = files + ["imgu0.bmp", "imgu1.bmp"], d1_keys + ["k_u0", "k_u1"]   # two unassigned images
    pool.append("k_u0")
    pool_set = {k for k in pool if k}
    pd.DataFrame({"path": [f"/elsewhere/test/{f}" for f in all_files], "identity": "old"}).to_parquet(
        d1 / "manifest.parquet")
    pd.DataFrame({"file": files, "identity": idents}).to_csv(tmp / "d1_relabel.csv", index=False)
    n_per1 = [n for _, _, n, _ in F1]
    for model, dim in (("base", 6), ("large", 9)):
        np.save(d1 / f"emb_{model}.npy", np.vstack([_clusters(rng, n_per1, dim), rng.normal(size=(2, dim))]))
    pd.DataFrame({"name": [f"p{i}.png" for i in range(len(pool))], "key": pool}).to_parquet(overlap / to.POOL_A_KEYS)
    pd.DataFrame({"file": all_files, "identity": idents + ["", ""], "key": all_keys,
                  "in_pool": [k in pool_set for k in all_keys]}).to_parquet(overlap / to.D1_MATCHES)
    kt = pd.DataFrame(table, columns=["path", "identity", "source", "key"]).assign(exclusion="none")
    kt.to_parquet(keys / "compilation_keys.parquet", index=False)
    (tmp / "compilation").mkdir()
    kt[["path", "identity", "source"]].to_csv(tmp / "compilation" / "MANIFEST.csv", index=False)
    manifest_sha = _sha(tmp / "compilation" / "MANIFEST.csv")
    (keys / "compilation_keys.json").write_text(json.dumps({
        "compilation_manifest_sha256": manifest_sha, "labelled_rows": len(kt), "unreadable": 0,
        "distinct_keys": int(kt["key"].nunique()), "keys_under_several_identities": 1}))
    order1 = sorted(i for i, *_ in F1)
    routes, src1 = {i: r for i, _, _, r in F1}, {i: s for i, s, _, _ in F1}
    worst = to.tally([routes[i] is not None for i in order1], [src1[i] for i in order1], "in_pool")
    unique = to.tally([routes[i] not in (None, "ambiguous") for i in order1], [src1[i] for i in order1], "in_pool")
    assert unique["in_pool"] == worst["in_pool"] - 1                        # a lookup of the unique-key tally fails
    (overlap / "a.json").write_text(json.dumps({
        "d1_fingers": dict(unique, worst_case=worst),
        "inputs_sha256": {"d1_manifest": _sha(d1 / "manifest.parquet"), "d1_relabel_map": _sha(tmp / "d1_relabel.csv"),
                          "compilation_manifest": manifest_sha, "compilation_key_table": _sha(keys / "compilation_keys.parquet")},
        "outputs_sha256": {"pool_a_keys": _sha(overlap / to.POOL_A_KEYS),
                           "d1_exact_matches": _sha(overlap / to.D1_MATCHES)}}))
    # second corpus: a flagged finger has one image at or above tau (t1:5 exactly at tau)
    ids2, src2, paths2, maxcos = [], [], [], []
    for ident, src, n_img, flagged in F2:
        for j in range(n_img):
            paths2.append(f"images/{src}/{len(paths2)}.png")
            ids2.append(ident)
            src2.append(src)
            top = TAU if ident == "t1:5:R1" else 0.999
            maxcos.append((top if j == 0 else 0.5) if flagged else 0.9969)
    pd.DataFrame({"identity": ids2, "source": src2, "path": paths2}).to_parquet(d2 / "manifest.parquet")
    n_per2 = [n for _, _, n, _ in F2]
    for model, dim in (("fingervit", 10), ("deepprint", 4)):
        np.save(d2 / f"emb_{model}.npy", eg.l2_normalize(_clusters(rng, n_per2, dim)).astype(np.float32))
    pd.DataFrame({"path": paths2, "max_cosine": maxcos, "pool_name": "q.png"}).to_parquet(overlap / to.D2_NEAREST)
    order2 = sorted(i for i, *_ in F2)
    flagged2, s2 = {i: f for i, _, _, f in F2}, {i: s for i, s, _, _ in F2}
    (overlap / "b.json").write_text(json.dumps({
        "tau_rule": {"tau": TAU, "separable": True},
        "d2": {"by_threshold": [
            {"name": "positives p0.1", "value": 0.99, "fingers": {"above": 15, "of": 15}},
            {"name": "tau = positives p1", "value": TAU,
             "fingers": to.tally([flagged2[i] for i in order2], [s2[i] for i in order2], "above")}]},
        "inputs_sha256": {"d2_manifest": _sha(d2 / "manifest.parquet"), "d2_emb_fingervit": _sha(d2 / "emb_fingervit.npy")}}))
    # the encoders' isotropy configs, with split sizes that fit the fixture corpora
    configs = tmp / "configs"
    configs.mkdir()
    for name, seeds in SPLITS.items():
        corpus, model = name.split("_")
        data = ({"kind": "sin", "root": "/data/primary", "model": model, "relabel": "/data/relabel/d1_relabel.csv"}
                if corpus == "d1" else {"kind": "sin", "root": "/data/corpus_d2", "model": model})
        (configs / f"isotropy_{name}.yaml").write_text(yaml.safe_dump(
            {"name": name, "data": data, "split": {"n_dev": 4, "n_tgt": 5, "min_tgt_images": 3, "seeds": seeds}}))
    return {"overlap": overlap, "keys": keys / "compilation_keys.parquet", "d1": d1, "map": tmp / "d1_relabel.csv", "d2": d2,
            "out": out, "compilation_manifest": tmp / "compilation" / "MANIFEST.csv", "configs": configs}


def _run(f: dict, *extra: str) -> dict:
    cs.main(["--overlap", str(f["overlap"]), "--d1-overlap", str(f["overlap"] / "a.json"),
             "--d2-overlap", str(f["overlap"] / "b.json"), "--d1", str(f["d1"]), "--d1-map", str(f["map"]),
             "--compilation-keys", str(f["keys"]), "--compilation-manifest", str(f["compilation_manifest"]), "--d2", str(f["d2"]),
             "--configs", str(f["configs"]), "--out", str(f["out"]), "--json", str(f["out"] / "c.json"),
             "--label", LABEL, "--min-clean", "3", "--subsamples", "4", "--impostor-pairs", "60", "--seed", "5", *extra])
    return json.loads((f["out"] / "c.json").read_text())


def test_clean_subset_end_to_end(tmp_path):
    f = _fixture(tmp_path)
    res = _run(f)
    text = (f["out"] / "c.json").read_text()
    assert res["flags"]["d1"]["reproduced"] == {"flagged": 10, "of": 17}
    assert res["flags"]["d1"]["routes"] == {"via_compilation_key_table": 10, "via_d1_images": 1, "only_via_d1_images": 0}
    assert res["flags"]["d2"]["reproduced"] == {"flagged": 9, "of": 15} and res["flags"]["d2"]["tau"] == TAU
    fd1, fd2 = res["fingers"]["d1"], res["fingers"]["d2"]
    assert fd1["per_source"]["s1"] == {"fingers": 11, "clean": 5, "flagged": 6, "images": 29,
                                       "eligible": {"clean": 4, "flagged": 5},
                                       "dropped_by_image_rule": {"clean": 1, "flagged": 1}}
    assert fd1["all_sources"]["eligible"] == {"clean": 6, "flagged": 9} and fd1["all_sources"]["images"] == 41
    assert fd2["per_source"]["t2"] == {"fingers": 4, "clean": 1, "flagged": 3, "images": 13,
                                       "eligible": {"clean": 1, "flagged": 3},
                                       "dropped_by_image_rule": {"clean": 0, "flagged": 0}}
    assert res["loading"]["d1"] == {"manifest_rows": 43, "rows": 41, "dropped_without_identity": 2}
    assert res["strata"] == {"d1": ["all_sources", "s1"], "d2": ["all_sources", "t1"]}
    s = res["settings"]
    assert (s["seed"], s["subsamples"], s["min_images"], s["min_clean_fingers"], s["impostor_pairs"]) == (5, 4, 2, 3, 60)
    assert s["tau"] == TAU and s["range_percentiles"] == [2.5, 97.5] and "at least 2" in s["min_images_rule"]
    assert set(res["results"]) == {"d1", "d2"}
    assert set(res["results"]["d1"]) == {"base", "large"} and set(res["results"]["d2"]) == {"fingervit", "deepprint"}
    dims = {m: r["dim"] for c in res["results"].values() for m, r in c.items()}
    assert dims == {"base": 6, "large": 9, "fingervit": 10, "deepprint": 4}
    for corpus, models in res["results"].items():
        a, b = (models[m]["strata"] for m in models)
        for stratum in res["strata"][corpus]:
            r = a[stratum]
            assert {n: v["label"] for n, v in r["spaces"].items()} == {"raw": RAW, "prekey": PREKEY}
            assert all(set(v["statistics"]) == set(cs.STATISTICS) for v in r["spaces"].values())
            assert set(r["counts"]) == {"fingers", "images", "genuine_pairs", "impostor_pairs"}
            assert r["counts"] == b[stratum]["counts"]          # paired: same fingers, images and pairs
            for space in ("raw", "prekey"):
                for k in ("pr_one_image_per_finger", "pr_all_images"):
                    assert r["spaces"][space]["statistics"][k]["samples"] == b[stratum]["spaces"]["raw"][
                        "statistics"][k]["samples"]
            assert r["fingers"]["subsample_size"] == r["fingers"]["clean"] == r["counts"]["fingers"]["clean"]
    r = res["results"]["d1"]["base"]["strata"]["s1"]
    assert r["fingers"] == {"clean": 4, "flagged": 5, "subsample_size": 4}
    assert r["counts"]["impostor_pairs"]["clean"] == 54                     # all 54 exist: fewer than requested
    pr = res["results"]["d2"]["fingervit"]["strata"]["all_sources"]["spaces"]["raw"]["statistics"]
    assert res["results"]["d2"]["fingervit"]["strata"]["all_sources"]["counts"]["impostor_pairs"]["clean"] == 60
    one, every = pr["pr_one_image_per_finger"], pr["pr_all_images"]          # the sample count next to each value
    assert one["samples"] == {"clean": 6, "flagged_min": 6, "flagged_mean": 6.0, "flagged_max": 6}
    assert (one["max_possible_clean"], one["bounded_by_samples"]) == (5, True)
    assert every["samples"]["clean"] == 19 and (every["max_possible_clean"], every["bounded_by_samples"]) == (10, False)
    dp = res["results"]["d2"]["deepprint"]["strata"]["all_sources"]["spaces"]["raw"]["statistics"]
    assert (dp["pr_one_image_per_finger"]["max_possible_clean"], dp["pr_one_image_per_finger"]["bounded_by_samples"]) \
        == (4, False)
    # the pre-key record of every encoder: its config, the first seed and the DEV size of that seed's split
    for corpus, models in res["results"].items():
        root, relabel = (f["d1"], f["map"]) if corpus == "d1" else (f["d2"], None)
        for model, rm in models.items():
            name, seed = f"isotropy_{corpus}_{model}.yaml", SPLITS[f"{corpus}_{model}"][0]
            split = make_split(load_sin_corpus(root, model, relabel=relabel).ids, 4, 5, seed, 3)
            assert rm["prekey"] == {"config": name, "config_sha256": _sha(f["configs"] / name), "seed": seed,
                                    "n_dev": 4, "n_tgt": 5, "min_tgt_images": 3, "dev_fingers": 4,
                                    "dev_images": int(len(split.rows(split.dev_ids)))}
    # composition of the pooled stratum: fingers and genuine pairs per source, no identity
    comp = res["composition"]["d1"]["all_sources"]
    assert comp["clean"] == {"s1": {"fingers": 4, "genuine_pairs": 12}, "s2": {"fingers": 2, "genuine_pairs": 2}}
    assert set(comp["flagged_subsamples"]) == {"s1", "s2"}
    assert sum(v["genuine_pairs"] for v in comp["clean"].values()) == res["results"]["d1"]["base"]["strata"][
        "all_sources"]["counts"]["genuine_pairs"]["clean"]
    assert set(res["composition"]["d2"]) == {"all_sources", "t1"}

    # the wiring: the reported values equal a direct computation with the recorded seeds, in both spaces
    seeds = res["seeds"]["d1"]["all_sources"]
    emb = load_sin_corpus(f["d1"], "large", relabel=f["map"])
    xc, _ = cs.prekey_space(emb.x, emb.ids, {"n_dev": 4, "n_tgt": 5, "min_tgt_images": 3, "seeds": SPLITS["d1_large"]})
    clean_ids = np.array(sorted(["s1:0:R1", "s1:1:R1", "s1:2:R1", "s1:3:R1", "s2:0:L1", "s2:1:L1"]))
    flagged_ids = np.array(sorted([f"s1:{k}:R1" for k in range(4, 9)] + [f"s2:{k}:L1" for k in range(2, 6)]))
    subs = cs.matched_subsamples(flagged_ids, 6, [q["subsample"] for q in seeds["flagged"]])
    for space, x in (("raw", emb.x), ("prekey", xc)):
        want = cs.subset_stats(x, emb.ids, clean_ids, 60, seeds["clean"]["impostors"],
                               seeds["clean"]["one_image_per_finger"], cs.AUC_BINS)
        flag = [cs.subset_stats(x, emb.ids, sub, 60, q["impostors"], q["one_image_per_finger"], cs.AUC_BINS)
                for sub, q in zip(subs, seeds["flagged"])]
        got = res["results"]["d1"]["large"]["strata"]["all_sources"]["spaces"][space]["statistics"]
        for k in cs.STATISTICS:
            assert got[k]["clean"] == pytest.approx(want[k]) and got[k]["flagged_mean"] == pytest.approx(
                np.mean([q[k] for q in flag]))
    raw_auc = res["results"]["d1"]["large"]["strata"]["all_sources"]["spaces"]["raw"]["statistics"]["auc"]["clean"]
    assert raw_auc != res["results"]["d1"]["large"]["strata"]["all_sources"]["spaces"]["prekey"]["statistics"]["auc"][
        "clean"]                                                            # the centring changes the scores

    inputs = res["inputs_sha256"]
    assert set(inputs) == {"d1_manifest", "d1_relabel_map", "d1_emb_base", "d1_emb_large", "compilation_key_table",
                           "compilation_key_table_metadata", "compilation_manifest", "pool_a_keys", "d1_exact_matches",
                           "d1_overlap_aggregates", "d2_manifest", "d2_emb_fingervit", "d2_emb_deepprint",
                           "d2_pool_maxima", "d2_overlap_aggregates"}
    assert inputs["d2_pool_maxima"] == _sha(f["overlap"] / to.D2_NEAREST)
    assert inputs["d1_emb_large"] == _sha(f["d1"] / "emb_large.npy")
    assert inputs["d1_overlap_aggregates"] == _sha(f["overlap"] / "a.json")
    assert inputs["compilation_manifest"] == _sha(f["compilation_manifest"])
    assert inputs["compilation_key_table_metadata"] == _sha(f["keys"].with_suffix(".json"))
    s = res["settings"]
    assert s["spaces"] == {"raw": RAW, "prekey": PREKEY} and "first seed" in s["prekey_rule"]
    assert "C(k, 2)" in s["composition_note"] and "single draw" in s["single_draw"]
    assert "both spaces" in s["participation_ratio"]
    flags1 = pd.read_parquet(f["out"] / "d1_finger_flags.parquet")
    assert {"identity", "source", "flagged", "via_compilation", "via_d1", "images", "eligible"} <= set(flags1)
    assert len(flags1) == 17 and len(pd.read_parquet(f["out"] / "d2_finger_flags.parquet")) == 15
    assert res["outputs_sha256"] == {"d1_finger_flags": _sha(f["out"] / "d1_finger_flags.parquet"),
                                     "d2_finger_flags": _sha(f["out"] / "d2_finger_flags.parquet")}
    assert res["provenance"]["script_sha256"] == _sha(ROOT / "scripts/data/clean_subset.py")
    imported = res["provenance"]["imported_sha256"]                       # library modules by name, not by path
    assert imported["ijiss_btp.data.loaders"] == _sha(ROOT / "src/ijiss_btp/data/loaders.py")
    assert imported["training_overlap.py"] == _sha(ROOT / "scripts/data/training_overlap.py")
    assert res["pools"] == {"d1": to.POOL_A_LABEL, "d2": to.POOL_B_LABEL} and res["data_source"] == LABEL
    _assert_no_leak(text, ("img0", "imgu0", "elsewhere", "images/", "/data/", str(tmp_path)))


def test_clean_subset_stops_on_a_degenerate_stratum_before_loading_embeddings(tmp_path):
    f = _fixture(tmp_path)
    for name in ("emb_base.npy", "emb_large.npy"):                          # never needed: the check comes first
        (f["d1"] / name).unlink()
    for name in ("emb_deepprint.npy",):
        (f["d2"] / name).unlink()
    with pytest.raises(SystemExit, match=r"degenerate strata: D2 t2: 1 clean \(at least 2 needed\)"):
        _run(f, "--min-clean", "1")                                         # t2 (one clean finger) becomes a stratum
    assert not (f["out"] / "c.json").exists()


def test_clean_subset_stops_when_the_flags_do_not_reproduce(tmp_path):
    f = _fixture(tmp_path)
    rec = json.loads((f["overlap"] / "b.json").read_text())
    rec["d2"]["by_threshold"][1]["fingers"]["above"] = 8
    (f["overlap"] / "b.json").write_text(json.dumps(rec))
    for name in ("emb_base.npy", "emb_large.npy"):                          # never needed: the check comes first
        (f["d1"] / name).unlink()
    (f["d2"] / "emb_deepprint.npy").unlink()
    with pytest.raises(SystemExit, match="do not reproduce.*D2 9 of 15 flagged here, 8 of 15 recorded"):
        _run(f)
    assert not (f["out"] / "c.json").exists()


def test_clean_subset_stops_when_an_input_differs_from_the_recorded_one(tmp_path):
    f = _fixture(tmp_path)
    rec = json.loads((f["overlap"] / "a.json").read_text())
    rec["outputs_sha256"]["d1_exact_matches"] = "0" * 64
    (f["overlap"] / "a.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="recorded another d1_exact_matches"):
        _run(f)
    assert not (f["out"] / "c.json").exists()
