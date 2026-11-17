# tests/test_control_subject_check.py
"""Subject-sensitivity check of the control encoder (scripts/data/control_subject_check.py).
Every manifest, flag table and embedding here is a synthetic fixture; no real biometric data is used."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import clean_subset as cs  # noqa: E402
import label_check as lc  # noqa: E402

spec = importlib.util.spec_from_file_location("control_subject_check", ROOT / "scripts" / "data" / "control_subject_check.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

LABEL = "synthetic: tiny test fixture, not real data"
DIM = 16
N_CLEAN, N_FLAGGED, N_POLYU = 3, 9, 6
SPLIT = {"n_dev": 5, "n_tgt": 6, "min_tgt_images": 3, "seeds": [201, 202]}


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fixture(seed=0):
    """Synthetic D2: casia_v5 has 3 clean and 9 flagged fingers (3 images each) and one flagged finger with a single
    image; polyu_cross has 6 fingers.  Returns (manifest, subject flags, {encoder: embeddings})."""
    rng = np.random.default_rng(seed)
    rows, flag_rows = [], []
    for k in range(N_CLEAN + N_FLAGGED):
        ident = f"casia_v5:s{k}:R"
        flag_rows.append((ident, "casia_v5", k >= N_CLEAN))
        rows += [(f"c{k}_{j}.bmp", ident, "casia_v5", "dev") for j in range(3)]
    flag_rows.append(("casia_v5:single:R", "casia_v5", True))
    rows.append(("single.bmp", "casia_v5:single:R", "casia_v5", "dev"))
    for k in range(N_POLYU):
        ident = f"polyu_cross:p{k}:R"
        flag_rows.append((ident, "polyu_cross", False))
        rows += [(f"p{k}_{j}.bmp", ident, "polyu_cross", "dev") for j in range(3)]
    man = pd.DataFrame(rows, columns=["path", "identity", "source", "device"])
    flags = pd.DataFrame(flag_rows, columns=["identity", "source", "subject_shared"])
    codes = pd.factorize(man["identity"])[0]
    embs = {}
    for name, noise in (("ctrlvit", 0.3), ("fingervit", 0.4), ("deepprint", 0.5)):
        centres = unit(rng.normal(size=(codes.max() + 1, DIM)))
        embs[name] = unit(centres[codes] + noise * rng.normal(size=(len(man), DIM)))
    return man, flags, embs


def write_configs(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for m in ("ctrlvit", "fingervit", "deepprint"):
        cfg = {"name": f"d2_{m}", "data": {"kind": "sin", "root": "/data/corpus_d2", "model": m}, "split": SPLIT}
        (d / f"isotropy_d2_{m}.yaml").write_text(yaml.safe_dump(cfg))
    return d


def write_run(root: Path, man, embs, models):
    root.mkdir(parents=True, exist_ok=True)
    man.to_parquet(root / "manifest.parquet", index=False)
    for m in models:
        np.save(root / f"emb_{m}.npy", embs[m])
    return root


# ---- flags and strata ----
def test_stratum_frame_keeps_the_stratum_and_maps_subject_sharing_to_flagged():
    man, flags, _ = fixture()
    f = sc.stratum_frame(flags, man["identity"], "casia_v5", min_images=2)
    assert set(f["source"]) == {"casia_v5"}
    by = f.set_index("identity")
    assert not by.loc["casia_v5:s0:R", "flagged"] and by.loc["casia_v5:s5:R", "flagged"]
    assert by.loc["casia_v5:s0:R", "images"] == 3
    # one image: kept in the table but not eligible (no genuine pair)
    assert by.loc["casia_v5:single:R", "images"] == 1 and not by.loc["casia_v5:single:R", "eligible"]
    clean, flagged = cs.stratum_fingers(f, "casia_v5")
    assert len(clean) == N_CLEAN and len(flagged) == N_FLAGGED   # the single-image flagged finger is dropped


def test_stratum_frame_rejects_a_manifest_identity_without_a_flag():
    man, flags, _ = fixture()
    with pytest.raises(ValueError):
        sc.stratum_frame(flags[flags["identity"] != "casia_v5:s0:R"], man["identity"], "casia_v5", 2)


def test_stratum_frame_rejects_an_unknown_stratum():
    man, flags, _ = fixture()
    with pytest.raises(ValueError):
        sc.stratum_frame(flags, man["identity"], "nope", 2)


# ---- the comparison ----
def test_check_reuses_the_clean_subset_statistics_on_the_same_fingers_images_and_pairs():
    man, flags, embs = fixture()
    ids = man["identity"].to_numpy(dtype=str)
    f = sc.stratum_frame(flags, ids, "casia_v5", 2)
    res = sc.check_encoder(embs["ctrlvit"], ids, f, SPLIT, "casia_v5", subsamples=4, n_imp=500, seed=7)
    assert res["fingers"] == {"clean": N_CLEAN, "flagged": N_FLAGGED, "subsample_size": N_CLEAN}
    assert set(res["spaces"]) == {"raw", "prekey"}
    seeds = cs.stratum_seeds(7, sc.CORPUS, "casia_v5", 4)
    clean, _ = cs.stratum_fingers(f, "casia_v5")
    want = cs.subset_stats(embs["ctrlvit"], ids, clean, 500, seeds["clean"]["impostors"],
                           seeds["clean"]["one_image_per_finger"], cs.AUC_BINS)
    raw = res["spaces"]["raw"]["statistics"]
    for k in cs.STATISTICS:
        assert raw[k]["clean"] == pytest.approx(want[k])
    assert res["prekey_record"]["seed"] == 201 and res["prekey_record"]["n_dev"] == 5


def test_check_is_deterministic_and_the_pairs_do_not_depend_on_the_encoder():
    man, flags, embs = fixture()
    ids = man["identity"].to_numpy(dtype=str)
    f = sc.stratum_frame(flags, ids, "casia_v5", 2)
    a = sc.check_encoder(embs["ctrlvit"], ids, f, SPLIT, "casia_v5", 3, 300, 1)
    b = sc.check_encoder(embs["ctrlvit"], ids, f, SPLIT, "casia_v5", 3, 300, 1)
    assert a == b
    c = sc.check_encoder(embs["fingervit"], ids, f, SPLIT, "casia_v5", 3, 300, 1)
    assert c["counts"] == a["counts"]   # the same genuine and impostor pairs for every encoder


def test_stratum_check_refuses_more_clean_than_flagged_fingers():
    man, flags, _ = fixture()
    ids = man["identity"].to_numpy(dtype=str)
    f = sc.stratum_frame(flags.assign(subject_shared=~flags["subject_shared"]), ids, "casia_v5", 2)  # 9 clean, 3 flagged
    with pytest.raises(SystemExit):
        sc.stratum_check(f, "casia_v5")


# ---- PolyU ----
def test_polyu_auc_equals_the_label_check_auc_of_that_source():
    man, _, embs = fixture()
    got = sc.source_auc(man, embs, "polyu_cross")
    full = lc.label_check(man, embs)
    for m in embs:
        want = full["models"][m]["per_source"]["polyu_cross"]
        assert got[m]["auc"] == want["auc"] and got[m]["genuine_pairs"] == want["genuine_pairs"]
        assert got[m]["impostor_pairs"] == want["impostor_pairs"]
        assert 0.5 < got[m]["auc"] <= 1.0


def test_source_auc_rejects_a_missing_source():
    man, _, embs = fixture()
    with pytest.raises(ValueError):
        sc.source_auc(man, embs, "absent")


# ---- PolyU descriptive statistics ----
def test_descriptive_stats_reuse_subset_stats_on_the_source_fingers_in_both_spaces():
    man, flags, embs = fixture()
    ids = man["identity"].to_numpy(dtype=str)
    res = sc.descriptive_stats(embs["ctrlvit"], ids, man, SPLIT, "polyu_cross", n_imp=500, seed=7, min_images=2)
    assert res["fingers"] == N_POLYU and set(res["spaces"]) == {"raw", "prekey"}
    fingers = np.array(sorted(set(i for i in ids if i.startswith("polyu_cross:"))))
    seeds = sc.descriptive_seeds(7, "polyu_cross")
    want = cs.subset_stats(embs["ctrlvit"], ids, fingers, 500, seeds["impostors"], seeds["one_image_per_finger"],
                           cs.AUC_BINS)
    for k in cs.STATISTICS:
        assert res["spaces"]["raw"][k] == pytest.approx(want[k])
    xc, rec = cs.prekey_space(embs["ctrlvit"], ids, SPLIT)
    want_pk = cs.subset_stats(xc, ids, fingers, 500, seeds["impostors"], seeds["one_image_per_finger"], cs.AUC_BINS)
    for k in cs.STATISTICS:
        assert res["spaces"]["prekey"][k] == pytest.approx(want_pk[k])
    assert res["counts"]["genuine_pairs"] == want["genuine_pairs"] and res["prekey_record"] == rec


def test_descriptive_stats_reject_a_missing_source():
    man, _, embs = fixture()
    ids = man["identity"].to_numpy(dtype=str)
    with pytest.raises(ValueError):
        sc.descriptive_stats(embs["ctrlvit"], ids, man, SPLIT, "absent", 500, 7, 2)


def test_main_reports_descriptive_polyu_statistics_for_every_encoder_and_keeps_the_label_check_block(tmp_path):
    man, flags, embs = fixture()
    rc, out = run_main(tmp_path, man, flags, embs)
    assert rc == 0
    r = json.loads(out.read_text())
    assert set(r["polyu"]["ctrlvit"]) >= {"auc", "genuine_pairs", "mean_genuine"}
    d = r["polyu_descriptive"]
    assert set(d) == {"ctrlvit", "fingervit", "deepprint"}
    for m in d:
        assert d[m]["source"] == "polyu_cross" and d[m]["fingers"] == N_POLYU
        assert set(d[m]["spaces"]["raw"]) >= set(cs.STATISTICS)
    assert r["polyu"]["ctrlvit"]["auc"] == pytest.approx(d["ctrlvit"]["spaces"]["raw"]["auc"], abs=0.01)
    assert "polyu_cross:" not in out.read_text()


# ---- command ----
def run_main(tmp_path, man, flags, embs, extra=(), other_manifest=None):
    ctrl = write_run(tmp_path / "ctrl", man, embs, ["ctrlvit"])
    base = write_run(tmp_path / "base", man if other_manifest is None else other_manifest, embs,
                     ["fingervit", "deepprint"])
    flag_file = tmp_path / "d2_subject_flags.parquet"
    flags.to_parquet(flag_file, index=False)
    out = tmp_path / "out.json"
    rc = sc.main(["--d2-ctrl", str(ctrl), "--d2-base", str(base), "--flags", str(flag_file),
                  "--configs", str(write_configs(tmp_path / "configs")), "--json", str(out), "--label", LABEL,
                  "--subsamples", "4", "--impostor-pairs", "400", "--min-images", "2", *extra])
    return rc, out


def test_main_writes_aggregates_with_provenance_and_no_identity(tmp_path):
    man, flags, embs = fixture()
    rc, out = run_main(tmp_path, man, flags, embs, ["--expect-clean", "3", "--expect-flagged", "9"])
    assert rc == 0
    text = out.read_text()
    r = json.loads(text)
    assert r["data_source"] == LABEL
    assert set(r["encoders"]) == {"ctrlvit", "fingervit", "deepprint"}
    assert r["stratum"]["name"] == "casia_v5" and r["stratum"]["fingers"]["eligible"] == {"clean": 3, "flagged": 9}
    assert set(r["polyu"]) == {"ctrlvit", "fingervit", "deepprint"} and r["polyu"]["ctrlvit"]["source"] == "polyu_cross"
    h = hashlib.sha256((tmp_path / "ctrl" / "emb_ctrlvit.npy").read_bytes()).hexdigest()
    assert r["inputs_sha256"]["d2_ctrl/emb_ctrlvit.npy"] == h
    assert r["provenance"]["script_sha256"] == hashlib.sha256(
        (ROOT / "scripts/data/control_subject_check.py").read_bytes()).hexdigest()
    assert r["settings"]["subsamples"] == 4 and r["settings"]["impostor_pairs"] == 400
    for ident in man["identity"]:
        assert ident not in text
    assert ".bmp" not in text


def test_main_stops_when_the_stratum_size_differs_from_the_expectation(tmp_path):
    man, flags, embs = fixture()
    rc, out = run_main(tmp_path, man, flags, embs, ["--expect-clean", "56", "--expect-flagged", "2181"])
    assert rc == 1 and not out.exists()


def test_main_refuses_to_overwrite_an_existing_output(tmp_path):
    man, flags, embs = fixture()
    (tmp_path / "out.json").write_text("{}")
    rc, out = run_main(tmp_path, man, flags, embs)
    assert rc == 2 and out.read_text() == "{}"


def test_main_refuses_baselines_with_another_manifest(tmp_path):
    man, flags, embs = fixture()
    rc, out = run_main(tmp_path, man, flags, embs, other_manifest=man.assign(path=man["path"] + "x"))
    assert rc == 1 and not out.exists()
