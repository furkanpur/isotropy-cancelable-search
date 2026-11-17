"""Joint stage of the isotropy experiment, its run configurations and the parity checker (synthetic data only)."""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ijiss_btp.experiments.isotropy import (
    JOINT_EXTRA_R,
    joint_budgets,
    joint_preconditioners,
    load_config,
    main,
    run,
    valid_budgets,
)
from ijiss_btp.precondition import make_preconditioner

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CHECKER = ROOT / "scripts" / "experiments" / "check_joint_parity.py"
WRAPPERS = sorted(CONFIGS.glob("joint_*.yaml"))

# A small synthetic configuration (never a result): two seeds and keys, a whitening pre-conditioner, an FI whose
# dimension caps the p grid, one pre-conditioner outside the joint set and a shifted AUX pool.
SYN = {
    "name": "joint_synthetic",
    "data": {"kind": "synthetic", "n_ids": 260, "per_id": 3, "dim": 24, "decay": 1.0, "seed": 0,
             "shifted_aux": {"kind": "synthetic", "n_ids": 40, "per_id": 2, "dim": 24, "decay": 0.5, "seed": 5,
                             "near_dup_cos": 0.95}},
    "split": {"n_dev": 80, "n_tgt": 120, "min_tgt_images": 3, "seeds": [1, 2]},
    "keys_per_seed": 2,
    "fmr": [0.01, 0.001],
    "preconditioners": ["none", "pca:16:1", "pca:16:0.5", "fi:12:20"],
    "schemes": ["lrp:full", "iom:64"],
    "attacks": {
        "anchor": {"r_grid": [5, 20], "variants": ["exact", "fresh"], "decoders": ["ridge", "proj"],
                   "min_targets": 50},
        "zeropair": {"p_grid": [4, 8, 16], "methods": ["eigen"], "ica_max_dim": 24, "expensive_keys": 1},
        "postcompromise": {"enabled": False},
    },
    "bootstrap": {"n_boot": 50},
}
JOINT_PRE_NAMES = ["none", "pca16_a1", "fi12_p20"]
ENDPOINT = ["genuine", "proxy", "rel", "rel_lo", "rel_hi", "successes", "trials", "cp_upper", "pair_successes",
            "pair_trials"]
JOINT_COLUMNS = (["seed", "key", "pre", "scheme", "m", "k", "method", "aux", "p", "r", "n_targets", "rec_cos_z",
                  "rec_abs_cos_z", "rec_cos2_z", "rec_cos"]
                 + [f"e1_{c}" for c in ENDPOINT] + [f"e2_{c}" for c in ENDPOINT])
KEYS = ["seed", "key", "pre", "scheme", "r"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The synthetic configuration run once with the existing stages and once with the joint stage only."""
    base = tmp_path_factory.mktemp("joint_stage")
    run(copy.deepcopy(SYN), base / "isotropy")
    run(copy.deepcopy(SYN), base / "joint", stages=["joint"])
    return base / "isotropy", base / "joint"


def test_joint_stage_reproduces_the_anchor_stage_colluders_victims_and_thresholds(runs):
    iso, joint_dir = runs
    anchor = pd.read_csv(iso / "anchor.csv")
    joint = pd.read_csv(joint_dir / "joint.csv")
    a = anchor[(anchor.variant == "exact") & (anchor.decoder == "proj") & (anchor.scheme == "lrp_full")
               & anchor.pre.isin(JOINT_PRE_NAMES)].drop(columns=["variant", "decoder"])
    j = joint[(joint.method == "proj") & joint.r.isin(SYN["attacks"]["anchor"]["r_grid"])]
    assert len(a) == 2 * 2 * 3 * 2
    got = j.drop(columns=["method", "aux", "p"]).sort_values(KEYS).reset_index(drop=True)
    want = a.sort_values(KEYS).reset_index(drop=True)[got.columns]
    # the same colluders (cosine), victims (n_targets, trials) and thresholds (genuine acceptance), bit for bit
    pd.testing.assert_frame_equal(got, want, check_exact=True)


def test_joint_stage_grid_victims_and_methods(runs):
    _, joint_dir = runs
    joint = pd.read_csv(joint_dir / "joint.csv")
    assert list(joint.columns) == JOINT_COLUMNS
    assert set(joint.scheme) == {"lrp_full"} and sorted(set(joint.pre)) == sorted(JOINT_PRE_NAMES)
    assert set(joint.method) == {"proj", "align_anchor", "joint", "eigen_oracle"}
    assert (joint.n_targets == 120 - 20).all() and (joint.e1_trials == joint.n_targets).all()
    budgets = [1, 2, 5, 20]
    for method in ("proj", "align_anchor", "joint"):
        assert sorted(set(joint[joint.method == method].r)) == budgets
    assert set(joint[joint.method == "eigen_oracle"].r) == {0}
    proj = joint[joint.method == "proj"]
    assert proj.aux.isna().all() and (proj.p == 0).all()
    other = joint[joint.method != "proj"]
    assert set(other.aux) == {"same_dist", "shifted"}
    p_by_pre = other.groupby("pre").p.apply(lambda s: sorted(set(s))).to_dict()
    assert p_by_pre == {"none": [4, 8, 16], "pca16_a1": [4, 8, 16], "fi12_p20": [4, 8, 12]}   # capped at m = 12
    n_p = {"none": 3, "pca16_a1": 3, "fi12_p20": 3}
    per_block = sum(2 * n_p[p] * (1 + 2 * len(budgets)) + len(budgets) for p in n_p)
    assert len(joint) == 2 * 2 * per_block
    assert not joint.duplicated(["seed", "key", "pre", "method", "aux", "p", "r"]).any()
    # with r >= m anchors the span is the whole pre-key space, so the joint decoder is the projection decoder
    full = joint[(joint.pre == "fi12_p20") & (joint.r == 20)]
    for _, g in full.groupby(["seed", "key"]):
        ref = g[g.method == "proj"].rec_cos_z.iloc[0]
        np.testing.assert_allclose(g[g.method == "joint"].rec_cos_z, ref, atol=1e-12)


def test_joint_run_writes_only_its_table_and_metadata(runs):
    _, joint_dir = runs
    assert {f.name for f in joint_dir.iterdir()} == {"joint.csv", "metadata.json"}
    meta = json.loads((joint_dir / "metadata.json").read_text())
    assert meta["stages"] == ["joint"] and meta["config"] == SYN
    assert meta["data_source"].startswith("synthetic") and "git_sha" in meta and meta["runtime_s"] > 0
    assert meta["joint"] == {"scheme": "lrp:full", "preconditioners": ["none", "pca:16:1", "fi:12:20"],
                             "anchors": "exact", "extra_r": [1, 2],
                             "methods": ["proj", "align_anchor", "joint", "eigen_oracle"], "attack_fmr": 0.01}
    assert set(meta["split_counts"]) == {"1", "2"} and set(meta["aux_sizes"]) == {"same_dist", "shifted"}


def test_existing_stages_record_their_selection_and_write_no_joint_table(runs):
    iso, _ = runs
    assert {f.name for f in iso.iterdir()} == {"utility.csv", "anchor.csv", "zeropair.csv", "postcompromise.csv",
                                               "spectra.json", "metadata.json"}
    meta = json.loads((iso / "metadata.json").read_text())
    assert meta["stages"] == ["utility", "anchor", "zeropair", "postcompromise"] and "joint" not in meta


def test_a_run_never_overwrites_an_existing_run_directory(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run(copy.deepcopy(SYN), tmp_path, stages=["joint"])
    assert {f.name for f in tmp_path.iterdir()} == {"metadata.json"}
    assert (tmp_path / "metadata.json").read_text() == "{}"


def test_stage_selection_from_the_command_line(tmp_path):
    cfg = tmp_path / "syn.yaml"
    cfg.write_text(yaml.safe_dump(SYN))
    main(["--config", str(cfg), "--out", str(tmp_path / "out"), "--stages", "joint"])
    assert {f.name for f in (tmp_path / "out").iterdir()} == {"joint.csv", "metadata.json"}
    with pytest.raises(ValueError, match="run together"):
        main(["--config", str(cfg), "--out", str(tmp_path / "partial"), "--stages", "anchor", "joint"])
    with pytest.raises(SystemExit):
        main(["--config", str(cfg), "--out", str(tmp_path / "bad"), "--stages", "nonsense"])


def test_joint_budgets_add_one_and_two_and_keep_the_largest_budget():
    assert JOINT_EXTRA_R == (1, 2)
    assert joint_budgets([5, 10, 640]) == [1, 2, 5, 10, 640]
    assert joint_budgets([2, 5]) == [1, 2, 5]
    with pytest.raises(ValueError, match="victims"):
        joint_budgets([1])   # r = 2 would exceed the largest budget and move a victim into the colluders
    with pytest.raises(ValueError, match="victims"):
        joint_budgets([])


def test_joint_preconditioners_budgets_and_scheme_on_every_committed_configuration():
    found = sorted(CONFIGS.glob("isotropy_d*.yaml"))
    assert len(found) == 10
    for f in found:
        cfg = yaml.safe_load(f.read_text())
        names = [make_preconditioner(s).name for s in joint_preconditioners(cfg["preconditioners"])]
        p0 = "p384" if "deepprint" in f.name else "p512"
        assert names == ["none", "pca128_a1", "pca256_a1", f"fi64_{p0}", f"fi128_{p0}", f"fi256_{p0}"], f.name
        assert "lrp:full" in cfg["schemes"]
        acfg = cfg["attacks"]["anchor"]
        valid = valid_budgets(acfg, cfg["split"]["n_tgt"])
        assert valid == acfg["r_grid"] and joint_budgets(valid) == [1, 2] + acfg["r_grid"]
    smoke = yaml.safe_load((CONFIGS / "isotropy_smoke.yaml").read_text())
    assert joint_preconditioners(smoke["preconditioners"]) == ["none", "fi:12:20"]


@pytest.mark.skipif(not WRAPPERS, reason="the joint run configurations are not part of this copy")
def test_wrapper_configurations_add_only_the_stage_to_the_committed_configurations(tmp_path):
    committed = sorted(f.name for f in CONFIGS.glob("isotropy_d*.yaml"))
    assert sorted(f.name.replace("joint_", "isotropy_", 1) for f in WRAPPERS) == committed
    for w in WRAPPERS:
        base_name = w.name.replace("joint_", "isotropy_", 1)
        assert yaml.safe_load(w.read_text()) == {"base": base_name, "stages": ["joint"]}
        assert load_config(w) == {**yaml.safe_load((CONFIGS / base_name).read_text()), "stages": ["joint"]}
    plain = CONFIGS / "isotropy_smoke.yaml"
    assert load_config(plain) == yaml.safe_load(plain.read_text())
    bad = tmp_path / "joint_bad.yaml"
    bad.write_text(yaml.safe_dump({"base": str(plain), "stages": ["joint"], "fmr": [0.5]}))
    with pytest.raises(ValueError, match="only"):
        load_config(bad)


def _checker():
    spec = importlib.util.spec_from_file_location("check_joint_parity", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mutated(src: Path, dst: Path, fn) -> Path:
    """Copy of a run directory whose joint.csv is replaced by fn(joint table)."""
    shutil.copytree(src, dst)
    fn(pd.read_csv(dst / "joint.csv")).to_csv(dst / "joint.csv", index=False)
    return dst


@pytest.mark.skipif(not CHECKER.exists(), reason="the parity checker is not part of this copy")
def test_parity_checker_accepts_the_reproducing_run_and_rejects_mismatches(runs, tmp_path):
    iso, joint_dir = runs
    chk = _checker()
    assert chk.check(joint_dir, iso, atol=1e-9) == []
    proc = subprocess.run([sys.executable, str(CHECKER), str(joint_dir), str(iso)], capture_output=True, text=True)
    assert proc.returncode == 0 and "parity: OK" in proc.stdout, proc.stdout + proc.stderr

    def first_proj(df, r=5):
        return df.index[(df.method == "proj") & (df.r == r)][0]

    def shifted(column, delta, r=5):
        def fn(df):
            df.loc[first_proj(df, r), column] += delta
            return df
        return fn

    cases = {"e1_successes": shifted("e1_successes", 1), "rec_cos_z": shifted("rec_cos_z", 1e-6, r=20),
             "e2_rel": shifted("e2_rel", 1e-6), "missing": lambda df: df.drop(index=first_proj(df))}
    for name, fn in cases.items():
        bad = _mutated(joint_dir, tmp_path / name, fn)
        problems = chk.check(bad, iso, atol=1e-9)
        assert problems and any(name in p for p in problems), (name, problems)
    proc = subprocess.run([sys.executable, str(CHECKER), str(tmp_path / "e1_successes"), str(iso)],
                          capture_output=True, text=True)
    assert proc.returncode == 1 and "parity: FAILED" in proc.stdout and "e1_successes" in proc.stdout

    def change_uncompared(df):
        df.loc[df.r == 1, "e1_rel"] += 0.5
        df.loc[df.method == "joint", "rec_cos_z"] += 0.1
        return df

    # rows of the extra budgets (r = 1, 2) and of the other methods are not compared
    assert chk.check(_mutated(joint_dir, tmp_path / "uncompared", change_uncompared), iso, atol=1e-9) == []
    other = tmp_path / "other_config"
    shutil.copytree(joint_dir, other)
    meta = json.loads((other / "metadata.json").read_text())
    meta["config"]["fmr"] = [0.001, 0.01]
    (other / "metadata.json").write_text(json.dumps(meta))
    assert any("config" in p for p in chk.check(other, iso, atol=1e-9))


def _with_metadata(src: Path, dst: Path, fn) -> Path:
    """Copy of a run directory whose metadata.json is changed in place by fn."""
    shutil.copytree(src, dst)
    meta = json.loads((dst / "metadata.json").read_text())
    fn(meta)
    (dst / "metadata.json").write_text(json.dumps(meta))
    return dst


@pytest.mark.skipif(not CHECKER.exists(), reason="the parity checker is not part of this copy")
def test_parity_checker_compares_every_field_that_fixes_the_data(runs, tmp_path):
    iso, joint_dir = runs
    chk = _checker()
    changes = {
        "data_source": lambda m: m.update(data_source=m["data_source"] + ":other"),
        "data_files": lambda m: m.update(data_files={"emb.npy": "0" * 64}),
        "corpus_counts": lambda m: m["corpus_counts"].update(images=m["corpus_counts"]["images"] + 1),
        "split_counts": lambda m: m["split_counts"]["2"].update(aux_images=m["split_counts"]["2"]["aux_images"] + 1),
        "aux_sizes": lambda m: m["aux_sizes"].update(shifted=m["aux_sizes"]["shifted"] - 1),
        "shifted_aux_source": lambda m: m.update(shifted_aux_source=m["shifted_aux_source"] + ":other"),
        "shifted_aux_removed": lambda m: m.update(shifted_aux_removed=m["shifted_aux_removed"] + 1),
        "secondary_removed_shared_identity": lambda m: m.update(secondary_removed_shared_identity=3),
    }
    for field, fn in changes.items():
        bad = _with_metadata(joint_dir, tmp_path / field, fn)
        problems = chk.check(bad, iso, atol=1e-9)
        assert len(problems) == 1 and problems[0].startswith(f"{field} differs"), (field, problems)
    proc = subprocess.run([sys.executable, str(CHECKER), str(tmp_path / "aux_sizes"), str(iso)], capture_output=True,
                          text=True)
    assert proc.returncode == 1 and "parity: FAILED" in proc.stdout and "aux_sizes differs" in proc.stdout


@pytest.mark.skipif(not CHECKER.exists(), reason="the parity checker is not part of this copy")
def test_parity_checker_rejects_a_defective_committed_side(runs, tmp_path):
    iso, joint_dir = runs
    chk = _checker()

    def committed(name, fn):
        dst = tmp_path / name
        shutil.copytree(iso, dst)
        fn(pd.read_csv(dst / "anchor.csv")).to_csv(dst / "anchor.csv", index=False)
        return dst

    def first_proj(df):
        return df.index[(df.variant == "exact") & (df.decoder == "proj") & (df.scheme == "lrp_full")
                        & (df.pre == "none")][0]

    def threshold_moved(df):
        df.loc[first_proj(df), "e2_genuine"] += 0.01
        return df

    cases = {"missing in the committed anchor.csv": lambda df: df.drop(index=first_proj(df)),
             "duplicate rows in the committed anchor.csv": lambda df: pd.concat([df, df.loc[[first_proj(df)]]]),
             "e2_genuine": threshold_moved}
    for text, fn in cases.items():
        problems = chk.check(joint_dir, committed(text.split()[0], fn), atol=1e-9)
        assert any(text in p for p in problems), (text, problems)


@pytest.mark.skipif(not CHECKER.exists(), reason="the parity checker is not part of this copy")
def test_parity_checker_reports_the_environment_without_failing_on_it(runs, tmp_path):
    iso, joint_dir = runs
    chk = _checker()
    problems, summary = chk.parity(joint_dir, iso)
    assert problems == [] and summary["environment"] == []
    other = _with_metadata(joint_dir, tmp_path / "env", lambda m: m.update(numpy="0.0.0", omp_num_threads="64"))
    problems, summary = chk.parity(other, iso)
    assert problems == [] and summary["environment"] == ["numpy", "omp_num_threads"]
    proc = subprocess.run([sys.executable, str(CHECKER), str(other), str(iso)], capture_output=True, text=True)
    assert proc.returncode == 0 and "parity: OK" in proc.stdout
    assert "environment differs (informational): numpy, omp_num_threads" in proc.stdout
    proc = subprocess.run([sys.executable, str(CHECKER), str(joint_dir), str(iso)], capture_output=True, text=True)
    assert "environment: same image freeze, Python, NumPy, threads and CPU" in proc.stdout


def test_a_command_line_selection_must_not_contradict_the_configuration(tmp_path):
    cfg = {**copy.deepcopy(SYN), "stages": ["joint"]}
    with pytest.raises(ValueError, match="contradicts"):
        run(copy.deepcopy(cfg), tmp_path / "core", stages=["utility", "anchor", "zeropair", "postcompromise"])
    assert not (tmp_path / "core").exists()
    run(copy.deepcopy(cfg), tmp_path / "joint", stages=["joint"])
    assert {f.name for f in (tmp_path / "joint").iterdir()} == {"joint.csv", "metadata.json"}
