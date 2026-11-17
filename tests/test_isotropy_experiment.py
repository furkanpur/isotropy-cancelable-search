# tests/test_isotropy_experiment.py
import copy
import json

import pandas as pd

from ijiss_btp.experiments.isotropy import run, stable_seed

SMOKE = {
    "name": "smoke",
    "data": {"kind": "synthetic", "n_ids": 260, "per_id": 3, "dim": 24, "decay": 1.0, "seed": 0},
    "split": {"n_dev": 80, "n_tgt": 120, "min_tgt_images": 3, "seeds": [1]},
    "keys_per_seed": 1,
    "fmr": [0.01],
    "preconditioners": ["none", "fi:12:20"],
    "schemes": ["lrp:full", "biohash:full", "iom:64"],
    "attacks": {
        "anchor": {"r_grid": [5, 20], "variants": ["exact", "fresh"], "decoders": ["ridge", "proj", "mlp"],
                   "min_targets": 50, "mlp_subset": {"pres": ["none"], "schemes": ["lrp:full"], "r": [20]}},
        "zeropair": {"p_grid": [4, 8], "methods": ["eigen", "ica", "wp"], "ica_max_dim": 24, "expensive_keys": 1},
        "postcompromise": {"enabled": True, "keys": 1},
    },
    "bootstrap": {"n_boot": 100},
}


def test_smoke_run_writes_expected_tables(tmp_path):
    run(copy.deepcopy(SMOKE), tmp_path)
    for name in ("utility", "anchor", "zeropair", "postcompromise"):
        assert (tmp_path / f"{name}.csv").exists()
    assert len(pd.read_csv(tmp_path / "utility.csv")) == 2 * 3
    anchor = pd.read_csv(tmp_path / "anchor.csv")
    assert {"e1_rel", "e1_rel_lo", "e1_rel_hi", "e1_cp_upper", "e2_rel", "variant", "decoder", "r"} <= set(anchor.columns)
    assert set(anchor["decoder"]) == {"ridge", "proj", "mlp"}
    assert "e1_pair_trials" in anchor.columns
    assert {"eer", "tmr", "fmr_actual", "rank1"} <= set(pd.read_csv(tmp_path / "utility.csv").columns)
    assert {"aux", "method", "p", "e1_rel", "e1_cp_upper"} <= set(pd.read_csv(tmp_path / "zeropair.csv").columns)
    assert {"method", "in_system_accept", "e1_rel", "e2_rel"} <= set(pd.read_csv(tmp_path / "postcompromise.csv").columns)
    assert (anchor.groupby(["seed", "key", "r"])["n_targets"].nunique() == 1).all()
    assert set(pd.read_csv(tmp_path / "zeropair.csv")["method"]) >= {"eigen", "eigen_oracle", "wp", "ica"}
    pc = pd.read_csv(tmp_path / "postcompromise.csv")
    assert set(pc["method"]) == {"analytic", "learned"}
    assert (pc[pc.method == "analytic"]["in_system_accept"] > 0.9).all()
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["config"]["name"] == "smoke" and meta["runtime_s"] > 0
    assert {"omp_num_threads", "cpu_model", "aux_sizes", "shifted_aux_removed"} <= set(meta)
    assert isinstance(meta["cpu_model"], str) and meta["cpu_model"]
    assert meta["aux_sizes"]["same_dist"] > 0
    assert meta["shifted_aux_removed"] is None
    spectra = json.loads((tmp_path / "spectra.json").read_text())
    assert "raw_centered_normalized" in spectra and "fi12_p20" in spectra["preconditioned"]


def test_smoke_run_is_deterministic(tmp_path):
    run(copy.deepcopy(SMOKE), tmp_path / "a")
    run(copy.deepcopy(SMOKE), tmp_path / "b")
    for name in ("anchor", "zeropair", "utility", "postcompromise"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a" / f"{name}.csv"),
                                      pd.read_csv(tmp_path / "b" / f"{name}.csv"), check_exact=False, rtol=1e-9)
    assert json.loads((tmp_path / "a" / "spectra.json").read_text()) == json.loads((tmp_path / "b" / "spectra.json").read_text())


def test_smoke_run_writes_only_aggregates(tmp_path):
    run(copy.deepcopy(SMOKE), tmp_path)
    assert {f.name for f in tmp_path.iterdir()} == {"utility.csv", "anchor.csv", "zeropair.csv",
                                                    "postcompromise.csv", "spectra.json", "metadata.json"}


def test_stable_seed_is_stable():
    assert stable_seed("x", 1) == stable_seed("x", 1) != stable_seed("x", 2)


def test_rec_cos_z_recorded(tmp_path):
    import numpy as np

    run(copy.deepcopy(SMOKE), tmp_path)
    anchor = pd.read_csv(tmp_path / "anchor.csv")
    zero = pd.read_csv(tmp_path / "zeropair.csv")
    for df in (anchor, zero):
        v = df["rec_cos_z"].to_numpy(dtype=float)
        assert np.isfinite(v).all() and (np.abs(v) <= 1 + 1e-9).all()
    none = anchor[anchor.pre == "none"]
    assert len(none) > 0
    np.testing.assert_allclose(none["rec_cos_z"], none["rec_cos"], atol=1e-9)


def test_sign_agnostic_cosines_recorded(tmp_path):
    import numpy as np

    run(copy.deepcopy(SMOKE), tmp_path)
    for name in ("anchor", "zeropair"):
        df = pd.read_csv(tmp_path / f"{name}.csv")
        a, c2, c = (df[k].to_numpy(dtype=float) for k in ("rec_abs_cos_z", "rec_cos2_z", "rec_cos_z"))
        assert np.isfinite(a).all() and np.isfinite(c2).all()
        assert (a >= np.abs(c) - 1e-12).all()
        assert ((c2 >= 0) & (c2 <= 1 + 1e-12)).all()
    assert (pd.read_csv(tmp_path / "anchor.csv")["decoder"] == "proj").any()


def test_shifted_aux_metadata(tmp_path):
    cfg = copy.deepcopy(SMOKE)
    cfg["data"]["shifted_aux"] = {"kind": "synthetic", "n_ids": 40, "per_id": 2, "dim": 24, "decay": 1.0, "seed": 0,
                                  "near_dup_cos": 0.95}
    run(cfg, tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert isinstance(meta["shifted_aux_removed"], int) and meta["shifted_aux_removed"] >= 0
    assert meta["aux_sizes"]["shifted"] == 80 - meta["shifted_aux_removed"]


def test_fmr1_configs_match_base():
    from pathlib import Path

    import yaml

    cdir = Path(__file__).resolve().parents[1] / "configs"
    found = sorted(cdir.glob("isotropy_*_fmr1.yaml"))
    assert len(found) == 5
    for f in found:
        base = yaml.safe_load((cdir / f.name.replace("_fmr1", "")).read_text())
        cfg = yaml.safe_load(f.read_text())
        assert cfg["name"] == base["name"] + "_fmr1" and cfg["fmr"] == [0.01, 0.001]
        for d in (base, cfg):
            d.pop("name"), d.pop("fmr")
        assert cfg == base
        assert "proj" in cfg["attacks"]["anchor"]["decoders"]


def test_configs_use_the_relabelled_first_corpus_and_an_identity_disjoint_secondary_corpus():
    from pathlib import Path

    import yaml

    cdir = Path(__file__).resolve().parents[1] / "configs"
    d1 = sorted(cdir.glob("isotropy_d1_*.yaml"))
    assert len(d1) == 4
    for f in d1:
        data = yaml.safe_load(f.read_text())["data"]
        assert data["relabel"] == "/data/relabel/d1_relabel.csv" and data.get("relabel_key", "basename") == "basename"
        sh = data["shifted_aux"]
        assert sh["relabel"] == "/data/relabel/secondary_relabel.csv" and sh["relabel_key"] == "path"
        assert sh["exclude_primary_identities"] is True and sh["model"] == data["model"] and sh["near_dup_cos"] == 0.95
    scale = sorted(cdir.glob("scale_d1_*.yaml"))
    assert len(scale) == 2
    for f in scale:
        cfg = yaml.safe_load(f.read_text())
        assert cfg["data"]["relabel"] == "/data/relabel/d1_relabel.csv"
        dis = cfg["distractors"]
        if dis["kind"] == "sin":
            assert dis["relabel"] == "/data/relabel/secondary_relabel.csv" and dis["relabel_key"] == "path"
            assert dis["exclude_primary_identities"] is True
            # every size below the rows left after the exclusion, so requested = used
            assert dis["sizes"] == [1000, 2000, 3000]
    d2 = sorted(cdir.glob("isotropy_d2_*.yaml"))
    assert len(d2) == 6
    for f in d2:
        data = yaml.safe_load(f.read_text())["data"]
        assert data == {"kind": "sin", "root": "/data/corpus_d2", "model": data["model"]}
        assert data["model"] in ("fingervit", "deepprint", "ctrlvit") and data["model"] in f.name
