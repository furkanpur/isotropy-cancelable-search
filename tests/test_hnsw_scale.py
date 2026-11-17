import importlib.util
import json
import subprocess
import sys

import pandas as pd
import pytest
import yaml

CFG = {
    "data": {"kind": "synthetic", "n_ids": 200, "per_id": 3, "dim": 32, "decay": 1.0, "seed": 0},
    "split": {"n_dev": 60, "n_tgt": 100, "seed": 1},
    "preconditioner": "fi:16:24",
    "schemes": ["lrp:full", "biohash:half", "iom:64"],
    "distractors": {"kind": "synthetic_gaussian", "dim": 32, "sizes": [200, 500], "seed": 2, "near_dup_cos": 0.95},
    "hnsw": {"m": 16, "ef_construction": 100, "ef_search": [64]},
}


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
def test_scale_run_outputs_and_recall(tmp_path):
    # faiss runs in a subprocess: on macOS its bundled libomp crashes when torch's libomp is
    # already loaded in the same pytest process (tests/test_btp_pipeline.py trains a torch model).
    cfg_path = tmp_path / "scale.yaml"
    cfg_path.write_text(yaml.safe_dump(CFG))
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "ijiss_btp.experiments.hnsw_scale",
                    "--config", str(cfg_path), "--out", str(out)], check=True)
    df = pd.read_csv(out / "scale.csv")
    assert len(df) == 3 * 2 * 1
    assert (df[df.scheme == "lrp_full"].recall10_vs_exact >= 0.8).all()
    assert (df.index_serialized_mb > 0).all() and (df.latency_ms_per_query > 0).all()
    assert {"n_distractors_requested", "recall10_tie_aware"} <= set(df.columns)
    assert (df.recall10_tie_aware >= df.recall10_vs_exact).all()
    assert "index_mb" not in df.columns
    meta = json.loads((out / "metadata.json").read_text())
    assert isinstance(meta["config"], dict) and meta["config"]["distractors"]["near_dup_cos"] == 0.95
    for k in ("started_utc", "git_sha", "image_freeze", "python", "numpy", "faiss", "cpu_model", "omp_num_threads",
              "data_files", "distractors_available_before", "distractors_removed", "distractors_available_after"):
        assert k in meta
    assert meta["distractors_removed"] == 0
    assert (meta["distractors_available_before"] - meta["distractors_removed"]
            == meta["distractors_available_after"])
    # every requested size is available: one row per requested size, requested = used (rows as before)
    assert meta["distractor_sizes_requested_to_used"] == {"200": 200, "500": 500}
    assert set(zip(df.n_distractors, df.n_distractors_requested)) == {(200, 200), (500, 500)}


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
def test_scale_run_removes_exact_copies_of_primary(tmp_path):
    # distractors from the same synthetic generator and seed as the primary: every row is an exact copy
    cfg = {**CFG, "schemes": ["lrp:full"],
           "distractors": {"kind": "synthetic", "n_ids": 200, "per_id": 3, "dim": 32, "decay": 1.0, "seed": 0,
                           "sizes": [100, 300], "near_dup_cos": 0.95}}
    cfg_path = tmp_path / "scale.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "ijiss_btp.experiments.hnsw_scale",
                        "--config", str(cfg_path), "--out", str(out)], capture_output=True, text=True)
    assert (out / "metadata.json").exists(), r.stderr[-2000:]
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["distractors_removed"] > 0
    assert meta["distractors_available_after"] == meta["distractors_available_before"] - meta["distractors_removed"]
