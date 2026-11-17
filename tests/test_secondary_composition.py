# tests/test_secondary_composition.py
"""Secondary-corpus composition after the relabel and after removing the fingers shared with the first corpus."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/data"))
import secondary_composition as sc  # noqa: E402


def test_labels_families_and_sources():
    assert sc.dataset_label("fvc2000/DB1") == "FVC2000 DB1"
    assert (sc.family_of("fvc2004/DB4"), sc.source_of("fvc2004/DB4")) == ("FVC2004", "FVC")
    assert (sc.dataset_label("sd301a"), sc.family_of("sd301a"), sc.source_of("sd301a")) == ("SD301a", "SD301a", "SD301")
    assert (sc.dataset_label("sd302d"), sc.family_of("sd302d"), sc.source_of("sd302d")) == ("SD302d", "SD302d", "SD302")
    assert (sc.dataset_label("other"), sc.family_of("other"), sc.source_of("other")) == ("other", "other", "other")


# rows: FVC2000 DB1 (two rows of a shared finger, one kept), FVC2000 DB4 (outside every path-map entry), SD302b and
# SD302d holding the same two fingers (one shared, one kept), one SD302d row without a unique identity, one SD301a row
DATASET = ["fvc2000/DB1"] * 3 + ["fvc2000/DB4"] + ["sd302b"] * 2 + ["sd302d"] * 3 + ["sd301a"]
ENTRY = [0, 0, 0, -1, 1, 1, 1, 1, 1, 2]
SEC_ID = ["f:1", "f:1", "f:2", "", "s:1", "s:2", "s:1", "s:2", "", "t:1"]
D1_IDS = ["f:1", "s:1", "x:9"]
COUNT_KEYS = ("rows", "dropped_by_relabel", "dropped_outside_path_map", "dropped_inside_path_map",
              "rows_after_relabel", "fingers_after_relabel", "rows_shared", "fingers_shared", "rows_kept",
              "fingers_kept")


def _counts(entry):
    return tuple(entry[k] for k in COUNT_KEYS)


def test_composition_per_dataset_family_source_and_union_totals():
    out = sc.composition(DATASET, ENTRY, SEC_ID, D1_IDS)
    by_ds = {d["label"]: d for d in out["by_dataset"]}
    assert list(by_ds) == ["FVC2000 DB1", "FVC2000 DB4", "SD301a", "SD302b", "SD302d"]
    assert _counts(by_ds["FVC2000 DB1"]) == (3, 0, 0, 0, 3, 2, 2, 1, 1, 1)
    assert _counts(by_ds["FVC2000 DB4"]) == (1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
    assert _counts(by_ds["SD302b"]) == (2, 0, 0, 0, 2, 2, 1, 1, 1, 1)
    assert _counts(by_ds["SD302d"]) == (3, 1, 0, 1, 2, 2, 1, 1, 1, 1)
    assert _counts(by_ds["SD301a"]) == (1, 0, 0, 0, 1, 1, 0, 0, 1, 1)
    assert by_ds["FVC2000 DB1"]["family"] == "FVC2000" and by_ds["SD302d"]["source"] == "SD302"
    fam = {f["label"]: f for f in out["by_family"]}
    assert list(fam) == ["FVC2000", "SD301a", "SD302b", "SD302d"]
    assert _counts(fam["FVC2000"]) == (4, 1, 1, 0, 3, 2, 2, 1, 1, 1)
    assert fam["FVC2000"]["datasets"] == ["FVC2000 DB1", "FVC2000 DB4"]
    src = {s["label"]: s for s in out["by_source"]}
    assert list(src) == ["FVC", "SD301", "SD302"]
    assert _counts(src["SD302"]) == (5, 1, 0, 1, 4, 2, 2, 1, 2, 1)   # fingers counted once over SD302b and SD302d
    assert src["SD302"]["families"] == ["SD302b", "SD302d"]
    assert _counts(out["totals"]) == (10, 2, 1, 1, 8, 5, 4, 2, 4, 3)
    # per-family finger counts do not add up: SD302b and SD302d share their fingers
    assert sum(f["fingers_kept"] for f in out["by_family"]) == 4 > out["totals"]["fingers_kept"]
    text = json.dumps(out)
    assert not any(i in text for i in ("f:1", "f:2", "s:1", "s:2", "t:1", "x:9"))


def test_composition_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="rows"):
        sc.composition(DATASET[:-1], ENTRY, SEC_ID, D1_IDS)


ISO_META = {"config": {"name": "iso"}, "git_sha": "abc", "started_utc": "2026-01-01T00:00:00",
            "secondary_removed_shared_identity": 4, "aux_sizes": {"same_dist": 9, "shifted": 3},
            "shifted_aux_removed": 1,
            "shifted_aux_source": "sin:secondary:m:relabel=s.csv:key=path:dropped=2:shared_ids_removed=4:dedup<0.95"}
SCALE_META = {"config": {"name": "sc"}, "git_sha": "abc", "started_utc": "2026-01-01T00:00:00",
              "secondary_removed_shared_identity": 4, "distractors_loaded": 8,
              "distractor_source": "sin:secondary:m:relabel=s.csv:key=path:dropped=2:shared_ids_removed=4:dedup<0.95"}
SUMMARY = {"matched": 8, "true_fingers": 5, "unmatched": 1, "ambiguous": 1, "paths_unmapped": 1}


def test_run_metadata_counts_for_isotropy_and_scale_runs():
    iso = sc.run_metadata_counts(ISO_META)
    assert (iso["kind"], iso["shared_rows"], iso["kept_rows_before_near_duplicate_filter"], iso["relabel_dropped"]) \
        == ("isotropy", 4, 4, 2)
    assert iso["run"] == {"config": "iso", "git_sha": "abc", "started_utc": "2026-01-01T00:00:00"}
    scale = sc.run_metadata_counts(SCALE_META)
    assert (scale["kind"], scale["shared_rows"], scale["kept_rows_before_near_duplicate_filter"]) == ("scale", 4, 4)
    for bad in ({**ISO_META, "secondary_removed_shared_identity": None},
                {k: v for k, v in ISO_META.items() if k != "secondary_removed_shared_identity"},
                {"secondary_removed_shared_identity": 4}):
        with pytest.raises(ValueError, match="metadata"):
            sc.run_metadata_counts(bad)


def test_checks_pass_on_matching_totals_and_stop_on_any_mismatch():
    totals = sc.composition(DATASET, ENTRY, SEC_ID, D1_IDS)["totals"]
    res = sc.check(totals, sc.run_metadata_counts(ISO_META), SUMMARY)
    assert all(v["match"] for v in res.values())
    assert res["shared_rows_vs_run_metadata"] == {"computed": 4, "reference": 4, "match": True}
    assert res["fingers_after_relabel_vs_map_summary"] == {"computed": 5, "reference": 5, "match": True}
    assert set(sc.check(totals, sc.run_metadata_counts(SCALE_META))) == {
        "shared_rows_vs_run_metadata", "kept_rows_vs_run_metadata", "relabel_dropped_vs_run_metadata"}
    with pytest.raises(ValueError, match="shared_rows_vs_run_metadata"):
        sc.check(totals, sc.run_metadata_counts({**ISO_META, "secondary_removed_shared_identity": 5}))
    with pytest.raises(ValueError, match="kept_rows_vs_run_metadata"):
        sc.check(totals, sc.run_metadata_counts({**ISO_META, "shifted_aux_removed": 2}))
    with pytest.raises(ValueError, match="fingers_after_relabel_vs_map_summary"):
        sc.check(totals, sc.run_metadata_counts(ISO_META), {**SUMMARY, "true_fingers": 6})


def _write_inputs(tmp_path, meta):
    paths = ["/hostdata/F0/DB1/1.tif", "/hostdata/F0/DB1/2.tif", "/hostdata/F0/DB1/3.tif", "/hostdata/F0/DB4/1.tif",
             "/hostdata/N2/b/1.png", "/hostdata/N2/b/2.png", "/hostdata/N2/d/1.png", "/hostdata/N2/d/2.png",
             "/hostdata/N2/d/3.png", "/hostdata/N1/a/1.png"]
    pd.DataFrame({"path": paths, "dataset": DATASET, "identity": ["origlabel"] * len(paths)}).to_parquet(
        tmp_path / "manifest.parquet")
    pd.DataFrame({"file": [p for p, i in zip(paths, SEC_ID) if i], "identity": [i for i in SEC_ID if i]}).to_csv(
        tmp_path / "sec_map.csv", index=False)
    pd.DataFrame({"file": ["a.png", "b.png", "c.png"], "identity": D1_IDS}).to_csv(tmp_path / "d1_map.csv", index=False)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "summary.json").write_text(json.dumps(SUMMARY))
    return ["--sec-manifest", str(tmp_path / "manifest.parquet"), "--sec-map", str(tmp_path / "sec_map.csv"),
            "--d1-map", str(tmp_path / "d1_map.csv"), "--path-map", "/hostdata/F0/DB1=/c/f",
            "--path-map", "/hostdata/N2=/c/n2", "--path-map", "/hostdata/N1=/c/n1",
            "--run-metadata", str(tmp_path / "meta.json"), "--relabel-summary", str(tmp_path / "summary.json")]


def test_cli_writes_aggregates_with_provenance(tmp_path):
    out = tmp_path / "out" / "composition.json"
    sc.main(_write_inputs(tmp_path, ISO_META) + ["--out", str(out)])
    res = json.loads(out.read_text())
    assert _counts(res["totals"]) == (10, 2, 1, 1, 8, 5, 4, 2, 4, 3)
    assert all(v["match"] for v in res["checks"].values())
    prov = res["provenance"]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    assert prov["secondary_map_sha256"] == sha(tmp_path / "sec_map.csv")
    assert prov["d1_map_sha256"] == sha(tmp_path / "d1_map.csv")
    assert prov["secondary_manifest_sha256"] == sha(tmp_path / "manifest.parquet")
    assert prov["run_metadata_sha256"] == sha(tmp_path / "meta.json")
    assert prov["relabel_summary_sha256"] == sha(tmp_path / "summary.json")
    assert prov["script_sha256"] == sha(ROOT / "scripts/data/secondary_composition.py")
    assert prov["run"] == {"kind": "isotropy", "config": "iso", "git_sha": "abc", "started_utc": "2026-01-01T00:00:00"}
    assert {"git_sha", "image", "image_id", "numpy", "pandas", "path_map_entries"} <= set(prov)
    text = out.read_text()
    assert "hostdata" not in text and ".png" not in text and ".tif" not in text and "origlabel" not in text
    assert not any(i in text for i in ("f:1", "s:2", "t:1", "x:9"))


def test_cli_stops_before_writing_when_a_check_fails(tmp_path):
    out = tmp_path / "out" / "composition.json"
    args = _write_inputs(tmp_path, {**ISO_META, "secondary_removed_shared_identity": 5}) + ["--out", str(out)]
    with pytest.raises(ValueError, match="shared_rows_vs_run_metadata"):
        sc.main(args)
    assert not out.exists()
