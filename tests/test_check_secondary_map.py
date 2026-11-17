# tests/test_check_secondary_map.py
"""The secondary-map check: per path-map entry, identity agreement on rows that duplicate a first-corpus row."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/data"))
import build_relabel_map as brm  # noqa: E402
import check_secondary_map as csm  # noqa: E402


def test_match_index_is_the_rule_map_path_uses():
    maps = brm.parse_path_map(["/hostdata/db=/c/db", "/hostdata/db/DB1=/c/db/DB1_B"])
    assert brm.match_index("/hostdata/db/DB1/1_1.tif", maps) == 1
    assert brm.match_index("/hostdata/db/DB2/1_1.tif", maps) == 0
    assert brm.match_index("/hostdata/dbx/1.tif", maps) is None
    assert brm.map_path("/hostdata/db/DB1/1_1.tif", maps) == "/c/db/DB1_B/1_1.tif"


def test_entry_index():
    maps = brm.parse_path_map(["/hostdata/A=/c/a", "/hostdata/B=/c/b"])
    got = csm.entry_index(["/hostdata/A/1.png", "/hostdata/B/x/2.png", "/hostdata/C/3.png", "/hostdata/A"], maps)
    assert got.tolist() == [0, 1, -1, 0]


def test_nearest_returns_the_best_uncentred_cosine_and_its_row():
    d1 = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    sec = np.array([[3.0, 0.0], [0.0, -1.0], [2.0, 2.1]])
    cos, row = csm.nearest(sec, d1, block=2)
    assert row.tolist() == [0, 0, 2]   # [0, -1]: cosine 0 with row 0 beats -1 (row 1) and -0.707 (row 2)
    np.testing.assert_allclose(cos[:2], [1.0, 0.0], atol=1e-12)
    assert 0.999 < cos[2] < 1.0


def _rows():
    # entry 0 (dataset dsA): 4 rows; entry 1 (dsB1, dsB2): 2 rows without duplicates; 2 unmapped rows (dsC)
    dataset = np.array(["dsA"] * 4 + ["dsB1", "dsB2"] + ["dsC"] * 2)
    entry = np.array([0, 0, 0, 0, 1, 1, -1, -1])
    sec_id = np.array(["id:a:01", "id:x:09", "id:c:03", "id:d:04", "id:e:05", "", "", ""])
    base = (np.array([1.0, 0.9995, 0.96, 0.5, 0.2, 0.3, 0.9999, 0.1]),
            np.array(["id:a:01", "id:b:02", "id:c:03", "", "id:e:05", "id:f:06", "id:g:07", ""]))
    large = (np.array([0.9991, 0.9, 0.97, 0.99, 0.1, 0.1, 0.5, 0.1]),
             np.array(["id:a:01", "id:b:02", "", "id:d:04", "", "", "", ""]))
    return dataset, entry, sec_id, {"base": base, "large": large}


def test_aggregate_counts_per_entry_and_flags_entries_without_duplicates():
    dataset, entry, sec_id, nn = _rows()
    out = csm.aggregate(dataset, entry, sec_id != "", sec_id, nn, n_entries=2, thresholds=(0.999, 0.95))
    a, b = out["entries"]
    assert (a["label"], a["rows"], a["matched"]) == ("dsA", 4, 4)
    assert a["models"]["base"]["0.999"] == {"duplicate_rows": 2, "comparable": 2, "same_identity": 1}
    assert a["models"]["base"]["0.95"] == {"duplicate_rows": 3, "comparable": 3, "same_identity": 2}
    assert a["models"]["large"]["0.999"] == {"duplicate_rows": 1, "comparable": 1, "same_identity": 1}
    assert a["models"]["large"]["0.95"] == {"duplicate_rows": 3, "comparable": 2, "same_identity": 2}
    assert a["no_exact_duplicate_coverage"] is False
    assert (b["label"], b["rows"], b["matched"]) == ("dsB1+dsB2", 2, 1)
    assert b["no_exact_duplicate_coverage"] is True
    assert b["models"]["base"]["0.95"]["duplicate_rows"] == 0
    assert out["unmapped"] == [{"label": "dsC", "rows": 2, "matched": 0}]
    tot = out["totals"]
    assert (tot["rows"], tot["matched"]) == (8, 5)
    assert tot["models"]["base"]["0.999"] == {"duplicate_rows": 3, "comparable": 2, "same_identity": 1}
    assert out["entries_without_exact_duplicate_coverage"] == ["dsB1+dsB2"]
    assert out["thresholds"] == [0.999, 0.95] and out["models"] == ["base", "large"]
    assert "id:" not in json.dumps(out)   # aggregates only: no identity value


def test_aggregate_rejects_misaligned_inputs():
    dataset, entry, sec_id, nn = _rows()
    with pytest.raises(ValueError, match="rows"):
        csm.aggregate(dataset[:-1], entry, sec_id != "", sec_id, nn, n_entries=2)


def _sin(root, x, man):
    root.mkdir(parents=True)
    for m, arr in x.items():
        np.save(root / f"emb_{m}.npy", np.asarray(arr, dtype=np.float32))
    man.to_parquet(root / "manifest.parquet")


def test_cli_writes_aggregates_and_provenance(tmp_path):
    rng = np.random.default_rng(0)
    d1x = {m: rng.standard_normal((5, 6)) for m in ("base", "large")}
    d1_paths = [f"/elsewhere/d1/{i}.png" for i in range(5)]
    _sin(tmp_path / "d1", d1x, pd.DataFrame({"identity": ["old"] * 5, "path": d1_paths}))
    pd.DataFrame({"file": [f"{i}.png" for i in range(4)], "identity": ["g:1", "g:2", "g:3", "g:4"]}).to_csv(
        tmp_path / "d1_map.csv", index=False)                      # D1 row 4 has no compilation identity
    # secondary: rows 0-1 copy D1 rows 0-1 (entry A), row 2 copies D1 row 2 under another identity (entry B),
    # row 3 is fresh (entry B), row 4 lies outside every path-map entry
    secx = {m: np.vstack([d1x[m][[0, 1, 2]], rng.standard_normal((2, 6))]) for m in d1x}
    sec_paths = ["/hostdata/A/1.png", "/hostdata/A/2.png", "/hostdata/B/1.png", "/hostdata/B/2.png", "/hostdata/Z/1.png"]
    _sin(tmp_path / "sec", secx, pd.DataFrame({"identity": ["o"] * 5, "path": sec_paths,
                                              "dataset": ["dsA", "dsA", "dsB", "dsB", "dsZ"]}))
    pd.DataFrame({"file": sec_paths[:4], "identity": ["g:1", "g:2", "g:9", "g:5"]}).to_csv(
        tmp_path / "sec_map.csv", index=False)
    out = tmp_path / "out" / "check.json"
    csm.main(["--sec-root", str(tmp_path / "sec"), "--sec-map", str(tmp_path / "sec_map.csv"),
              "--d1-root", str(tmp_path / "d1"), "--d1-map", str(tmp_path / "d1_map.csv"),
              "--path-map", "/hostdata/A=/c/a", "--path-map", "/hostdata/B=/c/b", "--out", str(out)])
    res = json.loads(out.read_text())
    a, b = res["entries"]
    assert (a["label"], a["rows"], a["matched"]) == ("dsA", 2, 2)
    assert a["models"]["base"]["0.999"] == {"duplicate_rows": 2, "comparable": 2, "same_identity": 2}
    assert b["models"]["large"]["0.999"] == {"duplicate_rows": 1, "comparable": 1, "same_identity": 0}
    assert res["unmapped"] == [{"label": "dsZ", "rows": 1, "matched": 0}]
    prov = res["provenance"]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    assert prov["secondary_map_sha256"] == sha(tmp_path / "sec_map.csv")
    assert prov["d1_map_sha256"] == sha(tmp_path / "d1_map.csv")
    assert prov["secondary_manifest_sha256"] == sha(tmp_path / "sec" / "manifest.parquet")
    assert prov["d1_manifest_sha256"] == sha(tmp_path / "d1" / "manifest.parquet")
    assert prov["script_sha256"] == sha(ROOT / "scripts/data/check_secondary_map.py")
    assert prov["models"] == ["base", "large"] and prov["thresholds"] == [0.999, 0.95]
    assert prov["path_map_entries"] == 2 and {"numpy", "pandas", "git_sha"} <= set(prov)
    text = out.read_text()
    assert "hostdata" not in text and "elsewhere" not in text and ".png" not in text and "g:" not in text
