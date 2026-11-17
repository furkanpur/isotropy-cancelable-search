# tests/test_control_selection.py
"""Training-set selection, validation split and D2 disjointness checks (SYNTHETIC fixtures: no real biometric data)."""
import numpy as np
import pandas as pd
import pytest

from ijiss_btp.train.selection import (disjointness, select_training_rows, subject_flags, subject_key,
                                       validation_split)


def _man(rows):
    cols = ["path", "identity", "source", "subject", "finger", "device", "exclusion"]
    return pd.DataFrame(rows, columns=cols)


def _idt(rows):
    return pd.DataFrame(rows, columns=["identity", "partition"])


def _keys(man):
    # synthetic content keys: the path stem
    return pd.DataFrame({"path": man["path"], "key": [p.split("/")[-1].split(".")[0] for p in man["path"]]})


def _fixture():
    man = _man([
        ["i/a1.png", "s:1:L", "s", "1", "L", "d", "none"],
        ["i/a2.png", "s:1:L", "s", "1", "L", "d", "none"],
        ["i/a3.png", "s:1:L", "s", "1", "L", "d", "spoof"],      # excluded
        ["i/b1.png", "s:2:L", "s", "2", "L", "d", "none"],       # holdout finger
        ["i/c1.png", "s:3:L", "s", "3", "L", "d", "none"],
        ["i/z1.png", "s:9:L", "s", "9", "L", "d", "none"],       # not requested
    ])
    idt = _idt([["s:1:L", "development"], ["s:2:L", "holdout"], ["s:3:L", "development"], ["s:9:L", "development"]])
    return man, idt


def test_partition_exclusion_and_counts():
    man, idt = _fixture()
    rows, c = select_training_rows(man, idt, ["s:1:L", "s:2:L", "s:3:L", "s:404:L"], _keys(man))
    assert sorted(rows["path"]) == ["i/a1.png", "i/a2.png", "i/c1.png"]
    assert "key" in rows.columns
    assert c == {"rows_before": 4, "identities_requested": 4, "identities_found": 3,
                 "identities_dropped_partition": 1, "rows_dropped_partition": 1,
                 "duplicate_rows_collapsed": 0, "keys_shared_across_fingers": 0,
                 "rows_with_shared_keys": 0, "rows": 3, "identities": 2}


def test_duplicates_collapse_within_a_finger_only():
    man = _man([
        ["i/x_b.png", "s:1:L", "s", "1", "L", "d", "none"],
        ["i/x_a.png", "s:1:L", "s", "1", "L", "d", "none"],
        ["i/y.png", "s:1:L", "s", "1", "L", "d", "none"],
        ["i/x_c.png", "s:2:L", "s", "2", "L", "d", "none"],      # same key under another finger: kept
    ])
    keys = pd.DataFrame({"path": man["path"], "key": ["K", "K", "Y", "K"]})
    idt = _idt([["s:1:L", "development"], ["s:2:L", "development"]])
    rows, c = select_training_rows(man, idt, ["s:1:L", "s:2:L"], keys)
    assert sorted(rows["path"]) == ["i/x_a.png", "i/x_c.png", "i/y.png"]   # first sorted path kept
    assert c["duplicate_rows_collapsed"] == 1 and c["rows"] == 3 and c["rows_before"] == 4


def test_cross_finger_shared_keys_are_counted_and_kept():
    man = _man([
        ["i/p.png", "s:1:L", "s", "1", "L", "d", "none"],
        ["i/q.png", "s:1:L", "s", "1", "L", "d", "none"],
        ["i/r.png", "s:2:L", "s", "2", "L", "d", "none"],
        ["i/t.png", "s:3:L", "s", "3", "L", "d", "none"],
        ["i/u.png", "s:3:L", "s", "3", "L", "d", "none"],
    ])
    keys = pd.DataFrame({"path": man["path"], "key": ["K", "A", "K", "K", "B"]})   # K under three fingers
    idt = _idt([["s:1:L", "development"], ["s:2:L", "development"], ["s:3:L", "development"]])
    rows, c = select_training_rows(man, idt, ["s:1:L", "s:2:L", "s:3:L"], keys)
    assert len(rows) == 5 and c["duplicate_rows_collapsed"] == 0
    assert c["keys_shared_across_fingers"] == 1 and c["rows_with_shared_keys"] == 3


def test_missing_key_is_an_error():
    man, idt = _fixture()
    with pytest.raises(ValueError):
        select_training_rows(man, idt, ["s:1:L"], _keys(man).iloc[1:])


def _rows_for_split():
    recs = []
    for src, n in (("a", 25), ("b", 7)):
        for i in range(n):
            for _ in range(2):
                recs.append({"identity": f"{src}:{i:02d}:L", "source": src})
    return pd.DataFrame(recs)


def test_validation_split_constant_per_finger_sized_and_deterministic():
    rows = _rows_for_split()
    v = validation_split(rows)
    assert v.dtype == bool and v.index.equals(rows.index)
    per = rows.assign(v=v).groupby("identity")["v"].nunique()
    assert (per == 1).all()
    nval = rows.assign(v=v).drop_duplicates("identity").groupby("source")["v"].sum()
    assert nval["a"] == int(np.floor(0.1 * 25 + 0.5)) == 3 and nval["b"] == int(np.floor(0.1 * 7 + 0.5)) == 1
    assert validation_split(rows).equals(v)
    assert not validation_split(rows, seed=1).equals(v)


def test_validation_split_does_not_depend_on_row_order():
    rows = _rows_for_split()
    v = validation_split(rows)
    shuf = rows.sample(frac=1.0, random_state=0)
    assert validation_split(shuf).sort_index().equals(v.sort_index())


def test_subject_key_drops_the_last_component():
    s = subject_key(pd.Series(["casia_v5:000:L1", "fvc2000:12:R", "plain"]))
    assert s.tolist() == ["casia_v5:000", "fvc2000:12", "plain"]


def _d2():
    return pd.DataFrame({
        "identity": ["casia_v5:000:R1", "casia_v5:000:R1", "casia_v5:050:L1", "fvc2000:12:L", "casia_ageing:7:L"],
        "source": ["casia_v5", "casia_v5", "casia_v5", "fvc2000", "casia_ageing"],
        "subject": ["000", "000", "050", "12", "7"],
        "path": ["d/1", "d/2", "d/3", "d/4", "d/5"]})


def _train():
    return pd.DataFrame({
        "identity": ["casia_v5:000:L1", "casia_v5:001:L1", "casia_ageing:7:L"],
        "source": ["casia_ageing", "casia_v5", "casia_ageing"],   # first: a finger linked across sources
        "subject": ["0000", "001", "7"],
        "key": ["t1", "t2", "t3"]})


def test_disjointness_reports_identity_content_and_subject_overlap():
    keys = pd.DataFrame({"path": ["d/1", "d/2", "d/3", "d/4", "d/5"], "key": ["u1", "u2", "t2", "u4", "u5"]})
    r = disjointness(_train(), _d2(), keys)
    assert r["d2_rows"] == 5 and r["d2_keys_checked"] == 5
    assert r["identity_overlap"] == 1                      # casia_ageing:7:L in both
    assert r["content_overlap"] == 1                       # key t2
    assert r["subjects"]["by_subject_key"] == {"train": 3, "d2": 4, "shared": 2}   # casia_v5:000, casia_ageing:7
    assert r["subjects"]["by_source_subject"] == {"train": 3, "d2": 4, "shared": 1}   # only (casia_ageing, 7): the link is missed
    assert r["d2_fingers_with_shared_subject"] == {"total": 2, "by_source": {"casia_ageing": 1, "casia_v5": 1}}
    assert r["d2_fingers_without_shared_subject"] == {"total": 2, "by_source": {"casia_v5": 1, "fvc2000": 1}}


def test_disjointness_zero_when_disjoint():
    keys = pd.DataFrame({"path": ["d/1", "d/2", "d/3", "d/4", "d/5"], "key": list("vwxyz")})
    train = _train().iloc[[1]]
    r = disjointness(train, _d2(), keys)
    assert r["identity_overlap"] == 0 and r["content_overlap"] == 0
    assert r["subjects"]["by_subject_key"]["shared"] == 0
    assert r["d2_fingers_with_shared_subject"] == {"total": 0, "by_source": {}}


def test_subject_flags_one_row_per_d2_identity():
    f = subject_flags(_train(), _d2())
    assert list(f.columns) == ["identity", "source", "subject_shared"]
    assert len(f) == 4 and f["identity"].is_unique
    got = dict(zip(f["identity"], f["subject_shared"]))
    assert got == {"casia_v5:000:R1": True, "casia_v5:050:L1": False, "fvc2000:12:L": False, "casia_ageing:7:L": True}


@pytest.mark.parametrize("case", ["path_missing", "null_key"])
def test_disjointness_refuses_a_d2_row_without_a_key(case):
    keys = pd.DataFrame({"path": ["d/1", "d/2", "d/3", "d/4", "d/5"], "key": list("vwxyz")})
    if case == "path_missing":
        keys = keys.iloc[:4]
    else:
        keys.loc[4, "key"] = None
    with pytest.raises(ValueError, match="no content key"):
        disjointness(_train(), _d2(), keys)
