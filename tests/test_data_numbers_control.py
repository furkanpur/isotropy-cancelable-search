"""Control-encoder data keys (ctrl-vit): scripts/manuscript/data_numbers.py, one function per aggregate in
results/data/control/.

The fixtures are small dicts shaped like the committed aggregates (results/data/control/*.json). They are synthetic test
fixtures, not data; several values mirror the committed aggregates so that the expected keys match the report, the others
are invented. No test here reads a data file.
"""
import json
from pathlib import Path

import pytest

import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dn = _load("data_numbers", "scripts/manuscript/data_numbers.py")
mn = _load("make_numbers_dnc", "scripts/manuscript/make_numbers.py")


def _keys(fn, *args) -> dict:
    c = mn.Collector()
    fn(*args, c.put)
    return c.d


def _src(f, i, tf, ti, vf, vi):
    return {"fingers": f, "images": i, "train_fingers": tf, "train_images": ti, "val_fingers": vf, "val_images": vi}


# training_set.json (Task C1), reduced to two sources whose totals equal the committed totals
TRAIN_SET = {
    "fingers": 3684, "images": 27342, "train_fingers": 3315, "train_images": 24588, "val_fingers": 369,
    "val_images": 2754,
    "by_source": {"casia_v5": _src(3000, 15000, 2700, 13500, 300, 1500),
                  "sd302": _src(684, 12342, 615, 11088, 69, 1254)},
    "devices": {"distinct": 2, "images_by_device": {"": 15000, "C": 6000, "E": 6342}},
    "images_per_finger": {"min": 5, "median": 6.0, "max": 30},
    "disjointness": {"identity_overlap": 0, "content_overlap": 0},
    "selection": {"identities": 3684, "identities_found": 3694, "identities_dropped_partition": 10,
                  "rows_dropped_partition": 80, "duplicate_rows_collapsed": 0},
    "probe": {"unusable": 0},
}


def _epochs():
    rows = [{"epoch": 0, "val_eer": 0.290245}]
    rows += [{"epoch": e, "val_eer": 0.29 - 0.009 * e} for e in range(1, 27)]
    rows += [{"epoch": 27, "val_eer": 0.03103207810320781}]
    rows += [{"epoch": e, "val_eer": 0.04} for e in range(28, 31)]
    return rows


# training_summary.json (the frozen training record)
TRAIN_SUMMARY = {"epochs_requested": 30, "epochs_run": 30, "selected_epoch": 27,
                 "selected_val_eer": 0.03103207810320781, "pretrained_val_eer": 0.290245,
                 "improved_over_pretrained": True, "epochs": _epochs(),
                 "fingers": {"train": 3315, "val": 369}, "images": {"train": 24588, "val": 2754}}


def test_control_training_keys():
    d = _keys(dn.control_training, TRAIN_SET, TRAIN_SUMMARY)
    assert d["control.train.images"] == "27{,}342" and d["control.train.fingers"] == "3{,}684"
    assert d["control.train.devices"] == d["control.train.devices.recorded"] == "2" and d["control.train.devices.unrecordedimages"] == "15{,}000" and d["control.train.sources"] == "2"
    assert d["control.train.source.casia_v5.images"] == "15{,}000"
    assert d["control.train.source.sd302.fingers"] == "684"
    assert d["control.train.val.images"] == "2{,}754" and d["control.train.val.fingers"] == "369"
    assert d["control.train.fitted.images"] == "24{,}588" and d["control.train.fitted.fingers"] == "3{,}315"
    assert (d["control.train.perfinger.min"], d["control.train.perfinger.median"],
            d["control.train.perfinger.max"]) == ("5", "6", "30")
    assert d["control.train.dropped.partition"] == "10" and d["control.train.dropped.partitionrows"] == "80"
    assert d["control.train.dropped.duplicates"] == "0" and d["control.train.dropped.unusable"] == "0"
    assert d["control.fit.epochs"] == "30" and d["control.fit.selectedepoch"] == "27"
    assert d["control.fit.valeer.pretrained"] == "0.290245" and d["control.fit.valeer.selected"] == "0.031032"
    # the paper prints EER fractions with four decimals; these are new keys, and the six-decimal ones stay
    assert d["control.fit.valeer.pretrained.r4"] == "0.2902" and d["control.fit.valeer.selected.r4"] == "0.0310"


def test_control_training_renders_a_fractional_median_with_one_decimal():
    ts = json.loads(json.dumps(TRAIN_SET))
    ts["images_per_finger"]["median"] = 6.5
    assert _keys(dn.control_training, ts, TRAIN_SUMMARY)["control.train.perfinger.median"] == "6.5"


@pytest.mark.parametrize("edit, match", [
    (lambda ts, s: ts["by_source"]["sd302"].update(fingers=683), "per source"),
    (lambda ts, s: (ts.update(fingers=3685), ts["by_source"]["sd302"].update(fingers=685)), "train and validation"),
    (lambda ts, s: s["fingers"].update(train=3314), "summary"),
    (lambda ts, s: ts["selection"].update(identities_dropped_partition=9), "partition"),
    (lambda ts, s: ts["devices"].update(distinct=3), "devices"),
    (lambda ts, s: ts["disjointness"].update(identity_overlap=1), "overlap"),
    (lambda ts, s: s.update(selected_epoch=26), "selected epoch"),
    (lambda ts, s: s["epochs"][5].update(val_eer=0.01), "lowest"),
    (lambda ts, s: s.update(epochs_run=29), "epochs"),
    (lambda ts, s: s.update(improved_over_pretrained=False), "pretrained"),
])
def test_control_training_refuses_inconsistent_records(edit, match):
    ts, s = json.loads(json.dumps(TRAIN_SET)), json.loads(json.dumps(TRAIN_SUMMARY))
    edit(ts, s)
    with pytest.raises(ValueError, match=match):
        _keys(dn.control_training, ts, s)


def test_control_training_selects_the_earliest_of_tied_epochs_and_ignores_nan():
    s = json.loads(json.dumps(TRAIN_SUMMARY))
    s["epochs"][28]["val_eer"] = s["selected_val_eer"]          # a later tie does not change the earliest epoch
    s["epochs"][29]["val_eer"] = float("nan")
    _keys(dn.control_training, TRAIN_SET, s)
    s["selected_epoch"] = 28
    with pytest.raises(ValueError, match="selected epoch"):
        _keys(dn.control_training, TRAIN_SET, s)


def _src_counts(**by):
    return {"total": sum(by.values()), "by_source": dict(by)}


def _fl(fingers, reaching, gap_max):
    return {"fingers": fingers, "reaching_own_genuine_mean": reaching,
            "max_minus_own_genuine_mean": {"n": fingers, "max": gap_max}}


def _quant(median, mean):
    return {"n": 3, "mean": mean, "quantiles": {"0.5": median}}


def _sens(auc_c, auc_f, auc_d, fingers, removed):
    return {"images": fingers * 5, "fingers": fingers, "removed_fingers": removed,
            "encoders": {"ctrlvit": {"auc": auc_c}, "fingervit": {"auc": auc_f}, "deepprint": {"auc": auc_d}}}


# overlap_audit.json (Task C3, C4), reduced to three sources (14 D2 fingers, 100 images)
AUDIT = {
    "identity_overlap": 0, "content_overlap": 0, "d2_fingers": 14, "d2_rows": 100,
    "near_duplicates": {
        "threshold": 0.9384497767090806, "d2_images": 100,
        "images_at_or_above": _src_counts(casia_v5=0, sd302=0, polyu_cross=0),
        "fingers_at_or_above": _src_counts(casia_v5=0, sd302=0, polyu_cross=0),
        "max_cosine": {"overall": {"n": 100, "max": 0.8957546949386597},
                       "by_source": {"casia_v5": {"n": 60}, "sd302": {"n": 30}, "polyu_cross": {"n": 10}}}},
    "finger_level": {
        "d2_fingers": 14, "null_reference_source": "polyu_cross",
        "overall": _fl(14, 3, 0.07624096660042748),
        "by_source": {"casia_v5": _fl(9, 2, 0.07624096660042748), "sd302": _fl(4, 1, 0.05385495123903),
                      "polyu_cross": _fl(1, 0, -0.17080203839028063)}},
    "characterisation": {
        "flagged": {"overall": {"n": 3, "same_subject_key": 2, "position_comparable": 3, "same_finger_position": 1,
                                "own_genuine_mean": _quant(0.32461305077093405, 0.33860642406725733)}},
        "all": {"overall": {"n": 14, "same_subject_key": 9, "position_comparable": 12, "same_finger_position": 5,
                            "own_genuine_mean": _quant(0.5610759919709785, 0.5618894537614892)}}},
    "label_structure": {
        "candidates": {"total": 54}, "candidates_among_flagged": 0,
        "candidates_without_position_requirement": {"total": 128},
        "candidates_without_position_requirement_among_flagged": 4},
    "sensitivity": {"subsets": {
        "a_as_is": _sens(0.9954, 0.8935, 0.7389, 14, 0),
        "b_without_flagged": _sens(0.9957, 0.8948, 0.7405, 11, 3),
        "c_without_flagged_subjects": _sens(0.9957, 0.8961, 0.7432, 9, 5),
        "d_without_structural_candidates": _sens(0.9953, 0.8937, 0.7394, 12, 2)}},
    "subjects": {
        "d2_fingers_with_shared_subject": {"total": 12, "by_source": {"casia_v5": 8, "sd302": 4}},
        "d2_fingers_without_shared_subject": {"total": 2, "by_source": {"casia_v5": 1, "polyu_cross": 1}},
        "c1_subjects": {"by_subject_key": {"d2": 718, "shared": 710, "train": 759},
                        "by_source_subject": {"d2": 720, "shared": 704, "train": 780}},
        "agreement": {"flags_vs_recomputed": True, "c1_json_vs_recomputed": True}},
}


def test_control_audit_keys_image_level():
    d = _keys(dn.control_audit, AUDIT)
    assert d["control.audit.identityoverlap"] == "0" and d["control.audit.contentoverlap"] == "0"
    assert d["control.audit.nd.p999"] == "0.938450" and d["control.audit.nd.maxcos"] == "0.896"
    # zero observed hits carry the one-sided 95% Clopper-Pearson bound, 1 - 0.05**(1/n), in the key itself (status rule)
    assert d["control.audit.nd.images"] == dn.zero_count(1 - 0.05 ** (1 / 100), "image") and d["control.audit.nd.images.of"] == "100"
    assert d["control.audit.nd.fingers"] == dn.zero_count(1 - 0.05 ** (1 / 14), "finger")
    assert d["control.audit.nd.fingers"] == "0 (CP95 $\\leq 1.9\\times10^{-1}$ per finger)"
    assert d["control.audit.nd.fingers.of"] == "14"
    assert d["control.audit.nd.source.sd302.images"] == dn.zero_count(1 - 0.05 ** (1 / 30), "image")   # the source's n
    assert d["control.audit.nd.source.polyu_cross.images"] == dn.zero_count(1 - 0.05 ** (1 / 10), "image")
    assert d["control.audit.nd.images.bound"] == dn.sci(1 - 0.05 ** (1 / 100))
    assert d["control.audit.nd.fingers.bound"] == dn.sci(1 - 0.05 ** (1 / 14))


def test_control_audit_a_nonzero_hit_count_has_no_bound():
    bad = json.loads(json.dumps(AUDIT))
    nd = bad["near_duplicates"]
    nd["images_at_or_above"] = _src_counts(casia_v5=2, sd302=0, polyu_cross=0)
    d = _keys(dn.control_audit, bad)
    assert d["control.audit.nd.images"] == "2" and "control.audit.nd.images.bound" not in d
    assert d["control.audit.nd.source.casia_v5.images"] == "2"
    assert d["control.audit.nd.source.sd302.images"].startswith("0 (CP95")


def test_control_audit_keys_finger_level_and_label_structure():
    d = _keys(dn.control_audit, AUDIT)
    assert d["control.audit.fl.flagged"] == "3" and d["control.audit.fl.maxgap"] == "0.0762"
    assert d["control.audit.fl.source.casia_v5.flagged"] == "2" and d["control.audit.fl.source.sd302.flagged"] == "1"
    # a zero count of observed events carries its bound per unit; the null source has 1 finger in this fixture
    assert d["control.audit.fl.source.polyu_cross.flagged"] == dn.zero_count(1 - 0.05, "finger")
    assert d["control.audit.fl.null.flagged"] == dn.zero_count(1 - 0.05, "finger")
    assert d["control.audit.fl.null.maxgap"] == "-0.1708"      # negative values use a plain hyphen, as zp.*.oraclegap
    assert d["control.audit.ls.candidates"] == "54"
    assert d["control.audit.ls.flagged"] == dn.zero_count(1 - 0.05 ** (1 / 54), "candidate")
    assert d["control.audit.ls.nopos.candidates"] == "128" and d["control.audit.ls.nopos.flagged"] == "4"   # non-zero: bare


def test_control_audit_keys_characterisation_use_the_eligible_denominators():
    d = _keys(dn.control_audit, AUDIT)
    assert d["control.char.flagged.n"] == "3" and d["control.char.flagged.samesubject"] == "2"
    assert d["control.char.flagged.samesubject.pct"] == "66.7"
    assert d["control.char.flagged.sameposition"] == "1" and d["control.char.flagged.sameposition.pct"] == "33.3"
    # same-subject share among the D2 fingers that have a training subject (12), not among all 14
    assert d["control.char.all.samesubject"] == "9" and d["control.char.all.eligible"] == "12"
    assert d["control.char.all.samesubject.pct"] == "75.0"
    # same finger position among the fingers whose position is comparable (12), not among all 14
    assert d["control.char.all.sameposition"] == "5" and d["control.char.all.positioncomparable"] == "12"
    assert d["control.char.all.sameposition.pct"] == "41.7"
    assert d["control.char.owngen.flagged.median"] == "0.325" and d["control.char.owngen.all.median"] == "0.561"
    assert d["control.char.owngen.flagged.mean"] == "0.339" and d["control.char.owngen.all.mean"] == "0.562"


def test_control_audit_keys_sensitivity():
    d = _keys(dn.control_audit, AUDIT)
    assert d["control.sens.a.d2_ctrlvit.auc"] == "0.9954" and d["control.sens.d.d2_deepprint.auc"] == "0.7394"
    assert d["control.sens.b.fingers"] == "11" and d["control.sens.b.removed"] == "3"
    assert d["control.sens.c.images"] == "45"
    # largest absolute change of the label-check AUC from (a) across (b), (c), (d), per encoder and over all encoders
    assert d["control.sens.maxchange.d2_ctrlvit"] == "0.0003" and d["control.sens.maxchange.d2_fingervit"] == "0.0026"
    assert d["control.sens.maxchange.d2_deepprint"] == "0.0043" and d["control.sens.maxchange"] == "0.0043"


def test_control_audit_keys_subjects():
    d = _keys(dn.control_audit, AUDIT)
    assert d["control.subj.shared"] == "710" and d["control.subj.d2subjects"] == "718"
    assert d["control.subj.d2subjects.bysource"] == "720"   # (source, subject) pairs, not subject keys
    # subject keys with no training finger: they hold the fingers counted as unshared below
    assert d["control.subj.d2subjects.unshared"] == "8"
    assert d["control.subj.trainsubjects"] == "759"
    assert d["control.subj.d2fingers.shared"] == "12" and d["control.subj.d2fingers.unshared"] == "2"
    assert d["control.subj.d2fingers.unshared.polyu_cross"] == "1" and d["control.subj.d2fingers.unshared.casia_v5"] == "1"
    assert d["control.subj.d2fingers.shared.casia_v5"] == "8" and d["control.subj.d2fingers.shared.sd302"] == "4"
    assert "control.subj.d2fingers.shared.polyu_cross" not in d


@pytest.mark.parametrize("edit, match", [
    (lambda a: a.update(identity_overlap=1), "overlap"),
    (lambda a: a.update(content_overlap=2), "overlap"),
    (lambda a: a["finger_level"]["by_source"]["sd302"].update(reaching_own_genuine_mean=2), "flagged"),
    (lambda a: a["characterisation"]["flagged"]["overall"].update(n=4), "flagged"),
    (lambda a: a["near_duplicates"]["fingers_at_or_above"]["by_source"].update(sd302=1), "per source"),
    (lambda a: a["subjects"]["agreement"].update(flags_vs_recomputed=False), "agree"),
    (lambda a: a["characterisation"]["all"]["overall"].update(same_subject_key=13), "eligible"),
    (lambda a: a["sensitivity"]["subsets"]["a_as_is"].update(removed_fingers=1), "as is"),
])
def test_control_audit_refuses_inconsistent_aggregates(edit, match):
    bad = json.loads(json.dumps(AUDIT))
    edit(bad)
    with pytest.raises(ValueError, match=match):
        _keys(dn.control_audit, bad)


def _stat(clean, mean, lo, hi, outside, bounded=None):
    s = {"clean": clean, "flagged_mean": mean, "flagged_p2.5": lo, "flagged_p97.5": hi,
         "clean_outside_flagged_range": outside}
    return s if bounded is None else {**s, "bounded_by_samples": bounded}


def _space(auc, eer):
    return {"statistics": {"auc": _stat(auc, 0.9967, 0.9947, 0.9988, True),
                           "eer": _stat(eer, 0.0244, 0.0155, 0.0352, True),
                           "mean_genuine": _stat(0.5813, 0.5497, 0.5339, 0.5712, True),
                           "mean_impostor": _stat(0.0385, 0.0092, 0.0062, 0.0125, True),
                           "pr_all_images": _stat(41.3119, 64.6416, 62.5785, 67.6123, True, True),
                           "pr_one_image_per_finger": _stat(29.3647, 39.241, 37.8188, 40.2218, True, True)}}


def _enc(auc, eer):
    return {"counts": {name: {"clean": n, "flagged_min": n, "flagged_max": n}   # size-matched flagged subsamples
                       for name, n in (("fingers", 1), ("genuine_pairs", 560), ("images", 280))},
            "spaces": {"raw": _space(auc, eer), "prekey": _space(auc + 0.0003, eer - 0.0005)}}


def _pspace(auc, eer, pr1, pr2):
    return {"auc": auc, "eer": eer, "pr_all_images": pr1, "pr_one_image_per_finger": pr2}


def _polyu_enc(auc, eer):
    return {"fingers": 336, "counts": {"images": 2976},
            "spaces": {"raw": _pspace(auc, eer, 59.0608, 53.9248),
                       "prekey": _pspace(auc + 0.0007, eer - 0.003, 58.8209, 53.734)}}


# subject_check.json (Task C4): a casia_v5 stratum for three encoders, the polyu label-check block and descriptive block
SUBJECT = {
    "stratum": {"name": "casia_v5", "fingers": {"clean": 1, "flagged": 8, "fingers": 9}},
    "encoders": {"ctrlvit": _enc(0.9878, 0.047), "fingervit": _enc(0.9085, 0.174), "deepprint": _enc(0.7629, 0.304)},
    "polyu": {"ctrlvit": {"auc": 0.9923, "fingers": 336}, "fingervit": {"auc": 0.9302, "fingers": 336},
              "deepprint": {"auc": 0.8931, "fingers": 336}},
    "polyu_descriptive": {"ctrlvit": _polyu_enc(0.9923, 0.04303), "fingervit": _polyu_enc(0.93, 0.1),
                          "deepprint": _polyu_enc(0.8931, 0.18485)},
}


def test_control_subject_check_keys():
    d = _keys(dn.control_subject_check, SUBJECT)
    assert d["control.subj.casia.clean"] == "1" and d["control.subj.casia.flagged"] == "8"
    assert d["control.subj.casia.genuinepairs"] == "560" and d["control.subj.casia.images"] == "280"
    k = "control.subj.casia.raw.d2_ctrlvit"
    assert d[f"{k}.eer.clean"] == "0.0470" and d[f"{k}.eer.flaggedmean"] == "0.0244"
    assert d[f"{k}.auc.clean"] == "0.9878" and d[f"{k}.auc.outside"] == "yes"
    assert d[f"{k}.prall.clean"] == "41.31" and d[f"{k}.prone.bounded"] == "yes" and f"{k}.eer.bounded" not in d
    assert d["control.subj.casia.prekey.d2_deepprint.eer.flaggedlo"] == "0.0155"
    assert d["control.subj.casia.raw.d2_fingervit.auc.clean"] == "0.9085"


def test_control_subject_check_polyu_keys_are_descriptive():
    d = _keys(dn.control_subject_check, SUBJECT)
    assert d["control.polyu.fingers"] == "336" and d["control.polyu.images"] == "2{,}976"
    assert d["control.polyu.raw.d2_ctrlvit.eer"] == "0.0430" and d["control.polyu.prekey.d2_ctrlvit.eer"] == "0.0400"
    assert d["control.polyu.raw.d2_ctrlvit.prall"] == "59.06" and d["control.polyu.raw.d2_ctrlvit.prone"] == "53.92"
    assert d["control.polyu.raw.d2_deepprint.eer"] == "0.1848"
    assert not any(k.endswith(".outside") for k in d if k.startswith("control.polyu."))


def test_control_subject_check_refuses_flagged_subsamples_that_are_not_size_matched():
    bad = json.loads(json.dumps(SUBJECT))
    bad["encoders"]["fingervit"]["counts"]["images"]["flagged_max"] = 281
    with pytest.raises(ValueError, match="fingervit.*size-matched"):
        _keys(dn.control_subject_check, bad)


def test_control_subject_check_refuses_inconsistent_aggregates():
    bad = json.loads(json.dumps(SUBJECT))
    bad["encoders"]["deepprint"]["counts"]["genuine_pairs"]["clean"] = 559
    with pytest.raises(ValueError, match="deepprint"):
        _keys(dn.control_subject_check, bad)
    bad = json.loads(json.dumps(SUBJECT))
    bad["polyu_descriptive"]["fingervit"]["fingers"] = 335
    with pytest.raises(ValueError, match="polyu"):
        _keys(dn.control_subject_check, bad)


def _lc_ctrl(auc=0.9954):
    return {"n_images": 100, "n_identities": 14,
            "models": {"ctrlvit": {"all": {"genuine_pairs": 10, "impostor_pairs": 90, "auc": auc, "mean_impostor": 0.0108,
                                           "same_device": {"pairs": 6, "auc": 0.997},
                                           "cross_device": {"pairs": 4, "auc": 0.9904}},
                                   "per_source": {"polyu_cross": {"genuine_pairs": 5, "auc": 0.9923,
                                                                  "mean_impostor": 0.1705,
                                                                  "same_device": {"pairs": 5, "auc": 0.9923},
                                                                  "cross_device": None}}}}}


LABEL_CTRL = _lc_ctrl()


def test_control_label_check_keys_follow_the_label_check_family():
    d = _keys(dn.d2_label_check, LABEL_CTRL)
    assert d["d2.auc.d2_ctrlvit.all"] == "0.995" and d["d2.auc.d2_ctrlvit.cross"] == "0.990"
    assert d["d2.auc.d2_ctrlvit.polyu_cross.all"] == "0.992" and d["d2.imp.d2_ctrlvit.polyu_cross"] == "0.171"
    assert "d2.auc.d2_ctrlvit.polyu_cross.cross" not in d


def test_control_label_check_merges_into_the_label_check_and_checks_the_corpus():
    merged = dn.merge_label_checks({"n_images": 100, "n_identities": 14, "models": {"fingervit": {}}}, LABEL_CTRL)
    assert set(merged["models"]) == {"fingervit", "ctrlvit"}
    with pytest.raises(ValueError, match="corpus"):
        dn.merge_label_checks({"n_images": 99, "n_identities": 14, "models": {}}, LABEL_CTRL)
    with pytest.raises(ValueError, match="already"):
        dn.merge_label_checks({"n_images": 100, "n_identities": 14, "models": {"ctrlvit": {}}}, LABEL_CTRL)


RELABEL_FINGERS = {"true_fingers": 3694}
EXTRACT_FINGERS = {"composition": {"n_identities": 14}}


def _checks(**over):
    args = dict(relabel=RELABEL_FINGERS, extract=EXTRACT_FINGERS, train_set=TRAIN_SET, audit=AUDIT,
                label_check=None, subject=SUBJECT, label_check_ctrl=LABEL_CTRL)
    args.update(over)
    return dn.control_checks(**args)


def _label_ref(fv=0.8935, dp=0.7389, fv_polyu=0.9302, dp_polyu=0.8931):
    """The baseline encoders' label check (the AUCs the audit's subset (a) and the polyu block must agree with)."""
    return {"models": {"fingervit": {"all": {"auc": fv}, "per_source": {"polyu_cross": {"auc": fv_polyu}}},
                       "deepprint": {"all": {"auc": dp}, "per_source": {"polyu_cross": {"auc": dp_polyu}}}}}


def test_control_checks_pass_on_consistent_aggregates():
    _checks(label_check=_label_ref())


@pytest.mark.parametrize("over, match", [
    ({"relabel": {"true_fingers": 3693}}, "control.train.fingers"),
    ({"extract": {"composition": {"n_identities": 15}}}, "d2.fingers"),
    ({"label_check_ctrl": _lc_ctrl(0.9955)}, "label check"),
])
def test_control_checks_refuse_disagreeing_aggregates(over, match):
    with pytest.raises(ValueError, match=match):
        _checks(label_check=_label_ref(), **over)


def test_control_checks_compare_the_baseline_aucs_with_the_audit():
    with pytest.raises(ValueError, match="fingervit"):
        _checks(label_check=_label_ref(fv=0.8936))
    with pytest.raises(ValueError, match="polyu.*deepprint"):
        _checks(label_check=_label_ref(dp_polyu=0.8932))


def test_control_checks_tie_the_subject_stratum_and_the_null_source_to_the_audit():
    bad = json.loads(json.dumps(SUBJECT))
    bad["stratum"]["fingers"]["clean"] = 2          # the audit has 1 casia_v5 finger without a training subject
    with pytest.raises(ValueError, match="stratum casia_v5"):
        _checks(label_check=_label_ref(), subject=bad)
    bad = json.loads(json.dumps(SUBJECT))
    bad["stratum"]["fingers"]["flagged"] = 9
    with pytest.raises(ValueError, match="stratum casia_v5"):
        _checks(label_check=_label_ref(), subject=bad)
    bad = json.loads(json.dumps(AUDIT))
    bad["finger_level"]["by_source"]["polyu_cross"]["fingers"] = 2
    bad["finger_level"]["by_source"]["casia_v5"]["fingers"] = 8
    with pytest.raises(ValueError, match="polyu_cross finger lacks"):
        _checks(label_check=_label_ref(), audit=bad)


def test_control_checks_compare_the_polyu_block_with_the_label_check():
    bad = json.loads(json.dumps(SUBJECT))
    bad["polyu"]["fingervit"]["auc"] = 0.9301
    with pytest.raises(ValueError, match="polyu"):
        _checks(label_check=_label_ref(), subject=bad)


def test_every_new_key_is_a_valid_collector_key():
    c = mn.Collector()
    dn.control_training(TRAIN_SET, TRAIN_SUMMARY, c.put)
    dn.control_audit(AUDIT, c.put)
    dn.control_subject_check(SUBJECT, c.put)
    assert all(k.startswith("control.") for k in c.d)
    assert len(c.d) > 150   # the Collector refuses a malformed or duplicate key at put, so every key here is valid
    # a bare zero is a census or a structural fact; every zero that states an observed absence carries its bound
    bare = sorted(k for k, v in c.d.items() if v == "0")
    assert bare == ["control.audit.contentoverlap", "control.audit.identityoverlap", "control.sens.a.removed",
                    "control.train.dropped.duplicates", "control.train.dropped.unusable"], bare
