"""Data-provenance keys (scripts/manuscript/data_numbers.py): one function per aggregate in results/data/.

The fixtures are small dicts shaped like the committed aggregates. Many fixture values mirror committed aggregate
values, so that the expected keys match the reports; the others are invented. No test here reads a data file.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dn = _load("data_numbers", "scripts/manuscript/data_numbers.py")
mn = _load("make_numbers_dn", "scripts/manuscript/make_numbers.py")


def _keys(fn, *args) -> dict:
    """Run one family function through the make_numbers Collector (key syntax and duplicate checks included)."""
    c = mn.Collector()
    fn(*args, c.put)
    return c.d


def test_formats_come_from_make_numbers():
    """data_numbers defines no count or pct of its own: it uses make_numbers' helpers, and its share of n in `of`
    has its own name, so pct keeps make_numbers' meaning (a fraction in percent)."""
    assert dn.count.__module__ == "make_numbers" and dn.pct.__module__ == "make_numbers"
    for n in (7, 2009, 11147, 1000000):
        assert dn.count(n) == mn.count(n)
    assert dn.pct(0.71242) == mn.pct(0.71242) == "71.2"
    assert dn.share_pct(2237, 3140) == "71.2" and dn.sci(2.9957277863525433e-06) == "$3.0\\times10^{-6}$"
    assert dn.sci(5.641521224714513e-05) == "$5.6\\times10^{-5}$"


RELABEL = {"n_images": 10359, "matched": 10233, "unmatched": 119, "ambiguous": 7, "true_fingers": 3694}


def test_relabel_d1_keys():
    d = _keys(dn.relabel_d1, RELABEL)
    assert d == {"relabel.d1.images": "10{,}359", "relabel.d1.placed": "10{,}233", "relabel.d1.unmatched": "119",
                 "relabel.d1.ambiguous": "7", "relabel.d1.dropped": "126", "relabel.d1.fingers": "3{,}694"}


def test_relabel_d1_refuses_counts_that_do_not_add_up():
    with pytest.raises(ValueError, match="add up"):
        _keys(dn.relabel_d1, {**RELABEL, "unmatched": 118})


EXTRACT = {
    "composition": {
        "n_images": 100, "n_identities": 14, "n_subjects": 9, "n_source_devices": 4,
        "impressions_per_finger_summary": {"min": 5, "max": 30, "mean": 7.1, "median": 5.0},
        "capture": {"contact": 70, "roll": 30}, "dpi": {"500": 30, "unknown": 70},
        "per_source": {"casia_v5": {"identities": 10, "images": 50}, "sd302": {"identities": 4, "images": 50}},
        "devices_by_source": {"casia_v5": {"": 50}, "sd302": {"C": 20, "E": 20, "G": 10}}},
    "compilation": {"excluded_sources": ["livdet"]},
    "selection_counts": {"identities_excluded_source": 3372, "rows_excluded_source": 37020,
                         "identities_excluded_d1": 3694, "identities_excluded_d1_in_folds": 3684,
                         "rows_excluded_d1": 27422, "identities_dropped_d1_map_duplicate": 8,
                         "identities_dropped_d1_map_duplicate_via_unselected_images": 8,
                         "images_dropped_d1_map_duplicate": 53, "identities_with_d1_image": 0},
}


def test_d2_corpus_keys():
    d = _keys(dn.d2_corpus, EXTRACT)
    assert d["d2.images"] == "100" and d["d2.fingers"] == "14" and d["d2.subjects"] == "9"
    assert d["d2.sources"] == "2" and d["d2.devices"] == "4"
    assert d["d2.perfinger.min"] == "5" and d["d2.perfinger.max"] == "30"
    assert d["d2.capture.roll"] == "30" and d["d2.dpi.500"] == "30" and d["d2.dpi.unknown"] == "70"
    assert d["d2.source.casia_v5.fingers"] == "10" and d["d2.source.casia_v5.fingers.pct"] == "71.4"
    assert d["d2.source.sd302.images"] == "50" and d["d2.source.sd302.images.pct"] == "50.0"
    assert d["d2.excluded.sources"] == "1" and d["d2.excluded.source.fingers"] == "3{,}372"
    assert d["d2.excluded.source.rows"] == "37{,}020"
    assert d["d2.excluded.firstcorpus.fingers"] == "3{,}694" and d["d2.excluded.firstcorpus.fingersinfolds"] == "3{,}684"
    assert d["d2.excluded.firstcorpus.rows"] == "27{,}422"
    assert d["d2.excluded.duplicate.fingers"] == "8" and d["d2.excluded.duplicate.images"] == "53"
    # the duplicate rule: every dropped finger was found through compilation images not selected for D2, and the
    # separate check against D1's own images dropped none
    assert d["d2.excluded.duplicate.viaunselected"] == "8" and d["d2.excluded.d1image.fingers"] == "0"
    # devices: named devices and sources without a device record, each of which counts as one device
    assert d["d2.devices.recorded"] == "3" and d["d2.devices.unrecorded"] == "1"


def test_d2_corpus_refuses_per_source_counts_that_do_not_add_up():
    bad = json.loads(json.dumps(EXTRACT))
    bad["composition"]["per_source"]["sd302"]["identities"] = 5
    with pytest.raises(ValueError, match="per source"):
        _keys(dn.d2_corpus, bad)


def test_d2_corpus_refuses_devices_that_do_not_add_up():
    bad = json.loads(json.dumps(EXTRACT))
    bad["composition"]["devices_by_source"]["sd302"]["K"] = 1
    with pytest.raises(ValueError, match="devices"):
        _keys(dn.d2_corpus, bad)


def _auc(auc, same=None, cross=None, pairs=10, imp=0.3):
    return {"genuine_pairs": pairs, "auc": auc, "mean_impostor": imp,
            "same_device": None if same is None else {"pairs": 6, "auc": same},
            "cross_device": None if cross is None else {"pairs": 4, "auc": cross}}


LABEL = {"n_images": 100, "n_identities": 14, "models": {
    "fingervit": {"all": {**_auc(0.8935, 0.9513, 0.7127, 53100, imp=0.2428),
                          "same_device": {"pairs": 40242, "auc": 0.9513}, "cross_device": {"pairs": 12858, "auc": 0.7127}},
                  "per_source": {"casia_v5": _auc(0.8682, same=0.8682, imp=0.3633),
                                 "sd302": _auc(0.6458, cross=0.6458, imp=0.2617),
                                 "polyu_cross": _auc(0.9302, same=0.9302, imp=0.7058),
                                 "crossmatch": _auc(0.9814, same=0.9814, imp=0.635)}},
    "deepprint": {"all": {**_auc(0.7389, imp=0.3761), "same_device": {"pairs": 40242, "auc": 0.8123},
                          "cross_device": {"pairs": 12858, "auc": 0.5092}},
                  "per_source": {"casia_v5": _auc(0.6434, same=0.6434, imp=0.4375),
                                 "sd302": _auc(0.5304, cross=0.5304, imp=0.3279),
                                 "polyu_cross": _auc(0.8931, same=0.8931, imp=0.5132),
                                 "crossmatch": _auc(0.9306, same=0.9306, imp=0.4285)}}}}


def test_d2_label_check_keys():
    d = _keys(dn.d2_label_check, LABEL)
    assert d["d2.auc.d2_fingervit.all"] == "0.893" and d["d2.auc.d2_fingervit.all.pairs"] == "53{,}100"
    assert d["d2.auc.d2_fingervit.same"] == "0.951" and d["d2.auc.d2_fingervit.same.pairs"] == "40{,}242"
    assert d["d2.auc.d2_fingervit.cross"] == "0.713" and d["d2.auc.d2_fingervit.cross.pairs"] == "12{,}858"
    assert d["d2.auc.d2_deepprint.cross"] == "0.509"
    assert d["d2.auc.d2_fingervit.sd302.all"] == "0.646" and d["d2.auc.d2_fingervit.sd302.cross"] == "0.646"
    assert "d2.auc.d2_fingervit.sd302.same" not in d and "d2.auc.d2_deepprint.casia_v5.cross" not in d
    # mean impostor cosine, overall and per source
    assert d["d2.imp.d2_fingervit.all"] == "0.243" and d["d2.imp.d2_deepprint.all"] == "0.376"
    assert d["d2.imp.d2_fingervit.polyu_cross"] == "0.706" and d["d2.imp.d2_fingervit.crossmatch"] == "0.635"
    assert d["d2.imp.d2_fingervit.casia_v5"] == "0.363" and d["d2.imp.d2_deepprint.sd302"] == "0.328"


SECONDARY = {"n_images": 14863, "matched": 14614, "unmatched": 242, "ambiguous": 7, "true_fingers": 2330,
             "paths_unmapped": 240, "images_unreadable": 2,
             "per_source": {"fvc2000": 240, "sd301": 2760, "sd302": 11134}}


def _entry(label, rows, matched, base, large=None, covered=True):
    large = base if large is None else large
    models = {m: {"0.999": {"duplicate_rows": v[0], "comparable": v[1], "same_identity": v[2]},
                  "0.95": {"duplicate_rows": v[0] + 1, "comparable": v[1], "same_identity": v[2]}}
              for m, v in (("base", base), ("large", large))}
    return {"label": label, "rows": rows, "matched": matched, "models": models, "no_exact_duplicate_coverage": not covered}


MAPPING = {
    "thresholds": [0.999, 0.95], "models": ["base", "large"],
    "entries": [_entry("fvc2002/DB1", 80, 80, (3, 3, 3)), _entry("fvc2004/DB2", 80, 80, (6, 6, 6), (7, 7, 7)),
                _entry("sd302b+sd302d", 11141, 11134, (1520, 1511, 1511))],
    "unmapped": [{"label": "fvc2000/DB4", "rows": 80, "matched": 0}, {"label": "fvc2002/DB4", "rows": 80, "matched": 0}],
    "totals": {"rows": 14863, "matched": 14614, "models": {
        "base": {"0.999": {"duplicate_rows": 2054, "comparable": 2009, "same_identity": 2009},
                 "0.95": {"duplicate_rows": 2116, "comparable": 2033, "same_identity": 2026}},
        "large": {"0.999": {"duplicate_rows": 2052, "comparable": 2009, "same_identity": 2009},
                  "0.95": {"duplicate_rows": 2084, "comparable": 2027, "same_identity": 2020}}}},
}


def test_secondary_keys():
    d = _keys(dn.secondary, SECONDARY, MAPPING)
    assert d["sec.images"] == "14{,}863" and d["sec.matched"] == "14{,}614" and d["sec.unmatched"] == "242"
    assert d["sec.unmapped"] == "240" and d["sec.unreadable"] == "2" and d["sec.ambiguous"] == "7"
    assert d["sec.dropped"] == "249" and d["sec.fingers"] == "2{,}330" and d["sec.source.sd302.rows"] == "11{,}134"
    assert d["sec.check.rows"] == "14{,}863" and d["sec.check.matched"] == "14{,}614"
    assert d["sec.check.entries"] == "3" and d["sec.check.entriescovered"] == "3"
    assert d["sec.check.unmapped.entries"] == "2" and d["sec.check.unmapped.rows"] == "160"
    assert d["sec.check.d1_base.exact.comparable"] == "2{,}009" and d["sec.check.d1_base.exact.agree"] == "2{,}009"
    assert d["sec.check.d1_base.exact.dup"] == "2{,}054" and d["sec.check.d1_large.near.agree"] == "2{,}020"
    assert d["sec.check.d1_base.mincover"] == "3" and d["sec.check.d1_base.entriesbelowten"] == "2"
    assert d["sec.check.d1_large.entriesbelowten"] == "2"
    # comparable images whose two labellings disagree (comparable minus agreeing), per encoder and threshold
    assert d["sec.check.d1_base.exact.disagree"] == "0" and d["sec.check.d1_base.near.disagree"] == "7"
    assert d["sec.check.d1_large.near.disagree"] == "7"


def test_secondary_counts_entries_without_exact_duplicate_coverage():
    bad = {**MAPPING, "entries": [*MAPPING["entries"][:2], _entry("x", 10, 10, (0, 0, 0), covered=False)]}
    d = _keys(dn.secondary, SECONDARY, bad)
    assert d["sec.check.entries"] == "3" and d["sec.check.entriescovered"] == "2"
    assert d["sec.check.d1_base.mincover"] == "0"


def _threshold(name, value, images, fingers):
    return {"name": name, "value": value, "images": {"above": images, "of": 1000, "per_source": {}},
            "fingers": {"above": fingers, "of": 100,
                        "per_source": {"casia_v5": {"above": 1, "of": 2}, "polyu_cross": {"above": 1, "of": 1},
                                       "crossmatch": {"above": 0, "of": 1}}}}


D1_OVERLAP = {"d1_images": {"in_pool": 8260, "of": 10233}, "d1_fingers": {"worst_case": {"in_pool": 3339, "of": 3694}},
              "pool": {"files": 90557}}
D2_OVERLAP = {
    "pool": {"files": 51938},
    "tau_rule": {"tau": 0.9972475611070859, "d2_genuine_p99.9": 0.9944923402292157, "separable": True},
    "d2": {"by_threshold": [_threshold("positives p0.1", 0.9908, 810, 84), _threshold("tau", 0.9972475611070859, 775, 84),
                            _threshold("positives p5", 0.9987, 719, 83)]},
    "calibration": {
        "positives": {"quantiles": {"n": 2034}, "by_source": {"casia_v5": {}, "sd301": {}}},
        "pairs_at_or_above_thresholds": [
            {"name": "positives p0.1", "value": 0.9908, "genuine": {"at_or_above": 472, "n": 53100},
             "impostor": {"at_or_above": 0, "n": 1000000, "upper_95_per_pair": 2.9957277863525433e-06}},
            {"name": "tau", "value": 0.9972475611070859, "genuine": {"at_or_above": 2, "n": 53100},
             "impostor": {"at_or_above": 0, "n": 1000000, "upper_95_per_pair": 2.9957277863525433e-06}}]},
}


def test_overlap_d1_keys():
    d = _keys(dn.overlap_d1, D1_OVERLAP)
    assert d == {"ovl.d1.images.n": "8{,}260", "ovl.d1.images.of": "10{,}233", "ovl.d1.images.pct": "80.7",
                 "ovl.d1.fingers.n": "3{,}339", "ovl.d1.fingers.of": "3{,}694", "ovl.d1.fingers.pct": "90.4",
                 "ovl.d1.pool": "90{,}557"}


def test_overlap_d2_keys():
    d = _keys(dn.overlap_d2, D2_OVERLAP)
    assert d["ovl.d2.images.n"] == "775" and d["ovl.d2.images.of"] == "1{,}000" and d["ovl.d2.images.pct"] == "77.5"
    assert d["ovl.d2.fingers.n"] == "84" and d["ovl.d2.fingers.pct"] == "84.0"
    assert d["ovl.d2.images.pctmin"] == "71.9" and d["ovl.d2.images.pctmax"] == "81.0"
    assert d["ovl.d2.fingers.pctmin"] == "83.0" and d["ovl.d2.fingers.pctmax"] == "84.0"
    assert d["ovl.d2.pool"] == "51{,}938"
    assert d["ovl.tau"] == "0.9972" and d["ovl.genp999"] == "0.9945" and d["ovl.margin"] == "0.0028"
    assert d["ovl.positives"] == "2{,}034"
    # a zero count carries its bound in the same key; the bound alone is in .bound
    assert d["ovl.impostor.atorabove"] == "0 (CP95 $\\leq 3.0\\times10^{-6}$ per pair)"
    assert d["ovl.impostor.n"] == "1{,}000{,}000" and d["ovl.impostor.bound"] == "$3.0\\times10^{-6}$"
    assert d["ovl.genuine.atorabove"] == "2" and d["ovl.genuine.n"] == "53{,}100" and "ovl.genuine.bound" not in d
    assert d["ovl.nocalib"] == "2"   # polyu_cross and crossmatch have no calibration pairs


def test_overlap_d2_refuses_a_missing_tau_row():
    bad = json.loads(json.dumps(D2_OVERLAP))
    bad["d2"]["by_threshold"] = bad["d2"]["by_threshold"][:1]
    with pytest.raises(ValueError, match="tau"):
        _keys(dn.overlap_d2, bad)


def _stat(clean, mean, lo, hi, outside, bounded=None):
    s = {"clean": clean, "flagged_mean": mean, "flagged_p2.5": lo, "flagged_p97.5": hi,
         "clean_outside_flagged_range": outside}
    return s if bounded is None else {**s, "bounded_by_samples": bounded}


def _space(auc):
    return {"statistics": {"auc": _stat(auc, 0.7555, 0.7354, 0.7800, True), "eer": _stat(0.26761, 0.3153, 0.2912, 0.3319, True),
                           "mean_genuine": _stat(0.16514, 0.1636, 0.1504, 0.1825, False),
                           "mean_impostor": _stat(0.0121305, 0.0024, 0.0019, 0.0028, True),
                           "pr_all_images": _stat(195.3965, 235.45, 230.15, 241.92, True, False),
                           "pr_one_image_per_finger": _stat(145.849, 170.79, 167.20, 174.03, True, True)}}


def _stratum(clean, flagged, auc):
    return {"fingers": {"clean": clean, "flagged": flagged, "subsample_size": clean},
            "spaces": {"raw": _space(auc + 0.0017), "prekey": _space(auc)}}


CLEAN = {"settings": {"subsamples": 20, "impostor_pairs": 200000},
         "results": {"d1": {m: {"strata": {"all_sources": _stratum(355, 2640, 0.80462),
                                           "casia_v5": _stratum(340, 1307, 0.7943)}} for m in ("base", "large")},
                     "d2": {m: {"strata": {"all_sources": _stratum(507, 2633, 0.9229)}}
                            for m in ("fingervit", "deepprint")}}}


def test_clean_subset_keys():
    d = _keys(dn.clean_subset, CLEAN)
    k = "cs.prekey.d1_base.all_sources"
    assert d[f"{k}.auc.clean"] == "0.8046" and d[f"{k}.auc.flaggedmean"] == "0.7555"
    assert d[f"{k}.auc.flaggedlo"] == "0.7354" and d[f"{k}.auc.flaggedhi"] == "0.7800" and d[f"{k}.auc.outside"] == "yes"
    assert d[f"{k}.eer.clean"] == "0.2676" and d[f"{k}.gen.outside"] == "no" and d[f"{k}.imp.clean"] == "0.0121"
    assert d[f"{k}.prall.clean"] == "195.40" and d[f"{k}.prone.flaggedhi"] == "174.03"
    assert d[f"{k}.prone.bounded"] == "yes" and d[f"{k}.prall.bounded"] == "no" and f"{k}.auc.bounded" not in d
    assert d["cs.raw.d1_base.all_sources.auc.clean"] == "0.8063"
    assert d["cs.prekey.d2_deepprint.all_sources.auc.clean"] == "0.9229"
    assert d["cs.d1.all_sources.fingers.clean"] == "355" and d["cs.d1.all_sources.fingers.flagged"] == "2{,}640"
    assert d["cs.d1.casia_v5.fingers.flagged"] == "1{,}307" and d["cs.d2.all_sources.fingers.flagged"] == "2{,}633"
    assert d["cs.subsamples"] == "20" and d["cs.impostorpairs"] == "200{,}000"


def test_clean_subset_digits_keep_an_outside_value_apart_from_its_bound():
    # clean exceeds p97.5 by 7.5e-6: at four decimals both print 0.3680, so this statistic gets five
    near = json.loads(json.dumps(CLEAN))
    st = near["results"]["d2"]["fingervit"]["strata"]["all_sources"]["spaces"]["raw"]["statistics"]
    st["mean_impostor"] = _stat(0.3679872, 0.3627488, 0.3569392, 0.3679797, True)
    d = _keys(dn.clean_subset, near)
    k = "cs.raw.d2_fingervit.all_sources.imp"
    assert (d[f"{k}.clean"], d[f"{k}.flaggedhi"], d[f"{k}.outside"]) == ("0.36799", "0.36798", "yes")
    assert d[f"{k}.flaggedlo"] == "0.35694" and d[f"{k}.flaggedmean"] == "0.36275"   # one precision per statistic
    assert d["cs.raw.d2_fingervit.all_sources.auc.clean"] == "0.9246"                # the others keep four decimals
    assert d["cs.raw.d2_deepprint.all_sources.imp.clean"] == "0.0121"


def test_clean_subset_refuses_an_outside_flag_that_contradicts_the_values():
    bad = json.loads(json.dumps(CLEAN))
    bad["results"]["d1"]["base"]["strata"]["casia_v5"]["spaces"]["raw"]["statistics"]["mean_genuine"][
        "clean_outside_flagged_range"] = True   # clean 0.16514 lies inside [0.1504, 0.1825]
    with pytest.raises(ValueError, match="outside"):
        _keys(dn.clean_subset, bad)


COMPOSITION = {
    "by_family": [
        {"label": "FVC2000", "fingers_kept": 18, "rows_kept": 144}, {"label": "SD301a", "fingers_kept": 39, "rows_kept": 426},
        {"label": "SD302b", "fingers_kept": 421, "rows_kept": 1263}, {"label": "SD302d", "fingers_kept": 421, "rows_kept": 1058}],
    "by_source": [
        {"label": "FVC", "fingers_kept": 18, "rows_kept": 144}, {"label": "SD301", "fingers_kept": 39, "rows_kept": 426},
        {"label": "SD302", "fingers_kept": 421, "rows_kept": 2321}],
    "totals": {"rows": 3200, "dropped_by_relabel": 9, "dropped_outside_path_map": 0, "dropped_inside_path_map": 9,
               "rows_after_relabel": 3191, "fingers_after_relabel": 600, "rows_shared": 300, "fingers_shared": 122,
               "rows_kept": 2891, "fingers_kept": 478},
    "checks": {"shared_rows_vs_run_metadata": {"computed": 300, "reference": 300, "match": True}},
}


def test_secondary_composition_keys():
    d = _keys(dn.secondary_composition, COMPOSITION)
    assert d["sec.comp.rows.total"] == "3{,}200" and d["sec.comp.rows.dropped"] == "9"
    assert d["sec.comp.rows.droppedoutside"] == "0" and d["sec.comp.rows.droppedinside"] == "9"
    assert d["sec.comp.rows.afterrelabel"] == "3{,}191" and d["sec.comp.fingers.afterrelabel"] == "600"
    assert d["sec.comp.rows.shared"] == "300" and d["sec.comp.fingers.shared"] == "122"
    assert d["sec.comp.rows.kept"] == "2{,}891" and d["sec.comp.fingers.kept"] == "478"
    assert d["sec.comp.family.sd302b.fingers.kept"] == "421" and d["sec.comp.family.sd301a.rows.kept"] == "426"
    assert d["sec.comp.source.sd302.fingers.kept"] == "421" and d["sec.comp.source.sd302.rows.kept"] == "2{,}321"
    assert d["sec.comp.source.fvc.fingers.kept"] == "18"


@pytest.mark.parametrize("path, value, match", [
    (("totals", "rows_kept"), 2890, "add up"),                      # after relabel != shared + kept
    (("totals", "dropped_inside_path_map"), 8, "add up"),           # dropped != outside + inside
    (("by_source", 2, "fingers_kept"), 420, "per source"),           # sources do not add up to the kept fingers
    (("checks", "shared_rows_vs_run_metadata", "match"), False, "check"),
])
def test_secondary_composition_refuses_inconsistent_counts(path, value, match):
    bad = json.loads(json.dumps(COMPOSITION))
    node = bad
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        _keys(dn.secondary_composition, bad)


def test_clean_subset_refuses_encoders_of_one_corpus_with_different_fingers():
    bad = json.loads(json.dumps(CLEAN))
    bad["results"]["d2"]["deepprint"]["strata"]["all_sources"]["fingers"]["flagged"] = 2632
    with pytest.raises(ValueError, match="cs.d2.all_sources.fingers.flagged"):
        _keys(dn.clean_subset, bad)


ctl = _load("test_data_numbers_control", "tests/test_data_numbers_control.py")


def _write_aggregates(root: Path) -> Path:
    data = root / "results/data"
    for rel, obj in (("relabel_d1/summary.json", RELABEL), ("corpus_d2/extract_summary.json", EXTRACT),
                     ("corpus_d2/label_check.json", LABEL), ("relabel_secondary/summary.json", SECONDARY),
                     ("relabel_secondary/mapping_check.json", MAPPING),
                     ("relabel_secondary/composition.json", COMPOSITION), ("overlap/d1_vs_ext_train.json", D1_OVERLAP),
                     ("overlap/d2_vs_packaged_train.json", D2_OVERLAP), ("overlap/clean_subset_stats.json", CLEAN),
                     ("control/label_check_ctrlvit.json", ctl.LABEL_CTRL), ("control/training_set.json", ctl.TRAIN_SET),
                     ("control/training_summary.json", ctl.TRAIN_SUMMARY), ("control/overlap_audit.json", ctl.AUDIT),
                     ("control/subject_check.json", ctl.SUBJECT)):
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_text(json.dumps(obj))
    return data


# the run-metadata keys that make_numbers writes before the data keys, here matching the fixture aggregates
RUN_KEYS = {"corpus.d1.images": "10{,}233", "corpus.d1.fingers": "3{,}694", "corpus.d2.images": "100",
            "corpus.d2.fingers": "14", "sec.rows.shared": "300", "sec.rows.kept": "2{,}891"}


def test_make_numbers_cross_checks_the_data_keys_against_the_run_metadata(tmp_path):
    """collect_data refuses aggregates that disagree with the run metadata: the D1 relabel against the D1 corpus
    loaded by the runs, the D2 extraction against the D2 corpus, the secondary composition against the secondary rows."""
    _write_aggregates(tmp_path)
    c = mn.Collector()
    for k, v in RUN_KEYS.items():
        c.put(k, v)
    assert mn.collect_data(c, tmp_path) is True
    for key, wrong in (("corpus.d1.images", "10{,}232"), ("corpus.d1.fingers", "3{,}693"), ("corpus.d2.images", "99"),
                       ("corpus.d2.fingers", "15"), ("sec.rows.shared", "301"), ("sec.rows.kept", "2{,}890")):
        c = mn.Collector()
        for k, v in {**RUN_KEYS, key: wrong}.items():
            c.put(k, v)
        with pytest.raises(ValueError, match=key.replace(".", r"\.")):
            mn.collect_data(c, tmp_path)
    c = mn.Collector()   # a run-metadata key that is missing is an error as well
    for k, v in RUN_KEYS.items():
        if k != "corpus.d1.fingers":
            c.put(k, v)
    with pytest.raises(ValueError, match=r"corpus\.d1\.fingers"):
        mn.collect_data(c, tmp_path)


def test_every_cross_check_pair_is_refused_on_its_own(tmp_path):
    """Each CROSS_CHECKS pair fails alone: the data-side key is changed in an otherwise consistent key set, so the
    refusal names that pair and not the first pair of the tuple."""
    _write_aggregates(tmp_path)
    c = mn.Collector()
    for k, v in RUN_KEYS.items():
        c.put(k, v)
    assert mn.collect_data(c, tmp_path) is True
    mn.cross_check(c.d)   # consistent: no refusal
    assert len(mn.CROSS_CHECKS) >= 8
    for data_key, run_key in mn.CROSS_CHECKS:
        bad = {**c.d, data_key: c.d[data_key] + "0"}
        with pytest.raises(ValueError, match=f"{data_key.replace('.', r'[.]')} = .*{run_key.replace('.', r'[.]')}") as e:
            mn.cross_check(bad)
        assert str(e.value).count("cross-check failed") == 1


def test_collect_needs_every_aggregate(tmp_path):
    data = _write_aggregates(tmp_path)
    c = mn.Collector()
    dn.collect(c.put, tmp_path)
    assert c.d["relabel.d1.placed"] == "10{,}233" and c.d["ovl.d2.images.pct"] == "77.5"
    assert c.d["cs.prekey.d1_large.casia_v5.prone.bounded"] == "yes" and c.d["sec.check.entries"] == "3"
    assert c.d["sec.comp.fingers.kept"] == "478"
    # the ctrl-vit families: label check merged in, training record, audit, subject check
    assert c.d["d2.auc.d2_ctrlvit.all"] == "0.995" and c.d["control.train.fingers"] == "3{,}684"
    assert c.d["control.audit.fl.flagged"] == "3" and c.d["control.subj.casia.flagged"] == "8"
    for rel in ("overlap/clean_subset_stats.json", "control/overlap_audit.json", "control/subject_check.json",
                "control/label_check_ctrlvit.json", "control/training_set.json", "control/training_summary.json"):
        (data / rel).rename(data / (rel + ".gone"))
        with pytest.raises(FileNotFoundError):
            dn.collect(mn.Collector().put, tmp_path)
        (data / (rel + ".gone")).rename(data / rel)


def test_make_numbers_leaves_the_data_keys_out_without_results_data(tmp_path):
    c = mn.Collector()
    assert mn.collect_data(c, tmp_path) is False and c.d == {}
