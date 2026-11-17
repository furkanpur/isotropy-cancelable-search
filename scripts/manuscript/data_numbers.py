"""Data-provenance keys of manuscript/generated/numbers.tex, read from the committed aggregates in results/data/.

scripts/manuscript/make_numbers.py loads this module and merges its keys when results/data/ exists. The repository and
the public code artifact both hold results/data/, and every aggregate below is then required. Each family function
maps parsed JSON to keys through `put(key, value)` (the make_numbers Collector, which
checks the key syntax and refuses a second, different value for a key); nothing is recomputed from data. Counts that
must add up are checked before any key is written.

Aggregates:
    results/data/relabel_d1/summary.json               D1 identity labels re-derived by content matching
    results/data/corpus_d2/extract_summary.json          D2 composition and selection
    results/data/corpus_d2/label_check.json              D2 genuine-vs-impostor AUC per encoder
    results/data/relabel_secondary/summary.json        secondary-corpus relabel
    results/data/relabel_secondary/mapping_check.json  check of the secondary labels on the images shared with D1
    results/data/relabel_secondary/composition.json    secondary corpus after the relabel: shared with D1 and kept
    results/data/overlap/d1_vs_ext_train.json          D1 against candidate training pool A (exact content key)
    results/data/overlap/d2_vs_packaged_train.json     D2 against candidate training pool B (finger-vit near-duplicates)
    results/data/overlap/clean_subset_stats.json       clean-subset statistics (raw and pre-key spaces)
    results/data/control/label_check_ctrlvit.json      D2 genuine-vs-impostor AUC of ctrl-vit (merged with label_check.json)
    results/data/control/training_set.json             ctrl-vit training set: composition, split, disjointness from D2
    results/data/control/training_summary.json         ctrl-vit training record: epochs, selection, validation EER
    results/data/control/overlap_audit.json            training-set audit against D2: near-duplicates, finger-level
        statistic, characterisation, label-structure scan, sensitivity AUCs, subject overlap
    results/data/control/subject_check.json            casia_v5 subject stratum for three encoders, polyu block

Key scheme (one family per line; <src> is a source name as the aggregate records it, <enc> an experiment encoder name:
d1_base, d1_large, d2_fingervit, d2_deepprint, d2_ctrlvit):
    relabel.d1.<images|placed|unmatched|ambiguous|dropped|fingers>   images in the manifest, placed on a finger,
        unmatched, ambiguous, dropped (unmatched + ambiguous), fingers
    d2.<images|fingers|subjects|sources|devices>, d2.perfinger.<min|max>, d2.capture.<type>, d2.dpi.<dpi>   D2 corpus
    d2.devices.<recorded|unrecorded>                     device names as the compilation records them, and entries
        without a device record (each such source counts as one device); together they make d2.devices
    d2.source.<src>.<fingers|images>[.pct]               per source, with its share of D2
    d2.excluded.sources, d2.excluded.source.<fingers|rows>   excluded sources, their identity keys and manifest rows
    d2.excluded.firstcorpus.<fingers|fingersinfolds|rows>    fingers of D1 excluded (all, inside the folds) and rows
    d2.excluded.duplicate.<fingers|images>               fingers dropped because a compilation image of the finger,
        selected for D2 or not, equals a compilation image of a D1 finger, and their selected images
    d2.excluded.duplicate.viaunselected                  of those fingers, the ones found through unselected images
    d2.excluded.d1image.fingers                          fingers dropped by the separate check against D1's own images
    d2.auc.<enc>.<all|same|cross>[.pairs]                genuine AUC over all, same-device and cross-device pairs
    d2.auc.<enc>.<src>.<all|same|cross>                  the same per source, where the source has such pairs
    d2.imp.<enc>.<all|src>                               mean impostor cosine of the label check, overall and per
        source
    sec.<images|matched|unmatched|unmapped|unreadable|ambiguous|dropped|fingers>, sec.source.<src>.rows   secondary
        corpus relabel: rows, matched, unmatched (of which unmapped paths and unreadable files), ambiguous, dropped
        (unmatched + ambiguous), fingers; matched rows per source
    sec.check.<rows|matched|entries|entriescovered>, sec.check.unmapped.<entries|rows>   mapping check: rows, matched
        rows, path-map entries, entries with exact-duplicate coverage, rows outside every entry
    sec.check.<enc>.<exact|near>.<dup|comparable|agree|disagree>   duplicates of a D1 image, comparable (both maps
        assign a finger), agreeing, disagreeing (comparable minus agreeing); exact is the uncentred-cosine threshold
        0.999, near is 0.95
    sec.check.<enc>.<mincover|entriesbelowten>           smallest per-entry number of comparable exact duplicates, and
        the number of entries with fewer than ten
    sec.comp.rows.<total|dropped|droppedoutside|droppedinside>   composition: secondary-manifest rows, rows dropped by
        the relabel, of which outside every path-map entry and inside one
    sec.comp.<rows|fingers>.<afterrelabel|shared|kept>   union totals after the relabel, shared with D1, kept
    sec.comp.<family|source>.<label>.<fingers|rows>.kept   kept per family (fvc2000, ..., sd301a, sd302b, sd302d) and
        per source (fvc, sd301, sd302); sd302b and sd302d hold the same fingers, so family fingers do not add up
    ovl.d1.<images|fingers>.<n|of|pct>, ovl.d1.pool      D1 in candidate pool A (fingers: worst case), pool files
    ovl.d2.<images|fingers>.<n|of|pct>, ovl.d2.<images|fingers>.<pctmin|pctmax>, ovl.d2.pool   D2 in candidate pool B
        at tau, and the range of the share over the reported thresholds; pool files
    ovl.<tau|genp999|margin>, ovl.positives              tau, the D2 genuine 99.9th percentile, tau minus that
        percentile, the number of calibration positives
    ovl.<genuine|impostor>.<n|atorabove>[, .bound]       D2 genuine and impostor pairs at or above tau; when none
        reaches it, .atorabove reads `0 (CP95 $\\leq m.m\\times10^{e}$ per pair)` with the one-sided 95% Clopper-Pearson
        bound per pair (make_numbers' status rule), and .bound holds that bound alone
    ovl.nocalib                                          D2 sources without calibration pairs
    cs.<space>.<enc>.<stratum>.<stat>.<clean|flaggedmean|flaggedlo|flaggedhi|outside>   clean-subset statistics:
        clean value, mean and 2.5th / 97.5th percentiles over the flagged subsamples, clean outside that range;
        space raw or prekey; stratum all_sources or casia_v5; stat auc, eer, gen (mean genuine cosine), imp (mean
        impostor cosine), prall (participation ratio, all images), prone (one image per finger)
    cs.<space>.<enc>.<stratum>.<prall|prone>.bounded     the participation ratio is bounded by its sample count
    cs.<d1|d2>.<stratum>.fingers.<clean|flagged>, cs.<subsamples|impostorpairs>

    control.train.<images|fingers|sources>, control.train.source.<src>.<images|fingers>   ctrl-vit training set
    control.train.devices[.recorded|.unrecordedimages]   named devices of the training set (devices and devices.recorded
        are the same number, comparable only with d2.devices.recorded: the training aggregate has no devices by source,
        so d2.devices' rule of one device per source without a device record cannot be applied), and the images
        without a device record
    control.train.<val|fitted>.<images|fingers>          the validation fingers and the fingers the weights were fitted on
    control.train.perfinger.<min|median|max>, control.train.dropped.<partition|partitionrows|duplicates|unusable>
        fingers dropped by the partition (and their rows), duplicate rows collapsed, unusable images
    control.fit.<epochs|selectedepoch>, control.fit.valeer.<pretrained|selected>[.r4]   training record, EER with six
        decimals (.r4: four decimals, the precision of the paper)
    control.audit.<identityoverlap|contentoverlap>       D2 fingers and images in the training set (identity, content key)
    control.audit.nd.<p999|maxcos>                       near-duplicate threshold (D2 genuine 99.9th percentile under
        ctrl-vit) and the largest cosine of a D2 image to a training image
    control.audit.nd.<images|fingers>[.of|.bound]        D2 images and fingers reaching the threshold, of how many;
        when none does, the count renders as `0 (CP95 $\\leq x$ per image)` (per finger), as ovl.*.atorabove does, and
        .bound holds the one-sided 95% Clopper-Pearson bound x alone
    control.audit.nd.source.<src>.images                 the same per source (the bound per image of that source)
    control.audit.fl.<flagged|maxgap>, control.audit.fl.source.<src>.flagged   D2 fingers whose best training finger
        reaches their own genuine mean (finger-level statistic), the largest gap, per source; a zero count of a source
        renders with its bound per finger
    control.audit.fl.null.<flagged|maxgap>               the null-reference source (polyu_cross: no training subject);
        a zero renders with its bound per finger; the gap is negative and renders with a plain hyphen, as zp.*.oraclegap
    control.char.flagged.<n|samesubject|sameposition|positioncomparable>[.pct]   flagged fingers: the best training
        finger shares the subject key, the finger position. The denominator of the subject share is all flagged
        fingers, because the audit does not record how many of them have a training subject: the share among the
        eligible flagged fingers lies between this share (72%) and 100%
    control.char.all.<samesubject|eligible|sameposition|positioncomparable>[.pct]   the same among all D2 fingers, with
        the eligible denominators: fingers that have a training subject, fingers with a comparable position
    control.char.owngen.<flagged|all>.<median|mean>      own genuine mean cosine of the fingers
    control.audit.ls.<candidates|flagged>, control.audit.ls.nopos.<candidates|flagged>   label-structure scan, with and
        without the finger-position requirement; flagged = among the fingers of the finger-level statistic (a zero
        renders with its bound per candidate)
    control.sens.<a|b|c|d>.<images|fingers|removed>, control.sens.<a|b|c|d>.<enc>.auc   label-check AUC of the subsets
        (a as is, b without flagged fingers, c also without their subjects, d without label-structure candidates)
    control.sens.maxchange[.<enc>]                       largest absolute AUC change of b, c, d from a (per encoder, all)
    control.subj.<shared|d2subjects|trainsubjects>       subject keys shared with training, in D2, in the training set
    control.subj.d2subjects.bysource                     D2 (source, subject) pairs of the training-set build (C1)
        Three D2 subject counts exist and differ by definition: control.subj.d2subjects counts subject keys (the
        compilation's subject link; 718), control.subj.d2subjects.bysource counts (source, subject) pairs as C1's
        by_source_subject does (720), and d2.subjects counts the (source, subject) pairs of the extraction
        (extract_d2.py, 719); they are not meant to add up
    control.subj.d2subjects.unshared                     D2 subject keys with no training finger (d2 - shared); they
        hold the control.subj.d2fingers.unshared fingers, more than a person has, so a key is not a person everywhere
    control.subj.d2fingers.<shared|unshared>[.<src>]     D2 fingers whose subject has / has no training finger
    control.subj.casia.<clean|flagged|genuinepairs|images>   casia_v5 stratum of the subject check (clean = no training
        finger of the subject), clean pairs and images
    control.subj.casia.<space>.<enc>.<stat>.<field>      as cs.* for the subject check (raw and prekey space)
    control.polyu.<fingers|images>, control.polyu.<space>.<enc>.<eer|prall|prone>   descriptive statistics of polyu_cross

FORMATS: counts carry the thousands separator {,} of make_numbers.count; shares (.pct, .pctmin, .pctmax) in percent
with one decimal, computed from the counts (make_numbers.pct for a fraction, share_pct for n of a total); label-check
AUC and mean impostor cosine with three decimals; tau, the genuine 99.9th percentile and
the margin with four decimals (the margin is narrow; control.audit.nd.p999 has six); zero counts that state an
observed absence carry their bound in the key (zero_count), while a zero that is a census or a structural fact stays
bare: control.audit.<identityoverlap|contentoverlap> (every D2 key is compared with every training key, and the
training set is built disjoint), control.train.dropped.<duplicates|unusable> (counts of a deterministic preparation
step) and control.sens.a.removed (subset a is D2 as is); negative values (control.audit.fl.null.maxgap) use a plain
hyphen; the per-pair bound as $m.m\\times10^{e}$; clean-subset AUC, EER
(a fraction, not a percent) and cosines with four decimals, participation ratios with two, raised for a statistic
(its clean value, flagged mean and both percentiles together) until the rounded clean value lies outside the rounded
flagged range exactly when the outside flag says so; flags yes or no.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_numbers import count, pct  # noqa: E402  (one definition of the count and percent formats)

Put = Callable[[str, str], None]

AGGREGATES = {
    "relabel_d1": "relabel_d1/summary.json",
    "extract": "corpus_d2/extract_summary.json",
    "label_check": "corpus_d2/label_check.json",
    "secondary": "relabel_secondary/summary.json",
    "mapping": "relabel_secondary/mapping_check.json",
    "composition": "relabel_secondary/composition.json",
    "overlap_d1": "overlap/d1_vs_ext_train.json",
    "overlap_d2": "overlap/d2_vs_packaged_train.json",
    "clean_subset": "overlap/clean_subset_stats.json",
    "label_check_ctrl": "control/label_check_ctrlvit.json",
    "train_set": "control/training_set.json",
    "train_summary": "control/training_summary.json",
    "audit": "control/overlap_audit.json",
    "subject": "control/subject_check.json",
}
LABEL_CHECK_ENC = {"fingervit": "d2_fingervit", "deepprint": "d2_deepprint", "ctrlvit": "d2_ctrlvit"}
MAPPING_ENC = {"base": "d1_base", "large": "d1_large"}
THRESHOLDS = {"0.999": "exact", "0.95": "near"}
CS_STATS = {"auc": "auc", "eer": "eer", "mean_genuine": "gen", "mean_impostor": "imp", "pr_all_images": "prall",
            "pr_one_image_per_finger": "prone"}


# ---- formats (count and pct, a fraction in percent, come from make_numbers) ------------------------------------
def share_pct(n, of) -> str:
    """The share of n in a total `of`, in percent with one decimal, e.g. 71.2 (pct takes a fraction instead)."""
    return f"{100 * n / of:.1f}"


def dec(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


def sci_math(x: float) -> str:
    """A small rate as m.m\\times10^{e} (one decimal in the mantissa), for use inside math mode."""
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def sci(x: float) -> str:
    """A small rate as $m.m\\times10^{e}$."""
    return f"${sci_math(x)}$"


def zero_count(bound: float, unit: str = "pair") -> str:
    """A zero count with its one-sided 95% Clopper-Pearson bound per `unit` (make_numbers' status rule)."""
    return f"0 (CP95 $\\leq {sci_math(bound)}$ per {unit})"


def yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _relabel_counts(s: dict, name: str) -> None:
    if s["matched"] + s["unmatched"] + s["ambiguous"] != s["n_images"]:
        raise ValueError(f"{name}: matched + unmatched + ambiguous do not add up to the {s['n_images']} images")


# ---- families -----------------------------------------------------------------------------------------------
def relabel_d1(s: dict, put: Put) -> None:
    """D1 relabel summary: images in the manifest, placed on a finger, unmatched, ambiguous, dropped, fingers."""
    _relabel_counts(s, "relabel_d1")
    put("relabel.d1.images", count(s["n_images"]))
    put("relabel.d1.placed", count(s["matched"]))
    put("relabel.d1.unmatched", count(s["unmatched"]))
    put("relabel.d1.ambiguous", count(s["ambiguous"]))
    put("relabel.d1.dropped", count(s["unmatched"] + s["ambiguous"]))
    put("relabel.d1.fingers", count(s["true_fingers"]))


def d2_corpus(ex: dict, put: Put) -> None:
    """D2 composition (images, fingers, sources, capture, dpi, impressions per finger) and the selection exclusions."""
    comp, sel = ex["composition"], ex["selection_counts"]
    src = comp["per_source"]
    for field, total in (("identities", comp["n_identities"]), ("images", comp["n_images"])):
        if sum(v[field] for v in src.values()) != total:
            raise ValueError(f"extract_summary: {field} per source do not add up to {total}")
    names = [name for devices in comp["devices_by_source"].values() for name in devices]
    recorded, unrecorded = sum(1 for n in names if n), sum(1 for n in names if not n)
    if recorded + unrecorded != comp["n_source_devices"]:
        raise ValueError(f"extract_summary: {recorded} recorded devices and {unrecorded} sources without a device "
                         f"record do not add up to the {comp['n_source_devices']} devices")
    put("d2.images", count(comp["n_images"]))
    put("d2.fingers", count(comp["n_identities"]))
    put("d2.subjects", count(comp["n_subjects"]))
    put("d2.sources", count(len(src)))
    put("d2.devices", count(comp["n_source_devices"]))
    put("d2.devices.recorded", count(recorded))
    put("d2.devices.unrecorded", count(unrecorded))
    put("d2.perfinger.min", count(comp["impressions_per_finger_summary"]["min"]))
    put("d2.perfinger.max", count(comp["impressions_per_finger_summary"]["max"]))
    for kind, n in comp["capture"].items():
        put(f"d2.capture.{kind}", count(n))
    for dpi, n in comp["dpi"].items():
        put(f"d2.dpi.{dpi}", count(n))
    for name, v in src.items():
        put(f"d2.source.{name}.fingers", count(v["identities"]))
        put(f"d2.source.{name}.fingers.pct", share_pct(v["identities"], comp["n_identities"]))
        put(f"d2.source.{name}.images", count(v["images"]))
        put(f"d2.source.{name}.images.pct", share_pct(v["images"], comp["n_images"]))
    put("d2.excluded.sources", count(len(ex["compilation"]["excluded_sources"])))
    put("d2.excluded.source.fingers", count(sel["identities_excluded_source"]))
    put("d2.excluded.source.rows", count(sel["rows_excluded_source"]))
    put("d2.excluded.firstcorpus.fingers", count(sel["identities_excluded_d1"]))
    put("d2.excluded.firstcorpus.fingersinfolds", count(sel["identities_excluded_d1_in_folds"]))
    put("d2.excluded.firstcorpus.rows", count(sel["rows_excluded_d1"]))
    put("d2.excluded.duplicate.fingers", count(sel["identities_dropped_d1_map_duplicate"]))
    put("d2.excluded.duplicate.images", count(sel["images_dropped_d1_map_duplicate"]))
    put("d2.excluded.duplicate.viaunselected", count(sel["identities_dropped_d1_map_duplicate_via_unselected_images"]))
    put("d2.excluded.d1image.fingers", count(sel["identities_with_d1_image"]))


def d2_label_check(lc: dict, put: Put) -> None:
    """Genuine AUC per encoder over all, same-device and cross-device pairs, overall and per source, and the mean
    impostor cosine overall and per source."""
    for model, res in lc["models"].items():
        enc = LABEL_CHECK_ENC[model]
        a = res["all"]
        put(f"d2.auc.{enc}.all", dec(a["auc"], 3))
        put(f"d2.auc.{enc}.all.pairs", count(a["genuine_pairs"]))
        put(f"d2.imp.{enc}.all", dec(a["mean_impostor"], 3))
        for part, field in (("same", "same_device"), ("cross", "cross_device")):
            put(f"d2.auc.{enc}.{part}", dec(a[field]["auc"], 3))
            put(f"d2.auc.{enc}.{part}.pairs", count(a[field]["pairs"]))
        for name, s in res["per_source"].items():
            put(f"d2.auc.{enc}.{name}.all", dec(s["auc"], 3))
            put(f"d2.imp.{enc}.{name}", dec(s["mean_impostor"], 3))
            for part, field in (("same", "same_device"), ("cross", "cross_device")):
                if s[field] is not None:
                    put(f"d2.auc.{enc}.{name}.{part}", dec(s[field]["auc"], 3))


def secondary(s: dict, mc: dict, put: Put) -> None:
    """Secondary-corpus relabel summary and the mapping check against the images it shares with D1."""
    _relabel_counts(s, "relabel_secondary")
    put("sec.images", count(s["n_images"]))
    put("sec.matched", count(s["matched"]))
    put("sec.unmatched", count(s["unmatched"]))
    put("sec.unmapped", count(s["paths_unmapped"]))
    put("sec.unreadable", count(s["images_unreadable"]))
    put("sec.ambiguous", count(s["ambiguous"]))
    put("sec.dropped", count(s["unmatched"] + s["ambiguous"]))
    put("sec.fingers", count(s["true_fingers"]))
    for name, n in s["per_source"].items():
        put(f"sec.source.{name}.rows", count(n))
    entries = mc["entries"]
    put("sec.check.rows", count(mc["totals"]["rows"]))
    put("sec.check.matched", count(mc["totals"]["matched"]))
    put("sec.check.entries", count(len(entries)))
    put("sec.check.entriescovered", count(sum(not e["no_exact_duplicate_coverage"] for e in entries)))
    put("sec.check.unmapped.entries", count(len(mc["unmapped"])))
    put("sec.check.unmapped.rows", count(sum(u["rows"] for u in mc["unmapped"])))
    for model, enc in MAPPING_ENC.items():
        for thr, name in THRESHOLDS.items():
            t = mc["totals"]["models"][model][thr]
            put(f"sec.check.{enc}.{name}.dup", count(t["duplicate_rows"]))
            put(f"sec.check.{enc}.{name}.comparable", count(t["comparable"]))
            put(f"sec.check.{enc}.{name}.agree", count(t["same_identity"]))
            put(f"sec.check.{enc}.{name}.disagree", count(t["comparable"] - t["same_identity"]))
        cover = [e["models"][model]["0.999"]["comparable"] for e in entries]
        put(f"sec.check.{enc}.mincover", count(min(cover)))
        put(f"sec.check.{enc}.entriesbelowten", count(sum(c < 10 for c in cover)))


def _label(name: str) -> str:
    """A family or source label of the composition as a key part (FVC2000 -> fvc2000, SD301a -> sd301a)."""
    return name.lower()


def secondary_composition(comp: dict, put: Put) -> None:
    """Secondary corpus after the relabel: union totals of rows and fingers (after the relabel, shared with D1, kept),
    the rows dropped by the relabel, and the kept rows and fingers per family and per source."""
    t = comp["totals"]
    if (t["rows"] != t["dropped_by_relabel"] + t["rows_after_relabel"]
            or t["dropped_by_relabel"] != t["dropped_outside_path_map"] + t["dropped_inside_path_map"]
            or t["rows_after_relabel"] != t["rows_shared"] + t["rows_kept"]
            or t["fingers_after_relabel"] != t["fingers_shared"] + t["fingers_kept"]):
        raise ValueError("composition: the totals do not add up (rows, dropped, after the relabel, shared, kept)")
    for field in ("fingers_kept", "rows_kept"):   # sources hold disjoint fingers; families hold disjoint rows
        if sum(s[field] for s in comp["by_source"]) != t[field]:
            raise ValueError(f"composition: {field} per source do not add up to {t[field]}")
    if sum(f["rows_kept"] for f in comp["by_family"]) != t["rows_kept"]:
        raise ValueError(f"composition: rows_kept per family do not add up to {t['rows_kept']}")
    failed = [name for name, chk in comp["checks"].items() if not chk["match"]]
    if failed:
        raise ValueError(f"composition: check {', '.join(failed)} does not match")
    put("sec.comp.rows.total", count(t["rows"]))
    put("sec.comp.rows.dropped", count(t["dropped_by_relabel"]))
    put("sec.comp.rows.droppedoutside", count(t["dropped_outside_path_map"]))
    put("sec.comp.rows.droppedinside", count(t["dropped_inside_path_map"]))
    for part, field in (("afterrelabel", "after_relabel"), ("shared", "shared"), ("kept", "kept")):
        put(f"sec.comp.rows.{part}", count(t[f"rows_{field}"]))
        put(f"sec.comp.fingers.{part}", count(t[f"fingers_{field}"]))
    for group, rows in (("family", comp["by_family"]), ("source", comp["by_source"])):
        for x in rows:
            put(f"sec.comp.{group}.{_label(x['label'])}.fingers.kept", count(x["fingers_kept"]))
            put(f"sec.comp.{group}.{_label(x['label'])}.rows.kept", count(x["rows_kept"]))


def overlap_d1(o: dict, put: Put) -> None:
    """D1 images and fingers (worst case) with an exact copy in candidate training pool A."""
    for part, x in (("images", o["d1_images"]), ("fingers", o["d1_fingers"]["worst_case"])):
        put(f"ovl.d1.{part}.n", count(x["in_pool"]))
        put(f"ovl.d1.{part}.of", count(x["of"]))
        put(f"ovl.d1.{part}.pct", share_pct(x["in_pool"], x["of"]))
    put("ovl.d1.pool", count(o["pool"]["files"]))


def _at_tau(rows: list[dict], tau: float, what: str) -> dict:
    hits = [r for r in rows if r["value"] == tau]
    if len(hits) != 1:
        raise ValueError(f"d2_vs_packaged_train: {len(hits)} rows of {what} at tau = {tau}, expected 1")
    return hits[0]


def overlap_d2(o: dict, put: Put) -> None:
    """D2 images and fingers with a finger-vit near-duplicate in candidate training pool B at tau, the range over the
    reported thresholds, the calibration (tau, the D2 genuine 99.9th percentile, positives) and the pairs at tau."""
    rule, rows = o["tau_rule"], o["d2"]["by_threshold"]
    tau, p999 = rule["tau"], rule["d2_genuine_p99.9"]
    row = _at_tau(rows, tau, "d2.by_threshold")
    for part in ("images", "fingers"):
        x = row[part]
        put(f"ovl.d2.{part}.n", count(x["above"]))
        put(f"ovl.d2.{part}.of", count(x["of"]))
        put(f"ovl.d2.{part}.pct", share_pct(x["above"], x["of"]))
        shares = [r[part]["above"] / r[part]["of"] for r in rows]
        put(f"ovl.d2.{part}.pctmin", pct(min(shares)))
        put(f"ovl.d2.{part}.pctmax", pct(max(shares)))
    put("ovl.d2.pool", count(o["pool"]["files"]))
    put("ovl.tau", dec(tau, 4))
    put("ovl.genp999", dec(p999, 4))
    put("ovl.margin", dec(tau - p999, 4))
    cal = o["calibration"]
    put("ovl.positives", count(cal["positives"]["quantiles"]["n"]))
    pairs = _at_tau(cal["pairs_at_or_above_thresholds"], tau, "calibration.pairs_at_or_above_thresholds")
    for kind in ("genuine", "impostor"):
        p = pairs[kind]
        put(f"ovl.{kind}.n", count(p["n"]))
        if p["at_or_above"] == 0:   # a zero count carries its bound in the same key; .bound is the bound alone
            put(f"ovl.{kind}.atorabove", zero_count(p["upper_95_per_pair"]))
            put(f"ovl.{kind}.bound", sci(p["upper_95_per_pair"]))
        else:
            put(f"ovl.{kind}.atorabove", count(p["at_or_above"]))
    put("ovl.nocalib", count(len(set(row["fingers"]["per_source"]) - set(cal["positives"]["by_source"]))))


def cs_digits(v: dict, digits: int, name: str) -> int:
    """Decimals for the four values of one clean-subset statistic: `digits`, raised until the rounded clean value lies
    outside the rounded flagged range exactly when the aggregate's outside flag says so (rounding can otherwise make
    an outside clean value print equal to a percentile). The flag must agree with the unrounded values."""
    clean, lo, hi, flag = v["clean"], v["flagged_p2.5"], v["flagged_p97.5"], v["clean_outside_flagged_range"]
    if (clean < lo or clean > hi) != flag:
        raise ValueError(f"clean_subset_stats: {name}: the outside flag {flag} contradicts the values")
    for d in range(digits, 13):
        c, a, b = (float(dec(x, d)) for x in (clean, lo, hi))
        if (c < a or c > b) == flag:
            return d
    raise ValueError(f"clean_subset_stats: {name}: no precision separates the clean value from the flagged range")


def clean_subset(cs: dict, put: Put) -> None:
    """Clean-subset statistics in both spaces for every encoder and stratum, and the finger counts per stratum."""
    put("cs.subsamples", count(cs["settings"]["subsamples"]))
    put("cs.impostorpairs", count(cs["settings"]["impostor_pairs"]))
    for corpus, models in cs["results"].items():
        for model, res in models.items():
            enc = f"{corpus}_{model}"
            for stratum, st in res["strata"].items():
                put(f"cs.{corpus}.{stratum}.fingers.clean", count(st["fingers"]["clean"]))
                put(f"cs.{corpus}.{stratum}.fingers.flagged", count(st["fingers"]["flagged"]))
                for space, sp in st["spaces"].items():
                    for stat, short in CS_STATS.items():
                        k = f"cs.{space}.{enc}.{stratum}.{short}"
                        v = sp["statistics"][stat]
                        digits = cs_digits(v, 2 if short.startswith("pr") else 4, k)
                        put(f"{k}.clean", dec(v["clean"], digits))
                        put(f"{k}.flaggedmean", dec(v["flagged_mean"], digits))
                        put(f"{k}.flaggedlo", dec(v["flagged_p2.5"], digits))
                        put(f"{k}.flaggedhi", dec(v["flagged_p97.5"], digits))
                        put(f"{k}.outside", yes(v["clean_outside_flagged_range"]))
                        if "bounded_by_samples" in v:
                            put(f"{k}.bounded", yes(v["bounded_by_samples"]))


# ---- ctrl-vit control encoder (results/data/control/) ----------------------------------------------------------
def num(x) -> str:
    """An integral value as a count, any other as one decimal (a median of impression counts can be x.5)."""
    return count(int(x)) if float(x) == int(x) else f"{x:.1f}"


def cp95_zero(n: int) -> float:
    """One-sided 95% Clopper-Pearson upper bound of a rate with 0 events in n trials: 1 - 0.05**(1/n)."""
    return 1 - 0.05 ** (1 / n)


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def control_training(ts: dict, summary: dict, put: Put) -> None:
    """ctrl-vit training set (training_set.json) and training record (training_summary.json)."""
    src, sel = ts["by_source"], ts["selection"]
    for field, total in (("fingers", ts["fingers"]), ("images", ts["images"]), ("train_fingers", ts["train_fingers"]),
                         ("train_images", ts["train_images"]), ("val_fingers", ts["val_fingers"]),
                         ("val_images", ts["val_images"])):
        _check(sum(v[field] for v in src.values()) == total, f"training_set: {field} per source do not add up to {total}")
    _check(ts["train_fingers"] + ts["val_fingers"] == ts["fingers"]
           and ts["train_images"] + ts["val_images"] == ts["images"],
           "training_set: train and validation do not add up to the training set")
    _check(summary["fingers"]["train"] == ts["train_fingers"] and summary["fingers"]["val"] == ts["val_fingers"]
           and summary["images"] == {"train": ts["train_images"], "val": ts["val_images"]},
           "training_summary: fingers or images disagree with training_set")
    _check(sel["identities_found"] - sel["identities_dropped_partition"] == ts["fingers"] == sel["identities"],
           "training_set: fingers found minus fingers dropped by partition is not the training set (partition)")
    dev = ts["devices"]
    named = [name for name in dev["images_by_device"] if name]   # "" collects the images without a device record
    _check(dev["distinct"] == len(named) and sum(dev["images_by_device"].values()) == ts["images"],
           "training_set: devices do not add up")
    _check(ts["disjointness"]["identity_overlap"] == 0 and ts["disjointness"]["content_overlap"] == 0,
           "training_set: the training set overlaps D2 (identity or content overlap is not 0)")
    ep = summary["epochs"]
    _check(summary["epochs_run"] == summary["epochs_requested"] == len(ep) - 1,
           "training_summary: epochs run, requested and recorded differ")
    valid = [(r["val_eer"], r["epoch"]) for r in ep if not math.isnan(r["val_eer"])]
    best = min(valid)   # the lowest validation EER, ties to the earliest epoch; NaN is never selected
    chosen = ep[summary["selected_epoch"]]
    _check(chosen["epoch"] == summary["selected_epoch"] == best[1] and chosen["val_eer"] == summary["selected_val_eer"],
           f"training_summary: selected epoch {summary['selected_epoch']} is not the earliest epoch of the lowest "
           f"validation EER (epoch {best[1]})")
    _check(best[0] == summary["selected_val_eer"], "training_summary: the lowest validation EER is not the selected one")
    _check(summary["improved_over_pretrained"] and summary["selected_val_eer"] < summary["pretrained_val_eer"]
           and ep[0]["val_eer"] == summary["pretrained_val_eer"],
           "training_summary: the selected epoch is not better than the pretrained model")
    put("control.train.images", count(ts["images"]))
    put("control.train.fingers", count(ts["fingers"]))
    put("control.train.devices", count(dev["distinct"]))
    put("control.train.devices.recorded", count(dev["distinct"]))
    put("control.train.devices.unrecordedimages", count(dev["images_by_device"].get("", 0)))
    put("control.train.sources", count(len(src)))
    for name, v in src.items():
        put(f"control.train.source.{name}.images", count(v["images"]))
        put(f"control.train.source.{name}.fingers", count(v["fingers"]))
    put("control.train.val.images", count(ts["val_images"]))
    put("control.train.val.fingers", count(ts["val_fingers"]))
    put("control.train.fitted.images", count(ts["train_images"]))
    put("control.train.fitted.fingers", count(ts["train_fingers"]))
    for part in ("min", "median", "max"):
        put(f"control.train.perfinger.{part}", num(ts["images_per_finger"][part]))
    put("control.train.dropped.partition", count(sel["identities_dropped_partition"]))
    put("control.train.dropped.partitionrows", count(sel["rows_dropped_partition"]))
    put("control.train.dropped.duplicates", count(sel["duplicate_rows_collapsed"]))
    put("control.train.dropped.unusable", count(ts["probe"]["unusable"]))
    put("control.fit.epochs", count(summary["epochs_run"]))
    put("control.fit.selectedepoch", count(summary["selected_epoch"]))
    put("control.fit.valeer.pretrained", dec(summary["pretrained_val_eer"], 6))
    put("control.fit.valeer.selected", dec(summary["selected_val_eer"], 6))
    # the precision of the paper's EER fractions (four decimals); the six-decimal keys above stay for the record
    put("control.fit.valeer.pretrained.r4", dec(summary["pretrained_val_eer"], 4))
    put("control.fit.valeer.selected.r4", dec(summary["selected_val_eer"], 4))


def put_hits(put: Put, key: str, hits: int, n: int, unit: str, bound_key: bool = False) -> None:
    """Write `key` as a count of observed events among n units. Zero events render as zero_count (with the one-sided 95%
    Clopper-Pearson bound per unit, 1 - 0.05**(1/n)); `bound_key` adds `key.bound` holding the bound alone."""
    if hits:
        put(key, count(hits))
        return
    put(key, zero_count(cp95_zero(n), unit))
    if bound_key:
        put(f"{key}.bound", sci(cp95_zero(n)))


def control_audit(au: dict, put: Put) -> None:
    """Training-set audit against D2: overlap, near-duplicates, finger-level statistic, characterisation,
    label-structure scan, sensitivity AUCs and subject overlap."""
    _check(au["identity_overlap"] == 0 and au["content_overlap"] == 0,
           "overlap_audit: identity or content overlap with D2 is not 0")
    put("control.audit.identityoverlap", count(au["identity_overlap"]))
    put("control.audit.contentoverlap", count(au["content_overlap"]))
    nd = au["near_duplicates"]
    put("control.audit.nd.p999", dec(nd["threshold"], 6))
    put("control.audit.nd.maxcos", dec(nd["max_cosine"]["overall"]["max"], 3))
    for part, hits, n in (("images", nd["images_at_or_above"], nd["d2_images"]),
                          ("fingers", nd["fingers_at_or_above"], au["d2_fingers"])):
        _check(sum(hits["by_source"].values()) == hits["total"],
               f"overlap_audit: near-duplicate {part} per source do not add up to {hits['total']}")
        put_hits(put, f"control.audit.nd.{part}", hits["total"], n, part[:-1], bound_key=True)
        put(f"control.audit.nd.{part}.of", count(n))
    per_source = nd["max_cosine"]["by_source"]
    _check(sum(v["n"] for v in per_source.values()) == nd["d2_images"],
           "overlap_audit: D2 images per source do not add up")
    for name, n in nd["images_at_or_above"]["by_source"].items():
        put_hits(put, f"control.audit.nd.source.{name}.images", n, per_source[name]["n"], "image")
    fl = au["finger_level"]
    flagged = fl["overall"]["reaching_own_genuine_mean"]
    _check(sum(v["reaching_own_genuine_mean"] for v in fl["by_source"].values()) == flagged
           and sum(v["fingers"] for v in fl["by_source"].values()) == fl["overall"]["fingers"] == au["d2_fingers"]
           and au["characterisation"]["flagged"]["overall"]["n"] == flagged,
           "overlap_audit: flagged fingers per source, overall and characterised do not add up")
    put("control.audit.fl.flagged", count(flagged))
    put("control.audit.fl.maxgap", dec(fl["overall"]["max_minus_own_genuine_mean"]["max"], 4))
    for name, v in fl["by_source"].items():
        put_hits(put, f"control.audit.fl.source.{name}.flagged", v["reaching_own_genuine_mean"], v["fingers"], "finger")
    null = fl["by_source"][fl["null_reference_source"]]
    put_hits(put, "control.audit.fl.null.flagged", null["reaching_own_genuine_mean"], null["fingers"], "finger")
    put("control.audit.fl.null.maxgap", dec(null["max_minus_own_genuine_mean"]["max"], 4))   # negative: plain hyphen
    ls = au["label_structure"]
    put("control.audit.ls.candidates", count(ls["candidates"]["total"]))
    put_hits(put, "control.audit.ls.flagged", ls["candidates_among_flagged"], ls["candidates"]["total"], "candidate")
    put("control.audit.ls.nopos.candidates", count(ls["candidates_without_position_requirement"]["total"]))
    put_hits(put, "control.audit.ls.nopos.flagged", ls["candidates_without_position_requirement_among_flagged"],
             ls["candidates_without_position_requirement"]["total"], "candidate")
    subj = au["subjects"]
    shared, unshared = subj["d2_fingers_with_shared_subject"], subj["d2_fingers_without_shared_subject"]
    _check(shared["total"] + unshared["total"] == au["d2_fingers"]
           and sum(shared["by_source"].values()) == shared["total"]
           and sum(unshared["by_source"].values()) == unshared["total"],
           "overlap_audit: D2 fingers with and without a shared subject do not add up")
    _check(all(subj["agreement"].values()), "overlap_audit: the subject flags do not agree with the recomputed ones")
    ch = au["characterisation"]
    f, a = ch["flagged"]["overall"], ch["all"]["overall"]
    _check(a["same_subject_key"] <= shared["total"] and a["same_finger_position"] <= a["position_comparable"]
           and f["same_subject_key"] <= f["n"] and f["same_finger_position"] <= f["position_comparable"],
           "overlap_audit: characterisation counts exceed their eligible denominators (eligible)")
    put("control.char.flagged.n", count(f["n"]))
    put("control.char.flagged.samesubject", count(f["same_subject_key"]))
    put("control.char.flagged.samesubject.pct", share_pct(f["same_subject_key"], f["n"]))
    put("control.char.flagged.positioncomparable", count(f["position_comparable"]))
    put("control.char.flagged.sameposition", count(f["same_finger_position"]))
    put("control.char.flagged.sameposition.pct", share_pct(f["same_finger_position"], f["position_comparable"]))
    put("control.char.all.samesubject", count(a["same_subject_key"]))
    put("control.char.all.eligible", count(shared["total"]))
    put("control.char.all.samesubject.pct", share_pct(a["same_subject_key"], shared["total"]))
    put("control.char.all.positioncomparable", count(a["position_comparable"]))
    put("control.char.all.sameposition", count(a["same_finger_position"]))
    put("control.char.all.sameposition.pct", share_pct(a["same_finger_position"], a["position_comparable"]))
    for group, x in (("flagged", f), ("all", a)):
        put(f"control.char.owngen.{group}.median", dec(x["own_genuine_mean"]["quantiles"]["0.5"], 3))
        put(f"control.char.owngen.{group}.mean", dec(x["own_genuine_mean"]["mean"], 3))
    subsets = {name.split("_")[0]: v for name, v in au["sensitivity"]["subsets"].items()}
    _check(set(subsets) == {"a", "b", "c", "d"}, "overlap_audit: sensitivity subsets are not a, b, c, d")
    _check(subsets["a"]["removed_fingers"] == 0 and subsets["a"]["fingers"] == au["d2_fingers"],
           "overlap_audit: sensitivity subset (a) is not D2 as is")
    for sub, v in subsets.items():
        put(f"control.sens.{sub}.images", count(v["images"]))
        put(f"control.sens.{sub}.fingers", count(v["fingers"]))
        put(f"control.sens.{sub}.removed", count(v["removed_fingers"]))
        for model, e in v["encoders"].items():
            put(f"control.sens.{sub}.{LABEL_CHECK_ENC[model]}.auc", dec(e["auc"], 4))
    changes = {}
    for model, enc in LABEL_CHECK_ENC.items():
        base = subsets["a"]["encoders"][model]["auc"]
        changes[enc] = max(abs(subsets[s]["encoders"][model]["auc"] - base) for s in "bcd")
        put(f"control.sens.maxchange.{enc}", dec(changes[enc], 4))
    put("control.sens.maxchange", dec(max(changes.values()), 4))
    put("control.subj.shared", count(subj["c1_subjects"]["by_subject_key"]["shared"]))
    put("control.subj.d2subjects", count(subj["c1_subjects"]["by_subject_key"]["d2"]))
    put("control.subj.trainsubjects", count(subj["c1_subjects"]["by_subject_key"]["train"]))
    put("control.subj.d2subjects.bysource", count(subj["c1_subjects"]["by_source_subject"]["d2"]))
    keys = subj["c1_subjects"]["by_subject_key"]
    put("control.subj.d2subjects.unshared", count(keys["d2"] - keys["shared"]))   # keys with no training finger
    put("control.subj.d2fingers.shared", count(shared["total"]))
    put("control.subj.d2fingers.unshared", count(unshared["total"]))
    for name, n in shared["by_source"].items():
        put(f"control.subj.d2fingers.shared.{name}", count(n))
    for name, n in unshared["by_source"].items():
        put(f"control.subj.d2fingers.unshared.{name}", count(n))


def control_subject_check(sc: dict, put: Put) -> None:
    """casia_v5 subject stratum (clean against size-matched flagged subsamples) for each encoder, and the descriptive
    polyu_cross statistics (a source with no training subject, so no comparison set)."""
    st = sc["stratum"]["fingers"]
    put("control.subj.casia.clean", count(st["clean"]))
    put("control.subj.casia.flagged", count(st["flagged"]))
    sizes = set()
    for model, res in sc["encoders"].items():
        enc = LABEL_CHECK_ENC[model]
        c = res["counts"]
        for field, v in c.items():   # the flagged subsamples are size-matched to the clean stratum
            _check(v["flagged_min"] == v["flagged_max"] == v["clean"],
                   f"subject_check: {model}: the flagged subsamples are not size-matched in {field}")
        sizes.add((c["fingers"]["clean"], c["genuine_pairs"]["clean"], c["images"]["clean"]))
        _check(len(sizes) == 1 and c["fingers"]["clean"] == st["clean"],
               f"subject_check: {model}: the clean stratum differs from the other encoders' or the stratum's")
        for space, sp in res["spaces"].items():
            for stat, short in CS_STATS.items():
                k = f"control.subj.casia.{space}.{enc}.{short}"
                v = sp["statistics"][stat]
                digits = cs_digits(v, 2 if short.startswith("pr") else 4, k)
                put(f"{k}.clean", dec(v["clean"], digits))
                put(f"{k}.flaggedmean", dec(v["flagged_mean"], digits))
                put(f"{k}.flaggedlo", dec(v["flagged_p2.5"], digits))
                put(f"{k}.flaggedhi", dec(v["flagged_p97.5"], digits))
                put(f"{k}.outside", yes(v["clean_outside_flagged_range"]))
                if "bounded_by_samples" in v:
                    put(f"{k}.bounded", yes(v["bounded_by_samples"]))
    ((_, pairs, images),) = sizes
    put("control.subj.casia.genuinepairs", count(pairs))
    put("control.subj.casia.images", count(images))
    desc = sc["polyu_descriptive"]
    shape = {(r["fingers"], r["counts"]["images"]) for r in desc.values()}
    _check(len(shape) == 1 and {r["fingers"] for r in sc["polyu"].values()} == {next(iter(shape))[0]},
           "subject_check: polyu fingers or images differ between encoders or blocks")
    ((fingers, images),) = shape
    put("control.polyu.fingers", count(fingers))
    put("control.polyu.images", count(images))
    for model, res in desc.items():
        enc = LABEL_CHECK_ENC[model]
        for space, sp in res["spaces"].items():
            put(f"control.polyu.{space}.{enc}.eer", dec(sp["eer"], 4))
            put(f"control.polyu.{space}.{enc}.prall", dec(sp["pr_all_images"], 2))
            put(f"control.polyu.{space}.{enc}.prone", dec(sp["pr_one_image_per_finger"], 2))


def merge_label_checks(main: dict, ctrl: dict) -> dict:
    """The label check of the baseline encoders with the ctrl-vit label check; both must be on the same D2."""
    _check((main["n_images"], main["n_identities"]) == (ctrl["n_images"], ctrl["n_identities"]),
           "label checks: the ctrl-vit label check is not on the same corpus as label_check.json")
    both = set(main["models"]) & set(ctrl["models"])
    _check(not both, f"label checks: {', '.join(sorted(both))} already in label_check.json")
    return {**main, "models": {**main["models"], **ctrl["models"]}}


def control_checks(relabel: dict, extract: dict, train_set: dict, audit: dict, label_check: dict, subject: dict,
                   label_check_ctrl: dict) -> None:
    """Cross-checks between aggregates: the training fingers are D1's fingers minus the partition drops, the D2 fingers
    split into those with and without a training subject, the audit's baseline AUCs are the label checks' AUCs, and the
    polyu block of the subject check is the label check's polyu_cross AUC."""
    sel = train_set["selection"]
    _check(train_set["fingers"] == relabel["true_fingers"] - sel["identities_dropped_partition"],
           f"control.train.fingers {train_set['fingers']} is not relabel.d1.fingers {relabel['true_fingers']} minus "
           f"control.train.dropped.partition {sel['identities_dropped_partition']}")
    subj = audit["subjects"]
    d2 = extract["composition"]["n_identities"]
    _check(subj["d2_fingers_with_shared_subject"]["total"] + subj["d2_fingers_without_shared_subject"]["total"]
           == d2 == audit["d2_fingers"], f"control.subj.d2fingers.shared + unshared is not d2.fingers {d2}")
    stratum = subject["stratum"]
    in_stratum = stratum["fingers"]
    _check(in_stratum["clean"] == subj["d2_fingers_without_shared_subject"]["by_source"].get(stratum["name"], 0)
           and in_stratum["flagged"] == subj["d2_fingers_with_shared_subject"]["by_source"].get(stratum["name"], 0),
           f"subject_check stratum {stratum['name']} (clean {in_stratum['clean']}, flagged {in_stratum['flagged']}) "
           "differs from the audit's unshared and shared fingers of that source")
    null = audit["finger_level"]["null_reference_source"]
    _check(subj["d2_fingers_without_shared_subject"]["by_source"].get(null) == audit["finger_level"]["by_source"][null]["fingers"]
           and null not in subj["d2_fingers_with_shared_subject"]["by_source"],
           f"overlap_audit: not every {null} finger lacks a training subject")
    models = {**label_check["models"], **label_check_ctrl["models"]}
    base = audit["sensitivity"]["subsets"]["a_as_is"]["encoders"]
    for model, res in models.items():
        _check(round(res["all"]["auc"], 4) == base[model]["auc"],
               f"audit sensitivity (a) AUC of {model} differs from the label check")
    for model, p in subject["polyu"].items():
        ref = models[model]["per_source"]["polyu_cross"]["auc"]
        _check(round(p["auc"], 4) == round(ref, 4),
               f"subject_check polyu AUC of {model} {p['auc']} differs from the label check's polyu_cross AUC {ref}")


def collect(put: Put, root: Path) -> None:
    """All data-provenance keys from <root>/results/data/; every aggregate must exist."""
    data = Path(root) / "results/data"
    a = {name: json.loads((data / rel).read_text(encoding="utf-8")) for name, rel in AGGREGATES.items()}
    label_check = merge_label_checks(a["label_check"], a["label_check_ctrl"])
    relabel_d1(a["relabel_d1"], put)
    d2_corpus(a["extract"], put)
    d2_label_check(label_check, put)
    secondary(a["secondary"], a["mapping"], put)
    secondary_composition(a["composition"], put)
    overlap_d1(a["overlap_d1"], put)
    overlap_d2(a["overlap_d2"], put)
    clean_subset(a["clean_subset"], put)
    control_training(a["train_set"], a["train_summary"], put)
    control_audit(a["audit"], put)
    control_subject_check(a["subject"], put)
    control_checks(a["relabel_d1"], a["extract"], a["train_set"], a["audit"], a["label_check"], a["subject"],
                   a["label_check_ctrl"])
