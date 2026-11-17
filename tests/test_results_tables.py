"""Results record generator (scripts/figures/results_tables.py): data-source labels, scale and corpus tables."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/figures"))
_spec = importlib.util.spec_from_file_location("results_tables_r", ROOT / "scripts/figures/results_tables.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


def withheld_in(text: str) -> list:
    return []


D1 = "SIN 2026 image set, identity labels re-derived from each source's native naming by content matching"
D2 = ("development folds of an internal compilation with labels parsed from each source's native naming, "
      "finger-disjoint from D1")
SECONDARY = "secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed"


def _cells(table: str) -> list[dict]:
    """Rows of a Markdown table as {header: cell} dicts."""
    lines = table.splitlines()
    head = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    return [dict(zip(head, (c.strip() for c in line.strip().strip("|").split("|")))) for line in lines[2:]]


def test_dataset_labels_describe_the_corrected_corpora():
    assert D1 in rt.DATASET_LABEL["d1_base"] and D1 in rt.DATASET_LABEL["d1_large"]
    assert D2 in rt.DATASET_LABEL["d2_fingervit"] and D2 in rt.DATASET_LABEL["d2_deepprint"]
    assert SECONDARY in rt.SCALE_LABELS["scale_d1_base_secondary"]
    assert "SYNTHETIC" in rt.SCALE_LABELS["scale_d1_base_synthetic"]
    labels = [*rt.DATASET_LABEL.values(), *rt.SCALE_LABELS.values()]
    assert all("(real)" in lab for lab in labels if "SYNTHETIC" not in lab)
    assert not any(withheld_in(lab) for lab in labels)


def test_ctrl_vit_is_the_fifth_encoder_with_its_own_data_source_label():
    assert rt.BASE_ORDER == ["d1_base", "d1_large", "d2_fingervit", "d2_deepprint", "d2_ctrlvit"]
    assert rt.ENCODER_ORDER[4] == "d2_ctrlvit" and rt.ENCODER_ORDER[-1] == "d2_ctrlvit_fmr1"
    label = rt.DATASET_LABEL["d2_ctrlvit"]
    assert D2 in label and "(real)" in label
    assert "ctrl-vit: a control ViT trained by us on the compilation images of the D1 fingers" in label
    assert not withheld_in(label)


def test_encoder_names_agree_in_figures_and_tables():
    import isotropy_figures as fig
    mt = importlib.util.spec_from_file_location("make_tables_r", ROOT / "scripts/manuscript/make_tables.py")
    mod = importlib.util.module_from_spec(mt)
    mt.loader.exec_module(mod)
    assert fig.ENC_LAB["d2_ctrlvit"] == mod.ENC_LAB["d2_ctrlvit"] == "ctrl-vit"
    assert mod.ENCS == rt.BASE_ORDER and mod.DATA["d2_ctrlvit"] == "D2"


def test_figures_use_the_table_order_and_one_colour_per_encoder(tmp_path, monkeypatch):
    import json

    import isotropy_figures as fig
    assert fig.ENC_ORDER == rt.BASE_ORDER and fig.ENC_COLOR["d2_ctrlvit"] == "C4"
    assert len(set(fig.ENC_COLOR.values())) == len(rt.BASE_ORDER)
    assert fig.enc_rank("d2_ctrlvit") > fig.enc_rank("d2_deepprint") and fig.enc_rank("zz") > fig.enc_rank("d2_ctrlvit")
    # fig_spectra draws the same figure whatever the order of the run directories (synthetic fixture spectra)
    runs = []
    for i, name in enumerate(["d2_ctrlvit", "d1_base", "d2_fingervit"]):
        d = tmp_path / f"run{i}"
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"config": {"name": name}}))
        (d / "spectra.json").write_text(json.dumps({"raw_centered_normalized": {
            "cumvar_first256": [0.3 + 0.1 * i, 0.6 + 0.1 * i, 0.9], "participation_ratio": 10.0 + i}}))
        runs.append(d)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    outs = []
    for k, order in enumerate((runs, runs[::-1])):
        out = tmp_path / f"out{k}"
        out.mkdir()
        fig.fig_spectra(order, out)
        outs.append((out / "fig_spectra.pdf").read_bytes())
    assert outs[0] == outs[1]


def _scale_rows(pairs, schemes=("lrp_full", "iom_256"), efs=(64, 128)):
    return [{"scheme": s, "m": 128, "k": 128, "n_distractors": used, "n_distractors_requested": req,
             "gallery": 1200 + used, "ef_search": ef, "rank1_exact": 0.1, "rank1_hnsw": 0.1, "recall10_vs_exact": 1.0,
             "recall10_tie_aware": 1.0, "latency_ms_per_query": 0.01, "build_s": 0.1, "index_serialized_mb": 1.0}
            for s in schemes for req, used in pairs for ef in efs]


def _scale_run(tmp_path, name, meta, pairs):
    d = tmp_path / f"{name}-20990101-000000"
    d.mkdir()
    pd.DataFrame(_scale_rows(pairs)).to_csv(d / "scale.csv", index=False)
    return {"dir": d, "spec": None, "meta": {"config": {"name": name}, "distractor_source": "a:b:dedup<0.95",
                                             "cpu_model": "cpu", "omp_num_threads": "2", **meta}}


SECONDARY_META = {"distractors_loaded": 14614, "secondary_removed_shared_identity": 11387,
                  "distractors_available_before": 3000, "distractors_removed": 2, "distractors_available_after": 2998,
                  "distractor_sizes_requested_to_used": {"1000": 1000, "3000": 2998}}
SYNTHETIC_META = {"distractors_loaded": 100, "secondary_removed_shared_identity": None,
                  "distractors_available_before": 100, "distractors_removed": 0, "distractors_available_after": 100}


def test_scale_meta_table_shows_loaded_rows_and_both_removals(tmp_path):
    sec = _scale_run(tmp_path, "scale_d1_base_secondary", SECONDARY_META, [(1000, 1000), (3000, 2998)])
    syn = _scale_run(tmp_path, "scale_d1_base_synthetic", SYNTHETIC_META, [(10, 10), (100, 100)])
    meta, _ = rt.t_scale([sec, syn])
    s, y = _cells(meta)
    assert s["rows loaded (after the relabel step, if any)"] == "14614"
    assert s["removed: fingers shared with D1"] == "11387"
    assert s["drawn for the largest size"] == "3000"
    assert s["removed: near-duplicates of a D1 row (cos >= 0.95)"] == "2"
    assert s["available after the near-duplicate filter"] == "2998"
    assert s["requested -> used"] == "1000 -> 1000, 3000 -> 2998"
    # the shared-identity removal stands next to the near-duplicate removal
    head = list(s)
    assert head.index("removed: near-duplicates of a D1 row (cos >= 0.95)") \
        - head.index("removed: fingers shared with D1") == 2
    # no exclusion for synthetic distractors; the size mapping then comes from scale.csv alone
    assert y["removed: fingers shared with D1"] == "n/a (no exclusion)"
    assert y["requested -> used"] == "10 -> 10, 100 -> 100"


def test_scale_meta_table_refuses_a_mapping_that_contradicts_scale_csv(tmp_path):
    bad = {**SECONDARY_META, "distractor_sizes_requested_to_used": {"1000": 1000, "3000": 3000}}
    run = _scale_run(tmp_path, "scale_d1_base_secondary", bad, [(1000, 1000), (3000, 2998)])
    with pytest.raises(ValueError, match="requested"):
        rt.t_scale([run])


def test_size_map_accepts_capped_sizes_that_collapse_onto_one_size_used(tmp_path):
    # the runner runs each size used once, labelled with the largest request that maps to it (5000 and 13000 -> 3225)
    meta = {**SECONDARY_META, "distractor_sizes_requested_to_used": {"1000": 1000, "5000": 3225, "13000": 3225}}
    run = _scale_run(tmp_path, "scale_d1_base_secondary", meta, [(1000, 1000), (13000, 3225)])
    assert rt.size_map(run, pd.read_csv(run["dir"] / "scale.csv")) == {1000: 1000, 5000: 3225, 13000: 3225}
    (s,) = _cells(rt.t_scale([run])[0])
    assert s["requested -> used"] == "1000 -> 1000, 5000 -> 3225, 13000 -> 3225"
    (line,) = rt.capped_sizes([run])
    assert "5000 -> 3225, 13000 -> 3225" in line and "ran once" in line


def test_size_map_refuses_used_sizes_that_scale_csv_does_not_have(tmp_path):
    # the recorded mapping names a size used (2000) for which scale.csv has no row
    meta = {**SECONDARY_META, "distractor_sizes_requested_to_used": {"1000": 1000, "2000": 2000, "3000": 2998}}
    run = _scale_run(tmp_path, "scale_d1_base_secondary", meta, [(1000, 1000), (3000, 2998)])
    with pytest.raises(ValueError, match="disagree"):
        rt.size_map(run, pd.read_csv(run["dir"] / "scale.csv"))


def test_started_utc_is_required_in_the_provenance_table_and_the_title_date(tmp_path):
    run = _scale_run(tmp_path, "scale_d1_base_secondary",
                     {**SECONDARY_META, "git_sha": "a" * 40, "runtime_s": 7.0}, [(1000, 1000)])
    with pytest.raises(KeyError, match="started_utc"):
        rt.t_provenance([], [run])
    with pytest.raises(KeyError, match="started_utc"):
        rt.results_date([run])
    run["meta"]["started_utc"] = "2026-11-14T18:39:21+00:00"
    assert "2026-11-14T18:39:21+00:00" in rt.t_provenance([], [run]) and rt.results_date([run]) == "2026-11-14"


def _iso_run(name, *, shifted=True):
    split = {str(s): {"dev_identities": 10, "tgt_identities": 12, "aux_identities": 15, "dev_images": 20 + i,
                      "tgt_images": 40 - i, "aux_images": 41 + 2 * i} for i, s in enumerate((101, 102, 103))}
    meta = {"config": {"name": name}, "data_source": f"sin:primary:{name}",
            "corpus_counts": {"images": 102, "identities": 37, "identities_ge_min_tgt_images": 30},
            "split_counts": split, "aux_sizes": {"same_dist": 41}}
    if shifted:
        meta.update({"aux_sizes": {"same_dist": 41, "shifted": 33}, "shifted_aux_removed": 2,
                     "secondary_removed_shared_identity": 114, "shifted_aux_source": "sin:secondary:x:dedup<0.95"})
    return {"dir": Path(f"isotropy_{name}-20990101-000000"), "spec": None, "meta": meta}


def test_corpus_table_puts_the_shared_identity_removal_next_to_the_near_duplicate_removal():
    rows = _cells(rt.t_corpus([_iso_run("d1_base"), _iso_run("d2_fingervit", shifted=False)]))
    d1, d2 = rows
    assert d1["images"] == "102" and d1["fingers (with >= min_tgt_images images)"] == "37 (30)"
    assert d1["split fingers DEV/TGT/AUX"] == "10/12/15"
    assert d1["split images DEV/TGT/AUX (min-max over seeds)"] == "20-22 / 38-40 / 41-45"
    assert d1["shifted AUX rows"] == "33"
    assert d1["shifted AUX removed: fingers shared with D1"] == "114"
    assert d1["shifted AUX removed: near-duplicates (cos >= 0.95)"] == "2"
    head = list(d1)
    assert head.index("shifted AUX removed: near-duplicates (cos >= 0.95)") \
        == head.index("shifted AUX removed: fingers shared with D1") + 1
    assert d2["shifted AUX rows"] == d2["shifted AUX removed: fingers shared with D1"] == "n/a"


def test_corpus_table_refuses_split_identity_counts_that_vary_over_seeds():
    run = _iso_run("d1_base")
    run["meta"]["split_counts"]["102"]["aux_identities"] = 16
    with pytest.raises(ValueError, match="aux"):
        rt.t_corpus([run])


def test_results_date_is_the_latest_run_start():
    runs = [{"meta": {"started_utc": "2026-11-14T16:46:20+00:00"}},
            {"meta": {"started_utc": "2026-11-13T09:04:54+00:00"}}]
    assert rt.results_date(runs) == "2026-11-14"


def test_committed_results_record_uses_the_corrected_labels():
    md = (ROOT / "results/isotropy/RESULTS.md").read_text()
    for text in (D1, D2, SECONDARY, "SYNTHETIC"):
        assert text in md, text
    assert "test split; see the paper" not in md and "SIN-2026 primary corpus" not in md
    # D2 spans all development folds, and "second corpus" means D2 elsewhere, not the secondary corpus
    assert "identity-disjoint fold" not in md and "second corpus" not in md
    assert "the secondary corpus, secondary corpus" not in md and "(secondary corpus (" not in md
