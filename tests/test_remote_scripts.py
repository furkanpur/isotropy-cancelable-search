# tests/test_remote_scripts.py
"""The private docker runners in scripts/remote/ keep the container rules (skipped where they are absent)."""
import re
import subprocess
from pathlib import Path

import pytest

REMOTE = Path(__file__).resolve().parents[1] / "scripts" / "remote"
RUNNERS = {"run_extract.sh": True, "run_label_check.sh": False, "run_relabel_map.sh": False,
           "run_overlap.sh": True, "run_clean_subset.sh": False,   # name: uses the GPU
           "run_secondary_composition.sh": False, "run_control.sh": True}
# launchers that take an output location as an argument, and the variable that holds it on the data host
OUTPUT_ARG = {"run_extract.sh": "RUN", "run_label_check.sh": "OUTDIR", "run_relabel_map.sh": "OUT",
              "run_overlap.sh": "RUN", "run_clean_subset.sh": "RUN", "run_secondary_composition.sh": "OUT",
              "run_control.sh": "OUTDIR"}
GUARD = re.compile(r"# >>> output guard.*?# <<< output guard\n", re.S)

pytestmark = pytest.mark.skipif(not REMOTE.is_dir(), reason="the private runners are not part of this copy")


@pytest.mark.parametrize("name, gpu", sorted(RUNNERS.items()))
def test_runner_keeps_the_container_rules(name, gpu):
    path = REMOTE / name
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "--network none" in text and '--user "$(id -u):$(id -g)"' in text
    assert ":ro" in text and "CANVEC_HOST" in text
    assert ("--gpus all" in text) == gpu


def test_relabel_runner_has_a_path_mode_for_the_secondary_corpus():
    text = (REMOTE / "run_relabel_map.sh").read_text()
    assert "secondary)" in text and "--key path" in text and "--path-map" in text
    assert "secondary_relabel.csv" in text and "d1_relabel.csv" in text
    assert "invocation.txt" in text and "exit.txt" in text
    # the mapping check runs with the same path-map entries as the map build
    assert "secondary-check)" in text and "check_secondary_map.py" in text


def test_overlap_runner_uses_the_gpu_image_only_for_the_embedding_step():
    text = (REMOTE / "run_overlap.sh").read_text()
    cases = dict(re.findall(r"^  (\w+)\)\n(.*?);;", text, re.S | re.M))
    assert set(cases) == {"exact", "coarse", "embed", "measure", "deepprint"}
    assert {name for name, body in cases.items() if "--gpus all" in body} == {"embed"}
    assert "canvec/extract:gpu" in cases["embed"] and "IMG=canvec/exp:core" in text
    assert all("--expect-files" in cases[name] for name in ("exact", "coarse"))
    assert "invocation.txt" in text and "exit.txt" in text and "run_$STEP.sh" in text
    assert "/out/d2_vs_packaged_train.json" in cases["measure"]
    assert "--compilation /data/compilation" in cases["deepprint"] and "--flx-readme" in cases["deepprint"]
    assert "PYTHONPATH=/work/src" in text and "PYTHONPATH" not in cases["embed"]


def test_clean_subset_runner_runs_the_cpu_image_with_read_only_inputs():
    text = (REMOTE / "run_clean_subset.sh").read_text()
    assert "IMG=canvec/exp:core" in text and "canvec/extract:gpu" not in text and "--gpus" not in text
    assert "PYTHONPATH=/work/src" in text and "python scripts/data/clean_subset.py" in text
    assert "--json /out/clean_subset_stats.json" in text
    assert "/data/overlap/d1_vs_ext_train.json" in text and "/data/overlap/d2_vs_packaged_train.json" in text
    assert '-v "$COMPILATION/MANIFEST.csv:/data/compilation/MANIFEST.csv:ro"' in text and "--compilation-manifest /data/compilation/MANIFEST.csv" in text
    mounts = re.findall(r'-v "([^"]+)"', text)
    assert len(mounts) == 8 and [m for m in mounts if not m.endswith(":ro")] == ["$RUN:/out"]
    assert "invocation.txt" in text and "exit.txt" in text and "stdout.log" in text and "run_$STEP.sh" in text


def test_config_runner_passes_the_relabel_and_second_corpus_mounts():
    path = REMOTE / "run_config.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "CANVEC_RELABEL_ROOT" in text and "CANVEC_D2_ROOT" in text
    assert "exit.txt" in text and "invocation.txt" in text and "docker/run.sh" in text


def test_output_guard_is_identical_and_applied_in_every_launcher():
    blocks = {}
    for name, var in OUTPUT_ARG.items():
        text = (REMOTE / name).read_text()
        found = GUARD.findall(text)
        assert len(found) == 1, name
        blocks[name] = found[0]
        assert f'under_canvec_runs "${var}" ||' in text, name
        assert text.index(f'under_canvec_runs "${var}" ||') < text.index("docker run" if "docker run" in text
                                                                            else "CMD=(docker"), name
    assert len(set(blocks.values())) == 1


@pytest.mark.parametrize("rel, ok", [
    ("canvec-runs/run-1", True), ("canvec-runs/a/b", True),
    ("canvec-runs", False), ("canvec-runs/", False), ("canvec-runs/../x", False), ("canvec-runs/./x", False),
    ("canvec-runsX/y", False), ("other/canvec-runs/x", False), ("ABS:/tmp/x", False), ("REL:canvec-runs/x", False),
])
def test_output_guard_accepts_only_paths_below_canvec_runs(tmp_path, rel, ok):
    block = GUARD.findall((REMOTE / "run_relabel_map.sh").read_text())[0]
    if rel.startswith("ABS:"):
        path = rel[4:]
    elif rel.startswith("REL:"):
        path = rel[4:]
    else:
        path = f"{tmp_path}/{rel}"
    r = subprocess.run(["bash", "-c", block + 'under_canvec_runs "$1"', "_", path],
                       env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert (r.returncode == 0) == ok, (path, r.stderr)


def test_control_runner_fetches_weights_with_network_and_no_data_and_trains_offline_on_the_gpu():
    text = (REMOTE / "run_control.sh").read_text()
    cases = dict(re.findall(r"^  ([\w|-]+)\)\n(.*?);;", text, re.S | re.M))
    fetch, gpu = cases["fetch-weights"], cases["train|embed"]
    assert "--network none" not in fetch and "HF_HUB_OFFLINE=0" in fetch and "fetch_weights.py" in fetch
    assert [m for m in re.findall(r'-v "([^"]+)"', fetch) if not m.endswith(":ro")] == ["$OUTDIR:/out"]
    assert "/data" not in fetch
    assert "refusing to reuse" in gpu and gpu.index("refusing to reuse") < gpu.index("> \"$OUTDIR/stdout.log\"")
    assert "--gpus all" in gpu and "--network none" in gpu and "PYTHONPATH=/work/src:/opt/flx" in gpu
    assert [m for m in re.findall(r'-v "([^"]+)"', gpu) if not m.endswith(":ro")] == ["$OUTDIR:/out"]


def test_control_runner_audit_runs_offline_on_the_cpu_with_read_only_inputs():
    text = (REMOTE / "run_control.sh").read_text()
    cases = dict(re.findall(r"^  ([\w|-]+)\)\n(.*?);;", text, re.S | re.M))
    audit = cases["audit"]
    assert "control_audit.py" in audit and "--network none" in audit and "--gpus" not in audit
    assert "canvec/exp:core" in audit and "refusing to reuse" in audit
    assert [m for m in re.findall(r'-v "([^"]+)"', audit) if not m.endswith(":ro")] == ["$OUTDIR:/out"]


def test_control_runner_subject_check_runs_offline_on_the_cpu_with_read_only_inputs():
    text = (REMOTE / "run_control.sh").read_text()
    cases = dict(re.findall(r"^  ([\w|-]+)\)\n(.*?);;", text, re.S | re.M))
    subject = cases["subject"]
    assert "control_subject_check.py" in subject and "--network none" in subject and "--gpus" not in subject
    assert "canvec/exp:core" in subject and "refusing to reuse" in subject
    assert "/data/d2_ctrl:ro" in subject and "/data/d2_base:ro" in subject and "d2_subject_flags.parquet" in subject
    assert [m for m in re.findall(r'-v "([^"]+)"', subject) if not m.endswith(":ro")] == ["$OUTDIR:/out"]
    assert "CANVEC_IMAGE=" in subject and "CANVEC_IMAGE_ID=" in subject


def test_label_check_runner_writes_a_provenance_sidecar_and_keeps_output_apart_from_the_corpus():
    text = (REMOTE / "run_label_check.sh").read_text()
    assert '.provenance.json' in text and "docker image inspect" in text and "platform.python_version()" in text
    assert "refusing an output directory equal to the corpus directory" in text
    sidecar = text[text.index("P_IMAGE="):]
    assert "--network none" in text[text.index("-e P_IMAGE=") - 200:text.index("-e P_IMAGE=")] and "-v " not in sidecar
    audit = dict(re.findall(r"^  ([\w|-]+)\)\n(.*?);;", (REMOTE / "run_control.sh").read_text(), re.S | re.M))["audit"]
    assert "CANVEC_IMAGE=" in audit and "CANVEC_IMAGE_ID=" in audit
