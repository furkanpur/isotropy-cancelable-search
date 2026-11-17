# tests/test_control_training.py
"""Training library (ArcFace head, layer-wise parameter groups, schedule) and the ctrl-vit train/embed script.
All images and backbones here are synthetic stand-ins; no real biometric data is used."""
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from ijiss_btp.train.arcface import ArcFaceHead  # noqa: E402
from ijiss_btp.train.optim import best_epoch, vit_param_groups, warmup_cosine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ---- ArcFace -------------------------------------------------------------------------------------------------
def test_arcface_logits_shape_and_values_on_a_2d_example():
    head = ArcFaceHead(2, 2, s=30.0, m=0.3)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    theta = 0.4
    emb = torch.tensor([[math.cos(theta), math.sin(theta)]])
    out = head(emb, torch.tensor([0]))
    assert out.shape == (1, 2)
    assert out[0, 0].item() == pytest.approx(30.0 * math.cos(theta + 0.3), abs=1e-4)
    assert out[0, 1].item() == pytest.approx(30.0 * math.sin(theta), abs=1e-4)


def test_arcface_uses_the_linear_fallback_beyond_pi_minus_m():
    head = ArcFaceHead(2, 2, s=30.0, m=0.3)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    theta = math.pi - 0.1   # cos < cos(pi - m)
    emb = torch.tensor([[math.cos(theta), math.sin(theta)]])
    out = head(emb, torch.tensor([0]))
    cos = math.cos(theta)
    assert out[0, 0].item() == pytest.approx(30.0 * (cos - math.sin(math.pi - 0.3) * 0.3), abs=1e-3)


def test_arcface_gradients_reach_the_weights_and_the_embedding():
    head = ArcFaceHead(4, 3)
    emb = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1])
    torch.nn.functional.cross_entropy(head(emb, labels), labels).backward()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0 and emb.grad is not None


# ---- parameter groups and schedule ----------------------------------------------------------------------------
SHAPES = {"embeddings.cls_token": (1, 1, 3), "embeddings.patch_embeddings.projection.weight": (3, 3, 2, 2),
          "embeddings.patch_embeddings.projection.bias": (3,), "embeddings.position_embeddings": (1, 5, 3),
          "encoder.layer.0.attention.attention.query.weight": (3, 3), "encoder.layer.11.mlp.fc2.bias": (3,),
          "encoder.layer.11.mlp.fc2.weight": (3, 3), "layernorm.weight": (3,),
          "embeddings.mask_token": (1, 3), "encoder.layer.0.layer_scale1.lambda1": (3,)}


def _groups():
    named = [(n, torch.nn.Parameter(torch.zeros(*sh))) for n, sh in SHAPES.items()]
    return named, vit_param_groups(named, num_layers=12, lr=2e-5, layer_decay=0.75, weight_decay=0.05)


def _find(named, groups, name):
    p = dict(named)[name]
    return next(g for g in groups if any(q is p for q in g["params"]))


def test_param_group_learning_rate_multipliers():
    named, groups = _groups()
    for name, layer_id in [("embeddings.cls_token", 0), ("embeddings.patch_embeddings.projection.weight", 0),
                           ("encoder.layer.0.attention.attention.query.weight", 1),
                           ("encoder.layer.11.mlp.fc2.weight", 12), ("layernorm.weight", 13)]:
        g = _find(named, groups, name)
        assert g["lr"] == pytest.approx(2e-5 * 0.75 ** (13 - layer_id))
        assert g["base_lr"] == g["lr"] and "weight_decay" in g


def test_param_group_weight_decay_assignment():
    named, groups = _groups()
    wd = {n: _find(named, groups, n)["weight_decay"] for n, _ in named}
    for n in ("embeddings.patch_embeddings.projection.weight", "encoder.layer.0.attention.attention.query.weight",
              "encoder.layer.11.mlp.fc2.weight"):
        assert wd[n] == 0.05, n
    for n in ("embeddings.cls_token", "embeddings.position_embeddings", "embeddings.patch_embeddings.projection.bias",
              "encoder.layer.11.mlp.fc2.bias", "layernorm.weight", "embeddings.mask_token",
              "encoder.layer.0.layer_scale1.lambda1"):
        assert wd[n] == 0.0, n
    assert sum(len(g["params"]) for g in groups) == len(SHAPES)


def test_param_groups_reject_unknown_names():
    with pytest.raises(ValueError):
        vit_param_groups([("pooler.dense.weight", torch.nn.Parameter(torch.zeros(2, 2)))], 12, 1e-3, 0.75, 0.05)


def test_warmup_cosine_values():
    assert warmup_cosine(0, 100, 10) == 0.0
    assert warmup_cosine(5, 100, 10) == pytest.approx(0.5)
    assert warmup_cosine(10, 100, 10) == pytest.approx(1.0)
    assert warmup_cosine(55, 100, 10) == pytest.approx(0.5)
    assert warmup_cosine(100, 100, 10) == pytest.approx(0.0, abs=1e-12)


def test_best_epoch_ties_go_to_the_earliest_and_nan_is_skipped():
    assert best_epoch([0.3, 0.1, 0.1, 0.2]) == 1
    assert best_epoch([float("nan"), 0.4, 0.2, float("nan")]) == 2
    with pytest.raises(ValueError):
        best_epoch([float("nan"), float("inf")])


# ---- script ---------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tc():
    spec = importlib.util.spec_from_file_location("train_control", ROOT / "scripts" / "train" / "train_control.py")
    mod = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "train_control", mod)
        spec.loader.exec_module(mod)
        yield mod


class FakeBackbone(torch.nn.Module):
    """Synthetic stand-in for Dinov2Model: parameter names follow the Dinov2 layout.
    nan: "always" (NaN output), "train" (NaN only in train mode) or None."""

    def __init__(self, nan=None):
        super().__init__()
        self.nan = nan
        self.embeddings = torch.nn.Linear(3 * 8 * 8, 768)
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(768, 768)])
        self.layernorm = torch.nn.LayerNorm(768)

    def forward(self, pixel_values=None):
        h = self.embeddings(pixel_values.flatten(1))
        h = self.layernorm(self.encoder.layer[0](h))
        if self.nan == "always" or (self.nan == "train" and self.training):
            h = h * float("nan")
        return SimpleNamespace(last_hidden_state=h[:, None, :])

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")
        (Path(path) / "config.json").write_text("{}")


@pytest.fixture()
def synthetic(tmp_path, tc, monkeypatch):
    """Synthetic random PNGs: 8 fingers x 4 images; fingers f0..f5 train, f6..f7 validation."""
    state = SimpleNamespace(nan=None, transform_calls=[])

    def fake_load(weights_dir):
        torch.manual_seed(1)
        return FakeBackbone(state.nan)

    def fake_transform(train):
        state.transform_calls.append(bool(train))

        def f(im):
            a = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
            x = torch.from_numpy(a).permute(2, 0, 1)
            return x + 0.5 if train else x   # the training transform visibly differs from the eval one
        return f
    monkeypatch.setattr(tc, "load_backbone", fake_load)
    monkeypatch.setattr(tc, "make_transform", fake_transform)
    rng = np.random.default_rng(0)
    root = tmp_path / "root"
    (root / "images").mkdir(parents=True)
    rows = []
    for f in range(8):
        for k in range(4):
            rel = f"images/f{f}_{k}.png"
            Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(root / rel)
            rows.append({"path": rel, "identity": f"f{f}", "source": "synthetic" if f % 2 == 0 else "synthetic-b", "subject": f"s{f}", "finger": "1",
                         "device": "d", "key": f"{f}{k}", "val": f >= 6})
    man = tmp_path / "train_manifest.parquet"
    pd.DataFrame(rows).to_parquet(man, index=False)
    weights = tmp_path / "weights"
    weights.mkdir()
    return SimpleNamespace(root=root, manifest=man, weights=weights, tmp=tmp_path, state=state, load=fake_load)


def test_validation_eer_is_zero_for_separable_embeddings_and_deterministic(tc):
    ids = np.repeat(np.arange(6), 3)
    emb = np.eye(6)[ids] + 0.01 * np.random.default_rng(0).normal(size=(18, 6))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert tc.validation_eer(emb, ids, 1) == 0.0
    rnd = np.random.default_rng(1).normal(size=(18, 6))
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    assert tc.validation_eer(rnd, ids, 5) == tc.validation_eer(rnd, ids, 5)


def test_embed_rows_returns_unit_norm_rows_in_input_order_with_the_eval_transform(tc, synthetic):
    model = FakeBackbone().eval()
    paths = [str(synthetic.root / f"images/f{f}_0.png") for f in range(5)]
    a = tc.embed_rows(model, paths, batch=2, workers=0)
    b = tc.embed_rows(model, paths[::-1], batch=3, workers=0)
    assert a.dtype == np.float32 and a.shape == (5, 768)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a, b[::-1], atol=1e-5)
    assert synthetic.state.transform_calls == [False, False]


def _train_args(s, out, **kw):
    a = ["train", "--smoke", "--manifest", str(s.manifest), "--image-root", str(s.root), "--weights", str(s.weights),
         "--out", str(out), "--epochs", "2", "--batch", "8", "--workers", "0"]
    for k, v in kw.items():
        flag = f"--{k.replace('_', '-')}"
        if flag in a:
            a[a.index(flag) + 1] = str(v)
        else:
            a += [flag, str(v)]
    return a


def _summary(out):
    return json.loads((out / "training_summary.json").read_text())


def test_train_writes_summary_with_recipe_epochs_selection_and_provenance(tc, synthetic):
    (synthetic.weights / "FETCH.json").write_text(json.dumps({"revision": "abc"}))
    out = synthetic.tmp / "out"
    assert tc.main(_train_args(synthetic, out)) == 0
    s = _summary(out)
    assert s["status"] == "ok" and s["smoke"] is True and "synthetic" in s["data_source"]
    for k in ("SEED", "LR_BACKBONE", "LR_HEAD", "LAYER_DECAY", "WEIGHT_DECAY", "ARC_S", "ARC_M", "IMAGE_SIZE",
              "AFFINE_INTERPOLATION"):
        assert k in s["recipe"]
    assert s["recipe"]["SEED"] == 20261115 and s["recipe"]["ARC_S"] == 30.0 and s["recipe"]["LR_BACKBONE"] == 2e-5
    assert s["recipe_overrides"] == {"epochs": 2, "batch": 8}
    assert s["epochs_run"] == 2 and len(s["epochs"]) == 3 and s["epochs"][0]["epoch"] == 0
    assert all(math.isfinite(e["val_eer"]) for e in s["epochs"])
    assert all(math.isfinite(e["train_loss"]) and e["lr_head"] > 0 and e["lr_backbone_max"] > 0 for e in s["epochs"][1:])
    assert s["pretrained_val_eer"] == s["epochs"][0]["val_eer"]
    assert isinstance(s["improved_over_pretrained"], bool)
    assert len(s["weights_sha256"]) == 64 and set(s["weights_files_sha256"]) == {"model.pt", "config.json"}
    assert s["weights_fetch"] == {"revision": "abc"} and len(s["weights_input"]["sha256"]) == 64
    assert s["input_images"]["images"] == 32 and len(s["input_images"]["combined_sha256"]) == 64
    assert len(s["data_sha256"]["manifest"]) == 64 and "environment" in s and s["seed"] == 20261115
    assert s["workers"] == 0 and s["argv"][0] == "train" and "not bitwise reproducible" in s["reproducibility"]
    assert "TF32 disabled" in s["eval_precision"] and "LayerScale" in s["param_groups_note"]
    assert sum(g["tensors"] for g in s["param_groups"]) == 7   # 6 backbone tensors + the head
    assert {"seconds_model_load", "seconds_epoch0_eval", "seconds_per_epoch"} <= set(s)
    assert s["fingers"] == {"train": 6, "val": 2, "limit_fingers": None}
    assert tc.weights_digest(out / "best")["combined"] == s["weights_sha256"]
    # validation and the epoch-0 evaluation use the eval transform; training builds the augmenting one once
    assert synthetic.state.transform_calls.count(True) == 1 and synthetic.state.transform_calls[0] is False


def test_train_keeps_the_best_epoch_by_the_validation_eer(tc, synthetic, monkeypatch):
    snaps = []
    real_embed = tc.embed_rows

    def spy(model, *args, **kw):
        snaps.append({k: v.clone() for k, v in model.state_dict().items()})
        return real_embed(model, *args, **kw)
    seq = iter([0.3, 0.1, 0.2])
    monkeypatch.setattr(tc, "embed_rows", spy)
    monkeypatch.setattr(tc, "validation_eer", lambda emb, ids, seed: next(seq))
    out = synthetic.tmp / "outb"
    assert tc.main(_train_args(synthetic, out)) == 0
    s = _summary(out)
    assert s["selected_epoch"] == 1 and s["selected_val_eer"] == 0.1 and s["improved_over_pretrained"] is True
    saved = torch.load(out / "best" / "model.pt", weights_only=True)
    assert all(torch.equal(saved[k], snaps[1][k]) for k in saved)
    assert not all(torch.equal(saved[k], snaps[2][k]) for k in saved)


def test_train_records_micro_batching_as_an_override(tc, synthetic):
    out = synthetic.tmp / "outm"
    assert tc.main(_train_args(synthetic, out, micro_batch=4, epochs=1)) == 0
    s = _summary(out)
    assert s["micro_batch"] == 4 and s["grad_accumulation"] == 2 and s["recipe_overrides"]["micro_batch"] == 4


def test_train_limit_fingers_flags_a_smoke_run(tc, synthetic):
    out = synthetic.tmp / "out2"
    assert tc.main(["train", "--manifest", str(synthetic.manifest), "--image-root", str(synthetic.root),
                    "--weights", str(synthetic.weights), "--out", str(out), "--epochs", "1", "--batch", "8",
                    "--workers", "0", "--limit-fingers", "3"]) == 0
    s = _summary(out)
    assert s["smoke"] is True and s["fingers"] == {"train": 3, "val": 2, "limit_fingers": 3}
    assert s["weights_fetch"] is None


def test_train_fails_with_a_non_finite_loss(tc, synthetic):
    synthetic.state.nan = "train"   # finite in eval mode, so the pretrained evaluation passes and the loss branch runs
    out = synthetic.tmp / "out3"
    assert tc.main(_train_args(synthetic, out, limit_fingers=3)) != 0
    s = _summary(out)
    assert s["status"] == "failed" and s["failure"].startswith("non-finite loss") and s["pretrained_val_eer"] <= 1


def test_train_never_selects_a_nan_model(tc, synthetic):
    synthetic.state.nan = "always"
    out = synthetic.tmp / "out4"
    assert tc.main(_train_args(synthetic, out)) != 0
    s = _summary(out)
    assert s["status"] == "failed" and "pretrained validation EER" in s["failure"] and not (out / "best").exists()


def test_train_refuses_to_overwrite_a_checkpoint_or_summary(tc, synthetic):
    out = synthetic.tmp / "out5"
    assert tc.main(_train_args(synthetic, out)) == 0
    before = (out / "training_summary.json").read_bytes(), (out / "best" / "model.pt").read_bytes()
    assert tc.main(_train_args(synthetic, out)) == 2
    assert before == ((out / "training_summary.json").read_bytes(), (out / "best" / "model.pt").read_bytes())


def test_train_stops_when_the_weights_differ_from_fetch_json(tc, synthetic):
    (synthetic.weights / "model.bin").write_bytes(b"x")
    (synthetic.weights / "FETCH.json").write_text(json.dumps({"files_sha256": {"model.bin": "0" * 64}}))
    out = synthetic.tmp / "out6"
    assert tc.main(_train_args(synthetic, out)) == 1
    s = _summary(out)
    assert s["status"] == "failed" and "model.bin" in s["failure"] and not (out / "best").exists()


def test_a_full_run_needs_cuda_and_a_clean_commit(tc, synthetic, monkeypatch):
    args = [a for a in _train_args(synthetic, synthetic.tmp / "out7") if a != "--smoke"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert tc.main(args) == 2
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("CANVEC_GIT_SHA", raising=False)
    assert tc.main(args) == 2
    monkeypatch.setenv("CANVEC_GIT_SHA", "abc123-dirty")
    assert tc.main(args) == 2
    assert not (synthetic.tmp / "out7").exists()


def _embed_args(s, out, *extra):
    return ["embed", "--smoke", "--manifest", str(s.manifest), "--image-root", str(s.root), "--weights", str(s.weights),
            "--out", str(out), "--batch", "5", "--workers", "0", *extra]


def test_embed_writes_embeddings_manifest_copy_and_self_consistency(tc, synthetic):
    out = synthetic.tmp / "emb"
    assert tc.main(_embed_args(synthetic, out)) == 0
    emb = np.load(out / "emb_ctrlvit.npy")
    assert emb.dtype == np.float32 and emb.shape == (32, 768)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert (out / "manifest.parquet").read_bytes() == synthetic.manifest.read_bytes()
    s = json.loads((out / "embed_summary.json").read_text())
    assert s["status"] == "ok" and s["manifest_sha256"] == s["input_manifest_sha256"]
    assert s["self_consistency"]["rows"] == 32 and s["self_consistency"]["min_cosine"] >= 0.9999
    assert len(s["weights_sha256"]) == 64 and s["rows"] == 32 and s["complete"] is True
    assert "synthetic" in s["data_source"] and s["input_images"]["images"] == 32 and s["argv"][0] == "embed"
    paths = [str(synthetic.root / p) for p in pd.read_parquet(synthetic.manifest)["path"]]
    ref = tc.embed_rows(synthetic.load(None).eval(), paths, 8, 0)
    assert np.allclose(emb, ref, atol=1e-5)
    assert set(synthetic.state.transform_calls) == {False}


def test_embed_limit_rows_is_recorded_as_an_incomplete_smoke_run(tc, synthetic):
    out = synthetic.tmp / "emb_lim"
    args = ["embed", "--manifest", str(synthetic.manifest), "--image-root", str(synthetic.root),
            "--weights", str(synthetic.weights), "--out", str(out), "--workers", "0", "--limit-rows", "7"]
    assert tc.main(args) == 0
    assert np.load(out / "emb_ctrlvit.npy").shape == (7, 768)
    s = json.loads((out / "embed_summary.json").read_text())
    assert s["limit_rows"] == 7 and s["rows"] == 7 and s["manifest_rows"] == 32
    assert s["smoke"] is True and s["complete"] is False
    assert tc.main(args) == 2   # existing output is not overwritten


def test_embed_rejects_non_finite_embeddings_and_writes_no_array(tc, synthetic):
    synthetic.state.nan = "always"
    out = synthetic.tmp / "emb_nan"
    assert tc.main(_embed_args(synthetic, out)) == 1
    s = json.loads((out / "embed_summary.json").read_text())
    assert s["status"] == "failed" and not (out / "emb_ctrlvit.npy").exists()
