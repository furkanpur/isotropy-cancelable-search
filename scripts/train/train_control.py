"""Train and apply the control encoder (ctrl-vit): DINOv2-base fine-tuned with ArcFace on the documented training set.

Subcommands (run inside a container on the data host; only aggregates leave it):
  train  --manifest train_manifest.parquet --image-root DIR --weights DIR --out DIR [--epochs N] [--batch N]
         Fine-tunes with the pre-registered recipe (module-level constants below, recorded in the summary), evaluates the
         pretrained model (epoch 0) and every epoch on the validation fingers, keeps the best checkpoint in <out>/best
         and writes <out>/training_summary.json after every epoch (exit 1 and status "failed" when the loss or an
         embedding is not finite).  Refuses to run when <out>/best or <out>/training_summary.json exists.
         A full run needs CUDA and a clean CANVEC_GIT_SHA; --smoke (implied by --limit-fingers N: the first N training
         and the first N validation fingers) lifts both and flags the summary "smoke".
  embed  --manifest M.parquet --image-root DIR --weights DIR --out DIR [--batch N]
         Writes emb_ctrlvit.npy (float32, l2-normalised, manifest order), a byte copy of the manifest as
         manifest.parquet and embed_summary.json (hashes, self-consistency on the first 200 rows).  Refuses to
         overwrite an existing output.  --limit-rows N (smoke: embeds the first N rows only) implies --smoke.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from extract_d2 import environment, sha256_file, software_versions  # noqa: E402  (same repository, scripts/data)

from ijiss_btp.eval.stats import eer, make_pairs  # noqa: E402
from ijiss_btp.train.arcface import ArcFaceHead  # noqa: E402
from ijiss_btp.train.optim import best_epoch, vit_param_groups, warmup_cosine  # noqa: E402

# ---- pre-registered recipe (fixed before training; changing a value is a deviation from it) -------------------------
BACKBONE = "facebook/dinov2-base"
EMB_DIM = 768
ARC_S = 30.0
ARC_M = 0.3
LR_BACKBONE = 2e-5
LAYER_DECAY = 0.75
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
EPOCHS = 30
WARMUP_EPOCHS = 1
BATCH = 128
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
AFFINE = {"degrees": 10, "translate": (0.05, 0.05), "scale": (0.9, 1.1), "fill": 255}
AFFINE_INTERPOLATION = "NEAREST (torchvision RandomAffine default; recorded, not changed)"
JITTER = {"brightness": 0.2, "contrast": 0.2}
SEED = 20261115
MAX_IMPOSTOR_PAIRS = 200_000
SELF_CHECK_ROWS = 200
SELF_CHECK_MIN_COSINE = 0.9999
UNIT_NORM_TOL = 1e-3
RECIPE = {k: v for k, v in globals().items() if k.isupper() and k not in ("RECIPE",) and isinstance(
    v, (str, int, float, tuple, dict))}
REPRODUCIBILITY = ("not bitwise reproducible: no deterministic-algorithm setting; augmentation seeds depend on the "
                   "number of data-loading workers")
PARAM_GROUP_NOTE = ("1-D tensors (biases, normalisation weights, LayerScale lambda) and cls_token, mask_token, "
                    "position_embeddings get weight_decay 0; layer id 0 = embeddings, i+1 = encoder.layer.i, "
                    "L+1 = final layernorm; lr = LR_BACKBONE * LAYER_DECAY ** (L + 1 - id); the ArcFace head is its own group")


# ---- seams (tests monkeypatch these; the heavy imports are lazy) -------------------------------------------------
def load_backbone(weights_dir):
    from transformers import Dinov2Model
    return Dinov2Model.from_pretrained(str(weights_dir))


def make_transform(train: bool):
    from torchvision import transforms as T
    steps = [T.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [T.RandomAffine(**AFFINE), T.ColorJitter(**JITTER)]
    steps += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(steps)


# ---- data ------------------------------------------------------------------------------------------------------
class _Images(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[str], transform, labels: Sequence[int] | None = None):
        self.paths, self.transform, self.labels = list(paths), transform, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return (x, self.labels[i]) if self.labels is not None else x


def _device(model) -> torch.device:
    return next(model.parameters()).device


def cls_embedding(model, x: torch.Tensor) -> torch.Tensor:
    return model(pixel_values=x).last_hidden_state[:, 0]


@contextlib.contextmanager
def fp32_eval():
    """Strict fp32 for evaluation: TF32 off for cuDNN and matmul (restored afterwards)."""
    prev = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev


@torch.no_grad()
def embed_rows(model, paths: Sequence[str], batch: int, workers: int) -> np.ndarray:
    """fp32 (TF32 off) eval-mode l2-normalised CLS vectors in input order."""
    was_training = model.training
    model.eval()
    dev = _device(model)
    loader = torch.utils.data.DataLoader(_Images(paths, make_transform(False)), batch_size=batch, shuffle=False,
                                         num_workers=workers)
    out = []
    with fp32_eval():
        for x in loader:
            out.append(torch.nn.functional.normalize(cls_embedding(model, x.to(dev)).float(), dim=1).cpu())
    model.train(was_training)
    return torch.cat(out).numpy().astype(np.float32) if out else np.zeros((0, EMB_DIM), np.float32)


def validation_eer(emb: np.ndarray, ids: np.ndarray, seed: int) -> float:
    ids = np.asarray(ids)
    rng = np.random.default_rng(seed)
    pairs_g, pairs_i = make_pairs(np.arange(len(ids)), ids, rng, MAX_IMPOSTOR_PAIRS)
    score = lambda p: np.einsum("ij,ij->i", emb[p[:, 0]], emb[p[:, 1]])  # noqa: E731
    return float(eer(score(pairs_g), score(pairs_i)))


def weights_digest(directory) -> dict:
    files = {p.name: sha256_file(p) for p in sorted(Path(directory).iterdir()) if p.is_file()}
    combined = hashlib.sha256("".join(f"{n} {h}\n" for n, h in sorted(files.items())).encode()).hexdigest()
    return {"files": files, "combined": combined}


def images_digest(root, rels: Sequence[str], workers: int = 8) -> dict:
    """One digest over the sorted (relative path, SHA-256 of the bytes) pairs of the images that are read."""
    uniq = sorted(set(rels))
    with ThreadPoolExecutor(max(1, workers)) as ex:
        hashes = list(ex.map(lambda r: sha256_file(Path(root) / r), uniq))
    combined = hashlib.sha256("".join(f"{r} {h}\n" for r, h in zip(uniq, hashes)).encode()).hexdigest()
    return {"images": len(uniq), "combined_sha256": combined}


def _read_fetch(weights: Path):
    f = weights / "FETCH.json"
    return json.loads(f.read_text()) if f.is_file() else None


def _fetch_mismatch(digest: dict, fetch) -> list[str]:
    """Files of the input weights directory whose hash differs from the one FETCH.json recorded."""
    if not fetch:
        return []
    return sorted(f for f, h in fetch.get("files_sha256", {}).items() if f in digest["files"] and digest["files"][f] != h)


def _dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _provenance() -> dict:
    return {"script_sha256": sha256_file(Path(__file__).resolve()), "env": environment(),
            "software": software_versions()}


def _data_source(sources, override: str | None) -> str:
    if override:
        return override
    if len(sources):
        return "manifest source column: " + ", ".join(sorted(sources))
    return "unspecified (the manifest has no source column and --data-source was not given)"


def _preflight(a: argparse.Namespace, outputs: Sequence[str], smoke: bool) -> int | None:
    """Refuse (exit code 2) to overwrite outputs, to run a full job without CUDA or from a dirty/unknown commit."""
    out = Path(a.out)
    clash = [n for n in outputs if (out / n).exists()]
    if clash:
        print(f"refusing to overwrite existing output in {out}: {', '.join(clash)}", file=sys.stderr)
        return 2
    if not smoke:
        if not torch.cuda.is_available():
            print("refusing a full run without CUDA (use --smoke for a test run)", file=sys.stderr)
            return 2
        sha = os.environ.get("CANVEC_GIT_SHA", "")
        if not sha or sha.endswith("-dirty"):
            print(f"refusing a full run from an unknown or dirty commit (CANVEC_GIT_SHA={sha!r})", file=sys.stderr)
            return 2
    return None


# ---- train -----------------------------------------------------------------------------------------------------
def _split(man: pd.DataFrame, limit: int | None):
    ids_tr = sorted(man.loc[~man["val"], "identity"].unique())
    ids_va = sorted(man.loc[man["val"], "identity"].unique())
    if limit is not None:
        ids_tr, ids_va = ids_tr[:limit], ids_va[:limit]
    return (man[man["identity"].isin(ids_tr) & ~man["val"]].reset_index(drop=True),
            man[man["identity"].isin(ids_va) & man["val"]].reset_index(drop=True), ids_tr, ids_va)


def _set_lr(opt, mult: float) -> tuple[float, float]:
    """Apply the schedule; returns (head lr, highest backbone lr = the final-layernorm group)."""
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * mult
    return opt.param_groups[-1]["lr"], max(opt.param_groups[:-1], key=lambda g: g["base_lr"])["lr"]


def _group_table(groups: Sequence[dict]) -> list[dict]:
    return [{"base_lr": g["base_lr"], "weight_decay": g["weight_decay"], "tensors": len(g["params"]),
             "elements": int(sum(p.numel() for p in g["params"]))} for g in groups]


def train(a: argparse.Namespace) -> int:
    t0 = time.time()
    smoke = bool(a.smoke or a.limit_fingers is not None)
    bad = _preflight(a, ["best", "training_summary.json"], smoke)
    if bad:
        return bad
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, weights = Path(a.image_root), Path(a.weights)
    man = pd.read_parquet(a.manifest)
    man["val"] = man["val"].astype(bool)
    tr, va, ids_tr, ids_va = _split(man, a.limit_fingers)
    label_of = {f: i for i, f in enumerate(ids_tr)}
    tr_paths = [str(root / p) for p in tr["path"]]
    va_paths = [str(root / p) for p in va["path"]]
    va_ids = va["identity"].to_numpy()
    micro = a.micro_batch or a.batch
    if a.batch % micro:
        print("--batch must be a multiple of --micro-batch", file=sys.stderr)
        return 2
    accum = a.batch // micro
    overrides = {k: v for k, v in (("epochs", a.epochs if a.epochs != EPOCHS else None),
                                   ("batch", a.batch if a.batch != BATCH else None),
                                   ("micro_batch", a.micro_batch if a.micro_batch not in (None, a.batch) else None))
                 if v is not None}
    wd = weights_digest(weights)
    fetch = _read_fetch(weights)
    summary: dict = {
        "status": "running", "smoke": smoke, "recipe": RECIPE, "recipe_overrides": overrides,
        "argv": a.argv, "workers": a.workers, "reproducibility": REPRODUCIBILITY,
        "seed": SEED, "epochs_requested": a.epochs, "batch": a.batch, "micro_batch": micro, "grad_accumulation": accum,
        "data_source": _data_source(man["source"].unique() if "source" in man else [], a.data_source),
        "data_sha256": {"manifest": sha256_file(a.manifest)},
        "fingers": {"train": len(ids_tr), "val": len(ids_va), "limit_fingers": a.limit_fingers},
        "images": {"train": len(tr), "val": len(va)}, "weights_fetch": fetch,
        "weights_input": {"sha256": wd["combined"], "files_sha256": wd["files"]},
        "eval_precision": "fp32, TF32 disabled for cuDNN and matmul", "param_groups_note": PARAM_GROUP_NOTE,
        "environment": _provenance(), "device": str(dev), "epochs": []}

    def finish(status: str, code: int, failure: str | None = None) -> int:
        summary["status"] = status
        if failure:
            summary["failure"] = failure
        summary["seconds"] = round(time.time() - t0, 1)
        _dump(out / "training_summary.json", summary)
        print(f"train {status}" + (f": {failure}" if failure else ""), flush=True)
        return code

    mism = _fetch_mismatch(wd, fetch)
    if mism:
        return finish("failed", 1, f"weights differ from FETCH.json: {', '.join(mism)}")

    def run() -> int:
        t = time.time()
        summary["input_images"] = images_digest(root, list(tr["path"]) + list(va["path"]), a.workers)
        summary["seconds_hashing"] = round(time.time() - t, 1)
        t = time.time()
        model = load_backbone(weights).to(dev)
        summary["seconds_model_load"] = round(time.time() - t, 1)
        best_dir = out / "best"
        val_eers: list[float] = []

        def evaluate(epoch: int, train_loss, lr_head, lr_backbone, seconds=None) -> float:
            emb = embed_rows(model, va_paths, a.batch, a.workers)
            v = validation_eer(emb, va_ids, SEED) if np.isfinite(emb).all() else float("nan")
            val_eers.append(v)
            summary["epochs"].append({"epoch": epoch, "train_loss": train_loss, "val_eer": v, "lr_head": lr_head,
                                      "lr_backbone_max": lr_backbone, "seconds": seconds})
            print(f"epoch {epoch}: loss {train_loss} val_eer {v:.5f} lr_head {lr_head}", flush=True)
            if math.isfinite(v) and best_epoch(val_eers) == epoch:
                shutil.rmtree(best_dir, ignore_errors=True)
                model.save_pretrained(best_dir)
            _dump(out / "training_summary.json", summary)
            return v

        t = time.time()
        pre = evaluate(0, None, None, None)
        summary["seconds_epoch0_eval"] = round(time.time() - t, 1)
        summary["pretrained_val_eer"] = pre
        if not math.isfinite(pre):
            return finish("failed", 1, "non-finite pretrained validation EER (non-finite embeddings)")

        names = [n for n, _ in model.named_parameters()]
        layers = 1 + max(int(n.split(".")[2]) for n in names if n.startswith("encoder.layer."))
        groups = vit_param_groups(model.named_parameters(), layers, LR_BACKBONE, LAYER_DECAY, WEIGHT_DECAY)
        head = ArcFaceHead(EMB_DIM, len(ids_tr), ARC_S, ARC_M).to(dev)
        summary["param_groups"] = _group_table(groups) + [{"base_lr": LR_HEAD, "weight_decay": WEIGHT_DECAY,
                                                           "tensors": 1, "elements": int(head.weight.numel())}]
        groups.append({"params": [head.weight], "lr": LR_HEAD, "base_lr": LR_HEAD, "weight_decay": WEIGHT_DECAY})
        opt = torch.optim.AdamW(groups)
        labels = [label_of[i] for i in tr["identity"]]
        gen = torch.Generator().manual_seed(SEED)
        loader = torch.utils.data.DataLoader(_Images(tr_paths, make_transform(True), labels), batch_size=micro,
                                             shuffle=True, drop_last=True, num_workers=a.workers, generator=gen,
                                             persistent_workers=a.workers > 0)
        steps_per_epoch = len(loader) // accum
        if steps_per_epoch < 1:
            return finish("failed", 1, "fewer images than one effective batch")
        total, warm = steps_per_epoch * a.epochs, steps_per_epoch * WARMUP_EPOCHS
        amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
        step, peak_mem = 0, 0
        for epoch in range(1, a.epochs + 1):
            te = time.time()
            model.train()
            losses, lrs, it = [], (0.0, 0.0), iter(loader)
            for _ in range(steps_per_epoch):
                lrs = _set_lr(opt, warmup_cosine(step, total, warm))
                opt.zero_grad(set_to_none=True)
                for _ in range(accum):
                    x, y = next(it)
                    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                    with torch.autocast(**amp):
                        emb = cls_embedding(model, x)
                    loss = torch.nn.functional.cross_entropy(head(emb.float(), y), y)
                    if not torch.isfinite(loss):
                        return finish("failed", 1, f"non-finite loss at epoch {epoch}, step {step}")
                    (loss / accum).backward()
                    losses.append(loss.item())
                opt.step()
                step += 1
            if dev.type == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
            evaluate(epoch, float(np.mean(losses)), lrs[0], lrs[1], round(time.time() - te, 1))
        sel = best_epoch(val_eers)
        d = weights_digest(best_dir)
        summary.update({"epochs_run": a.epochs, "selected_epoch": sel, "selected_val_eer": val_eers[sel],
                        "improved_over_pretrained": bool(val_eers[sel] < pre),
                        "weights_sha256": d["combined"], "weights_files_sha256": d["files"],
                        "steps_per_epoch": steps_per_epoch, "peak_gpu_memory_bytes": peak_mem,
                        "seconds_per_epoch": round(float(np.mean([e["seconds"] for e in summary["epochs"][1:]])), 1)})
        return finish("ok", 0)

    try:
        return run()
    except Exception as exc:   # e.g. out of memory: leave a record, then re-raise
        finish("failed", 1, f"{type(exc).__name__}: {exc}")
        raise


# ---- embed -----------------------------------------------------------------------------------------------------
def embed(a: argparse.Namespace) -> int:
    t0 = time.time()
    smoke = bool(a.smoke or a.limit_rows is not None)
    bad = _preflight(a, ["emb_ctrlvit.npy", "manifest.parquet", "embed_summary.json"], smoke)
    if bad:
        return bad
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Path(a.weights)
    man = pd.read_parquet(a.manifest)
    rels = list(man["path"])
    if a.limit_rows is not None:
        rels = rels[:a.limit_rows]
    paths = [str(Path(a.image_root) / p) for p in rels]
    d = weights_digest(weights)
    fetch = _read_fetch(weights)
    summary = {"status": "running", "smoke": smoke, "argv": a.argv, "workers": a.workers,
               "reproducibility": REPRODUCIBILITY, "rows": int(len(paths)), "manifest_rows": int(len(man)),
               "complete": len(paths) == len(man), "limit_rows": a.limit_rows,
               "data_source": _data_source(man["source"].unique() if "source" in man else [], a.data_source),
               "input_manifest_sha256": sha256_file(a.manifest), "weights_sha256": d["combined"],
               "weights_files_sha256": d["files"], "weights_fetch": fetch,
               "eval_precision": "fp32, TF32 disabled for cuDNN and matmul", "recipe": RECIPE,
               "environment": _provenance(), "device": str(dev)}

    def finish(status: str, code: int, failure: str | None = None) -> int:
        summary["status"] = status
        if failure:
            summary["failure"] = failure
        summary["seconds"] = round(time.time() - t0, 1)
        _dump(out / "embed_summary.json", summary)
        print(f"embed {status}" + (f": {failure}" if failure else ""), flush=True)
        return code

    mism = _fetch_mismatch(d, fetch)
    if mism:
        return finish("failed", 1, f"weights differ from FETCH.json: {', '.join(mism)}")
    summary["input_images"] = images_digest(a.image_root, rels, a.workers)
    model = load_backbone(weights).to(dev).eval()
    emb = embed_rows(model, paths, a.batch, a.workers)
    k = min(SELF_CHECK_ROWS, len(paths))
    again = embed_rows(model, paths[:k], max(1, a.batch // 2), a.workers)
    min_cos = float(np.min(np.einsum("ij,ij->i", emb[:k], again))) if k else 1.0
    summary["self_consistency"] = {"rows": int(k), "min_cosine": min_cos, "threshold": SELF_CHECK_MIN_COSINE}
    if not (np.isfinite(emb).all() and np.abs(np.linalg.norm(emb, axis=1) - 1.0).max(initial=0.0) <= UNIT_NORM_TOL):
        return finish("failed", 1, "embeddings are not finite and unit-norm")
    if min_cos < SELF_CHECK_MIN_COSINE:
        return finish("failed", 1, f"self-consistency minimum cosine {min_cos:.6f} below {SELF_CHECK_MIN_COSINE}")
    np.save(out / "emb_ctrlvit.npy", emb)
    shutil.copyfile(a.manifest, out / "manifest.parquet")
    summary.update({"manifest_sha256": sha256_file(out / "manifest.parquet"),
                    "emb_sha256": sha256_file(out / "emb_ctrlvit.npy")})
    print(f"self-consistency min cosine {min_cos:.6f}", flush=True)
    return finish("ok", 0)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "embed"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--image-root", required=True)
        p.add_argument("--weights", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--batch", type=int, default=BATCH)
        p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
        p.add_argument("--smoke", action="store_true", help="test run: allows CPU and an unclean commit; flagged in the summary")
        p.add_argument("--data-source", default=None, help="label for the summary (default: the manifest's source column)")
    t = sub.choices["train"]
    t.add_argument("--epochs", type=int, default=EPOCHS)
    t.add_argument("--micro-batch", type=int, default=None, help="micro-batch with gradient accumulation to --batch")
    t.add_argument("--limit-fingers", type=int, default=None, help="smoke runs only")
    e = sub.choices["embed"]
    e.add_argument("--copy-manifest", action="store_true", help="accepted no-op: the manifest is always copied")
    e.add_argument("--limit-rows", type=int, default=None, help="smoke runs only")
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    return train(a) if a.cmd == "train" else embed(a)


if __name__ == "__main__":
    sys.exit(main())
