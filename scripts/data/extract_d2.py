"""Extract finger-vit and DeepPrint embeddings for an identity-disjoint compilation fold, in the SIN layout.

Every step runs inside the extraction image (docker/Dockerfile.extract) on the data host, with the images and the
model weights mounted read-only, `--network none` and the invoking user (`--gpus all` for parity and extract):

  # 1. Content keys of every labelled compilation image (any exclusion, partition or source); kept on the data host.
  python scripts/data/extract_d2.py keys --compilation /data/compilation --out /out

  # 2. Parity gate: re-extract images of the old corpus and compare them with the features stored for them.
  python scripts/data/extract_d2.py parity --old-images /data/old_d2_test \
      --old-fingervit /data/old_results/finger-vit_result/features.npz \
      --old-deepprint /data/old_results/flx_result/features.npz --models /models --out /out

  # 3. Selection and extraction; refuses to run unless /out/parity.json passed in this environment, with this
  #    script and these weights.
  python scripts/data/extract_d2.py extract --compilation /data/compilation --models /models --out /out \
      --exclude-identities-from /data/relabel/d1_relabel.csv --compilation-keys /data/keys/compilation_keys.parquet \
      [--d1-manifest /data/d1/manifest.parquet --d1-images /data/d1_images]

Selection, in this order: compilation `real` rows with an identity and `exclusion == "none"`; LivDet live captures only;
sources in EXCLUDED_SOURCES dropped (LivDet: its identity key is not edition-qualified); the identities in the
first corpus's relabel map dropped; the fold rule on `partition == development` (all development folds (0-4), or
the largest prefix within 25,000 images, counted before the probe); then the images are probed (both loaders must
decode them), identities holding an exact image of the first corpus are dropped (when its images are given), and
the exact-duplicate rule is applied against every labelled compilation image: an identity any of whose labelled images
(selected or not) is identical to an image of a relabel-map identity is dropped, a selected image identical to an
image of any other compilation identity is dropped, and within one identity the first path is kept.  Identities need at least two images.  The result has no identity
of the relabel map and no image identical to an image of those identities (exact content key); first-corpus images
that the map could not assign to an identity are not covered by this rule.
Outputs in --out: `manifest.parquet` (identity, source, subject, finger, device, capture, dpi, path),
`emb_fingervit.npy` and `emb_deepprint.npy` (L2-normalised float32, row i belongs to manifest row i) and
`extract_summary.json`.  The manifest and the arrays are biometric data: they stay on the data host and are never
committed.  `parity.json` and `extract_summary.json` hold aggregates only.

The preprocessing reproduces the scripts that produced the original corpus features:
- finger-vit: PIL RGB, Resize((224, 224)), ToTensor, Normalize(mean=(0.761, 0.760, 0.760),
  std=(0.287, 0.287, 0.288)); the feature is last_hidden_state[:, 0] (CLS) of
  AutoModel.from_pretrained(<models>/finger-vit_giant).
- DeepPrint: flx LazilyAllocatedBinarizer(5.0) and flx pad_and_resize_to_deepprint_input_size (299 x 299), then
  DeepPrint_TexMinu(num_training_subjects=8000, num_dims=256) with the released best_model.pyt; texture and
  minutia embeddings are concatenated by flx EmbeddingLoader.combine (512-d), through flx extract_embeddings.
  Loading: the parity images go through the reference loader, flx SFingeLoader._load_image (OpenCV grayscale,
  bottom 32 rows cropped: the SFinGe footer); compilation images are read in full (OpenCV grayscale, no crop), as flx's
  loaders for real data do.  One adaptation: flx pads both sides by floor(difference / 2) and asserts a square,
  which fails for an odd height-width difference; such images get one zero column or row first
  (make_square_compatible).  The parity images all have an even difference and follow the flx path unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from build_relabel_map import content_key  # same directory: the content key of the relabel maps

try:  # flx exists only in the extraction image; the selection logic and the tests do not need it
    from flx.data.dataset import DataLoader as _FlxDataLoader
except ImportError:  # pragma: no cover - depends on the environment
    _FlxDataLoader = object

MANIFEST_COLUMNS = ["identity", "source", "subject", "finger", "device", "capture", "dpi", "path"]
LIVE_DIRS = frozenset({"live", "alive"})
SPOOF_TOKENS = ("fake", "spoof")
EXCLUDED_SOURCES = ("livdet",)
EXCLUDED_SOURCES_REASON = ("compilation LivDet identity key not edition-qualified; cross-subset genuine pairings at "
                           "chance level in the label check")
FOLD_CANDIDATES = ((0, 1, 2, 3, 4), (0, 1, 2, 3), (0, 1, 2), (0, 1), (0,))   # first within MAX_IMAGES wins
FOLD_RULE = "all development folds (0-4), or the largest prefix within 25,000 images"
MAX_IMAGES = 25_000
MIN_IMAGES = 2
REQUIRED_GE3 = 600
REQUIRED_FURTHER = 500
PARITY_THRESHOLD = 0.999          # the parity gate; deliberately not a command-line option
PARITY_N = 200
SEED = 20261114

FINGERVIT_DIR = "finger-vit_giant"
DEEPPRINT_DIR = "fixed-length-fingerprint-extractors"
FV_MEAN = [0.761, 0.760, 0.760]
FV_STD = [0.287, 0.287, 0.288]
DP_RIDGE_WIDTH = 5.0
DP_TRAINING_SUBJECTS = 8000
DP_DIMS = 256
FLX_UPSTREAM = "https://github.com/tim-rohwedder/fixed-length-fingerprint-extractors"
FLX_UPSTREAM_COMMIT = "7accfca1f33b9b42bfd220e43cd5bc13b4a7fa13"

PREPROCESSING = {
    "fingervit": ("PIL Image.open(path).convert('RGB'); torchvision Resize((224, 224)), ToTensor, "
                  "Normalize(mean=[0.761, 0.760, 0.760], std=[0.287, 0.287, 0.288]); "
                  "AutoModel.from_pretrained(finger-vit_giant), eval, no_grad; feature = last_hidden_state[:, 0] (CLS)"),
    "deepprint": ("compilation images: cv2.imread(path, IMREAD_GRAYSCALE), the full image (no footer crop, as flx's loaders "
                  "for real data), to_tensor; flx LazilyAllocatedBinarizer(5.0); one zero column (taller) or row "
                  "(wider) appended only when the height-width difference is odd, because flx asserts a square after "
                  "symmetric padding; flx pad_and_resize_to_deepprint_input_size (zero padding to a square, "
                  "antialiased resize to 299 x 299); DeepPrint_TexMinu(num_training_subjects=8000, num_dims=256), "
                  "released best_model.pyt, eval; feature = concat(texture 256, minutia 256) via flx "
                  "extract_embeddings and EmbeddingLoader.combine"),
    "deepprint_parity": ("parity images of the old corpus: the reference loader flx SFingeLoader._load_image "
                         "(cv2.imread(path, IMREAD_GRAYSCALE), bottom 32 rows cropped: the SFinGe footer, to_tensor), "
                         "then the same chain as for compilation images"),
    "stored": "each row L2-normalised, float32; row i belongs to manifest row i",
}


# ---- selection (pandas only, reads no image) ------------------------------------------------------------------
def livdet_live_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows outside LivDet, and for LivDet rows with a Live/Alive directory and no fake/spoof directory."""
    is_livdet = df["source"].astype(str).str.lower().str.contains("livdet", regex=False)
    dirs = df["path"].astype(str).str.lower().str.split("/").str[:-1]
    live = dirs.map(lambda parts: any(p in LIVE_DIRS for p in parts)).astype(bool)
    spoof = dirs.map(lambda parts: any(t in p for p in parts for t in SPOOF_TOKENS)).astype(bool)
    return ~is_livdet | (live & ~spoof)


def _fold_strings(folds: Iterable[int]) -> list[str]:
    return [str(int(f)) for f in folds]


def _identity_table(identities: pd.DataFrame) -> pd.DataFrame:
    idt = identities[["identity", "fold", "partition"]].astype(str)
    dup = idt["identity"].duplicated()
    if dup.any():
        raise ValueError(f"IDENTITIES has {int(dup.sum())} duplicate identity rows")
    return idt


def candidate_rows(manifest: pd.DataFrame, identities: pd.DataFrame, folds: Sequence[int],
                   partition: str = "development", exclude_sources: Sequence[str] = EXCLUDED_SOURCES,
                   exclude_identities: Iterable[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Rows with an identity and exclusion == none, LivDet live only, minus the excluded sources, minus the excluded
    identities, in `partition` and `folds` (no count filter).  The exclusions precede the fold filter."""
    counts = {"manifest_rows": int(len(manifest))}
    m = manifest[manifest["identity"].astype(str).str.strip() != ""]
    counts["rows_with_identity"] = int(len(m))
    m = m[m["exclusion"] == "none"]
    counts["rows_exclusion_none"] = int(len(m))
    live = livdet_live_mask(m)
    counts["livdet_non_live_dropped"] = int((~live).sum())
    m = m[live]
    src = m["source"].astype(str).str.lower().isin([s.lower() for s in exclude_sources])
    counts["rows_excluded_source"] = int(src.sum())
    counts["identities_excluded_source"] = int(m.loc[src, "identity"].nunique())
    m = m[~src]
    idt = _identity_table(identities)
    if exclude_identities is not None:
        d1 = m["identity"].astype(str).isin(set(exclude_identities))
        counts["rows_excluded_d1"] = int(d1.sum())
        counts["identities_excluded_d1"] = int(m.loc[d1, "identity"].nunique())
        dropped = idt[idt["identity"].isin(set(m.loc[d1, "identity"]))]
        counts["identities_excluded_d1_in_folds"] = int(
            (dropped["partition"].eq(partition) & dropped["fold"].isin(_fold_strings(folds))).sum())
        m = m[~d1]
    m = m.drop(columns=[c for c in ("fold", "partition") if c in m.columns])
    m = m.merge(idt, on="identity", how="left", validate="many_to_one")
    counts["rows_without_identity_row"] = int(m["fold"].isna().sum())
    m = m[(m["partition"] == partition) & m["fold"].isin(_fold_strings(folds))]
    counts["rows_in_folds"] = int(len(m))
    return m, counts


DUPLICATE_KINDS = ("d1_map", "livdet", "other")


def dedup_exact(frame: pd.DataFrame, full: pd.DataFrame | None = None) -> tuple[pd.Series, dict]:
    """Keep-mask over a frame with columns identity, path and key (the probed images), aligned with its index.

    `full` lists every labelled compilation image as (identity, key, kind), kind being "d1_map" (an identity of the first
    corpus's relabel map), "livdet" or "other"; without it the frame itself is the reference, with kind "other".
    Only keys under another identity count; an identity's own copies are handled by rule 3.
    1. An identity loses all its images if any of its labelled images, selected or not (an excluded row counts
       too), has a key that occurs under a d1_map identity: the two labels may name the same finger.
    2. Otherwise a selected image is dropped if its key occurs under any other identity: kind "livdet" if one of
       those identities is a LivDet identity, else "other" (this includes other identities of the selection).
    3. Where a key repeats within one identity, only the first path in (identity, path) order is kept.
    """
    f = frame.sort_values(["identity", "path"], kind="mergesort").rename_axis("_row")
    ref = f[["identity", "key"]].assign(kind="other") if full is None else full[["identity", "key", "kind"]]
    pairs = ref.drop_duplicates()
    d1_pairs = pairs.loc[pairs["kind"].eq("d1_map"), ["key", "identity"]].rename(columns={"identity": "d1_identity"})
    own = pd.concat([pairs.loc[pairs["identity"].isin(set(f["identity"])), ["identity", "key"]],
                     f[["identity", "key"]]]).drop_duplicates()
    hit = own.merge(d1_pairs, on="key")
    d1_ids = set(hit.loc[hit["d1_identity"] != hit["identity"], "identity"])
    m = f[["identity", "key"]].reset_index().merge(pairs, on="key", how="inner", suffixes=("", "_other"))
    m = m[m["identity_other"] != m["identity"]]
    flags = (m.assign(d1=m["kind"].eq("d1_map"), lv=m["kind"].eq("livdet"), ot=m["kind"].eq("other"))
             .groupby("_row")[["d1", "lv", "ot"]].any().reindex(f.index, fill_value=False).astype(bool))
    via_selected = set(f.loc[flags["d1"], "identity"])
    rule1 = f["identity"].isin(d1_ids)
    lv = flags["lv"] & ~rule1
    ot = flags["ot"] & ~flags["lv"] & ~rule1
    rest = ~(rule1 | lv | ot)
    # Rules 1 and 2 remove every copy of a key within an identity, so a repeated key among the remaining rows has all
    # its copies there: the first copy in (identity, path) order is kept.
    within = f.duplicated(["identity", "key"], keep="first") & rest
    keep = rest & ~within
    counts = {"keys_shared_with_other_identities": int(f.loc[flags.any(axis=1), "key"].nunique()),
              "identities_dropped_d1_map_duplicate": len(d1_ids),
              "identities_dropped_d1_map_duplicate_via_unselected_images": len(d1_ids - via_selected),
              "images_dropped_d1_map_duplicate": int(rule1.sum()),
              "images_dropped_duplicate_livdet": int(lv.sum()),
              "images_dropped_duplicate_other": int(ot.sum()),
              "identities_with_duplicate_livdet": int(f.loc[lv, "identity"].nunique()),
              "identities_with_duplicate_other": int(f.loc[ot, "identity"].nunique()),
              "images_dropped_within_identity": int(within.sum())}
    return keep.rename_axis(frame.index.name).reindex(frame.index), counts


def select_compilation(manifest: pd.DataFrame, identities: pd.DataFrame, folds: Sequence[int],
                partition: str = "development", min_images: int = MIN_IMAGES,
                unreadable: Iterable[str] | None = None, exclude_sources: Sequence[str] = EXCLUDED_SOURCES,
                exclude_identities: Iterable[str] | None = None, keys: Mapping[str, str] | None = None,
                d1_keys: Iterable[str] | None = None,
                full_keys: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Candidate rows, minus unreadable paths, minus identities holding an exact first-corpus image (`d1_keys`),
    then the exact-duplicate rule of dedup_exact (`keys`: path -> key of the probed images; `full_keys`: every
    labelled compilation image as identity, key, kind), then identities with >= min_images images, sorted by
    (identity, path)."""
    m, counts = candidate_rows(manifest, identities, folds, partition, exclude_sources, exclude_identities)
    if unreadable is not None:
        bad = m["path"].isin(set(unreadable))
        counts["unreadable_dropped"] = int(bad.sum())
        m = m[~bad]
    if keys is not None:
        k = m["path"].map(keys)
        if k.isna().any():
            raise ValueError(f"no content key for {int(k.isna().sum())} probed images")
        m = m.assign(key=k)
        if d1_keys is not None:
            hit = m.loc[m["key"].isin(set(d1_keys)), "identity"].unique()
            d1 = m["identity"].isin(hit)
            counts["identities_with_d1_image"] = int(len(hit))
            counts["rows_dropped_d1_image"] = int(d1.sum())
            m = m[~d1]
        keep, dup = dedup_exact(m[["identity", "path", "key"]], full_keys)
        counts.update(dup)
        m = m[keep].drop(columns="key")
    n = m.groupby("identity")["path"].transform("size")
    counts["identities_below_min_images"] = int(m.loc[n < min_images, "identity"].nunique())
    m = m[n >= min_images].sort_values(["identity", "path"], kind="mergesort").reset_index(drop=True)
    if m["path"].duplicated().any():
        raise ValueError(f"{int(m['path'].duplicated().sum())} duplicate image paths in the selection")
    counts["selected_images"] = int(len(m))
    counts["selected_identities"] = int(m["identity"].nunique())
    return m, counts


def pick_folds(manifest: pd.DataFrame, identities: pd.DataFrame, max_images: int = MAX_IMAGES,
               candidates: Sequence[Sequence[int]] = FOLD_CANDIDATES, partition: str = "development",
               min_images: int = MIN_IMAGES, exclude_sources: Sequence[str] = EXCLUDED_SOURCES,
               exclude_identities: Iterable[str] | None = None) -> tuple[tuple[int, ...], dict]:
    """First candidate whose selection (after the source and identity exclusions, before the probe) has at most
    `max_images` images (the last candidate otherwise)."""
    sizes: dict[str, int] = {}
    for folds in candidates:
        sel, _ = select_compilation(manifest, identities, folds, partition, min_images,
                             exclude_sources=exclude_sources, exclude_identities=exclude_identities)
        sizes[",".join(str(int(f)) for f in folds)] = int(len(sel))
        if len(sel) <= max_images:
            return tuple(int(f) for f in folds), sizes
    return tuple(int(f) for f in candidates[-1]), sizes


def limit_per_source(sel: pd.DataFrame, n: int) -> pd.DataFrame:
    """Smoke-test subset: the first `n` identities (in selection order) of every source, with all their images."""
    keep = sel.drop_duplicates("identity").groupby("source").head(n)["identity"]
    return sel[sel["identity"].isin(keep)].reset_index(drop=True)


def check_identity_disjoint(sel: pd.DataFrame, identities: pd.DataFrame, folds: Sequence[int],
                            partition: str = "development") -> dict:
    """Every selected identity has its one IDENTITIES row in `partition` and `folds`, and a single source."""
    look = _identity_table(identities).set_index("identity")
    ids = pd.Index(sel["identity"].astype(str).unique())
    rows = look.reindex(ids)
    inside = rows["partition"].eq(partition) & rows["fold"].isin(_fold_strings(folds))
    if not inside.all():
        raise ValueError(f"{int((~inside).sum())} selected identities lie outside partition {partition!r}, "
                         f"folds {list(folds)}")
    n_src = sel.groupby("identity")["source"].nunique()
    if (n_src > 1).any():
        raise ValueError(f"{int((n_src > 1).sum())} identities span more than one source")
    return {"identities": int(len(ids)), "partition": partition, "folds": [int(f) for f in folds],
            "outside_folds": 0, "multi_source": 0}


def check_corpus_size(sel: pd.DataFrame, n_ge3: int = REQUIRED_GE3, n_further: int = REQUIRED_FURTHER) -> dict:
    """At least `n_ge3` identities with >= 3 images, plus more than `n_further` further identities."""
    per = sel.groupby("identity").size()
    ge3, n = int((per >= 3).sum()), int(per.size)
    res = {"n_identities_ge3": ge3, "n_identities": n, "required_ge3": n_ge3, "required_further_gt": n_further,
           "further_identities": n - n_ge3, "passed": bool(ge3 >= n_ge3 and n - n_ge3 > n_further)}
    if not res["passed"]:
        raise ValueError(f"corpus too small: {res}")
    return res


def _counts(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in sorted(s.astype(str).value_counts().items())}


def _counts_by(df: pd.DataFrame, key: str, col: str) -> dict:
    return {str(k): _counts(g[col]) for k, g in sorted(df.groupby(key), key=lambda t: str(t[0]))}


def composition(sel: pd.DataFrame) -> dict:
    """Aggregate composition of a selection (counts only: no path, no identity)."""
    if sel.empty:
        raise ValueError("empty selection")
    per = sel.groupby("identity").size()
    s = sel.assign(dpi=sel["dpi"].astype(str).str.strip().replace("", "unknown"))
    subj = sel[sel["subject"].astype(str).str.strip() != ""]
    ext = sel["path"].astype(str).str.rsplit(".", n=1).str[-1].str.lower()
    return {
        "n_images": int(len(sel)),
        "n_identities": int(per.size),
        "n_identities_ge3": int((per >= 3).sum()),
        "n_identities_eq2": int((per == 2).sum()),
        "impressions_per_finger_hist": {str(k): int(v) for k, v in sorted(Counter(per.tolist()).items())},
        "impressions_per_finger_summary": {"min": int(per.min()), "median": float(per.median()),
                                           "mean": round(float(per.mean()), 3), "max": int(per.max())},
        "per_source": {str(k): {"images": int(len(g)), "identities": int(g["identity"].nunique())}
                       for k, g in sorted(sel.groupby("source"), key=lambda t: str(t[0]))},
        "capture": _counts(sel["capture"]),
        "capture_by_source": _counts_by(sel, "source", "capture"),
        "devices_by_source": _counts_by(sel, "source", "device"),
        "n_source_devices": int(sel[["source", "device"]].drop_duplicates().shape[0]),
        "dpi": _counts(s["dpi"]),
        "dpi_by_source": _counts_by(s, "source", "dpi"),
        "n_subjects": int(subj[["source", "subject"]].drop_duplicates().shape[0]),
        "image_format": _counts(ext),
    }


# ---- SIN output ------------------------------------------------------------------------------------------------
def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_sin(out: str | Path, sel: pd.DataFrame, embs: Mapping[str, np.ndarray]) -> dict:
    """Write manifest.parquet and emb_<name>.npy (L2-normalised float32, aligned with the manifest rows)."""
    out, n = Path(out), len(sel)
    arrays = {}
    for name, x in embs.items():
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != n:
            raise ValueError(f"{name}: array of shape {x.shape} does not match {n} manifest rows")
        if not np.isfinite(x).all():
            raise ValueError(f"{name}: non-finite values")
        if (np.linalg.norm(x, axis=1) == 0).any():
            raise ValueError(f"{name}: zero vectors")
        arrays[name] = l2_normalize(x).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    sel[MANIFEST_COLUMNS].astype(str).reset_index(drop=True).to_parquet(out / "manifest.parquet", index=False)
    files = {"manifest.parquet": sha256_file(out / "manifest.parquet")}
    shapes = {}
    for name, y in arrays.items():
        np.save(out / f"emb_{name}.npy", y)
        files[f"emb_{name}.npy"] = sha256_file(out / f"emb_{name}.npy")
        shapes[name] = [int(v) for v in y.shape]
    return {"files": files, "shapes": shapes}


def extract_to_sin(sel: pd.DataFrame, image_root: str | Path,
                   encoders: Mapping[str, Callable[[list[str]], np.ndarray]], out: str | Path) -> dict:
    """Call each encoder on the selection's image paths (in manifest order) and write the SIN layout."""
    paths = [str(Path(image_root) / p) for p in sel["path"]]
    embs, seconds = {}, {}
    for name, encode in encoders.items():
        t0 = time.time()
        embs[name] = np.asarray(encode(paths))
        seconds[name] = round(time.time() - t0, 1)
        print(f"{name}: {embs[name].shape} in {seconds[name]:.0f} s", flush=True)
    info = write_sin(out, sel, embs)
    info["seconds"] = seconds
    return info


# ---- parity helpers ---------------------------------------------------------------------------------------------
def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(l2_normalize(a) * l2_normalize(b), axis=1)


def parity_stats(cos: np.ndarray, threshold: float = PARITY_THRESHOLD) -> dict:
    cos = np.asarray(cos, dtype=np.float64)
    med = float(np.median(cos))
    return {"n": int(cos.size), "median": med, "p5": float(np.percentile(cos, 5)), "min": float(cos.min()),
            "max": float(cos.max()), "mean": float(cos.mean()), "threshold": threshold,
            "passed": bool(med >= threshold)}


def sample_parity_keys(keys_a: Iterable[str], keys_b: Iterable[str], n: int, seed: int) -> list[str]:
    """`n` keys drawn without replacement (seeded) from the keys common to both feature files, sorted."""
    common = sorted(set(keys_a) & set(keys_b))
    if len(common) < n:
        raise ValueError(f"only {len(common)} keys are common to both feature files; {n} requested")
    return sorted(str(k) for k in np.random.default_rng(seed).choice(common, size=n, replace=False))


# ---- image probing and encoders (extraction image only) ----------------------------------------------------------
MIN_SIDE = 48   # the largest flx Gabor kernel is 41 pixels wide; every compilation image is far larger
PROBE_COLUMNS = ["usable", "mode", "height", "width", "key", "alpha_nonopaque"]


def _probe(path: str) -> tuple[bool, str, int, int, str | None, bool]:
    """(usable, PIL mode, height, width, content key, RGBA with a non-opaque pixel) of one image.  Usable: PIL and
    OpenCV decode it, its content key exists and both sides are at least MIN_SIDE pixels."""
    import cv2
    from PIL import Image
    try:
        with Image.open(path) as im:
            mode = im.mode
            im.convert("RGB")
            nonopaque = bool(mode == "RGBA" and im.getchannel("A").getextrema()[0] < 255)
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return False, mode, 0, 0, None, nonopaque
        h, w = int(g.shape[0]), int(g.shape[1])
        key = content_key(path)
        return bool(key is not None and min(h, w) >= MIN_SIDE), mode, h, w, key, nonopaque
    except Exception:
        return False, "unreadable", 0, 0, None, False


def probe_images(paths: Sequence[str], workers: int) -> tuple[pd.DataFrame, dict]:
    """One probe row per path (columns PROBE_COLUMNS, in input order) and aggregate stats."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_probe, paths, chunksize=64))
    df = pd.DataFrame(res, columns=PROBE_COLUMNS)
    ok = df["usable"].to_numpy(dtype=bool)
    stats = {"probed": int(len(df)), "unusable": int((~ok).sum()),
             "unusable_by_mode": _counts(df.loc[~ok, "mode"]), "pil_modes": _counts(df["mode"])}
    if ok.any():
        h, w = df.loc[ok, "height"], df.loc[ok, "width"]
        stats["height"] = {"min": int(h.min()), "median": float(h.median()), "max": int(h.max())}
        stats["width"] = {"min": int(w.min()), "median": float(w.median()), "max": int(w.max())}
    return df, stats


def rgba_report(sel: pd.DataFrame, probe: pd.DataFrame) -> dict:
    """PIL modes by source for the selection, and how many of its RGBA images have a non-opaque alpha pixel."""
    p = probe.set_index("path").reindex(sel["path"])
    f = pd.DataFrame({"source": sel["source"].to_numpy(), "mode": p["mode"].to_numpy(),
                      "nonopaque": p["alpha_nonopaque"].fillna(False).to_numpy(dtype=bool)})
    rgba = f["mode"] == "RGBA"
    nonop = rgba & f["nonopaque"]
    return {"pil_modes_by_source": _counts_by(f, "source", "mode"), "rgba_images": int(rgba.sum()),
            "rgba_nonopaque_alpha": int(nonop.sum()),
            "rgba_nonopaque_alpha_by_source": {str(k): int(v) for k, v in
                                               sorted(f.loc[nonop, "source"].value_counts().items())}}


def d1_image_keys(manifest: str | Path, images: str | Path, workers: int) -> tuple[set, dict]:
    """Content keys of the first corpus's images (manifest basenames under `images`)."""
    names = [os.path.basename(str(p)) for p in pd.read_parquet(manifest)["path"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        keys = list(pool.map(content_key, [str(Path(images) / n) for n in names], chunksize=64))
    found = {k for k in keys if k is not None}
    return found, {"manifest_sha256": sha256_file(manifest), "images": len(names),
                   "unreadable": sum(k is None for k in keys), "distinct_keys": len(found)}


class _MappedImages:
    """Map-style dataset over a path list (a torch DataLoader needs only __len__ and __getitem__)."""

    def __init__(self, paths: Sequence[str], load: Callable):
        self.paths, self.load = list(paths), load

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            return self.load(self.paths[i])
        except Exception as e:  # never echo the path: image-library messages contain it
            raise RuntimeError(f"row {i}: cannot load image ({type(e).__name__})") from None


class _FingerVitInput:
    """The finger-vit preprocessing of the original script (preprocess_image without the batch dimension)."""

    def __init__(self):
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                             transforms.Normalize(mean=FV_MEAN, std=FV_STD)])

    def __call__(self, path: str):
        from PIL import Image
        with Image.open(path) as im:
            image = im.convert("RGB")
        return self.transform(image)


def _require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run the container with --gpus all")
    return torch


def encode_fingervit(models: str | Path, paths: Sequence[str], batch_size: int = 1, workers: int = 8) -> np.ndarray:
    torch = _require_cuda()
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(Path(models) / FINGERVIT_DIR)).to("cuda")
    model.eval()
    loader = torch.utils.data.DataLoader(_MappedImages(paths, _FingerVitInput()), batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
    out, done = [], 0
    with torch.no_grad():
        for x in loader:
            out.append(model(pixel_values=x.to("cuda")).last_hidden_state[:, 0].cpu().numpy())
            done += len(x)
            if done % 2000 < len(x):
                print(f"fingervit: {done}/{len(paths)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def read_gray(path: str) -> np.ndarray:
    """8-bit grayscale image as flx's loaders read it: cv2.imread(path, IMREAD_GRAYSCALE)."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("image could not be decoded")
    return img


def gray_tensor(img: np.ndarray):
    """(1, H, W) float32 tensor in [0, 1] from an 8-bit grayscale array, computed as torchvision to_tensor does."""
    import torch
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit grayscale array, got {img.dtype} with shape {img.shape}")
    return torch.from_numpy(np.ascontiguousarray(img))[None].to(torch.float32).div(255)


def deepprint_input(path: str):
    """DeepPrint loading of a compilation image: the full grayscale image, without the footer crop (flx's loaders for real
    data, MCYT, FVC2004 and NIST SD4, read the whole image)."""
    return gray_tensor(read_gray(path))


class _RowImages(_FlxDataLoader):
    """flx data loader over an explicit path list: Identifier(subject=row, impression=0) loads the image of that
    row.  reference_loader=True uses flx SFingeLoader._load_image (OpenCV grayscale, bottom 32 rows cropped: the
    loader of the original features, for the parity images); False reads the full image (deepprint_input; compilation)."""

    def __init__(self, paths: Sequence[str], reference_loader: bool):
        from flx.data.dataset import Identifier, IdentifierSet
        self._paths = list(paths)
        self._reference = reference_loader
        self._ids = IdentifierSet([Identifier(i, 0) for i in range(len(self._paths))])

    @property
    def ids(self):
        return self._ids

    def get(self, identifier):
        row = identifier.subject
        try:
            if self._reference:
                from flx.data.image_loader import SFingeLoader
                return SFingeLoader._load_image(self._paths[row])
            return deepprint_input(self._paths[row])
        except Exception as e:  # never echo the path
            raise RuntimeError(f"row {row}: cannot load image ({type(e).__name__})") from None


def make_square_compatible(img):
    """Append one zero column (taller image) or one zero row (wider image) when the height-width difference is odd.

    flx pad_and_resize pads both sides by floor(difference / 2) and asserts a square result, so an odd difference
    fails inside flx.  Even differences, which include every image of the old corpus (224 x 224, 192 x 224 after
    the crop), are returned unchanged, so the parity images take exactly the flx path.  Zero is flx's fill value.
    """
    import torch.nn.functional as F
    h, w = int(img.shape[-2]), int(img.shape[-1])
    if (h - w) % 2 == 0:
        return img
    return F.pad(img, (0, 1, 0, 0) if h > w else (0, 0, 0, 1), mode="constant", value=0.0)


class _DeepPrintResize:
    """flx pad_and_resize_to_deepprint_input_size, preceded by make_square_compatible."""

    def __call__(self, img):
        from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
        return pad_and_resize_to_deepprint_input_size(make_square_compatible(img))


def batch_padded(paths: Sequence[str], batch_size: int) -> tuple[list[str], int]:
    """(rows to extract, number of real rows).  flx's texture head squeezes a batch of one image to a vector and then
    fails, so when the last batch would hold a single image the last path is appended once; the caller keeps only
    the first `n` embeddings."""
    rows = list(paths)
    n = len(rows)
    if n and n % batch_size == 1:
        rows.append(rows[-1])
    return rows, n


def flx_loader_args() -> dict:
    """flx's evaluation DataLoader settings, the ones extract_embeddings uses."""
    from flx.models.torch_helpers import get_dataloader_args
    args = get_dataloader_args(train=False)
    return {"batch_size": int(args["batch_size"]), "num_workers": int(args["num_workers"])}


def encode_deepprint(models: str | Path, paths: Sequence[str], reference_loader: bool) -> np.ndarray:
    """reference_loader=True: flx SFingeLoader loading (parity images); False: full compilation images."""
    torch = _require_cuda()
    if _FlxDataLoader is object:
        raise RuntimeError("flx is not importable")
    from flx.data.dataset import Dataset, Identifier
    from flx.data.embedding_loader import EmbeddingLoader
    from flx.data.transformed_image_loader import TransformedImageLoader
    from flx.extractor.fixed_length_extractor import get_DeepPrint_TexMinu
    from flx.image_processing.binarization import LazilyAllocatedBinarizer
    from flx.setup.paths import get_best_model_file
    model_dir = str(Path(models) / DEEPPRINT_DIR)
    if not os.path.isfile(get_best_model_file(model_dir)):  # flx itself only prints a warning and keeps random weights
        raise FileNotFoundError("DeepPrint best_model.pyt not found under the models directory")
    extractor = get_DeepPrint_TexMinu(num_training_subjects=DP_TRAINING_SUBJECTS, num_dims=DP_DIMS)
    extractor.load_best_model(model_dir)
    rows, n = batch_padded(paths, flx_loader_args()["batch_size"])
    image_loader = TransformedImageLoader(
        images=_RowImages(rows, reference_loader), poses=None,
        transforms=[LazilyAllocatedBinarizer(DP_RIDGE_WIDTH), _DeepPrintResize()])
    texture, minutia = extractor.extract(Dataset(image_loader, image_loader.ids))
    combined = EmbeddingLoader.combine(texture, minutia)
    out = np.stack([combined.get(Identifier(i, 0)) for i in range(n)]).astype(np.float32)
    del extractor
    torch.cuda.empty_cache()
    return out


# ---- provenance ----------------------------------------------------------------------------------------------
def software_versions() -> dict:
    v: dict = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "pyarrow", "PIL", "cv2", "torch", "torchvision", "transformers", "safetensors",
                "tokenizers", "huggingface_hub", "torchmetrics"):
        try:
            v[mod] = importlib.import_module(mod).__version__
        except Exception:
            v[mod] = None
    try:
        import torch
        v["cuda_runtime"] = torch.version.cuda
        v["cudnn"] = torch.backends.cudnn.version()
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        v["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        v["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    try:
        m = re.search(r"Kernel Module.*?\s(\d+\.\d+(?:\.\d+)*)\s", Path("/proc/driver/nvidia/version").read_text())
        v["nvidia_driver"] = m.group(1) if m else None
    except OSError:
        v["nvidia_driver"] = None
    try:
        import flx
        root = Path(flx.__file__).parent
        h = hashlib.sha256()
        for p in sorted(root.rglob("*.py")):
            h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
        v["flx_tree_sha256"] = h.hexdigest()
    except Exception:
        v["flx_tree_sha256"] = None
    return v


def code_hashes() -> dict:
    here = Path(__file__).resolve()
    repo = here.parents[2]
    out = {"extract_d2.py": sha256_file(here)}
    for rel in ("docker/Dockerfile.extract", "docker/requirements-extract.txt"):
        p = repo / rel
        out[rel] = sha256_file(p) if p.is_file() else None
    return out


def weight_hashes(models: str | Path) -> dict:
    m = Path(models)
    files = (f"{FINGERVIT_DIR}/config.json", f"{FINGERVIT_DIR}/model.safetensors", f"{DEEPPRINT_DIR}/best_model.pyt")
    return {f: sha256_file(m / f) for f in files}


def environment() -> dict:
    return {k: os.environ.get(f"CANVEC_{k.upper()}") for k in ("git_sha", "image", "image_id")}


def _dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


# ---- commands ------------------------------------------------------------------------------------------------
def cmd_keys(a: argparse.Namespace) -> int:
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lab, meta = build_compilation_keys(a.compilation, a.workers)
    lab.to_parquet(out / "compilation_keys.parquet", index=False)
    meta.update({"code": code_hashes(), "software": software_versions(), "env": environment(),
                 "seconds": round(time.time() - t0, 1)})
    _dump(out / "compilation_keys.json", meta)
    print(f"keys: {meta['labelled_rows']} labelled rows, {meta['unreadable']} unreadable, {meta['distinct_keys']} "
          f"distinct keys, {meta['keys_under_several_identities']} under several identities", flush=True)
    return 0


def cmd_parity(a: argparse.Namespace) -> int:
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zf, zd = np.load(a.old_fingervit), np.load(a.old_deepprint)
    keys = sample_parity_keys(zf.files, zd.files, a.n, a.seed)
    paths = [str(Path(a.old_images) / f"{k}.png") for k in keys]
    missing = sum(not os.path.isfile(p) for p in paths)
    if missing:
        raise SystemExit(f"{missing} of {len(paths)} parity images are missing")
    old = {"fingervit": np.stack([zf[k] for k in keys]), "deepprint": np.stack([zd[k] for k in keys])}
    new = {"fingervit": encode_fingervit(a.models, paths, a.fv_batch, a.workers),
           "deepprint": encode_deepprint(a.models, paths, reference_loader=True)}
    results = {name: parity_stats(cosine_rows(new[name], old[name])) for name in new}
    report = {
        "data_source": "real: images of the old D2 test set and the features the original scripts stored for them",
        "n": a.n, "seed": a.seed, "threshold": PARITY_THRESHOLD, "results": results,
        "passed": all(r["passed"] for r in results.values()),
        "dims": {"fingervit": [int(v) for v in new["fingervit"].shape], "deepprint": [int(v) for v in new["deepprint"].shape]},
        "old_features_sha256": {"fingervit": sha256_file(a.old_fingervit), "deepprint": sha256_file(a.old_deepprint)},
        "settings": {"fingervit_batch_size": a.fv_batch, "loader_workers": a.workers,
                     "deepprint_flx_loader": flx_loader_args(),
                     "deepprint_loader": "reference flx SFingeLoader._load_image (bottom 32 rows cropped)"},
        "preprocessing": PREPROCESSING, "code": code_hashes(), "weights_sha256": weight_hashes(a.models),
        "software": software_versions(), "env": environment(), "seconds": round(time.time() - t0, 1),
    }
    _dump(out / "parity.json", report)
    for name, r in results.items():
        print(f"parity {name}: median {r['median']:.6f} p5 {r['p5']:.6f} min {r['min']:.6f} "
              f"(n={r['n']}, pass={r['passed']})", flush=True)
    return 0 if report["passed"] else 1


ENCODERS = ("fingervit", "deepprint")
FEATURE_SOFTWARE = ("torch", "torchvision", "transformers", "numpy", "PIL", "cv2", "flx_tree_sha256")


def parity_problems(parity: dict, code: dict, weights: dict, fv_batch: int, env: dict, software: dict) -> list[str]:
    """Reasons to refuse an extraction with this parity record; an empty list accepts it.  The record must come from
    a passing run of this script with these weights, the same finger-vit batch size, the same image and the same
    versions of the software that determines the features."""
    problems = []
    results = parity.get("results") or {}
    if not parity.get("passed"):
        problems.append("parity did not pass")
    for name in ENCODERS:
        if name not in results:
            problems.append(f"parity: no {name} result")
        elif not results[name].get("passed"):
            problems.append(f"parity: {name} did not pass")
    if parity.get("threshold") != PARITY_THRESHOLD:
        problems.append("parity used another threshold")
    if parity.get("n") != PARITY_N:
        problems.append(f"parity used n={parity.get('n')}, expected {PARITY_N}")
    if (parity.get("code") or {}).get("extract_d2.py") != code.get("extract_d2.py"):
        problems.append("parity ran with another version of this script")
    if parity.get("weights_sha256") != weights:
        problems.append("parity ran with other weights")
    if (parity.get("settings") or {}).get("fingervit_batch_size") != fv_batch:
        problems.append("parity ran with another finger-vit batch size")
    if (parity.get("env") or {}).get("image_id") != env.get("image_id"):
        problems.append("parity ran in another image")
    for k in FEATURE_SOFTWARE:
        if (parity.get("software") or {}).get(k) != software.get(k):
            problems.append(f"parity ran with another {k}")
    return problems


def build_compilation_keys(compilation: str | Path, workers: int) -> tuple[pd.DataFrame, dict]:
    """Content key of every labelled compilation row (non-empty identity, whatever its exclusion, partition or source)."""
    compilation = Path(compilation)
    man = pd.read_csv(compilation / "MANIFEST.csv", dtype=str, keep_default_na=False)
    lab = man.loc[man["identity"].str.strip() != "", ["path", "identity", "source", "exclusion"]]
    lab = lab.reset_index(drop=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        lab["key"] = list(pool.map(content_key, [str(compilation / p) for p in lab["path"]], chunksize=64))
    ok = lab["key"].notna()
    per_key = lab[ok].groupby("key")["identity"].nunique()
    return lab, {"compilation_manifest_sha256": sha256_file(compilation / "MANIFEST.csv"), "labelled_rows": int(len(lab)),
                 "unreadable": int((~ok).sum()), "distinct_keys": int(per_key.size),
                 "keys_under_several_identities": int((per_key > 1).sum())}


def load_compilation_keys(path: str | Path, manifest_sha256: str, d1_ids: set) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Key table (identity, key, kind) for dedup_exact, the path -> key map, and the table's provenance."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("compilation_manifest_sha256") != manifest_sha256:
        raise SystemExit("the compilation key table was built from another MANIFEST.csv")
    t = pd.read_parquet(path)
    t = t[t["key"].notna()]
    kind = np.where(t["identity"].isin(d1_ids), "d1_map",
                    np.where(t["source"].str.lower() == "livdet", "livdet", "other"))
    info = {k: meta[k] for k in ("labelled_rows", "unreadable", "distinct_keys", "keys_under_several_identities")}
    info["table_sha256"] = sha256_file(path)
    return t[["identity", "key"]].assign(kind=kind), t.set_index("path")["key"], info


def cmd_extract(a: argparse.Namespace) -> int:
    t0 = time.time()
    out, compilation = Path(a.out), Path(a.compilation)
    out.mkdir(parents=True, exist_ok=True)
    code = code_hashes()
    if not a.dry_run:  # the parity gate guards every run that writes embeddings
        weights = weight_hashes(a.models)
        parity_path = Path(a.parity or out / "parity.json")
        parity = json.loads(parity_path.read_text())
        problems = parity_problems(parity, code, weights, a.fv_batch, environment(), software_versions())
        if problems:
            raise SystemExit("refusing to extract: " + "; ".join(problems))

    manifest_sha256 = sha256_file(compilation / "MANIFEST.csv")
    man = pd.read_csv(compilation / "MANIFEST.csv", dtype=str, keep_default_na=False)
    ids = pd.read_csv(compilation / "IDENTITIES.csv", dtype=str, keep_default_na=False)
    rm = pd.read_csv(a.exclude_identities_from, dtype=str, keep_default_na=False)
    if list(rm.columns[:2]) != ["file", "identity"]:
        raise SystemExit("the first-corpus relabel map needs the columns file,identity")
    d1_ids = set(rm["identity"])
    d1_map = {"sha256": sha256_file(a.exclude_identities_from), "rows": int(len(rm)), "identities": len(d1_ids)}
    d1_keys, d1_images = None, None
    if a.d1_manifest and a.d1_images:
        d1_keys, d1_images = d1_image_keys(a.d1_manifest, a.d1_images, a.workers)
    full_keys, table_keys, key_table = load_compilation_keys(a.compilation_keys, manifest_sha256, d1_ids)
    # The source and first-corpus exclusions apply inside every step, so the fold rule counts images after them.
    if a.folds == "auto":
        folds, fold_sizes = pick_folds(man, ids, exclude_identities=d1_ids)
    else:
        folds, fold_sizes = tuple(int(f) for f in a.folds.split(",")), {}
    cand, _ = candidate_rows(man, ids, folds, exclude_identities=d1_ids)
    pr, probe = probe_images([str(compilation / p) for p in cand["path"]], a.workers)
    pr.insert(0, "path", cand["path"].to_numpy())
    ok = pr["usable"].to_numpy(dtype=bool)
    mismatch = int((pr.loc[ok, "key"].to_numpy() != table_keys.reindex(pr.loc[ok, "path"]).to_numpy()).sum())
    if mismatch:
        raise SystemExit(f"{mismatch} probed content keys differ from the compilation key table")
    key_table["probed_keys_checked"] = int(ok.sum())
    sel, counts = select_compilation(man, ids, folds, unreadable=set(pr.loc[~ok, "path"]), exclude_identities=d1_ids,
                              keys=dict(zip(pr.loc[ok, "path"], pr.loc[ok, "key"])), d1_keys=d1_keys,
                              full_keys=full_keys)
    disjoint = check_identity_disjoint(sel, ids, folds)
    if a.limit_identities:
        sel = limit_per_source(sel, a.limit_identities)
        size = {"skipped": f"limited run: first {a.limit_identities} identities per source"}
    else:
        size = check_corpus_size(sel)
    comp = composition(sel)
    hw = pr.set_index("path").loc[sel["path"], ["height", "width"]].to_numpy()
    probe["selected_odd_height_width_difference"] = int(((hw[:, 0] - hw[:, 1]) % 2 != 0).sum())
    rgba = rgba_report(sel, pr)
    print(f"folds {list(folds)} (candidates {fold_sizes}); selected {comp['n_images']} images, "
          f"{comp['n_identities']} identities ({comp['n_identities_ge3']} with >= 3); "
          f"unusable images {probe['unusable']}; odd height-width difference "
          f"{probe['selected_odd_height_width_difference']}; RGBA {rgba['rgba_images']} "
          f"(non-opaque {rgba['rgba_nonopaque_alpha']})", flush=True)
    unassigned = (d1_images["images"] - d1_map["rows"]) if d1_images else None
    provenance = {
        "fold_rule": FOLD_RULE if a.folds == "auto" else f"explicit --folds {a.folds}",
        "first_corpus_exclusion": (
            f"the {d1_map['identities']} identities in the first corpus's relabel map are excluded before the fold "
            "rule; after the probe, an identity any of whose labelled images (selected or not) is identical to an "
            "image of those identities (exact content key, full labelled compilation key table) is dropped; exact-image "
            "check against the first corpus's "
            + (f"{d1_images['images']} images: ran" if d1_images else "images: not run (no --d1-images)")),
        "deduplication": ("checked against every labelled compilation image (key table): an identity any of whose labelled "
                          "images (selected or not) is identical to an image of a relabel-map identity is dropped; a "
                          "selected image identical to an image of any other compilation identity (LivDet or other) is "
                          "dropped; within one identity the first path is kept"),
        "first_corpus_images_not_assigned": unassigned,
    }
    selection = {"folds": list(folds), "fold_candidates": fold_sizes, "selection_counts": counts,
                 "image_probe": probe, "rgba": rgba, "composition": comp, "corpus_requirements": size,
                 "d1_relabel_map": d1_map, "d1_images": d1_images, "compilation_key_table": key_table, **provenance}
    if a.dry_run:
        _dump(out / "selection_summary.json", selection)
        return 0

    encoders = {"fingervit": partial(encode_fingervit, a.models, batch_size=a.fv_batch, workers=a.workers),
                "deepprint": partial(encode_deepprint, a.models, reference_loader=False)}
    info = extract_to_sin(sel, compilation, encoders, out)
    contiguous = len(folds) > 1 and tuple(folds) == tuple(range(folds[0], folds[-1] + 1))
    folds_text = f"{folds[0]}-{folds[-1]}" if contiguous else ",".join(str(f) for f in folds)
    summary = {
        "data_source": (
            f"real: compilation `real` part (internal research corpus of real fingerprint images), development folds "
            f"{folds_text}, without LivDet, without the identities in the first corpus's relabel map and without any "
            "image identical to an image of those identities (exact content key); "
            + (f"{unassigned} first-corpus images could not be assigned to an identity"
               if unassigned is not None else "the number of first-corpus images without an identity is unknown")
            + "; no synthetic data"),
        "compilation": {"manifest_sha256": manifest_sha256,
                 "identities_sha256": sha256_file(compilation / "IDENTITIES.csv"),
                 "partition": "development", "folds_used": list(folds), "images_by_fold_candidate": fold_sizes,
                 "fold_rule": provenance["fold_rule"],
                 "fold_rule_counts": "pre-probe image counts after the source and first-corpus exclusions",
                 "exclusion": "none", "min_images_per_identity": MIN_IMAGES,
                 "excluded_sources": list(EXCLUDED_SOURCES), "excluded_sources_reason": EXCLUDED_SOURCES_REASON,
                 "first_corpus_exclusion": provenance["first_corpus_exclusion"],
                 "first_corpus_images_not_assigned": unassigned,
                 "d1_relabel_map": d1_map, "d1_images": d1_images, "compilation_key_table": key_table,
                 "deduplication": provenance["deduplication"],
                 "livdet_guard": "live captures only (Live/Alive directory); spoof images carry exclusion == spoof",
                 "limit_identities": a.limit_identities},
        "selection_counts": counts,
        "duplicate_basenames": int(sel["path"].map(os.path.basename).duplicated().sum()),
        "image_probe": probe,
        "rgba": rgba,
        "identity_disjointness": disjoint,
        "corpus_requirements": size,
        "composition": comp,
        "encoders": {
            "fingervit": {"dim": info["shapes"]["fingervit"][1], "batch_size": a.fv_batch,
                          "weights_sha256": {k: v for k, v in weights.items() if k.startswith(FINGERVIT_DIR)},
                          "preprocessing": PREPROCESSING["fingervit"]},
            "deepprint": {"dim": info["shapes"]["deepprint"][1], "batch_size": flx_loader_args()["batch_size"],
                          "weights_sha256": {k: v for k, v in weights.items() if k.startswith(DEEPPRINT_DIR)},
                          "preprocessing": PREPROCESSING["deepprint"],
                          "preprocessing_parity": PREPROCESSING["deepprint_parity"]},
            "stored": PREPROCESSING["stored"]},
        "flx_source": {"upstream": FLX_UPSTREAM, "upstream_commit": FLX_UPSTREAM_COMMIT,
                       "local_change": ("SFingeLoader/FVC2004Loader identifiers without the -1 offset; identifiers do "
                                        "not enter the embedding computation (rows are mapped by an explicit index)")},
        "outputs": info,
        "parity": {"n": parity["n"], "seed": parity["seed"], "threshold": parity["threshold"],
                   "results": parity["results"], "parity_json_sha256": sha256_file(parity_path)},
        "seed": SEED,
        "code": code, "software": software_versions(), "env": environment(),
        "seconds": round(time.time() - t0, 1),
    }
    _dump(out / "extract_summary.json", summary)
    print(f"wrote {out}/extract_summary.json", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keys", help="content keys of every labelled compilation image (kept on the data host)")
    k.add_argument("--compilation", required=True, help="compilation `real` directory (MANIFEST.csv, images/)")
    k.add_argument("--out", required=True, help="directory for compilation_keys.parquet and compilation_keys.json")
    k.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("parity", help="re-extract old-corpus images and compare with their stored features")
    p.add_argument("--old-images", required=True)
    p.add_argument("--old-fingervit", required=True)
    p.add_argument("--old-deepprint", required=True)
    p.add_argument("--n", type=int, default=PARITY_N)
    p.add_argument("--seed", type=int, default=SEED)
    e = sub.add_parser("extract", help="select the compilation fold and extract both encoders")
    e.add_argument("--compilation", required=True, help="compilation `real` directory (MANIFEST.csv, IDENTITIES.csv, images/)")
    e.add_argument("--exclude-identities-from", required=True,
                   help="relabel map (file,identity) of the first corpus: its compilation identities are excluded")
    e.add_argument("--compilation-keys", required=True,
                   help="compilation_keys.parquet from the keys command (compilation_keys.json beside it): the duplicate reference")
    e.add_argument("--d1-manifest", default=None, help="first-corpus manifest.parquet, for the exact-image check")
    e.add_argument("--d1-images", default=None, help="first-corpus image directory, for the exact-image check")
    e.add_argument("--parity", default=None, help="parity.json (default: <out>/parity.json)")
    e.add_argument("--folds", default="auto",
                   help="'auto' (all development folds (0-4), or the largest prefix within 25,000 images) "
                        "or an explicit list such as '0,1'")
    e.add_argument("--limit-identities", type=int, default=0, help="smoke test: first N identities of each source")
    e.add_argument("--dry-run", action="store_true", help="selection and image probe only; no embeddings")
    for q in (p, e):
        q.add_argument("--models", required=True, help="directory with finger-vit_giant/ and "
                                                      "fixed-length-fingerprint-extractors/")
        q.add_argument("--out", required=True)
        q.add_argument("--fv-batch", type=int, default=1, help="finger-vit batch size (the original script used 1)")
        q.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    return {"keys": cmd_keys, "parity": cmd_parity, "extract": cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
