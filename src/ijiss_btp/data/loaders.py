"""Embedding loaders. Real biometric embeddings are only ever loaded inside the container."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Embeddings:
    """Embedding matrix (n, d), one identity label per row, and a provenance string."""

    x: np.ndarray
    ids: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or len(self.x) != len(self.ids):
            raise ValueError(f"shape mismatch: x {self.x.shape}, ids {np.shape(self.ids)}")


def _l2n(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


RELABEL_KEYS = ("basename", "path")


def load_sin_corpus(root: str | Path, model: str, relabel: str | Path | None = None,
                    relabel_key: str | None = None) -> Embeddings:
    """SIN 2026 layout: emb_<model>.npy and manifest.parquet with 'identity' and 'path' columns.

    relabel: optional CSV with columns `file,identity`. With relabel_key "basename" (the default when a map is given)
    `file` is matched against the basename of the manifest path; with "path" it is matched against the exact manifest
    `path` string (for corpora whose basenames repeat across folders). The identity is replaced and rows without a map
    entry are dropped; the key mode and the dropped count are recorded in the provenance string. A duplicate `file`
    entry in the map raises, and so does a relabel_key without a map.
    """
    if relabel is None and relabel_key is not None:
        raise ValueError(f"relabel_key without relabel: {relabel_key!r} given but no relabel map")
    relabel_key = relabel_key or "basename"
    if relabel_key not in RELABEL_KEYS:
        raise ValueError(f"relabel_key must be one of {RELABEL_KEYS}, got {relabel_key!r}")
    root = Path(root)
    x = np.load(root / f"emb_{model}.npy").astype(np.float64)
    man = pd.read_parquet(root / "manifest.parquet")
    ids = man["identity"].to_numpy().astype(str)
    source = f"sin:{root.name}:{model}"
    if relabel is not None:
        rm = pd.read_csv(relabel, dtype=str, keep_default_na=False)
        if rm["file"].duplicated().any():
            raise ValueError(f"duplicate file entries in relabel map {Path(relabel).name}")
        if (rm["identity"].str.strip().isin(["", "nan", "NaN"])).any():
            raise ValueError(f"empty or NaN identity in relabel map {Path(relabel).name}")
        mapping = dict(zip(rm["file"], rm["identity"]))
        if relabel_key == "path":
            key = man["path"].astype(str)
        else:
            key = man["path"].map(lambda p: os.path.basename(str(p)))
        keep = key.isin(mapping).to_numpy()
        ids = key[keep].map(mapping).to_numpy().astype(str)
        x = x[keep]
        source += f":relabel={Path(relabel).name}:key={relabel_key}:dropped={int((~keep).sum())}"
    return Embeddings(x, ids, source)


def load_keyed_npz(path: str | Path) -> Embeddings:
    """Per-image npz with keys '<identity>_<impression>' (fingerprint-model-benchmark-tool)."""
    z = np.load(path, allow_pickle=False)
    keys = sorted(z.files)
    x = np.stack([z[k] for k in keys]).astype(np.float64)
    ids = np.array([k.rsplit("_", 1)[0] for k in keys])
    return Embeddings(x, ids, f"keyed_npz:{Path(path).parent.name}")


def load_features_by_id(path: str | Path, n_ids: int, per_id: int = 1, seed: int = 0) -> Embeddings:
    """Pickled {identity: [vector, ...]} file (fingerprint-evaluate-tools).

    Security: allow_pickle can execute code embedded in the file. Use it only for the author's
    own files produced by fingerprint-evaluate-tools on the experiment server, loaded inside the read-only,
    network-less container; never for downloaded or third-party files.
    """
    obj = np.load(path, allow_pickle=True)["features_by_id"].item()
    keys = sorted(obj)
    pick = np.sort(np.random.default_rng(seed).choice(len(keys), size=min(n_ids, len(keys)), replace=False))
    xs: list[np.ndarray] = []
    ids: list[str] = []
    for i in pick:
        vecs = list(obj[keys[i]])[:per_id]
        xs.extend(vecs)
        ids.extend([keys[i]] * len(vecs))
    return Embeddings(np.asarray(xs, dtype=np.float64), np.asarray(ids), f"features_by_id:{Path(path).name}")


def synthetic_clusters(n_ids: int, per_id: int, dim: int, decay: float, seed: int) -> Embeddings:
    """Gaussian identity clusters with a power-law spectrum (decay 0 = isotropic). Tests and smoke runs only."""
    rng = np.random.default_rng(seed)
    spectrum = 1.0 / np.arange(1, dim + 1) ** decay
    centers = rng.standard_normal((n_ids, dim)) * np.sqrt(spectrum) * 3.0
    x = np.repeat(centers, per_id, 0) + rng.standard_normal((n_ids * per_id, dim)) * np.sqrt(spectrum) * 1.2
    ids = np.repeat(np.array([f"s{i:05d}" for i in range(n_ids)]), per_id)
    return Embeddings(x, ids, f"synthetic:decay={decay:g}")


def center_normalize(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Subtract the public DEV mean, then L2-normalize each row."""
    return _l2n(x - mean)


def drop_shared_identities(sec: Embeddings, prim: Embeddings) -> Embeddings:
    """Remove rows of `sec` whose identity occurs in `prim` (both must use one identity namespace, e.g. relabelled)."""
    keep = ~np.isin(sec.ids, np.unique(prim.ids))
    return Embeddings(sec.x[keep], sec.ids[keep], f"{sec.source}:shared_ids_removed={int((~keep).sum())}")


def exclude_primary(spec: dict, sec: Embeddings, prim: Embeddings | None,
                    prim_spec: dict | None) -> tuple[Embeddings, int | None]:
    """Apply `exclude_primary_identities` of a secondary-corpus spec (shifted AUX, distractors).

    Without the flag, `sec` is returned unchanged with None. With it, every row of `sec` whose identity occurs in the
    loaded primary corpus `prim` is dropped and the removed count is returned (0 when no identity is shared). The
    identities are compared as strings, so both specs must carry a relabel map that puts the two corpora in one
    identity namespace; a spec without `relabel` raises instead of comparing two namespaces (which would remove nothing
    without notice). The callers apply it before the near-duplicate filter.
    """
    if not spec.get("exclude_primary_identities", False):
        return sec, None
    for name, s in (("primary", prim_spec), ("secondary", spec)):
        if not (s or {}).get("relabel"):
            raise ValueError(f"exclude_primary_identities: the {name} corpus spec has no relabel map, so the two "
                             "corpora are not in one identity namespace")
    if prim is None:
        raise ValueError("exclude_primary_identities needs the loaded primary corpus")
    out = drop_shared_identities(sec, prim)
    return out, int(len(sec.x) - len(out.x))


def drop_near_duplicates(sec: Embeddings, prim: Embeddings, thr: float) -> Embeddings:
    """Remove rows of `sec` whose uncentered cosine to any row of `prim` is >= thr."""
    s, p = _l2n(sec.x), _l2n(prim.x)
    keep = np.ones(len(s), dtype=bool)
    for i in range(0, len(s), 2048):
        keep[i:i + 2048] = (s[i:i + 2048] @ p.T).max(1) < thr
    return Embeddings(sec.x[keep], sec.ids[keep], f"{sec.source}:dedup<{thr:g}")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
