"""Label-consistency check of a SIN-layout corpus (aggregates only).

Reads manifest.parquet (identity, source, device) and emb_<model>.npy from --root and writes one JSON file:
- identity structure per source: images, identities, identities seen on several devices, devices per identity;
- per encoder, for all pairs and per source: the AUC of all genuine pairs, of same-device and of cross-device
  genuine pairs, against the impostor pairs (within the source for the per-source rows);
- per encoder and per (source, device a, device b) pair with at least --min-pairs genuine pairs: the AUC of the
  genuine pairs against the impostor pairs of the same device pair.
Correctly labelled identities score clearly above impostors of the same device pair; merged different fingers give
an AUC near 0.5.  Cosine scores are binned over [-1, 1]; ties within a bin count one half.  The output holds no
path, identity or per-image value.

Usage:  python scripts/data/label_check.py --root DIR --out FILE [--models fingervit deepprint]
            [--bins 4000] [--min-pairs 30] [--label "<data source>"]
Run it inside the container on the data host: it reads embeddings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def bin_index(s: np.ndarray, bins: int) -> np.ndarray:
    return np.clip(((np.asarray(s, dtype=np.float64) + 1.0) * 0.5 * bins).astype(np.int64), 0, bins - 1)


def hist_auc(hg: np.ndarray, hi: np.ndarray) -> float | None:
    """P(genuine > impostor) + 0.5 P(same bin), from two histograms over the same bins."""
    hg, hi = np.asarray(hg, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    ng, ni = hg.sum(), hi.sum()
    if ng == 0 or ni == 0:
        return None
    below = np.cumsum(hi) - hi
    return float((hg * (below + 0.5 * hi)).sum() / (ng * ni))


def _mean(h: np.ndarray, bins: int) -> float | None:
    n = h.sum()
    return None if n == 0 else round(float((h * ((np.arange(bins) + 0.5) / bins * 2.0 - 1.0)).sum() / n), 4)


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def identity_structure(man: pd.DataFrame) -> dict:
    g = man.groupby("identity")
    src = g["source"].first()
    ndev = g["device"].nunique(dropna=False)
    out = {}
    for s in sorted(src.unique(), key=str):
        sel = src == s
        out[str(s)] = {"images": int((man["source"] == s).sum()), "identities": int(sel.sum()),
                       "multi_device_identities": int((ndev[sel] > 1).sum()),
                       "max_devices_per_identity": int(ndev[sel].max())}
    return out


def _entry(gen, same, cross, imp, bins) -> dict:
    def sub(h):
        return None if h.sum() == 0 else {"pairs": int(h.sum()), "auc": _r(hist_auc(h, imp))}
    return {"genuine_pairs": int(gen.sum()), "impostor_pairs": int(imp.sum()), "auc": _r(hist_auc(gen, imp)),
            "same_device": sub(same), "cross_device": sub(cross),
            "mean_genuine": _mean(gen, bins), "mean_impostor": _mean(imp, bins)}


def _model_check(man: pd.DataFrame, x: np.ndarray, bins: int, min_pairs: int, block: int) -> dict:
    n = len(man)
    e = np.asarray(x, dtype=np.float32)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    ident = pd.factorize(man["identity"].astype(str))[0]
    src, src_names = pd.factorize(man["source"].astype(str))
    dev, dev_names = pd.factorize(man["source"].astype(str) + "|" + man["device"].fillna("").astype(str))
    ns, nd = len(src_names), len(dev_names)
    a_gen, a_same, a_cross, a_imp = (np.zeros(bins, np.int64) for _ in range(4))
    s_gen, s_same, s_cross, s_imp = (np.zeros(ns * bins, np.int64) for _ in range(4))
    d_gen, d_imp = np.zeros(nd * nd * bins, np.int64), np.zeros(nd * nd * bins, np.int64)
    cols = np.arange(n)[None, :]
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        b = bin_index(e[i0:i1] @ e.T, bins)
        up = cols > np.arange(i0, i1)[:, None]
        gen = (ident[i0:i1, None] == ident[None, :]) & up
        imp = ~gen & up
        same_src = src[i0:i1, None] == src[None, :]
        same_dev = dev[i0:i1, None] == dev[None, :]
        for h, m in ((a_gen, gen), (a_same, gen & same_dev), (a_cross, gen & ~same_dev), (a_imp, imp)):
            h += np.bincount(b[m], minlength=bins)
        sc = src[i0:i1, None] * bins + b
        for h, m in ((s_gen, gen & same_src), (s_same, gen & same_dev), (s_cross, gen & same_src & ~same_dev),
                     (s_imp, imp & same_src)):
            h += np.bincount(sc[m], minlength=ns * bins)
        lo = np.minimum(dev[i0:i1, None], dev[None, :])
        hi = np.maximum(dev[i0:i1, None], dev[None, :])
        dc = (lo * nd + hi) * bins + b
        d_gen += np.bincount(dc[gen], minlength=nd * nd * bins)
        d_imp += np.bincount(dc[imp], minlength=nd * nd * bins)

    s_gen, s_same, s_cross, s_imp = (h.reshape(ns, bins) for h in (s_gen, s_same, s_cross, s_imp))
    per_source = {str(name): _entry(s_gen[k], s_same[k], s_cross[k], s_imp[k], bins)
                  for k, name in sorted(enumerate(src_names), key=lambda t: str(t[1]))}
    d_gen, d_imp = d_gen.reshape(nd, nd, bins), d_imp.reshape(nd, nd, bins)
    pairs = []
    for x_ in range(nd):
        for y_ in range(x_, nd):
            ng = int(d_gen[x_, y_].sum())
            if ng < min_pairs:
                continue
            (sa, da), (sb, db) = sorted([tuple(dev_names[x_].split("|", 1)), tuple(dev_names[y_].split("|", 1))])
            pairs.append({"source": sa if sa == sb else f"{sa}/{sb}", "devices": da if x_ == y_ else f"{da} x {db}",
                          "same_device": x_ == y_, "genuine_pairs": ng, "impostor_pairs": int(d_imp[x_, y_].sum()),
                          "auc": _r(hist_auc(d_gen[x_, y_], d_imp[x_, y_]))})
    pairs.sort(key=lambda r: (r["source"], r["devices"]))
    return {"all": _entry(a_gen, a_same, a_cross, a_imp, bins), "per_source": per_source, "device_pairs": pairs}


def label_check(man: pd.DataFrame, embeddings: Mapping[str, np.ndarray], bins: int = 4000, min_pairs: int = 30,
                block: int = 512) -> dict:
    """Aggregate label check of one corpus; `embeddings` maps an encoder name to its (n, d) array."""
    man = man.reset_index(drop=True)
    for name, x in embeddings.items():
        if len(x) != len(man):
            raise ValueError(f"{name}: {len(x)} rows for {len(man)} manifest rows")
    return {"n_images": int(len(man)), "n_identities": int(man["identity"].nunique()), "bins": bins,
            "min_pairs": min_pairs, "identity_structure": identity_structure(man),
            "models": {name: _model_check(man, x, bins, min_pairs, block) for name, x in embeddings.items()}}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo: Path) -> str | None:
    """CANVEC_GIT_SHA, else the GIT_SHA file the sync script writes next to the code, else None."""
    if os.environ.get("CANVEC_GIT_SHA"):
        return os.environ["CANVEC_GIT_SHA"]
    f = Path(repo) / "GIT_SHA"
    return f.read_text().strip() if f.is_file() else None


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="directory with manifest.parquet and emb_<model>.npy")
    ap.add_argument("--out", required=True, help="output JSON file")
    ap.add_argument("--models", nargs="+", default=["fingervit", "deepprint"])
    ap.add_argument("--bins", type=int, default=4000)
    ap.add_argument("--min-pairs", type=int, default=30)
    ap.add_argument("--label", default="", help="data-source label written into the output")
    a = ap.parse_args(argv)
    root = Path(a.root)
    man = pd.read_parquet(root / "manifest.parquet", columns=["identity", "source", "device"])
    embs = {m: np.load(root / f"emb_{m}.npy") for m in a.models}
    here = Path(__file__).resolve()
    res = {"data_source": a.label,
           "inputs_sha256": {f.name: _sha256(f) for f in
                             [root / "manifest.parquet"] + [root / f"emb_{m}.npy" for m in a.models]},
           "provenance": {"script_sha256": _sha256(here), "git_sha": git_sha(here.parents[2]),
                          "numpy": np.__version__, "pandas": pd.__version__}}
    res.update(label_check(man, embs, a.bins, a.min_pairs))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"label check: {res['n_images']} images, {res['n_identities']} identities, models {list(embs)}")


if __name__ == "__main__":
    main()
