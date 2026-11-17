"""Protected-domain HNSW scaling with distractors (spec Section 6). Aggregate outputs only."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ijiss_btp.data.loaders import (
    Embeddings,
    center_normalize,
    drop_near_duplicates,
    exclude_primary,
    load_features_by_id,
    load_sin_corpus,
    sha256_file,
)
from ijiss_btp.data.splits import corpus_counts, make_split, split_counts
from ijiss_btp.eval.retrieval import tie_aware_recall
from ijiss_btp.experiments.isotropy import (
    _cpu_model,
    _data_file_hashes,
    _read_optional,
    load_data,
    spectrum_stats,
    stable_seed,
)
from ijiss_btp.index.hnsw import ProtectedVectorIndex
from ijiss_btp.precondition import make_preconditioner
from ijiss_btp.protect.keyed import KeyedScheme, parse_scheme

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def load_distractors(spec: dict, n_max: int, primary: Embeddings | None = None,
                     primary_spec: dict | None = None) -> tuple[Embeddings, int, int | None]:
    """Distractor rows, at most n_max of them.

    Returns (rows, rows loaded before the identity exclusion, rows removed for an identity of the primary corpus or
    None). With `exclude_primary_identities` the exclusion comes before the random draw for SIN corpora, so the draw
    uses every eligible row; it needs the primary corpus and its spec (both specs must carry a relabel map).
    """
    kind = spec["kind"]
    if kind == "sin":
        emb = load_sin_corpus(spec["root"], spec["model"], spec.get("relabel"), spec.get("relabel_key"))
        pool, removed = exclude_primary(spec, emb, primary, primary_spec)
        pick = np.random.default_rng(spec.get("seed", 0)).permutation(len(pool.x))[:n_max]
        return Embeddings(pool.x[pick], pool.ids[pick], pool.source), len(emb.x), removed
    if kind == "features_by_id":
        emb = load_features_by_id(spec["path"], n_ids=n_max, per_id=1, seed=spec.get("seed", 0))
    elif kind == "synthetic":  # clustered synthetic rows from the isotropy generator (used by tests)
        full = load_data(spec)
        emb = Embeddings(full.x[:n_max], full.ids[:n_max], full.source)
    elif kind == "synthetic_gaussian":
        rng = np.random.default_rng(spec.get("seed", 0))
        emb = Embeddings(rng.standard_normal((n_max, spec["dim"])), np.array([f"d{i}" for i in range(n_max)]),
                         "synthetic_gaussian")
    else:
        raise ValueError(f"unknown distractor kind {kind!r}")
    out, removed = exclude_primary(spec, emb, primary, primary_spec)
    return out, len(emb.x), removed


def _distractor_hashes(spec: dict) -> dict:
    if spec["kind"] == "features_by_id":
        return {Path(spec["path"]).name: sha256_file(spec["path"])}
    return _data_file_hashes(spec) if spec["kind"] == "sin" else {}


def exact_topk(scheme: KeyedScheme, gallery: np.ndarray, probes: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    out = []
    for i in range(0, len(probes), chunk):
        sim = scheme.sim_matrix(probes[i:i + chunk], gallery)
        out.append(np.argsort(-sim, axis=1, kind="stable")[:, :k])
    return np.vstack(out)


def run_scale(cfg: dict, out_dir: Path) -> None:
    if faiss is None:
        raise ImportError("faiss is required for the scale experiment")
    faiss.omp_set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    emb = load_data(cfg["data"])
    sp = cfg["split"]
    split = make_split(emb.ids, sp["n_dev"], sp["n_tgt"], sp["seed"], sp.get("min_tgt_images", 3))
    dev_rows = split.rows(split.dev_ids)
    mean = emb.x[dev_rows].mean(0)
    xn = center_normalize(emb.x, mean)
    pre = make_preconditioner(cfg["preconditioner"]).fit(xn[dev_rows], emb.ids[dev_rows])
    tgt = np.asarray(split.tgt_ids)
    enr = np.array([split.enrolled_row(t) for t in tgt])
    probe = np.array([split.probe_rows(t)[0] for t in tgt])
    sizes = sorted(cfg["distractors"]["sizes"])
    # Relabel and identity exclusion happen in load_distractors (exclusion before the draw for SIN corpora).
    dis_all, n_loaded, shared_removed = load_distractors(cfg["distractors"], sizes[-1], emb, cfg["data"])
    # Drop distractors that duplicate a primary row (uncentered cosine >= near_dup_cos) before drawing subsets.
    dis = drop_near_duplicates(dis_all, emb, cfg["distractors"].get("near_dup_cos", 0.95))
    dn = center_normalize(dis.x, mean)
    zg, zp, zd = pre.transform(xn[enr]), pre.transform(xn[probe]), pre.transform(dn)
    # A requested size above the available rows is capped; capped sizes that coincide run once, labelled with the
    # largest requested size that maps to them (the full mapping goes to metadata.json).
    size_map = {int(n): int(min(n, len(dis.x))) for n in sizes}
    requested_for = {u: max(n for n, v in size_map.items() if v == u) for u in sorted(set(size_map.values()))}
    rows = []
    for scheme_spec in cfg["schemes"]:
        scheme = parse_scheme(scheme_spec)
        m = zg.shape[1]
        key = scheme.keygen(m, stable_seed("scale-key", scheme.name, m))
        cg, cp, cd = scheme.protect(zg, key), scheme.protect(zp, key), scheme.protect(zd, key)
        for n_used, n_d in requested_for.items():
            gallery = np.concatenate([cg, cd[:n_used]])
            labels = np.concatenate([np.arange(len(tgt)), np.full(n_used, -1)])
            exact = exact_topk(scheme, gallery, cp, 10)
            garr, binary = scheme.index_codes(gallery)
            parr, _ = scheme.index_codes(cp)
            dim = garr.shape[1] * 8 if binary else garr.shape[1]
            for ef in cfg["hnsw"]["ef_search"]:
                index = ProtectedVectorIndex(dim, m=cfg["hnsw"]["m"], ef_construction=cfg["hnsw"]["ef_construction"],
                                             ef_search=ef, is_binary=binary)
                build_s = index.add(garr)
                _, found, ms = index.search(parr, top_k=10)
                recall = float(np.mean([len(set(found[i]) & set(exact[i])) / 10.0 for i in range(len(found))]))
                recall_tie = tie_aware_recall(scheme, gallery, cp, exact, found, k=10)
                blob = faiss.serialize_index_binary(index.index) if binary else faiss.serialize_index(index.index)
                rows.append({
                    "scheme": scheme.name, "m": m, "k": scheme.k_for(m), "n_distractors": n_used,
                    "n_distractors_requested": n_d, "gallery": len(gallery), "ef_search": ef,
                    "rank1_exact": float(np.mean(labels[exact[:, 0]] == np.arange(len(tgt)))),
                    "rank1_hnsw": float(np.mean(labels[found[:, 0]] == np.arange(len(tgt)))),
                    "recall10_vs_exact": recall, "recall10_tie_aware": recall_tie,
                    "latency_ms_per_query": ms / len(parr),
                    "build_s": build_s, "index_serialized_mb": len(blob) / 1e6,
                })
    pd.DataFrame(rows).to_csv(out_dir / "scale.csv", index=False)
    meta = {
        "config": cfg,
        "data_source": emb.source,
        "distractor_source": dis.source,
        "corpus_counts": corpus_counts(emb.ids, sp.get("min_tgt_images", 3)),
        "split_counts": {str(sp["seed"]): split_counts(split)},
        "distractors_loaded": int(n_loaded),
        "distractors_available_before": int(len(dis_all.x)),
        "secondary_removed_shared_identity": shared_removed,
        "distractors_removed": int(len(dis_all.x) - len(dis.x)),
        "distractors_available_after": int(len(dis.x)),
        "distractor_sizes_requested_to_used": {str(n): u for n, u in size_map.items()},
        "spectrum_primary_dev": spectrum_stats(xn[dev_rows]),
        "spectrum_distractors": spectrum_stats(dn[: min(len(dn), 20000)]),
        "git_sha": os.environ.get("CANVEC_GIT_SHA", "unknown"),
        "image_freeze": _read_optional("/opt/canvec/requirements.freeze.txt"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "cpu_model": _cpu_model(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "data_files": {**(_data_file_hashes(cfg["data"]) if cfg["data"]["kind"] != "synthetic" else {}),
                       **{f"distractors:{k}": v for k, v in _distractor_hashes(cfg["distractors"]).items()}},
        "started_utc": started,
        "runtime_s": time.time() - t0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run_scale(yaml.safe_load(Path(args.config).read_text()), Path(args.out))


if __name__ == "__main__":
    main()
