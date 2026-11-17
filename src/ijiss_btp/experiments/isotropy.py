"""Isotropy experiment runner (spec Sections 5-7).

Writes aggregate CSV/JSON only; never writes embeddings, proxies or per-sample scores.

Stages: utility, anchor, zeropair and postcompromise (run together; the default) and joint (alone or with them;
`--stages joint`). The joint stage evaluates the joint alignment-plus-anchor collusion attack on LRP-full, exact
anchors and the pre-conditioners of `joint_preconditioners`, with the anchor stage's keys, thresholds, colluder order
and victims. Its table joint.csv has one row per (seed, key, pre, method, aux, p, r): method `proj` (the anchor stage's
projection decoder, for parity with anchor.csv; aux empty, p = 0), `align_anchor` (eigen alignment with anchor-voted
signs), `joint` (projection decoder on the anchor span plus align_anchor outside it) and `eigen_oracle` (true signs,
reference; r = 0); then n_targets, the pre-key cosines and the E1/E2 columns of anchor.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ijiss_btp.attacks.anchor import anchor_decode
from ijiss_btp.attacks.joint import AnchorSpan
from ijiss_btp.attacks.keyfree import EigenAligner, EigenBasis, ica_align_decoder, wasserstein_procrustes_decoder
from ijiss_btp.attacks.postcompromise import biohash_inverse, iom_preimage, lrp_lmmse_inverse
from ijiss_btp.data.loaders import (
    Embeddings,
    center_normalize,
    drop_near_duplicates,
    exclude_primary,
    load_keyed_npz,
    load_sin_corpus,
    sha256_file,
    synthetic_clusters,
)
from ijiss_btp.data.splits import corpus_counts, make_split, split_counts
from ijiss_btp.eval.endpoints import EndpointResult, _l2n, e1, e2
from ijiss_btp.eval.stats import (
    bootstrap_ratio_ci,
    clopper_pearson_upper,
    eer,
    make_pairs,
    rank1,
    rates_at_threshold,
    threshold_at_fmr,
)
from ijiss_btp.precondition import make_preconditioner
from ijiss_btp.protect.keyed import KeyedScheme, parse_scheme

CORE_STAGES = ("utility", "anchor", "zeropair", "postcompromise")
STAGES = CORE_STAGES + ("joint",)
JOINT_SCHEME = "lrp:full"
JOINT_EXTRA_R = (1, 2)
JOINT_METHODS = ("proj", "align_anchor", "joint", "eigen_oracle")


def _cos_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Pre-key-space cosine statistics: signed mean, mean |cos| and mean cos^2 over rows."""
    c = np.sum(_l2n(a) * _l2n(b), 1)
    return {"rec_cos_z": float(np.mean(c)), "rec_abs_cos_z": float(np.mean(np.abs(c))),
            "rec_cos2_z": float(np.mean(c ** 2))}


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from labels (independent of Python hash randomization)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def load_data(spec: dict) -> Embeddings:
    kind = spec["kind"]
    if kind == "sin":
        return load_sin_corpus(spec["root"], spec["model"], spec.get("relabel"), spec.get("relabel_key"))
    if kind == "keyed_npz":
        return load_keyed_npz(spec["path"])
    if kind == "synthetic":
        return synthetic_clusters(spec["n_ids"], spec["per_id"], spec["dim"], spec["decay"], spec["seed"])
    raise ValueError(f"unknown data kind {kind!r}")


def anisotropy_index(x: np.ndarray) -> float:
    """Scale-invariant ||S/(tr(S)/m) - I||_F / sqrt(m), S = covariance of the rows of x."""
    s = np.cov(x.T)
    m = s.shape[0]
    return float(np.linalg.norm(s / (np.trace(s) / m) - np.eye(m)) / np.sqrt(m))


def spectrum_stats(x: np.ndarray) -> dict:
    xc = x - x.mean(0)
    w = np.clip(np.linalg.eigvalsh(xc.T @ xc / max(len(x) - 1, 1))[::-1], 0.0, None)
    cum = np.cumsum(w) / w.sum()
    zc = xc / (xc.std(0) + 1e-12)
    return {
        "n": int(len(x)),
        "dim": int(x.shape[1]),
        "participation_ratio": float(w.sum() ** 2 / np.sum(w ** 2)),
        "dims_for_80pct_var": int(np.searchsorted(cum, 0.80) + 1),
        "dims_for_90pct_var": int(np.searchsorted(cum, 0.90) + 1),
        "top1_var_frac": float(w[0] / w.sum()),
        "anisotropy_index": anisotropy_index(x),
        "cumvar_first256": [float(c) for c in cum[:256]],
        "marginal_abs_skew": float(np.abs(np.mean(zc ** 3, 0)).mean()),
        "marginal_abs_exkurt": float(np.abs(np.mean(zc ** 4, 0) - 3.0).mean()),
    }


class SeedContext:
    """Identity split, DEV-mean centering and raw-space operating points for one split seed."""

    def __init__(self, emb: Embeddings, cfg: dict, seed: int, shifted: Embeddings | None) -> None:
        sp = cfg["split"]
        self.seed = seed
        self.split = make_split(emb.ids, sp["n_dev"], sp["n_tgt"], seed, sp.get("min_tgt_images", 3))
        self.ids = emb.ids
        self.dev_rows = self.split.rows(self.split.dev_ids)
        self.mean = emb.x[self.dev_rows].mean(0)
        self.xn = center_normalize(emb.x, self.mean)
        rng = np.random.default_rng(stable_seed("pairs", seed))
        self.dev_gen, self.dev_imp = make_pairs(self.dev_rows, emb.ids, rng)
        self.tgt_gen, self.tgt_imp = make_pairs(self.split.rows(self.split.tgt_ids), emb.ids, rng)
        self.tgt = np.asarray(self.split.tgt_ids)
        self.enr = np.array([self.split.enrolled_row(t) for t in self.tgt])
        self.known = np.array([self.split.known_row(t) for t in self.tgt])
        self.probes = np.stack([self.split.probe_rows(t) for t in self.tgt])
        imp_raw = np.sum(self.xn[self.dev_imp[:, 0]] * self.xn[self.dev_imp[:, 1]], 1)
        self.tau_raw = {f: threshold_at_fmr(imp_raw, f) for f in cfg["fmr"]}
        self.aux = {"same_dist": self.xn[self.split.rows(self.split.aux_ids)]}
        if shifted is not None:
            self.aux["shifted"] = center_normalize(shifted.x, self.mean)


def utility_rows(ctx: SeedContext, scheme: KeyedScheme, codes: np.ndarray, tag: dict, fmrs: list) -> list[dict]:
    g = scheme.sim_pairs(codes[ctx.tgt_gen[:, 0]], codes[ctx.tgt_gen[:, 1]])
    i = scheme.sim_pairs(codes[ctx.tgt_imp[:, 0]], codes[ctx.tgt_imp[:, 1]])
    dev_imp = scheme.sim_pairs(codes[ctx.dev_imp[:, 0]], codes[ctx.dev_imp[:, 1]])
    pr_rows = ctx.probes.reshape(-1)
    pr_lab = np.repeat(np.arange(len(ctx.tgt)), ctx.probes.shape[1])
    r1 = rank1(scheme.sim_matrix(codes[pr_rows], codes[ctx.enr]), pr_lab, np.arange(len(ctx.tgt)))
    e = eer(g, i)
    rows = []
    for f in fmrs:
        rates = rates_at_threshold(g, i, threshold_at_fmr(dev_imp, f))
        rows.append({**tag, "fmr_target": f, "eer": e, "tmr": rates["tmr"], "fmr_actual": rates["fmr"], "rank1": r1})
    return rows


def endpoint_cols(prefix: str, res: EndpointResult, n_boot: int, seed: int) -> dict:
    if res.genuine == 0:
        rel = lo = hi = float("nan")
    else:
        rel = res.rel
        lo, hi = bootstrap_ratio_ci(res.per_target_proxy, res.per_target_genuine, n_boot=n_boot, seed=seed)
    successes = int((res.per_target_proxy > 0).sum())
    trials = len(res.per_target_proxy)
    return {
        f"{prefix}_genuine": res.genuine,
        f"{prefix}_proxy": res.proxy,
        f"{prefix}_rel": rel,
        f"{prefix}_rel_lo": lo,
        f"{prefix}_rel_hi": hi,
        f"{prefix}_successes": successes,
        f"{prefix}_trials": trials,
        f"{prefix}_cp_upper": clopper_pearson_upper(successes, trials),
        f"{prefix}_pair_successes": res.n_success,
        f"{prefix}_pair_trials": res.n_trials,
    }


def _in_subset(subset: dict | None, pre_spec: str, scheme_spec: str, r: int) -> bool:
    if not subset:
        return False
    return pre_spec in subset.get("pres", []) and scheme_spec in subset.get("schemes", []) and r in subset.get("r", [])


class KeyedSystem:
    """One (seed, pre-conditioner, scheme, key) system as every stage sees it: enrolment under `key`, renewal to `key2`,
    the thresholds at the attack operating point (the first FMR of the configuration) and the E1/E2 scoring."""

    def __init__(self, ctx: SeedContext, pre, z: np.ndarray, scheme: KeyedScheme, key_idx: int, cfg: dict) -> None:
        m = z.shape[1]
        self.ctx, self.pre, self.scheme = ctx, pre, scheme
        self.tag = {"seed": ctx.seed, "key": key_idx, "pre": pre.name, "scheme": scheme.name, "m": m,
                    "k": scheme.k_for(m)}
        kseed = stable_seed("key", ctx.seed, key_idx, scheme.name, m)
        self.key, self.key2 = scheme.keygen(m, kseed), scheme.keygen(m, kseed + 1)
        self.codes, self.codes2 = scheme.protect(z, self.key), scheme.protect(z, self.key2)
        fmr = cfg["fmr"][0]
        self.tau_raw = ctx.tau_raw[fmr]
        self.tau1 = threshold_at_fmr(scheme.sim_pairs(self.codes[ctx.dev_imp[:, 0]], self.codes[ctx.dev_imp[:, 1]]),
                                     fmr)
        self.tau2 = threshold_at_fmr(scheme.sim_pairs(self.codes2[ctx.dev_imp[:, 0]], self.codes2[ctx.dev_imp[:, 1]]),
                                     fmr)
        self.n_boot = cfg.get("bootstrap", {}).get("n_boot", 1000)
        self.feats = scheme.features(self.codes)

    def endpoints(self, xhat: np.ndarray, idx: np.ndarray, salt: str) -> dict:
        ctx = self.ctx
        r1 = e1(ctx.xn, xhat, ctx.enr[idx], ctx.probes[idx], self.tau_raw)
        r2 = e2(self.scheme, self.pre, self.key2, self.codes2, xhat, ctx.enr[idx], ctx.probes[idx], self.tau2)
        bseed = stable_seed("boot", salt)
        return {"rec_cos": r1.rec_cos, **endpoint_cols("e1", r1, self.n_boot, bseed),
                **endpoint_cols("e2", r2, self.n_boot, bseed)}


def valid_budgets(acfg: dict, n_t: int) -> list[int]:
    """The anchor stage's budgets: the r of r_grid that leave at least min_targets victims among n_t targets."""
    return [r for r in acfg["r_grid"] if r <= n_t - acfg.get("min_targets", 100)]


def anchor_victims(ctx: SeedContext, key_idx: int, acfg: dict) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Budgets, colluder order and victims of the anchor stage: one permutation per (seed, key), colluders perm[:r]
    (nested), victims perm[max(valid_r):], shared across pre-conditioners and schemes."""
    n_t = len(ctx.tgt)
    valid_r = valid_budgets(acfg, n_t)
    perm = np.random.default_rng(stable_seed("anchors", ctx.seed, key_idx)).permutation(n_t)
    t_idx = perm[max(valid_r):] if valid_r else perm[:0]
    return valid_r, perm, t_idx


def joint_budgets(valid_r: list[int]) -> list[int]:
    """The anchor stage's budgets plus JOINT_EXTRA_R; the largest budget, and with it the victims, must not change."""
    if not valid_r:
        raise ValueError("no budget of r_grid leaves min_targets victims, so the anchor stage has no victims")
    budgets = sorted(set(valid_r) | set(JOINT_EXTRA_R))
    if max(budgets) != max(valid_r):
        raise ValueError(f"r = {max(budgets)} exceeds the largest budget {max(valid_r)} and would change the victims")
    return budgets


def joint_preconditioners(specs: list[str]) -> list[str]:
    """The joint stage's pre-conditioners in configuration order: none, PCA whitening (alpha 1) and FI."""
    def keep(spec: str) -> bool:
        kind, *args = spec.split(":")
        return spec == "none" or kind == "fi" or (kind == "pca" and float(args[1]) == 1.0)
    return [s for s in specs if keep(s)]


def select_stages(stages) -> tuple[str, ...]:
    """Validated stage selection in canonical order; None selects the core stages (utility to postcompromise)."""
    stages = tuple(CORE_STAGES if stages is None else stages)
    unknown = sorted(set(stages) - set(STAGES))
    if unknown or not stages:
        raise ValueError(f"unknown or empty stage selection {list(stages)}; choose from {list(STAGES)}")
    core = [s for s in CORE_STAGES if s in stages]
    if core and len(core) != len(CORE_STAGES):
        raise ValueError(f"the stages {', '.join(CORE_STAGES)} run together; got {', '.join(core)}")
    return tuple(s for s in STAGES if s in stages)


def run_block(ctx: SeedContext, pre_spec: str, pre, z: np.ndarray, aux_z: dict, scheme_spec: str,
              key_idx: int, cfg: dict, out: dict) -> None:
    """One (seed, pre-conditioner, scheme, key) block: utility, T1/T1', T0 and T2."""
    scheme = parse_scheme(scheme_spec)
    m = z.shape[1]
    try:
        k = scheme.k_for(m)
    except ValueError as exc:
        print(f"skip pre={pre.name} scheme={scheme.name}: {exc}", flush=True)
        return
    system = KeyedSystem(ctx, pre, z, scheme, key_idx, cfg)
    tag, key, codes, feats, endpoints = system.tag, system.key, system.codes, system.feats, system.endpoints
    if key_idx == 0:
        out["utility"] += utility_rows(ctx, scheme, codes, tag, cfg["fmr"])

    n_t = len(ctx.tgt)
    acfg = cfg["attacks"]["anchor"]
    valid_r, perm, t_idx = anchor_victims(ctx, key_idx, acfg)
    for r in valid_r:
        a_idx = perm[:r]
        for variant in acfg["variants"]:
            known = ctx.enr[a_idx] if variant == "exact" else ctx.known[a_idx]
            for decoder in acfg["decoders"]:
                if decoder == "mlp" and not _in_subset(acfg.get("mlp_subset"), pre_spec, scheme_spec, r):
                    continue
                zhat = anchor_decode(feats[ctx.enr[a_idx]], z[known], feats[ctx.enr[t_idx]], decoder, seed=r)
                salt = f"{ctx.seed}|{key_idx}|{pre.name}|{scheme.name}|{r}|{variant}|{decoder}"
                out["anchor"].append({**tag, "variant": variant, "decoder": decoder, "r": r,
                                      "n_targets": len(t_idx), **_cos_stats(zhat, z[ctx.enr[t_idx]]),
                                      **endpoints(pre.inverse(zhat), t_idx, salt)})

    zcfg = cfg["attacks"]["zeropair"]
    all_idx = np.arange(n_t)
    stolen = feats[ctx.enr]
    if scheme.kind != "iom":
        for aux_name, a in aux_z.items():
            aligner = EigenAligner(stolen, a)
            for p in zcfg["p_grid"]:
                if p > k:
                    continue
                for method, oracle in (("eigen", None), ("eigen_oracle", key)):
                    zh = aligner.decoder(p, oracle_key=oracle)(stolen)
                    xhat = pre.inverse(zh)
                    salt = f"{ctx.seed}|{key_idx}|{pre.name}|{scheme.name}|{aux_name}|{method}|{p}"
                    out["zeropair"].append({**tag, "aux": aux_name, "method": method, "p": p,
                                            **_cos_stats(zh, z[ctx.enr]), **endpoints(xhat, all_idx, salt)})
            if key_idx < zcfg.get("expensive_keys", 1) and scheme.kind == "lrp" and k == m:
                decs = []
                if "wp" in zcfg["methods"]:
                    decs.append(("wp", wasserstein_procrustes_decoder(stolen, a, seed=stable_seed("t0", ctx.seed, key_idx, pre.name, scheme.name, aux_name, "wp"))))
                if "ica" in zcfg["methods"] and m <= zcfg.get("ica_max_dim", 128):
                    decs.append(("ica", ica_align_decoder(stolen, a, seed=stable_seed("t0", ctx.seed, key_idx, pre.name, scheme.name, aux_name, "ica"))))
                for method, dec in decs:
                    salt = f"{ctx.seed}|{key_idx}|{pre.name}|{scheme.name}|{aux_name}|{method}"
                    zh = dec(stolen)
                    out["zeropair"].append({**tag, "aux": aux_name, "method": method, "p": m,
                                            **_cos_stats(zh, z[ctx.enr]),
                                            **endpoints(pre.inverse(zh), all_idx, salt)})

    pcfg = cfg["attacks"].get("postcompromise", {})
    if pcfg.get("enabled", False) and key_idx < pcfg.get("keys", 1):
        # T2 attacker holds the key, the stolen templates and public AUX (not DEV).
        a_same = aux_z["same_dist"]
        if scheme.kind == "lrp":
            analytic = lrp_lmmse_inverse(codes[ctx.enr], key, np.cov(a_same.T))
        elif scheme.kind == "biohash":
            analytic = biohash_inverse(codes[ctx.enr], key, float(np.std(a_same @ key)))
        else:
            analytic = iom_preimage(codes[ctx.enr], key, scheme.block)
        # Learned inverse: with the key, the attacker labels public AUX embeddings and fits a decoder.
        learned = anchor_decode(scheme.features(scheme.protect(a_same, key)), a_same, stolen, "ridge")
        for method, zhat in (("analytic", analytic), ("learned", learned)):
            in_system = float(np.mean(scheme.sim_pairs(scheme.protect(zhat, key), codes[ctx.enr]) >= system.tau1))
            salt = f"{ctx.seed}|{key_idx}|{pre.name}|{scheme.name}|t2|{method}"
            out["postcompromise"].append({**tag, "method": method, "in_system_accept": in_system,
                                          **endpoints(pre.inverse(zhat), all_idx, salt)})


def run_joint_block(ctx: SeedContext, pre, z: np.ndarray, aux_bases: dict, key_idx: int, cfg: dict,
                    out: dict) -> None:
    """Joint stage for one (seed, pre-conditioner, key) on LRP-full: the anchor stage's key, thresholds, colluders
    perm[:r] (exact anchors) and victims; aux_bases holds the AUX eigendecompositions of this (seed, pre-conditioner)."""
    system = KeyedSystem(ctx, pre, z, parse_scheme(JOINT_SCHEME), key_idx, cfg)
    valid_r, perm, t_idx = anchor_victims(ctx, key_idx, cfg["attacks"]["anchor"])
    budgets = joint_budgets(valid_r)
    p_grid = sorted({min(p, z.shape[1]) for p in cfg["attacks"]["zeropair"]["p_grid"]})
    victims = ctx.enr[t_idx]
    f_v, z_v = system.feats[victims], z[victims]
    stolen = EigenBasis.of(system.feats[ctx.enr])   # the whole stolen database, colluders included
    aligners = {name: EigenAligner(stolen, basis) for name, basis in aux_bases.items()}
    prefix = f"{ctx.seed}|{key_idx}|{pre.name}|{system.scheme.name}"

    def add(method: str, aux: str | None, p: int, r: int, zhat: np.ndarray, salt: str) -> None:
        out["joint"].append({**system.tag, "method": method, "aux": aux, "p": p, "r": r, "n_targets": len(t_idx),
                             **_cos_stats(zhat, z_v), **system.endpoints(pre.inverse(zhat), t_idx, salt)})

    for aux_name, aligner in aligners.items():
        for p in p_grid:
            add("eigen_oracle", aux_name, p, 0, aligner.decoder(p, oracle_key=system.key)(f_v),
                f"{prefix}|joint|{aux_name}|eigen_oracle|{p}")
    for r in budgets:
        a_rows = ctx.enr[perm[:r]]
        anchors = (system.feats[a_rows], z[a_rows])
        span = AnchorSpan(*anchors)
        z_proj = span.project(f_v)
        add("proj", None, 0, r, z_proj, f"{prefix}|{r}|exact|proj")   # the anchor stage's salt, hence its bootstrap
        for aux_name, aligner in aligners.items():
            for p in p_grid:
                z_aligned = aligner.decoder(p, anchors=anchors)(f_v)
                add("align_anchor", aux_name, p, r, z_aligned, f"{prefix}|joint|{aux_name}|align_anchor|{p}|{r}")
                add("joint", aux_name, p, r, span.joint(z_proj, z_aligned), f"{prefix}|joint|{aux_name}|joint|{p}|{r}")


def _read_optional(path: str) -> str | None:
    p = Path(path)
    return p.read_text() if p.exists() else None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _data_file_hashes(spec: dict) -> dict:
    files: dict = {}
    if spec["kind"] == "sin":
        for name in (f"emb_{spec['model']}.npy", "manifest.parquet"):
            files[name] = sha256_file(Path(spec["root"]) / name)
        if spec.get("relabel"):
            files[f"relabel:{Path(spec['relabel']).name}"] = sha256_file(spec["relabel"])
    elif spec["kind"] == "keyed_npz":
        files[Path(spec["path"]).name] = sha256_file(spec["path"])
    if spec.get("shifted_aux"):
        files.update({f"shifted:{k}": v for k, v in _data_file_hashes(spec["shifted_aux"]).items()})
    return files


def run(cfg: dict, out_dir: Path, stages=None) -> None:
    """Run the selected stages (default: the configuration's `stages`, else the core stages) into a new directory."""
    if stages is not None and cfg.get("stages") is not None and select_stages(stages) != select_stages(cfg["stages"]):
        raise ValueError(f"the stage selection {list(stages)} contradicts the configuration's stages {cfg['stages']}")
    stages = select_stages(cfg.get("stages") if stages is None else stages)
    core = any(s in CORE_STAGES for s in stages)
    joint_pres = joint_preconditioners(cfg["preconditioners"]) if "joint" in stages else []
    if "joint" in stages and (JOINT_SCHEME not in cfg["schemes"] or not joint_pres):
        raise ValueError(f"the joint stage needs the scheme {JOINT_SCHEME} and a pre-conditioner of "
                         "joint_preconditioners in the configuration")
    out_dir = Path(out_dir)
    if (out_dir / "metadata.json").exists():
        raise FileExistsError(f"{out_dir} already holds a run (metadata.json); write every run to a new directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    emb = load_data(cfg["data"])
    shifted = None
    shifted_removed = None
    shared_removed = None
    sh_spec = cfg["data"].get("shifted_aux")
    if sh_spec:
        # relabel (in load_data), then drop identities of the primary corpus, then near-duplicates of primary rows
        before, shared_removed = exclude_primary(sh_spec, load_data(sh_spec), emb, cfg["data"])
        shifted = drop_near_duplicates(before, emb, sh_spec.get("near_dup_cos", 0.95))
        shifted_removed = len(before.x) - len(shifted.x)
    out: dict[str, list] = {name: [] for name in stages}
    aux_sizes: dict = {}
    splits: dict = {}
    spectra: dict = {"raw_uncentered": spectrum_stats(emb.x), "preconditioned": {}} if core else {}
    for s_i, seed in enumerate(cfg["split"]["seeds"]):
        ctx = SeedContext(emb, cfg, seed, shifted)
        splits[str(seed)] = split_counts(ctx.split)
        if s_i == 0:
            aux_sizes = {name: int(len(a)) for name, a in ctx.aux.items()}
            if core:
                spectra["raw_centered_normalized"] = spectrum_stats(ctx.aux["same_dist"])
        for pre_spec in cfg["preconditioners"]:
            if not core and pre_spec not in joint_pres:
                continue
            pre = make_preconditioner(pre_spec).fit(ctx.xn[ctx.dev_rows], ctx.ids[ctx.dev_rows])
            z = pre.transform(ctx.xn)
            aux_z = {name: pre.transform(a) for name, a in ctx.aux.items()}
            if core:
                if s_i == 0:
                    st = spectrum_stats(aux_z["same_dist"])
                    spectra["preconditioned"][pre.name] = st
                for scheme_spec in cfg["schemes"]:
                    for key_idx in range(cfg["keys_per_seed"]):
                        run_block(ctx, pre_spec, pre, z, aux_z, scheme_spec, key_idx, cfg, out)
            if pre_spec in joint_pres:
                aux_bases = {name: EigenBasis.of(a) for name, a in aux_z.items()}   # one per (seed, pre, aux)
                for key_idx in range(cfg["keys_per_seed"]):
                    run_joint_block(ctx, pre, z, aux_bases, key_idx, cfg, out)
            print(f"[{time.time() - t0:8.1f}s] seed={seed} pre={pre.name} done", flush=True)
    for name, rows in out.items():
        pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    if core:
        (out_dir / "spectra.json").write_text(json.dumps(spectra, indent=2))
    meta = {
        "config": cfg,
        "stages": list(stages),
        **({"joint": {"scheme": JOINT_SCHEME, "preconditioners": joint_pres, "anchors": "exact",
                      "extra_r": list(JOINT_EXTRA_R), "methods": list(JOINT_METHODS), "attack_fmr": cfg["fmr"][0]}}
           if "joint" in stages else {}),
        "data_source": emb.source,
        "data_files": _data_file_hashes(cfg["data"]) if cfg["data"]["kind"] != "synthetic" else {},
        "git_sha": os.environ.get("CANVEC_GIT_SHA", "unknown"),
        "image_freeze": _read_optional("/opt/canvec/requirements.freeze.txt"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu_model": _cpu_model(),
        "corpus_counts": corpus_counts(emb.ids, cfg["split"].get("min_tgt_images", 3)),
        "split_counts": splits,
        "aux_sizes": aux_sizes,
        "shifted_aux_source": shifted.source if shifted is not None else None,
        "shifted_aux_removed": shifted_removed,
        "secondary_removed_shared_identity": shared_removed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "started_utc": started,
        "runtime_s": time.time() - t0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


def load_config(path: str | Path) -> dict:
    """Read a run configuration. A wrapper {base: <file next to it>, stages: [...]} reuses a configuration unchanged
    and adds only the stage selection."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" not in cfg:
        return cfg
    if set(cfg) != {"base", "stages"}:
        raise ValueError(f"{path}: a wrapper configuration holds only 'base' and 'stages'; got {sorted(cfg)}")
    base = yaml.safe_load((path.parent / cfg["base"]).read_text())
    if "base" in base or "stages" in base:
        raise ValueError(f"{path}: the base configuration {cfg['base']} must not be a wrapper or select stages")
    return {**base, "stages": list(cfg["stages"])}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=None,
                    help="stages to run (default: the configuration's 'stages', else the four core stages)")
    args = ap.parse_args(argv)
    run(load_config(args.config), Path(args.out), stages=args.stages)


if __name__ == "__main__":
    main()
