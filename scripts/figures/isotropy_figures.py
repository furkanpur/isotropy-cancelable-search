"""Paper figures from aggregated isotropy results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

from ijiss_btp.experiments.aggregate import r_at_level  # noqa: E402
from ijiss_btp.theory import collusion_cos_optimum  # noqa: E402

READABLE_TICKS = (30, 40, 60, 100, 200, 300, 400, 600)


def plain_log_ticks(ax, ticks=READABLE_TICKS) -> None:
    """Major ticks of a logarithmic y axis at the readable values within the view, printed as plain integers (not
    6 x 10^2); minor ticks keep their marks but carry no label."""
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def seed_curves(anchor: pd.DataFrame) -> pd.DataFrame:
    """Per-seed relative-success curves of a run's anchor rows: E1 and E2 averaged over keys at each (decoder, r)."""
    return anchor.groupby(["pre", "scheme", "variant", "decoder", "seed", "r"], as_index=False)[["e1_rel", "e2_rel"]].mean()


def seed_r50_range(curves: pd.DataFrame, pre: str, endpoint: str, scheme: str = "lrp_full",
                   variant: str = "exact") -> dict | None:
    """Per-seed r50 range of one (pre, scheme, variant, endpoint) cell of `seed_curves`; None if the cell is empty.

    For each split seed, r50 is interpolated per decoder with `aggregate.r_at_level` (log-linear) and the best decoder,
    the smallest r50, is kept (the rule of `best_decoder_status`). A seed where no decoder reaches 0.5 is censored and
    counts as r_max. Returns dict(lo, hi, rmax, n, n_cens, cens_any, cens_all).
    """
    sub = curves[(curves.pre == pre) & (curves.scheme == scheme) & (curves.variant == variant)]
    if not len(sub):
        return None
    rmax = float(sub.r.max())
    vals, cens = [], []
    for _, ps in sub.groupby("seed", sort=True):
        reached = [v for v in (r_at_level(d.r, d[f"{endpoint}_rel"]) for _, d in ps.groupby("decoder", sort=True))
                   if v is not None]
        cens.append(not reached)
        vals.append(min(reached) if reached else rmax)
    return {"lo": min(vals), "hi": max(vals), "rmax": rmax, "n": len(vals), "n_cens": int(sum(cens)),
            "cens_any": any(cens), "cens_all": all(cens)}


# One naming scheme in text, tables and figures: the names of the paper, not the code identifiers.
ENC_LAB = {"d1_base": "D1 base", "d1_large": "D1 large", "d2_fingervit": "finger-vit", "d2_deepprint": "DeepPrint",
           "d2_ctrlvit": "ctrl-vit"}
# One encoder order (the order of the tables) and one colour per encoder in every figure and every copy of a figure
ENC_ORDER = ["d1_base", "d1_large", "d2_fingervit", "d2_deepprint", "d2_ctrlvit"]
ENC_COLOR = {e: f"C{i}" for i, e in enumerate(ENC_ORDER)}   # C0..C4 of the matplotlib colour cycle


def enc_rank(name: str) -> tuple:
    """Sort key: the position in ENC_ORDER; an unknown encoder follows, alphabetically."""
    return (ENC_ORDER.index(name) if name in ENC_ORDER else len(ENC_ORDER), name)


PRE_LAB = {"none": "none", "pca128": "PCA$_{128}$", "pca256": "PCA$_{256}$", "pca128h": "PCA$_{128}^{1/2}$",
           "fi64": "FI$_{64}$", "fi128": "FI$_{128}$", "fi256": "FI$_{256}$", "rbig128": "RBIG$_{128}$",
           "firbig128": "FI+RBIG$_{128}$"}
# Okabe-Ito colour-blind-safe palette and one marker per pre-conditioner
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#882255"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]


def _enc_label(name: str) -> str:
    return ENC_LAB.get(name, name)


def _name(run_dir: Path) -> str:
    return json.loads((run_dir / "metadata.json").read_text())["config"]["name"]


def _spec(run_dir: Path) -> dict:
    return json.loads((run_dir / "spectra.json").read_text())["raw_centered_normalized"]


def anchor_decoder(df: pd.DataFrame) -> str:
    """Decoder drawn in the anchor-theory figure: the projection decoder, ridge only if proj rows are absent."""
    return "proj" if (df.decoder == "proj").any() else "ridge"


def wider_interval(x, col: str):
    """Reported 95% interval of an E1/E2-type relative-success column: the elementwise union (wider) of the
    identity-level interval (`{col}_cons_lo/_cons_hi`) and the seed-level Student-t interval (`{col}_lo/_hi`).

    lower = min(lo_cons, lo_t), upper = max(hi_cons, hi_t); a non-finite component is ignored. Works on a Series
    (one row, returns floats) or a DataFrame (returns two Series). Truncation at 0 is left to the formatter.
    """
    lo = np.fmin(x[f"{col}_cons_lo"], x[f"{col}_lo"])
    hi = np.fmax(x[f"{col}_cons_hi"], x[f"{col}_hi"])
    if np.ndim(lo) == 0:
        return float(lo), float(hi)
    return lo, hi


def best_decoder_status(summary: Path, endpoint: str, scheme: str = "lrp_full", variant: str = "exact") -> pd.DataFrame:
    """One row per (encoder, pre) for an endpoint ("e1" or "e2") with the best decoder and a status.

    Best decoder = smallest r50 over the decoders present (if none reaches 0.5: the decoder with the highest
    mean relative success at r_max). The key-known (T2) level is the mean endpoint relative success of the T2
    method with the highest mean (summary_postcompromise.csv, same encoder/pre/scheme); `ceiling` is that mean and
    `ceiling_hi` the upper bound of its reported interval (wider of the identity-level and seed-level intervals, `wider_interval`). Status: "reached" if some
    decoder reaches 0.5; "backmap_limited" if not, the scheme is lrp_full with exact anchors and ceiling_hi < 0.5
    (this bounds only attacks that reconstruct the pre-key vector and use the public LMMSE back-map); else
    "censored" (0.5 not reached up to r_max). An observed r50 is never overridden.
    Columns: encoder, pre, scheme, variant, m, decoder, r50, rmax, ceiling, ceiling_hi, status.
    """
    r50 = pd.read_csv(summary / "r50.csv")
    anchor = pd.read_csv(summary / "summary_anchor.csv")
    post_path = summary / "summary_postcompromise.csv"
    top_rows = pd.DataFrame(columns=["encoder", "pre"]).set_index(["encoder", "pre"])
    if post_path.exists():
        post = pd.read_csv(post_path)
        post = post[post.scheme == scheme].sort_values([f"{endpoint}_rel", "method"], ascending=[False, True],
                                                       kind="stable")
        top_rows = post.groupby(["encoder", "pre"]).head(1).set_index(["encoder", "pre"])
    r50 = r50[(r50.scheme == scheme) & (r50.variant == variant)]
    a = anchor[(anchor.scheme == scheme) & (anchor.variant == variant)]
    rows = []
    for (enc, pre), part in r50.groupby(["encoder", "pre"], sort=True):
        ap = a[(a.encoder == enc) & (a.pre == pre)]
        rmax = float(ap["r"].max())
        part = part.sort_values("decoder", kind="stable")
        reached = part[np.isfinite(part[f"r50_{endpoint}"])]
        if len(reached):
            best = reached.sort_values(f"r50_{endpoint}", kind="stable").iloc[0]
        else:
            top = ap[ap["r"] == rmax].set_index("decoder")[f"{endpoint}_rel"].sort_index()
            dec = top.idxmax() if len(top) and top.notna().any() else part.iloc[0].decoder
            best = part[part.decoder == dec].iloc[0]
        c = c_hi = np.nan
        if (enc, pre) in top_rows.index:
            c = float(top_rows.loc[(enc, pre), f"{endpoint}_rel"])
            c_hi = wider_interval(top_rows.loc[(enc, pre)], f"{endpoint}_rel")[1]
        v = float(best[f"r50_{endpoint}"])
        status = ("reached" if np.isfinite(v)
                  else "backmap_limited" if scheme == "lrp_full" and variant == "exact" and np.isfinite(c_hi) and c_hi < 0.5
                  else "censored")
        rows.append({"encoder": enc, "pre": pre, "scheme": scheme, "variant": variant, "m": int(best.m),
                     "decoder": best.decoder, "r50": v, "rmax": rmax, "ceiling": c, "ceiling_hi": c_hi, "status": status})
    return pd.DataFrame(rows, columns=["encoder", "pre", "scheme", "variant", "m", "decoder", "r50", "rmax",
                                       "ceiling", "ceiling_hi", "status"])


# Print widths of the manuscript figures as fractions of the 390 pt text width (the \includegraphics widths of
# manuscript/sections/results.tex); a figure drawn at its print width prints its fonts at their nominal size.
TEXT_WIDTH_IN = 390 / 72
SPECTRA_PRINT = 0.75
ANCHOR_PRINT = 0.8
BUDGET_PRINT = 0.75


def fig_spectra(run_dirs, out: Path, width: float = SPECTRA_PRINT * TEXT_WIDTH_IN) -> None:
    fig, ax = plt.subplots(figsize=(width, 0.63 * width))
    for d in sorted(map(Path, run_dirs), key=lambda d: enc_rank(_name(d))):
        spec = _spec(d)
        cum = np.asarray(spec["cumvar_first256"])
        ax.plot(np.arange(1, len(cum) + 1), cum, color=ENC_COLOR.get(_name(d)), label=f"{_enc_label(_name(d))} (PR={spec['participation_ratio']:.0f})")
    ax.set_xscale("log")
    ax.set_xlabel("principal components", fontsize=8)
    ax.set_ylabel("cumulative explained variance", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_spectra.pdf")
    plt.close(fig)


def fig_anchor_theory(summary: Path, out: Path, scheme: str = "lrp_full",
                      width: float = ANCHOR_PRINT * TEXT_WIDTH_IN) -> None:
    curves = pd.read_csv(summary / "summary_anchor.csv")
    base = curves[(curves.scheme == scheme) & (curves.variant == "exact")]
    dec = anchor_decoder(base)
    sel = base[base.decoder == dec]
    encs = sorted(sel.encoder.unique(), key=enc_rank)
    if not encs:
        return
    ncols = 2 if len(encs) > 1 else 1
    nrows = int(np.ceil(len(encs) / ncols))
    # at a width of 5.4 in, two rows of panels are 4.6 in tall and more rows share that height, so that the figure
    # leaves room for text on its page; the figure is drawn at its print width with that aspect ratio
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, min(2.1 * nrows + 0.4, 4.6) * width / 5.4), squeeze=False,
                             sharey=True)
    flat = list(axes.flat)
    for ax, enc in zip(flat, encs):
        part = sel[sel.encoder == enc]
        fi = [p for p in part.pre.unique() if p.startswith("fi128")]
        pca = [p for p in part.pre.unique() if p.startswith("pca128_a1")]
        configs = [("none", "-o")] + [(p, "-s") for p in fi[:1]] + [(p, "-^") for p in pca[:1]]
        rr = np.linspace(1, 127, 400)
        for pre, style in configs:
            df = part[part.pre == pre].sort_values("r")
            if not len(df):
                continue
            ax.plot(df.r, df.rec_cos_z, style, ms=3.5, lw=1.1, label=_pre_label(pre))
            if pre == "none":
                m_none = float(df.m.iloc[0])
                rn = np.linspace(1, m_none - 1, 400)
                ax.plot(rn, collusion_cos_optimum(rn, m_none), ":", color="grey", lw=1.1, label=r"$c^*(r,d)$")
        ax.plot(rr, collusion_cos_optimum(rr, 128), "-", color="black", lw=1.0, label=r"$c^*(r,128)$")
        ax.plot(rr, np.sqrt(rr / 128), "k--", lw=1.0, label=r"$\sqrt{r/128}$")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title(_enc_label(enc), fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in flat[len(encs):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("malicious enrollees $r$", fontsize=8)
    spare = flat[len(encs):]
    if spare:   # the legend takes a free panel, so that it covers no curve
        handles, labels = flat[0].get_legend_handles_labels()
        spare[0].set_visible(True)
        spare[0].axis("off")
        spare[0].legend(handles, labels, fontsize=8, loc="center", frameon=False)
    else:
        flat[0].legend(fontsize=7.5, loc="upper left")
    fig.supylabel(f"pre-key reconstruction cosine (T1 exact, {'projection' if dec == 'proj' else dec} decoder)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_anchor_theory.pdf")
    plt.close(fig)


def _best_zero_pair_points(z: pd.DataFrame, raws: dict, methods, scheme: str = "lrp_full") -> pd.DataFrame:
    """Per (encoder, pre): best E1 relative over `methods` and AUX; for zero-success cells the CP95 relative upper."""
    rows = []
    sel = z[z.method.isin(methods) & (z.scheme == scheme) & z.encoder.isin(raws)]
    for (enc, pre), part in sel.groupby(["encoder", "pre"]):
        top = part.sort_values(["best_e1_rel", "best_rec_cos_z"], ascending=False, kind="stable").iloc[0]
        zero = bool(top.best_e1_rel == 0.0)
        cp = np.nan
        if zero:
            raw = raws[enc]
            sub = raw[(raw.pre == pre) & (raw.scheme == scheme) & (raw.method == top.method) & (raw.aux == top.aux)]
            cp = float((sub["e1_cp_upper"] / sub["e1_genuine"]).max())
        rows.append({"encoder": enc, "pre": pre, "best_e1_rel": float(top.best_e1_rel), "zero": zero, "cp_rel": cp})
    return pd.DataFrame(rows, columns=["encoder", "pre", "best_e1_rel", "zero", "cp_rel"])


_SHORT = [(r"pca128_a0\.5", "pca128h"), (r"pca(\d+)_a1", r"pca\1"), (r"fi(\d+)_p\d+", r"fi\1"),
          (r"firbig128_p\d+_i2", "firbig128"), (r"rbig128_i2", "rbig128")]


def _short_pre(pre: str) -> str:
    """Canonical short key of a pre-conditioner name (the CSV keeps the full name)."""
    import re
    for pat, rep in _SHORT:
        if re.fullmatch(pat, pre):
            return re.sub(pat, rep, pre)
    return pre


def _pre_label(pre: str) -> str:
    """Paper name of a pre-conditioner for figure text."""
    return PRE_LAB.get(_short_pre(pre), pre)


def _place_labels(fig, ax, items, fontsize: float = 6) -> None:
    """Annotate (x, y, text) items without overlapping each other or the markers (layout only)."""
    import itertools
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = [Bbox.from_extents(px - 4, py - 4, px + 4, py + 4)
              for px, py in ax.transData.transform([(x, y) for x, y, _ in items])]
    angles = [0, 45, -45, 90, -90, 135, -135, 180]
    cands = [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for r, a in itertools.product((5, 10, 16, 24, 34, 46), angles)]
    for x, y, text in items:
        best = None
        for dx, dy in cands:
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                              ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top",
                              arrowprops=dict(arrowstyle="-", lw=0.3, color="grey") if np.hypot(dx, dy) > 12 else None)
            bb = ann.get_window_extent(renderer).expanded(1.05, 1.15)
            inside = ax.get_window_extent(renderer).contains(bb.x0, bb.y0) and ax.get_window_extent(renderer).contains(bb.x1, bb.y1)
            if inside and not any(bb.overlaps(o) for o in placed):
                best = bb
                break
            ann.remove()
        if best is not None:
            placed.append(best)


def fig_zeropair(summary: Path, run_dirs, out: Path) -> None:
    z = pd.read_csv(summary / "summary_zeropair.csv")
    pr = {_name(Path(d)): _spec(Path(d))["participation_ratio"] for d in run_dirs}
    raws = {_name(Path(d)): pd.read_csv(Path(d) / "zeropair.csv") for d in run_dirs}
    main_pts = _best_zero_pair_points(z, raws, ["eigen", "wp"])   # methods common to every pre
    ica = _best_zero_pair_points(z, raws, ["ica"])                # FastICA ran only for m <= 128
    for df_ in (main_pts, ica):
        df_["key"] = df_.pre.map(_short_pre)
    keys = [k for k in PRE_LAB if k in set(main_pts.key) | set(ica.key)] + sorted(
        (set(main_pts.key) | set(ica.key)) - set(PRE_LAB))
    color = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}
    marker = {k: MARKERS[i % len(MARKERS)] for i, k in enumerate(keys)}
    jit = {k: 10 ** (0.014 * (i - (len(keys) - 1) / 2)) for i, k in enumerate(keys)}   # horizontal offset, layout only
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    for key in keys:
        df = main_pts[main_pts.key == key]
        obs = df[~df.zero]
        if len(obs):
            ax.scatter([pr[e] * jit[key] for e in obs.encoder], obs.best_e1_rel, s=26, marker=marker[key],
                       color=color[key], edgecolors="white", linewidths=0.4, label=PRE_LAB.get(key, key), zorder=3)
        else:
            ax.scatter([], [], s=26, marker=marker[key], color=color[key], label=PRE_LAB.get(key, key))
        zr = df[df.zero]
        if len(zr):   # zero-success cells: open marker pointing down at the CP95 upper bound
            ax.scatter([pr[e] * jit[key] for e in zr.encoder], zr.cp_rel, s=30, marker="v", facecolors="none",
                       edgecolors=color[key], linewidths=1.0, zorder=3)
    if len(ica):
        ob = ica[~ica.zero]
        ax.scatter([pr[e] * jit[k] for e, k in zip(ob.encoder, ob.key)], ob.best_e1_rel, s=30, marker="*",
                   color="black", label="ICA ($m\\leq128$)", zorder=4)
        zi = ica[ica.zero]
        if len(zi):
            ax.scatter([pr[e] * jit[k] for e, k in zip(zi.encoder, zi.key)], zi.cp_rel, s=30, marker="v",
                       facecolors="none", edgecolors="black", linewidths=1.0, zorder=4)
    ax.scatter([], [], s=30, marker="v", facecolors="none", edgecolors="grey", label="0 successes (CP95 upper bound)")
    # zero-effort level of each encoder: target FMR divided by the mean raw genuine acceptance
    shown = False
    for d in run_dirs:
        name = _name(Path(d))
        fmr = json.loads((Path(d) / "metadata.json").read_text())["config"].get("fmr")
        gen = raws[name]["e1_genuine"].mean() if "e1_genuine" in raws[name] else 0.0
        if fmr and gen > 0:
            level = float(fmr[0]) / float(gen)
            ax.hlines(level, pr[name] / 1.2, pr[name] * 1.2, color="red", lw=1.8, zorder=2,
                      label=None if shown else "zero-effort level")
            shown = True
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = sorted(pr.items(), key=lambda kv: kv[1])
    ax.set_xticks([v for _, v in ticks])
    ax.set_xticklabels([f"{_enc_label(k)} (PR {v:.0f})" for k, v in ticks], fontsize=8, rotation=40, ha="right")
    ax.minorticks_off()
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("participation ratio of the encoder", fontsize=8)
    ax.set_ylabel("best key-free E1 relative success\n(eigen alignment, Wasserstein-Procrustes)", fontsize=8)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_zeropair.pdf", bbox_inches="tight")
    plt.close(fig)


# Marker of each pre-conditioner in the in-system budget figure: the shape gives m (circle: raw space), the fill the
# family (hollow: PCA whitening, filled: FI), so that every point is identified without a text label
PRE_MARKER = {"none": ("o", True), "pca128": ("s", False), "pca256": ("D", False), "fi64": ("v", True),
              "fi128": ("s", True), "fi256": ("D", True)}


def _tradeoff(summary: Path, out: Path, endpoint: str, stem: str, scheme: str = "lrp_full",
              keep: tuple[str, ...] | None = None, print_width: float | None = None,
              ranges: dict | None = None) -> None:
    """Collusion size r50 at `endpoint` against EER, one point per encoder and pre-conditioner; `keep` restricts the
    pre-conditioners to those whose short name (`_short_pre`) it lists. With `print_width` (inches) the figure is
    drawn at that width, the width at which the manuscript prints it, with fonts at their nominal print size, and the
    marker of PRE_MARKER identifies the pre-conditioner in place of a text label (layout only: the points are the
    same)."""
    u = pd.read_csv(summary / "summary_utility.csv")
    u = u[u.scheme == scheme].groupby(["encoder", "pre"], as_index=False)["eer"].mean()
    df = u.merge(best_decoder_status(summary, endpoint, scheme), on=["encoder", "pre"])
    if keep is not None:
        df = df[df.pre.map(_short_pre).isin(keep)]
    col = f"r50_{endpoint}"
    if print_width is not None:
        _tradeoff_marked(df, out, endpoint, stem, print_width, ranges)
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    points = []
    labels = []
    for enc, d in sorted(df.groupby("encoder"), key=lambda t: enc_rank(t[0])):
        color = ENC_COLOR.get(enc, "C7")
        d = d[np.isfinite(d.eer)]
        reached = d[d.status == "reached"]
        cens = d[d.status == "censored"]
        unat = d[d.status == "backmap_limited"]
        if len(reached):
            ax.scatter(reached.eer * 100, reached.r50, s=18, color=color, label=_enc_label(enc))
            labels += [(row.eer * 100, row.r50, _pre_label(row.pre)) for _, row in reached.iterrows()]
        if len(cens):
            ax.scatter(cens.eer * 100, cens.rmax, s=30, marker="^", color=color, label=None if len(reached) else _enc_label(enc))
            labels += [(row.eer * 100, row.rmax, f"{_pre_label(row.pre)} ($\\geq${row.rmax:g})") for _, row in cens.iterrows()]
        if len(unat):
            ax.scatter(unat.eer * 100, unat.rmax, s=30, marker="x", color=color,
                       label=None if len(reached) or len(cens) else _enc_label(enc))
            labels += [(row.eer * 100, row.rmax, f"{_pre_label(row.pre)} (BML)") for _, row in unat.iterrows()]
        for _, row in d.iterrows():
            points.append({"encoder": enc, "pre": row.pre, "eer": row.eer, col: row.r50,
                           "censored": row.status != "reached",
                           "y_plotted": row.r50 if row.status == "reached" else row.rmax,
                           "decoder": row.decoder, "status": row.status})
    ax.set_xlabel("EER (%)", fontsize=8)
    note = "; ^ censored, x back-map-limited" if (df.status != "reached").any() else ""
    ax.set_ylabel(f"$r_{{50}}$ malicious enrollees ({endpoint.upper()}, best decoder){note}", fontsize=7)
    ax.tick_params(labelsize=8)
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.margins(x=0.08, y=0.12)
    fig.tight_layout()
    _place_labels(fig, ax, labels)
    pd.DataFrame(points, columns=["encoder", "pre", "eer", col, "censored", "y_plotted", "decoder", "status"]).to_csv(
        out / f"{stem}_points.csv", index=False)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def _points_table(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """The points of a trade-off figure, in the order and with the columns of the committed point tables."""
    col = f"r50_{endpoint}"
    points = []
    for enc, d in sorted(df.groupby("encoder"), key=lambda t: enc_rank(t[0])):
        for _, row in d[np.isfinite(d.eer)].iterrows():
            points.append({"encoder": enc, "pre": row.pre, "eer": row.eer, col: row.r50,
                           "censored": row.status != "reached",
                           "y_plotted": row.r50 if row.status == "reached" else row.rmax,
                           "decoder": row.decoder, "status": row.status})
    return pd.DataFrame(points, columns=["encoder", "pre", "eer", col, "censored", "y_plotted", "decoder", "status"])


def _tradeoff_marked(df: pd.DataFrame, out: Path, endpoint: str, stem: str, width: float,
                     ranges: dict | None = None) -> None:
    """Print-size trade-off figure: colour gives the encoder, the PRE_MARKER shape and fill the pre-conditioner; only
    censored and back-map-limited points carry a text label. With `ranges` ((encoder, pre) -> per-seed r50 range), a
    vertical line spans the range of each point, and the point table gets the columns seed_lo and seed_hi."""
    pts = _points_table(df, endpoint)
    fig, ax = plt.subplots(figsize=(width, 0.68 * width))
    if ranges is not None:
        pts["seed_lo"] = [ranges.get((e, p), (np.nan, np.nan))[0] for e, p in zip(pts.encoder, pts.pre)]
        pts["seed_hi"] = [ranges.get((e, p), (np.nan, np.nan))[1] for e, p in zip(pts.encoder, pts.pre)]
        for _, row in pts[np.isfinite(pts.seed_lo)].iterrows():
            ax.vlines(row.eer * 100, row.seed_lo, row.seed_hi, colors=ENC_COLOR.get(row.encoder, "C7"),
                      linewidth=0.9, zorder=2)
    labels = []
    for _, row in pts.iterrows():
        shape, filled = PRE_MARKER.get(_short_pre(row.pre), ("X", True))
        color = ENC_COLOR.get(row.encoder, "C7")
        # hollow markers are transparent and drawn above the filled ones, so that coinciding points stay visible
        ax.scatter([row.eer * 100], [row.y_plotted], s=24, marker=shape, linewidths=1.0, zorder=3 if filled else 4,
                   facecolors=color if filled else "none", edgecolors=color)
        if row.status == "censored":
            labels.append((row.eer * 100, row.y_plotted, f"$\\geq${row.y_plotted:g}"))
        elif row.status == "backmap_limited":
            labels.append((row.eer * 100, row.y_plotted, "BML"))
    enc_handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=ENC_COLOR.get(e, "C7"), label=_enc_label(e))
                   for e in sorted(pts.encoder.unique(), key=enc_rank)]
    pre_keys = [k for k in PRE_MARKER if k in set(pts.pre.map(_short_pre))]
    pre_handles = [plt.Line2D([], [], ls="", marker=PRE_MARKER[k][0], ms=5, mew=1.0, color="0.25",
                              markerfacecolor="0.25" if PRE_MARKER[k][1] else "none", label=_pre_label(k))
                   for k in pre_keys]
    fig.legend(handles=enc_handles, fontsize=7, ncol=len(enc_handles), loc="upper center", bbox_to_anchor=(0.5, 1.0),
               frameon=False, handletextpad=0.1, columnspacing=0.9)
    ax.legend(handles=pre_handles, fontsize=7, ncol=3, loc="upper center", frameon=False, handletextpad=0.1,
              columnspacing=0.8, title="pre-conditioner", title_fontsize=7)
    ax.set_xlabel("EER (%)", fontsize=7.5)
    ax.set_ylabel(f"$r_{{50}}$ malicious enrollees ({endpoint.upper()}, best decoder)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_yscale("log")
    ax.margins(x=0.06, y=0.12)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.9)   # head room for the pre-conditioner legend above the points
    plain_log_ticks(ax)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))   # the top strip holds the encoder legend
    _place_labels(fig, ax, labels, fontsize=6.5)
    pts.to_csv(out / f"{stem}_points.csv", index=False)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def fig_tradeoff(summary: Path, out: Path, scheme: str = "lrp_full") -> None:
    _tradeoff(summary, out, "e1", "fig_tradeoff", scheme)


def fig_tradeoff_e2(summary: Path, out: Path, scheme: str = "lrp_full") -> None:
    _tradeoff(summary, out, "e2", "fig_tradeoff_e2", scheme)


# The pre-conditioners the focused paper analyses: no pre-conditioner, PCA whitening and FI at the protected dimensions
BUDGET_PRES = ("none", "pca128", "pca256", "fi64", "fi128", "fi256")


def fig_budget_e2(summary: Path, out: Path, scheme: str = "lrp_full", runs=None) -> None:
    """In-system collusion budget (E2) against EER for the pre-conditioners of BUDGET_PRES only; with the run
    directories, each point carries the range of its per-seed r50 (`seed_r50_range`, uncensored ranges only)."""
    ranges = None
    if runs:
        ranges = {}
        for d in map(Path, runs):
            curves = seed_curves(pd.read_csv(d / "anchor.csv"))
            for pre in curves.pre.unique():
                g = seed_r50_range(curves, pre, "e2", scheme)
                if g is not None and not g["cens_any"]:
                    ranges[(_name(d), pre)] = (g["lo"], g["hi"])
    # 0.75 of the 390 pt text width of the manuscript: the figure prints at its nominal font size
    _tradeoff(summary, out, "e2", "fig_budget_e2", scheme, keep=BUDGET_PRES, print_width=BUDGET_PRINT * TEXT_WIDTH_IN,
              ranges=ranges)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = Path(args.summary)
    fig_spectra(args.runs, out)
    fig_anchor_theory(summary, out)
    if (summary / "summary_zeropair.csv").exists():
        fig_zeropair(summary, args.runs, out)
    fig_tradeoff(summary, out)
    fig_tradeoff_e2(summary, out)
    fig_budget_e2(summary, out, runs=args.runs)


if __name__ == "__main__":
    main()
