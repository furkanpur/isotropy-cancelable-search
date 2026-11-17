"""Generate manuscript/generated/tab_*.tex from the same values as numbers.tex.

Every cell is a value of make_numbers.build() (the dictionary behind numbers.tex), so tables and \\R{} keys cannot
disagree. Each file holds complete `table` environments with a caption that states the data source and the operating
point; the reporting conventions (intervals, censoring, back-map-limited status, key-known level) are defined once in
the setup of the paper (label sec:setup-stats) and referred to from the captions by \\cref, never by section number.
Captions name the theory by \\cref (labels prop:symmetry, prop:collusion, prop:fisher), not by number,
because the focused paper moved some statements to the supplement. Tables input by supplementary.tex
(SUPPLEMENT_TABLES) refer to main-paper labels with the xr-hyper prefix M-. Tables that neither document inputs
(RECORD_ONLY) are kept for the repository record of the runs the paper does not analyse.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "manuscript/generated"
_spec = importlib.util.spec_from_file_location("make_numbers", ROOT / "scripts/manuscript/make_numbers.py")
mn = importlib.util.module_from_spec(_spec)
sys.modules["make_numbers"] = mn
_spec.loader.exec_module(mn)

ENCS = ["d1_base", "d1_large", "d2_fingervit", "d2_deepprint", "d2_ctrlvit"]
DATA = {"d1_base": "D1", "d1_large": "D1", "d2_fingervit": "D2", "d2_deepprint": "D2", "d2_ctrlvit": "D2"}
SCHEMES = ("lrp_full", "lrp_half", "biohash_half", "iom_256")
# One naming scheme in text, tables and figures: the names of the paper, not the code identifiers.
ENC_LAB = {"d1_base": "D1 base", "d1_large": "D1 large", "d2_fingervit": "finger-vit", "d2_deepprint": "DeepPrint",
           "d2_ctrlvit": "ctrl-vit"}
PRE_LAB = {"none": "none", "pca128": "PCA$_{128}$", "pca256": "PCA$_{256}$", "pca128h": "PCA$_{128}^{1/2}$",
           "fi64": "FI$_{64}$", "fi128": "FI$_{128}$", "fi256": "FI$_{256}$", "rbig128": "RBIG$_{128}$",
           "firbig128": "FI+RBIG$_{128}$"}
SCH_LAB = {"lrp_full": "LRP-full", "lrp_half": "LRP-half", "biohash_half": "BioHashing-half", "iom_256": "GRP-IoM"}
SRC = "Data source: real embeddings (D1 and D2, \\cref{sec:setup-data})."
OP = "FMR 0.1\\%"
CONV = "Reporting conventions: \\cref{sec:setup-stats}."


def esc(s: str) -> str:
    return s.replace("_", "\\_")


def enc(e: str) -> str:
    return ENC_LAB[e]


def pre(p: str) -> str:
    return PRE_LAB[p]


def sch(s_: str) -> str:
    return SCH_LAB[s_]


class Tab:
    def __init__(self, V: dict[str, str]):
        self.V = V
        self.used: set[str] = set()

    def v(self, key: str) -> str:
        if key not in self.V:
            raise KeyError(key)
        self.used.add(key)
        return self.V[key]

    def has(self, key: str) -> bool:
        return key in self.V

    def f(self, key: str) -> float:
        return float(self.v(key))

    def ci(self, key: str) -> str:
        """mean [lo, hi], or '0 (<= x)' for a zero-success value."""
        val = self.v(key)
        if val.startswith("0 successes"):
            return f"0 ($\\leq${self.v(key + '.hi')})"
        return f"{val} [{self.v(key + '.lo')}, {self.v(key + '.hi')}]"

    def short(self, key: str) -> str:
        val = self.v(key)
        return f"$\\leq${self.v(key + '.hi')}" if val.startswith("0 successes") else val

    def r50r(self, key: str, dec: bool = True) -> str:
        """r50 (best decoder) followed by the per-seed range [min--max]; a censored end already reads `$\\geq{}r_max$`
        (make_numbers' status rule); the range is left out when every seed is censored."""
        base = self.r50(key, dec)
        if self.v(key + ".seedcens") == "all":
            return base
        return f"{base} [{self.v(key + '.seedmin')}--{self.v(key + '.seedmax')}]"

    def r50(self, key: str, dec: bool = False) -> str:
        val = self.v(key)
        val = "BML" if val == "back-map-limited" else val
        return f"{val} ({self.v(key + '.dec')})" if dec else val


def table(label: str, caption: str, colspec: str, header: list[str], rows: list, resize: bool = False,
          size: str = "\\scriptsize", sideways: bool = False, tail: str = "") -> str:
    """rows: list of lists of cells, or the string 'MIDRULE' for a rule."""
    body = []
    for r in rows:
        body.append("\\midrule" if r == "MIDRULE" else " & ".join(r) + " \\\\")
    tab = "\\begin{tabular}{" + colspec + "}\n\\toprule\n" + "\n".join(header) + "\n\\midrule\n" + "\n".join(body) \
        + "\n\\bottomrule\n\\end{tabular}"
    if resize:
        bound = "\\textheight" if sideways else "\\linewidth"  # a sideways float is as wide as the page is tall
        tab = f"\\resizebox{{\\ifdim\\width>{bound}{bound}\\else\\width\\fi}}{{!}}{{%\n" + tab + "}"
    env = "sidewaystable" if sideways else "table"
    opt = "[htbp]" if not sideways else ""
    return (f"\\begin{{{env}}}{opt}\n\\centering\n" + size + "\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\renewcommand{\\arraystretch}{0.9}\n"
            f"{{\\footnotesize\\caption{{{caption}}}\\label{{tab:{label}}}}}\n{tab}\n{tail}\\end{{{env}}}\n")


BANNER = "% Generated by scripts/manuscript/make_tables.py. Do not edit by hand.\n"


def mid(rows: list, enc_changed: bool) -> None:
    if enc_changed and rows:
        rows.append("MIDRULE")


# ---- tables ------------------------------------------------------------------------------------
def tab_encoders(T: Tab) -> str:
    rows = []
    for e in ENCS:
        rows.append([enc(e), DATA[e], T.v(f"dim.{e}"), T.v(f"pr.{e}.raw.int"), T.v(f"aniso.{e}.raw"),
                     T.v(f"pr.{e}.fi128.int"), T.v(f"aniso.{e}.fi128"), T.v(f"n.{e}.targets"),
                     f"{T.v(f'util.{e}.none.lrp_full.eer')} [{T.v(f'util.{e}.none.lrp_full.eer.lo')}, "
                     f"{T.v(f'util.{e}.none.lrp_full.eer.hi')}]", T.v(f"gen.{e}")])
    # PR, the anisotropy index and g are defined in the text (sec:setup-data); the caption names only what the
    # columns hold, the data source and the operating point (no resize, short caption)
    cap = ("Encoders and their embedding spectra: participation ratio (PR) and anisotropy index of the centred, "
           "normalised embeddings, raw and after FI$_{128}$ (\\cref{sec:setup-data}); $n_{\\mathrm{tgt}}$ E1 trials per "
           "anchor-free attack row; raw-space EER of LRP-full with its 95\\% interval; and $g$, the raw-space genuine "
           f"acceptance at the attack threshold. {SRC} Operating point: {OP}; the spectra do not depend on it.")
    head = ["& & & \\multicolumn{2}{c}{Raw} & \\multicolumn{2}{c}{FI$_{128}$} & & & \\\\",
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
            "Encoder & Corpus & $d$ & PR & Aniso. & PR & Aniso. & $n_{\\mathrm{tgt}}$ & EER (\\%) & $g$ \\\\"]
    return table("encoders", cap, "llrrrrrrcr", head, rows, size="\\footnotesize")


PRES_UTIL = ["none", "pca128", "pca256", "pca128h", "fi64", "fi128", "fi256", "rbig128", "firbig128"]


def tab_utility(T: Tab) -> str:
    rows = []
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in PRES_UTIL:
            k = f"util.{e}.{p}.lrp_full"
            cells = [enc(e) if p == "none" else "", pre(p), T.v(k + ".m")]
            for col in ("eer", "tmr", "rank1"):
                cells.append(f"{T.v(f'{k}.{col}')} [{T.v(f'{k}.{col}.lo')}, {T.v(f'{k}.{col}.hi')}]")
            rows.append(cells)
    cap = ("Utility of LRP-full per pre-conditioner, in percent with 95\\% intervals (Student-$t$ over the seed means). "
           f"{SRC} Operating point: {OP}; TMR is the true-match rate at FMR 0.1\\%, Rank-1 the identification rate. "
           "Pre-conditioners: none; PCA whitening to $m=128$ (PCA$_{128}$) and $m=256$ (PCA$_{256}$) and "
           "half-strength whitening (PCA$_{128}^{1/2}$); Fisher-isotropic (FI) to $m=64,128,256$; RBIG alone "
           "(RBIG$_{128}$) and FI followed by RBIG (FI+RBIG$_{128}$). $m$ is the protected dimension.")
    return table("utility", cap, "llrccc", ["Encoder & Pre-conditioner & $m$ & EER (\\%) & TMR (\\%) & Rank-1 (\\%) \\\\"],
                 rows)


HASH_SCHEMES = ("lrp_half", "biohash_half", "iom_256")
HASH_PRES = ("none", "pca128", "fi128")


def tab_utility_hash(T: Tab) -> str:
    rows = []
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in HASH_PRES:
            for s_ in HASH_SCHEMES:
                k = f"util.{e}.{p}.{s_}"
                rows.append([enc(e) if (p == HASH_PRES[0] and s_ == HASH_SCHEMES[0]) else "",
                             pre(p) if s_ == HASH_SCHEMES[0] else "", sch(s_), T.v(k + ".m"),
                             f"{T.v(k + '.eer')} [{T.v(k + '.eer.lo')}, {T.v(k + '.eer.hi')}]",
                             f"{T.v(k + '.tmr')} [{T.v(k + '.tmr.lo')}, {T.v(k + '.tmr.hi')}]"])
    cap = ("Utility of LRP-half and the hashing schemes under three pre-conditioners (none, PCA$_{128}$, FI$_{128}$), in percent "
           f"with 95\\% Student-$t$ intervals over the seed means. {SRC} Operating point: {OP}; EER is the equal error "
           "rate and TMR the true-match rate at FMR 0.1\\%. $m$ is the dimension entering the scheme. The code length "
           "of BioHashing changes with the pre-conditioner ($d/2$ bits under none, 64 bits at $m=128$) and the other "
           "schemes have their own code lengths, so EER and TMR across rows compare protected systems of different "
           "length. The LRP-full rows are in \\cref{tab:utility}.")
    return table("utility_hash", cap, "lllrcc", ["Encoder & Pre-conditioner & Scheme & $m$ & EER (\\%) & TMR (\\%) \\\\"],
                 rows, resize=True)


PRES_COLL = ["none", "pca128", "fi128", "firbig128"]


def _collusion_part(T: Tab, label: str, ep: str, cap: str) -> str:
    rows = []
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in PRES_COLL:
            k = f"r50.{e}.{p}.lrp_full.exact"
            kf = f"r50.{e}.{p}.lrp_full.fresh"
            rows.append([enc(e) if p == "none" else "", pre(p), T.v(f"util.{e}.{p}.lrp_full.m"),
                         T.r50r(f"{k}.{ep}"), T.r50r(f"{kf}.{ep}"), T.ci(f"t2.{e}.{p}.lrp_full.{ep}")])
    head = [f"Encoder & Pre-conditioner & $m$ & \\multicolumn{{2}}{{c}}{{{ep.upper()} $r_{{50}}$ [min--max]}} & "
            f"T2 level {ep.upper()} \\\\",
            " & & & exact (T1) & fresh (T1$'$) & \\\\"]
    return table(label, cap, "llrlll", head, rows, resize=True)


def tab_collusion(T: Tab) -> str:
    common = ("Collusion endpoints for LRP-full: $r_{50}$, the number of malicious enrollees at which the mean relative "
              "success of the best decoder (named in parentheses) reaches $0.5$, for exact (T1) and fresh (T1$'$) "
              "anchors, and the key-known (T2) level. Square brackets give the range of the per-seed $r_{50}$ over the "
              "five split seeds; a censored seed counts as $r_{\\max}$ and is marked $\\geq$, and the range is omitted "
              "when all seeds are censored. The headline $r_{50}$ is computed on the seed-mean curve. "
              f"{SRC} Operating point: {OP}. {CONV}")
    cap1 = ("Raw-space endpoint E1. " + common + " The in-system endpoint E2 is in \\cref{tab:collusion_e2}.")
    cap2 = ("In-system endpoint E2 after key renewal. " + common + " The raw-space endpoint E1 is in \\cref{tab:collusion}.")
    return (BANNER + _collusion_part(T, "collusion", "e1", cap1) + "\n" + _collusion_part(T, "collusion_e2", "e2", cap2))


D1_ENCS = [e for e in ENCS if DATA[e] == "D1"]
D2_ENCS = [e for e in ENCS if DATA[e] == "D2"]


def _collusion_all_part(T: Tab, label: str, cap: str, cell, encs: list[str]) -> str:
    sub = "r50" if cell is None else "t2"
    rows = []
    for e in encs:
        mid(rows, e != encs[0])
        rows.append([f"\\multicolumn{{9}}{{l}}{{\\emph{{{enc(e)}}} ({DATA[e]})}}"])
        for p in PRES_UTIL:
            cells = [pre(p)]
            for s_ in SCHEMES:
                for ep in ("e1", "e2"):
                    cells.append(T.r50(f"r50.{e}.{p}.{s_}.exact.{ep}") if sub == "r50"
                                 else T.short(f"t2.{e}.{p}.{s_}.{ep}"))
            rows.append(cells)
    head = ["Pre-conditioner " + "".join(f"& \\multicolumn{{2}}{{c}}{{{sch(s_)}}} " for s_ in SCHEMES) + "\\\\",
            "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4)),
            " " + "".join("& E1 & E2 " for _ in SCHEMES) + "\\\\"]
    return table(label, cap, "l" + "rr" * 4, head, rows)


def tab_collusion_all(T: Tab) -> str:
    # five encoders in one table would not fit a page at a readable size: one table per corpus
    cap1 = ("Collusion size $r_{50}$ of E1 and E2 relative success (best decoder, exact anchors) for every "
            "pre-conditioner and scheme, D1 encoders (the D2 encoders are in \\cref{tab:collusion_all_d2}); the "
            "key-known levels are in \\cref{tab:collusion_all_t2,tab:collusion_all_t2_d2}. "
            f"{SRC} Operating point: {OP}. {CONV} Hashing-scheme numbers hold only for the decoders evaluated "
            "(\\cref{sec:disc-limitations}).")
    cap1b = ("Collusion size $r_{50}$ of E1 and E2 relative success for the D2 encoders; continuation of "
             f"\\cref{{tab:collusion_all}}, whose caption applies. {SRC} Operating point: {OP}. {CONV}")
    cap2 = ("Key-known (T2) levels of E1 and E2 relative success that accompany \\cref{tab:collusion_all}, for every "
            "pre-conditioner and scheme, D1 encoders (the D2 encoders are in \\cref{tab:collusion_all_t2_d2}). "
            "A level of the form ``$\\leq x$'' means zero observed successes and gives the "
            "one-sided 95\\% Clopper--Pearson upper bound. "
            f"{SRC} Operating point: {OP}. {CONV}")
    cap2b = ("Key-known (T2) levels for the D2 encoders; continuation of \\cref{tab:collusion_all_t2}, whose caption "
             f"applies. {SRC} Operating point: {OP}. {CONV}")
    return (BANNER + _collusion_all_part(T, "collusion_all", cap1, None, D1_ENCS)
            + "\n" + _collusion_all_part(T, "collusion_all_d2", cap1b, None, D2_ENCS)
            + "\n" + _collusion_all_part(T, "collusion_all_t2", cap2, True, D1_ENCS)
            + "\n" + _collusion_all_part(T, "collusion_all_t2_d2", cap2b, True, D2_ENCS))


def _prop2_part(T: Tab, label: str, cap: str, encs: list[str]) -> str:
    rows = []
    for e in encs:
        mid(rows, e != encs[0])
        for p in ("none", "pca128", "fi128"):
            m = T.v(f"util.{e}.{p}.lrp_full.m")
            for r in (20, 40, 80):
                k = f"ptwo.{e}.{p}.r{r}"
                last = f"+{T.v(k + '.excess')}" if p == "none" else T.v(k + ".dev")
                last = last.replace("+-", "$-$")
                cos = f"{T.v(k + '.cos')} [{T.v(k + '.cos.lo')}, {T.v(k + '.cos.hi')}]"   # Student-t over seed means
                rows.append([enc(e) if (p == "none" and r == 20) else "", pre(p) if r == 20 else "",
                             m if r == 20 else "", str(r), cos, T.v(k + ".cstar"), last, T.v(k + ".ridge")])
        T.v(f"ptwo.{e}.maxdev")
    return table(label, cap, "llrrlrrr",
                 ["Encoder & Pre-conditioner & $m$ & $r$ & $\\widehat{c}$ (projection) [95\\% interval] & $c^*$ & "
                  "dev./excess & $\\widehat{c}$ (ridge) \\\\"], rows)


def lead_claim_on_intervals(T: Tab) -> None:
    """The caption of tab_prop2 states the lead result on the interval ends: every whitened interval lies above
    c*(r,128), and at each r the raw-excess intervals of encoders adjacent in participation ratio do not overlap."""
    for e in ENCS:
        for p in ("pca128", "fi128"):
            for r in (20, 40, 80):
                k = f"ptwo.{e}.{p}.r{r}"
                assert T.f(k + ".cos.lo") > T.f(k + ".cstar"), k
    by_pr = sorted(ENCS, key=lambda e: -int(T.v(f"pr.{e}.raw.int")))
    for r in (20, 40, 80):
        for a, b in zip(by_pr, by_pr[1:]):
            assert T.f(f"ptwo.{a}.none.r{r}.excess.hi") < T.f(f"ptwo.{b}.none.r{r}.excess.lo"), (r, a, b)


def tab_prop2(T: Tab) -> str:
    lead_claim_on_intervals(T)
    cap = ("Check of \\cref{prop:collusion} (the exact collusion optimum $c^*(r,m)=\\mathbb{E}\\sqrt{\\mathrm{Beta}(r/2,(m-r)/2)}$): "
           "measured pre-key-space cosine $\\widehat{c}$ of the projection decoder, with its 95\\% Student-$t$ "
           "interval over the seed means (between-split; not identity-level), against $c^*$, exact anchors, "
           f"LRP-full. {SRC} The collusion cosine does not depend on the FMR. Every whitened interval lies above "
           "$c^*(r,128)$, and at each $r$ the raw-excess intervals of encoders adjacent in participation ratio do not "
           "overlap (the excess interval is the cosine interval minus $c^*(r,d)$). The last column is "
           "$|\\widehat{c}-c^*|$ for the whitened pre-conditioners (PCA$_{128}$, FI$_{128}$) and the signed excess "
           "$\\widehat{c}-c^*$ for none. The ridge column gives the cosine of the ridge-GCV decoder, which stays below "
           "the projection decoder at small $r$; we did not isolate the cause. The largest whitened deviation per "
           "encoder is in \\cref{M-tab:cstar}. D1 encoders here, D2 encoders in \\cref{tab:prop2_d2}.")
    cap_b = ("Check of \\cref{prop:collusion} for the D2 encoders; continuation of \\cref{tab:prop2}, whose caption applies. "
             f"{SRC}")
    return (BANNER + _prop2_part(T, "prop2", cap, D1_ENCS) + "\n" + _prop2_part(T, "prop2_d2", cap_b, D2_ENCS))


def tab_zeropair(T: Tab) -> str:
    rows = []
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in PRES_UTIL:
            k = f"zp.{e}.{p}"
            ica = T.ci(k + ".ica") if T.has(k + ".ica") else "n/a"
            gap = T.v(f"zp.{e}.oraclegap") if p == "none" else "--"
            rows.append([enc(e) if p == "none" else "", pre(p), T.ci(k + ".common"), ica,
                         f"{T.v(k + '.abscos')} / {T.v(k + '.uniformabscos')}", gap,
                         T.v(f"zp.{e}.impostor") if p == "none" else ""])
    cap = ("Key-free (zero-pair, T0) attacks on LRP-full: best E1 relative success, mean with the 95\\% interval "
           "(wider of the identity-level and seed-level intervals), or ``0 ($\\leq x$)'' for zero observed successes "
           f"with the one-sided 95\\% Clopper--Pearson upper bound $x$. {SRC} Operating point: {OP}. The maximum over "
           "attack settings is an optimistic envelope selected with ground truth. The common column takes the maximum "
           "over the methods that run for every pre-conditioner (eigen alignment, Wasserstein--Procrustes); ICA is "
           "reported separately because it ran only for $m\\leq128$; eigen alignment with oracle signs uses the key "
           "and appears only in the oracle-gap column, never as a key-free attack. $\\mathbb{E}|\\cos|$ is the mean "
           "absolute pre-key cosine at the selected configuration of the common methods, next to the value for a uniform direction (the reference of "
           "spherical symmetry). The zero-effort level is the target FMR divided by the genuine acceptance. "
           f"{CONV}")
    return table("zeropair", cap, "llllllr",
                 ["Encoder & Pre-conditioner & Common best E1 & ICA best E1 & $\\mathbb{E}|\\cos|$ / uniform & "
                  "Oracle gap & Zero-effort level \\\\"], rows, resize=True)


def tab_keycomp(T: Tab) -> str:
    rows = []
    insys = {T.v(f"t2.{e}.{p}.{s_}.insys{sfx}") for e in ENCS for p in ("none", "fi128") for s_ in SCHEMES
             for sfx in ("", ".lo", ".hi")}
    assert len(insys) == 1, insys   # the in-system column is constant, so it is stated once in the caption
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in ("none", "fi128"):
            for s_ in SCHEMES:
                k = f"t2.{e}.{p}.{s_}"
                rows.append([enc(e) if (p == "none" and s_ == SCHEMES[0]) else "", pre(p) if s_ == SCHEMES[0] else "",
                             sch(s_), T.ci(k + ".e1"), T.ci(k + ".e2")])
    cap = ("Key compromise (T2): the attacker knows the key. E1 and E2 relative success per scheme, for the best of the "
           "two T2 methods (analytic and learned inverse). The in-system acceptance of the attack, the maximum over the "
           f"T2 methods, is {next(iter(insys))} in every row and is therefore not tabulated. "
           f"{SRC} Operating point: {OP}. {CONV}")
    return table("keycomp", cap, "llccc",
                 ["Encoder & Pre-conditioner & Scheme & T2 E1 & T2 E2 \\\\"], rows)


SCALE_SCHEMES = ("lrp_full", "biohash_half", "iom_256")
SCALE_EF = (64, 128)


def scale_sizes(T: Tab, run: str) -> list[int]:
    """Distractor sizes of a scale run, read from its keys. make_numbers keys each scale.csv row on the size used, so
    these are the sizes of the run's scale.csv; every scheme and ef must have the same sizes."""
    pat = re.compile(rf"scale\.{run}\.(\w+)\.n(\d+)\.ef(\d+)\.used")
    found: dict[tuple, set] = {(s_, ef): set() for s_ in SCALE_SCHEMES for ef in SCALE_EF}
    for k in T.V:
        m = pat.fullmatch(k)
        if m:
            found.setdefault((m.group(1), int(m.group(3))), set()).add(int(m.group(2)))
    sizes = {frozenset(v) for v in found.values()}
    if len(sizes) != 1 or not next(iter(sizes)):
        raise ValueError(f"scale run {run}: the sizes differ between schemes or ef, or are missing: {found}")
    return sorted(next(iter(sizes)))


def _scale_part(T: Tab, label: str, cap: str, run: str) -> str:
    rows = []
    ns = scale_sizes(T, run)
    for s_ in SCALE_SCHEMES:
        for n in ns:
            for ef in SCALE_EF:
                k = f"scale.{run}.{s_}.n{n}.ef{ef}"
                rows.append([sch(s_) if (n == ns[0] and ef == SCALE_EF[0]) else "", T.v(k + ".requested"),
                             T.v(k + ".used"), f"{ef}", T.v(k + ".rankoneexact"), T.v(k + ".rankonehnsw"),
                             T.v(k + ".recallten"), T.v(k + ".latency"), T.v(k + ".mbone")])
        if s_ != SCALE_SCHEMES[-1]:
            rows.append("MIDRULE")
    return table(label, cap, "lrrrrrrrr",
                 ["Scheme & $n$ req. & $n_{\\mathrm{used}}$ & ef & Rank-1 exact & Rank-1 HNSW & Recall@10 & "
                  "Latency (ms) & Index (MB) \\\\"], rows, resize=True)


def tab_scale(T: Tab) -> str:
    ndef = ("Rank-1 (exact and HNSW), recall@10 against exact search (tie-aware), query latency and serialised index "
            "size per scheme, distractor count and \\texttt{efSearch} (ef). Targets are the real D1 identities. No attack "
            "runs here, so there is no FMR operating point. ")
    sizes = [f"scale.secondary.lrp_full.n{n}.ef{SCALE_EF[0]}" for n in scale_sizes(T, "secondary")]
    capped = [(T.v(k + ".requested"), T.v(k + ".used")) for k in sizes]
    capped = [(q, u) for q, u in capped if q != u]   # the capped-size story only when a size was capped
    cap1 = ("HNSW search with real distractors from the secondary corpus. " + ndef +
            f"Of the {T.v('scale.secondary.loaded')} secondary-corpus rows with an identity label, the "
            f"{T.v('scale.secondary.sharedremoved')} rows of fingers shared with D1 were removed before the seeded draw; "
            f"of the {T.v('scale.secondary.drawn')} rows drawn for the largest size, {T.v('scale.secondary.removed')} "
            "near-duplicates of a D1 row (cosine $\\geq0.95$) were removed"
            + ("" if not capped else ", so the size " + " and the size ".join(
                f"requested as {q} uses {u} distractors" for q, u in capped) + " ($n_{\\mathrm{used}}$)")
            + ". The participation ratio of the distractors differs from that of the primary "
            f"development population by {T.v('scale.secondary.prdiff')}\\%.")
    cap2 = ("HNSW search with \\emph{synthetic} distractors (synthetic fingerprint images embedded with the D1 base encoder); "
            "not real data and a stress test of the index, not a model of a real gallery. " + ndef +
            "The participation ratio of the distractors differs from that of the primary development population by "
            f"{T.v('scale.synthetic.prdiff')}\\%.")
    return (BANNER + _scale_part(T, "scale", cap1, "secondary") + "\n"
            + _scale_part(T, "scale_synthetic", cap2, "synthetic"))


def tab_fmr1(T: Tab) -> str:
    rows = []
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in ("none", "fi128"):
            k = f"fmrone.{e}.{p}"
            rows.append([enc(e) if p == "none" else "", pre(p), T.r50(k + ".lrp_full.exact.e1"),
                         T.r50(k + ".lrp_full.exact.e2"), T.ci(k + ".zp"),
                         f"{T.short(k + '.lrp_full.t2.e1')} / {T.short(k + '.lrp_full.t2.e2')}",
                         f"{T.short(k + '.biohash_half.t2.e1')} / {T.short(k + '.biohash_half.t2.e2')}"])
    cap = ("FMR 1\\% operating point: attack threshold at FMR 1\\% (separate runs, not pooled with the FMR 0.1\\% "
           "runs). $r_{50}$ for LRP-full with exact anchors (best decoder), best key-free E1 over the common methods "
           "(eigen alignment, Wasserstein--Procrustes; an optimistic envelope selected with ground truth), and the "
           "key-known (T2) levels of E1 and E2 for LRP-full and BioHashing-half, which are means without intervals in this table. "
           "Data source: real embeddings (D1 and D2, \\cref{sec:setup-data}). Operating point: FMR 1\\%. "
           f"{CONV} The zero-effort level, the target FMR divided by the genuine acceptance, is "
           + ", ".join(f"{T.v(f'fmrone.{e}.impostor')} ({enc(e)})" for e in ENCS) + ".")
    return table("fmr1", cap, "lllllll",
                 ["Encoder & Pre-conditioner & $r_{50}$ E1 & $r_{50}$ E2 & Key-free E1 & T2 E1 / E2, LRP-full & "
                  "T2 E1 / E2, BioHashing-half \\\\"], rows, resize=True)


def tab_synthetic_scaling(T: Tab) -> str:
    ms = sorted(int(k.split(".m")[-1]) for k in T.V if k.startswith("syn.cross.m"))
    qs = sorted({int(re.search(r"\.q(\d+)$", k).group(1)) for k in T.V if k.startswith(f"syn.ecos.m{ms[0]}.q")})
    aux = sorted(int(k.split(".q")[-1]) for k in T.V if k.startswith("syn.aux.q"))
    for m in ms:
        assert int(T.v(f"syn.ndev.m{m}")) == 32 * m, "caption states n_dev = 32 m in every row"
    # upper block: one column per m, rows are N/n_dev, so that no resizing is needed
    head = ["$N/n_{\\mathrm{dev}}$ " + "".join(f"& $m={m}$ " for m in ms) + "\\\\",
            "$n_{\\mathrm{dev}}$ " + "".join(f"& {T.v(f'syn.ndev.m{m}')} " for m in ms) + "\\\\"]
    rows = [[str(q)] + [T.v(f"syn.ecos.m{m}.q{q}") for m in ms] for q in qs]
    rows.append("MIDRULE")
    rows.append(["crossing"] + [T.v(f"syn.cross.m{m}") for m in ms])
    rows.append(["$m/4$"] + [T.v(f"syn.mdiv4.m{m}") for m in ms])
    first = table("synthetic_scaling", "", "r" + "r" * len(ms), head, rows, size="\\footnotesize")
    rows2 = [[str(q), T.v(f"syn.aux.q{q}")] for q in aux]
    if T.v("syn.aux.N") != "1000":  # the header below hard-codes N = 1000 n_dev
        raise ValueError(f"check5c ran with N = {T.v('syn.aux.N')} n_dev, not the 1000 n_dev of the table header")
    tab2 = ("\\vspace{1.5ex}\n\\footnotesize\n\\begin{tabular}{rr}\n\\toprule\n"
            "$n_{\\mathrm{aux}}/n_{\\mathrm{dev}}$ & $\\mathbb{E}\\cos$ (oracle sign, $N=1000\\,n_{\\mathrm{dev}}$) \\\\\n"
            "\\midrule\n" + "\n".join(" & ".join(r) + " \\\\" for r in rows2) + "\n\\bottomrule\n\\end{tabular}\n")
    cap = ("\\emph{Synthetic} simulation (Gaussian model with oracle covariance and signs); not biometric data. Upper block: "
           "mean cosine $\\mathbb{E}\\cos$ of the key-free reconstruction as the number of samples $N$ grows relative "
           "to the development size $n_{\\mathrm{dev}}$, for protected dimension $m$, with the crossing value of "
           "$N/n_{\\mathrm{dev}}$ at which $\\mathbb{E}\\cos$ reaches $0.5$ and the value $m/4$ for comparison. In "
           "every column $n_{\\mathrm{dev}}=32m$, so the table varies $m$ and $n_{\\mathrm{dev}}$ together and does "
           "not separate their effects. Lower block: $\\mathbb{E}\\cos$ against the auxiliary-set size. These "
           "simulations support the second-order leakage rate and \\cref{rem:sign} and are not evidence about any fingerprint "
           "encoder; no FMR operating point applies.")
    first = first.replace("\\caption{}", f"\\caption{{{cap}}}")
    first = first.replace("\\end{table}\n", tab2 + "\\end{table}\n")
    return first


def tab_cstar(T: Tab) -> str:
    """Main-text summary of the check of the collusion optimum: one row per encoder, by decreasing raw participation
    ratio; the full cells are in tab_prop2 (supplement)."""
    encs = sorted(ENCS, key=lambda e: -int(T.v(f"pr.{e}.raw.int")))
    rows = [[enc(e), DATA[e], T.v(f"dim.{e}"), T.v(f"pr.{e}.raw.int")]
            + [T.v(f"ptwo.{e}.none.r{r}.excess") for r in (20, 40, 80)] + [T.v(f"ptwo.{e}.maxdev")] for e in encs]
    cap = ("Collusion cosine against the isotropic optimum of \\cref{prop:collusion} (exact anchors, projection decoder, "
           f"LRP-full), one row per encoder in order of decreasing participation ratio (PR). {SRC} The cosine does not "
           "depend on the FMR. Raw excess: measured cosine minus $c^*(r,d)$ without pre-conditioning, at $r$ colluders. "
           "Whitened deviation: the largest $|\\widehat{c}-c^*(r,128)|$ over PCA$_{128}$, FI$_{128}$ and "
           "$r\\in\\{20,40,80\\}$; in every one of these cells the measured cosine lies above $c^*(r,128)$.")
    head = ["Encoder & Corpus & $d$ & PR (raw) & \\multicolumn{3}{c}{Raw excess over $c^*(r,d)$} & Whitened \\\\",
            "\\cmidrule(lr){5-7}",
            " & & & & $r=20$ & $r=40$ & $r=80$ & deviation \\\\"]
    return table("cstar", cap, "llrrrrrr", head, rows, size="\\footnotesize")


def tab_utility_lrp(T: Tab) -> str:
    """Main-text utility table: LRP-full EER and TMR without pre-conditioning and after whitening to m = 128, with the
    paired per-seed differences FI128 - PCA128 and FI128 - none."""
    rows = []
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in ("none", "pca128", "fi128"):
            k, kp = f"util.{e}.{p}.lrp_full", f"pair.{e}.fi128.{p}.lrp_full"
            paired = (["--", "--"] if p == "fi128" else
                      [f"{T.v(f'{kp}.{c}')} [{T.v(f'{kp}.{c}.lo')}, {T.v(f'{kp}.{c}.hi')}]" for c in ("eer", "tmr")])
            rows.append([enc(e) if p == "none" else "", pre(p)]
                        + [f"{T.v(f'{k}.{c}')} [{T.v(f'{k}.{c}.lo')}, {T.v(f'{k}.{c}.hi')}]" for c in ("eer", "tmr")]
                        + paired)
    cap = ("Utility of LRP-full in percent, with 95\\% Student-$t$ intervals over the seed means (between-split; "
           "not identity-level). "
           f"{SRC} Operating point: {OP}; EER is the equal error rate and TMR the true-match rate at FMR 0.1\\%. "
           "Pre-conditioners: none (raw space, $m=d$), PCA whitening (PCA$_{128}$) and Fisher-isotropic "
           "pre-conditioning (FI$_{128}$) to the protected dimension $m=128$. The last two columns give the paired "
           "difference FI$_{128}$ minus the row over the seeds, in percentage points, with its 95\\% Student-$t$ "
           "interval; a difference is resolved only when this interval excludes zero.")
    head = ["& & & & \\multicolumn{2}{c}{FI$_{128}$ minus row, paired} \\\\",
            "\\cmidrule(lr){5-6}",
            "Encoder & Pre-cond. & EER (\\%) & TMR (\\%) & EER & TMR \\\\"]
    # \scriptsize keeps the six columns within the text width without \resizebox (a nominal font size)
    return table("utility_lrp", cap, "llcccc", head, rows, size="\\scriptsize")


PRES_BUDGET = ["none", "pca128", "pca256", "fi64", "fi128", "fi256"]


def _upper_mark(T: Tab, key: str, m: str) -> str:
    """r50 with its range; a reached headline above m carries a dagger: it is interpolated between grid points above
    m, an upper estimate, since with m exact anchors the projection decoder recovers z (display only)."""
    cell, val = T.r50r(key), T.v(key)
    if val.isdigit() and int(val) > int(m.replace("{,}", "")):
        return cell.replace(val, val + "$^{\\dagger}$", 1)
    return cell


def tab_budget(T: Tab) -> str:
    """Supplement: the LRP-full collusion budget per pre-conditioner, both endpoints."""
    rows = []
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in PRES_BUDGET:
            k = f"r50.{e}.{p}.lrp_full.exact"
            m = T.v(f"util.{e}.{p}.lrp_full.m")
            rows.append([enc(e) if p == "none" else "", pre(p), m, _upper_mark(T, k + ".e1", m),
                         T.ci(f"t2.{e}.{p}.lrp_full.e1"), _upper_mark(T, k + ".e2", m), T.v(f"gen.{e}.{p}.e2")])
    cap = ("Collusion budget of LRP-full per pre-conditioner: $r_{50}$, the number of malicious enrollees with exact "
           "pre-key vectors (T1) at which the mean relative success of the best decoder (named in parentheses) "
           "reaches $0.5$, rounded to the nearest integer, at E1 and at E2 after key renewal, with the key-known (T2) "
           "level of E1, which decides the back-map-limited status (BML). Square brackets: range of the per-seed "
           "$r_{50}$; a censored seed counts as $r_{\\max}$ and is marked $\\geq$. $g_{\\mathrm{E2}}$: in-system "
           "genuine acceptance of the victims, so that at $r_{50}$ their proxies are accepted at about half of it "
           "(at E1, about half of a raw-space genuine acceptance close to $g$ of \\cref{M-tab:encoders}); the T2 "
           "levels use all targets. $^{\\dagger}$: interpolated between grid points above $m$, an upper estimate: $m$ "
           "exact anchors of rank $m$ let the projection decoder recover $z$, so the smallest sufficient budget is at most "
           "$m$ only where exact reconstruction reaches the endpoint on the same victims, which is not established for "
           "these E1 cells, where the back-map also matters "
           "(per-seed values above $m$ likewise). The T2 level is that of the best T2 method, and a learned proxy can exceed "
           f"one ({T.v('t2.d2_fingervit.none.lrp_full.e1')} for finger-vit without pre-conditioning). "
           f"{SRC} Operating point: {OP}. {CONV}")
    head = ["Encoder & Pre-conditioner & $m$ & E1 $r_{50}$ [min--max] & T2 level E1 & E2 $r_{50}$ [min--max] & "
            "$g_{\\mathrm{E2}}$ \\\\"]
    return table("budget", cap, "llrlllr", head, rows, resize=True)


def tab_keyfree(T: Tab) -> str:
    """Supplement: key-free (T0) success on LRP-full at both operating points, raw, PCA128 and FI128."""
    rows = []
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in ("none", "pca128", "fi128"):
            first = p == "none"
            rows.append([enc(e) if first else "", pre(p), T.ci(f"zp.{e}.{p}.common"),
                         "n/a" if first else T.ci(f"zp.{e}.{p}.ica"),
                         f"{T.v(f'zp.{e}.impostor')} / {T.v(f'zp.{e}.impostor.meas')}" if first else "",
                         T.ci(f"fmrone.{e}.{p}.zp"),
                         f"{T.v(f'fmrone.{e}.impostor')} / {T.v(f'fmrone.{e}.impostor.meas')}" if first else ""])
    cap = ("Key-free (T0) attacks on LRP-full: best E1 relative success, mean with the 95\\% interval (the wider of the "
           "identity-level and seed-level intervals), or ``0 ($\\leq x$)'' for zero observed successes with the "
           "one-sided 95\\% Clopper--Pearson upper bound $x$, a descriptive binomial reference limit (\\cref{M-sec:endpoints}). "
           "Each entry is the maximum over attack settings, selected "
           "with ground truth: an exploratory, optimistic envelope, whose interval does not account for the selection; "
           "no AUX-only baseline was run. Common: eigen alignment and Wasserstein--Procrustes; independent "
           "component analysis (ICA) ran only for $m\\leq128$ and at FMR 0.1\\%. Level: zero-effort level, nominal / "
           "measured (target FMR, or the impostor acceptance measured on the target split at the DEV threshold, over "
           "the genuine acceptance); the nominal level is approximate, because the FMR that the target population "
           "shows at the DEV threshold differs from the target FMR. The level comes from random impostor image pairs, "
           "the attack acceptance from each target with its designated probes, so the two are weighted differently. "
           f"{SRC} Operating points: FMR 0.1\\% and FMR 1\\% (separate runs). {CONV}")
    head = ["Encoder & Pre-conditioner & \\multicolumn{3}{c}{FMR 0.1\\%} & \\multicolumn{2}{c}{FMR 1\\%} \\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
            " & & Common & ICA & Level & Common & Level \\\\"]
    return table("keyfree", cap, "lllllll", head, rows, resize=True)


def tab_oracle(T: Tab) -> str:
    """Supplement: the sign-resolution reference eigen_oracle in the raw spaces and after whitening to m = 128."""
    def cell(k: str) -> str:
        return f"0 ($\\leq${T.v(k + '.hi')})" if T.v(k).startswith("0 successes") else T.v(k)
    rows = []
    for e in ENCS:
        if rows:
            rows.append("MIDRULE")
        rows.append([enc(e), PRE_LAB["none"], cell(f"oracle.{e}.e1max"), cell(f"oracle.{e}.none.e2max"),
                     cell(f"fmrone.oracle.{e}.e1max"), cell(f"fmrone.oracle.{e}.none.e2max"), T.v(f"oracle.{e}.cosmax"),
                     T.v(f"zp.{e}.oraclegap").replace("-", "$-$")])
        for pk in ("pca128", "pca256", "fi64", "fi128", "fi256"):
            rows.append(["", PRE_LAB[pk], *(cell(f"{fam}.{e}.{pk}.{ep}max") for fam in ("oracle", "fmrone.oracle")
                                             for ep in ("e1", "e2")), T.v(f"oracle.{e}.{pk}.cosmax"), "--"])
    cap = ("Sign-resolution reference: eigen alignment with the true signs from the key (LRP-full), a key-assisted "
           "diagnostic, never a key-free attack, in the raw spaces and after whitening. Largest mean "
           "relative success at E1 and E2 and largest pre-key cosine over 8 to 128 aligned components and the AUX "
           "pools; oracle gap (gain of the true signs) at the key-free eigen configuration (raw spaces). Means over "
           "seeds and keys, selected with ground truth; ``0 ($\\leq x$)'': zero successes, with the relative one-sided "
           "95\\% Clopper--Pearson bound. Without pre-conditioning E2 repeats E1. Joint estimators of alignment and "
           f"anchor regression, and chosen anchors, were not evaluated. {SRC} Operating points: FMR 0.1\\% and 1\\%.")
    head = ["& & \\multicolumn{2}{c}{FMR 0.1\\%} & \\multicolumn{2}{c}{FMR 1\\%} & & \\\\",
            "Encoder & Pre-cond. & E1 & E2 & E1 & E2 & Max.\\ cosine & Oracle gap \\\\"]
    return table("oracle", cap, "llrrrrrr", head, rows)


def tab_joint(T: Tab) -> str:
    """Supplement: r50 at E2 of the anchor-only projection decoder / the best evaluated attack per operating point and
    pre-conditioner (rows) and encoder (columns); the evidence is in Tables S7 to S10."""
    rows = []
    for fam in ("joint", "fmrone.joint"):
        if rows:
            rows.append("MIDRULE")
        for p in ("none", "pca128", "pca256", "fi64", "fi128", "fi256"):
            cells = [("FMR 0.1\\%" if fam == "joint" else "FMR 1\\%") if p == "none" else "", pre(p)]
            for e in ENCS:
                k = f"{fam}.{e}.{p}.e2"
                star = "$^{*}$" if T.v(k + ".robust") == "yes" else ""
                cells.append(f"{T.v(k + '.proj')} / {T.v(k + '.best')}{star}")
            rows.append(cells)
    cap = ("Joint alignment-plus-anchor attack on LRP-full (T1, exact anchors; E2 after key renewal): $r_{50}$ of the "
           "anchor-only projection decoder / of the best evaluated attack, one decimal. Per $r$, the best evaluated attack "
           "is the largest mean relative success over the projection decoder, eigen alignment with axis signs voted by the "
           "anchors (J1) and the joint decoder (J2), and their AUX and $p$ settings, selected with ground truth on the "
           "means over keys and seeds: an optimistic envelope, not a deployable selection procedure; interpolated "
           "$r_{50}$ are estimates. $^{*}$: the per-seed ranges of the two do not overlap, both when each seed re-selects its "
           "best setting and with the envelope's selections applied to every seed; elsewhere a difference is descriptive. Per-seed ranges, the budgets $r=1$, 2, 5 and the true-sign reference: "
           f"\\cref{{tab:joint_detail,tab:joint_detail_r25,tab:joint_detail_fmr1,tab:joint_detail_fmr1_r25}}. {SRC}")
    head = ["Operating point & Pre-cond. & " + " & ".join(enc(e) for e in ENCS) + " \\\\"]
    return table("joint", cap, "ll" + "r" * len(ENCS), head, rows, resize=True)


def tab_joint_detail(T: Tab, fam: str = "joint", part: str = "r1") -> str:
    """Supplement: the joint attack's evidence per encoder and pre-conditioner at one operating point. part "r1": both
    per-seed r50 ranges at E2, r = 1 and the true-sign reference; part "r25": r = 2 and 5. Each budget gives the
    projection decoder's and the best evaluated attack's E2 relative / absolute success with the pre-key cosine (and
    the selected method). No scaling, so the table prints at its nominal size."""
    def rel(k: str) -> str:
        v = T.v(k)
        return f"0 ($\\leq${v[len('0 successes (CP95 $' + chr(92) + 'leq$ '):-1]})" if v.startswith("0 successes") else v

    rs = (1,) if part == "r1" else (2, 5)
    rows = []
    for e in ENCS:
        mid(rows, e != ENCS[0])
        for p in ("none", "pca128", "pca256", "fi64", "fi128", "fi256"):
            k = f"{fam}.{e}.{p}.e2"
            cells = [enc(e) if p == "none" else "", pre(p)]
            if part == "r1":
                for who in ("proj", "best"):
                    cells.append(f"{T.v(k + '.' + who)} [{T.v(k + '.' + who + '.seedmin')}--{T.v(k + '.' + who + '.seedmax')}]")
            for r in rs:
                for who in ("proj", "best"):
                    kk = f"{k}.r{r}.{who}"
                    tail = f", {T.v(f'{k}.r{r}.method')}" if who == "best" else ""
                    cells.append(f"{rel(kk)} / {rel(kk + '.abs')} ({T.v(f'{k}.r{r}.cos.{who}')}{tail})")
            if part == "r1":
                ok = "oracle" if fam == "joint" else "fmrone.oracle"
                v = T.v(f"{ok}.{e}.{p}.e2max")
                cells.append(f"0 ($\\leq${T.v(f'{ok}.{e}.{p}.e2max.hi')})" if v.startswith("0 successes") else v)
            rows.append(cells)
    fm = fam.startswith("fmrone")
    op = "FMR 1\\%" if fm else "FMR 0.1\\%"
    label = ("joint_detail_fmr1" if fm else "joint_detail") + ("" if part == "r1" else "_r25")
    what = ("$r_{50}$ at E2 of the projection decoder and of the best evaluated attack with the range of the per-seed "
            "$r_{50}$ in brackets, $r=1$, and the true-sign reference" if part == "r1" else "$r=2$ and $r=5$")
    cap = (f"Joint alignment-plus-anchor attack at {op}, the evidence behind \\cref{{tab:joint}}: {what}. Per budget: E2 "
           "relative / absolute success (pre-key cosine; for the best attack also its method: proj, J1 alignment with "
           "anchor-voted signs, J2 joint decoder), means over keys within a seed and then over seeds, over the fixed "
           f"victims (\\R{{n.d1_base.victims}} per D1 run, \\R{{n.d2_fingervit.victims}} per D2 run). The best attack is "
           "selected with ground truth per $r$ on these seed means: an optimistic envelope, not a deployable selection "
           "procedure; interpolated $r_{50}$ are estimates. A per-seed range takes each seed's smallest $r_{50}$ over the "
           "settings, a selection more optimistic than the envelope's, so a point estimate can lie above its range. "
           "``0 ($\\leq x$) / 0 ($\\leq y$)'': zero successes, with the relative and the absolute one-sided 95\\% "
           "Clopper--Pearson bound, the largest over runs.")
    if part == "r1":
        cap += (" True sign: the largest mean E2 relative success of eigen alignment with the signs taken from the key "
                "(a diagnostic, not a joint estimator), over all targets (\\R{n.d1_base.targets} on D1, "
                "\\R{n.d2_fingervit.targets} on D2), not the fixed victims.")
        head = ["& & \\multicolumn{2}{c}{$r_{50}$ [per-seed range]} & \\multicolumn{2}{c}{$r=1$} & \\\\",
                "Encoder & Pre-cond. & proj & best & proj & best & true sign \\\\"]
    else:
        head = ["& & \\multicolumn{2}{c}{$r=2$} & \\multicolumn{2}{c}{$r=5$} \\\\",
                "Encoder & Pre-cond. & proj & best & proj & best \\\\"]
    cap += f" {SRC}"
    return table(label, cap, "l" * len(rows[-1]), head, rows, sideways=True).replace(
        "\\setlength{\\tabcolsep}{3pt}", "\\setlength{\\tabcolsep}{2pt}")


def tab_joint_detail_fmr1(T: Tab) -> str:
    return tab_joint_detail(T, "fmrone.joint")


def tab_joint_detail_r25(T: Tab) -> str:
    return tab_joint_detail(T, "joint", "r25")


def tab_joint_detail_fmr1_r25(T: Tab) -> str:
    return tab_joint_detail(T, "fmrone.joint", "r25")


SUPPLEMENT_TABLES = ("prop2", "budget", "keyfree", "joint", "joint_detail", "joint_detail_r25", "joint_detail_fmr1",
                     "joint_detail_fmr1_r25")  # input by supplementary.tex (Section S3)
MAIN_TABLES = ("encoders", "cstar", "utility_lrp")   # input by the sections of main.tex
# Not input by either document: the repository record of runs the focused paper does not analyse (HNSW at scale,
# hashing schemes, FMR 1% beyond the key-free summary, Gaussianisation variants, fresh captures, the synthetic
# simulation) and the long tables that the compact ones above replace.
RECORD_ONLY = ("utility", "utility_hash", "collusion", "collusion_all", "zeropair", "keycomp", "scale", "fmr1",
               "synthetic_scaling", "oracle")


def _to_main_refs(text: str) -> str:
    """References from the supplement to labels of main.tex carry the xr-hyper prefix M-."""
    return re.sub(r"\\([cC]ref)\{((?:sec|rem|prop|thm|lem|app|eq|fig):)", r"\\\1{M-\2", text)


BUILDERS = {"encoders": tab_encoders, "utility": tab_utility, "utility_hash": tab_utility_hash, "collusion": tab_collusion,
            "collusion_all": tab_collusion_all, "prop2": tab_prop2, "zeropair": tab_zeropair, "keycomp": tab_keycomp,
            "scale": tab_scale, "fmr1": tab_fmr1, "synthetic_scaling": tab_synthetic_scaling, "cstar": tab_cstar,
            "utility_lrp": tab_utility_lrp, "budget": tab_budget, "keyfree": tab_keyfree, "joint": tab_joint, "joint_detail": lambda T: tab_joint_detail(T, "joint"), "joint_detail_fmr1": tab_joint_detail_fmr1, "joint_detail_r25": tab_joint_detail_r25, "joint_detail_fmr1_r25": tab_joint_detail_fmr1_r25, "oracle": tab_oracle}
assert set(BUILDERS) == set(SUPPLEMENT_TABLES) | set(MAIN_TABLES) | set(RECORD_ONLY)


def main() -> int:
    V = mn.build()
    T = Tab(V)
    GEN.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        text = fn(T)
        if name in SUPPLEMENT_TABLES:
            text = _to_main_refs(text)
        if not text.startswith(BANNER):
            text = BANNER + text
        (GEN / f"tab_{name}.tex").write_text(text, encoding="utf-8")
    print(f"wrote {len(BUILDERS)} tables to {GEN} using {len(T.used)} keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
