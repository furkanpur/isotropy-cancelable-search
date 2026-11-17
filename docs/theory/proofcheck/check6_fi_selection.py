"""Check 6 (synthetic only): residual anisotropy Delta = W^T S_t^pop W - I on FRESH users for the repo's
FisherIsotropic (whiten p0 dims, then select the m smallest within-class directions on the SAME DEV data)
vs. PCA-whitening to m, vs. FI followed by re-whitening on a held-out DEV half.
Design-rule baseline: E||Delta||_F^2 ~ m(m+1)/n_dev for a whitening independent of the selection.
Also Prop 3 checks: W^T S_t W = I on DEV, and the restricted optimum equals Ky Fan in the p0 subspace.
Seed 7. Data source: synthetic Gaussian class model (no real embeddings).
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "src"))
from ijiss_btp.precondition import FisherIsotropic, PCAWhiten  # noqa: E402

rng = np.random.default_rng(7)
d, m, C, s = 128, 32, 500, 4
n = C * s


def haar(k):
    q, r = np.linalg.qr(rng.standard_normal((k, k)))
    return q * np.sign(np.diag(r))


def stats(W, St):
    D = W.T @ St @ W - np.eye(W.shape[1])
    f = np.sum(D ** 2)
    f0 = f - np.trace(D) ** 2 / len(D)
    return f, f0, np.trace(D) / len(D)


rows = {"PCA-whiten m": [], "FI (same DEV)": [], "FI + re-whiten held-out": []}
kyfan_gap = []
for rep in range(20):
    Qb, Qw = haar(d), haar(d)
    Sb = Qb @ np.diag(2.0 / np.arange(1, d + 1)) @ Qb.T
    Sw = Qw @ np.diag(np.linspace(0.2, 1.0, d)) @ Qw.T
    St_pop = Sb + Sw
    def draw(nc):
        mu = rng.multivariate_normal(np.zeros(d), Sb, nc)
        x = np.repeat(mu, s, 0) + rng.multivariate_normal(np.zeros(d), Sw, nc * s)
        return x, np.repeat(np.arange(nc), s)
    x, y = draw(C)
    pw = PCAWhiten(m, 1.0).fit(x)
    W_pca = pw.u / pw.scale
    fi = FisherIsotropic(m, d).fit(x, y)
    rows["PCA-whiten m"].append(stats(W_pca, St_pop))
    rows["FI (same DEV)"].append(stats(fi.proj, St_pop))
    # sample split: FI on half A, re-whiten its m outputs on half B
    a = y < C // 2
    fiA = FisherIsotropic(m, d).fit(x[a], y[a])
    zB = (x[~a] - x[~a].mean(0)) @ fiA.proj
    SB = zB.T @ zB / (len(zB) - 1)
    w, v = np.linalg.eigh(SB)
    W_split = fiA.proj @ v @ np.diag(w ** -0.5) @ v.T
    rows["FI + re-whiten held-out"].append(stats(W_split, St_pop))
    # Prop 3 (restricted problem): FI objective equals Ky Fan sum of top-m eigenvalues of whitened S_b
    xc = x - x.mean(0)
    St = xc.T @ xc / (n - 1)
    means = np.zeros_like(xc)
    for c in range(C):
        means[y == c] = xc[y == c].mean(0)
    Sb_dev = means.T @ means / (n - 1)
    lam, V = np.linalg.eigh(St)
    P = V / np.sqrt(lam)
    M = P.T @ Sb_dev @ P
    kyfan = np.sort(np.linalg.eigvalsh(M))[::-1][:m].sum()
    kyfan_gap.append(np.trace(fi.proj.T @ Sb_dev @ fi.proj) - kyfan)
    if rep == 0:
        print("rep0: max|W^T S_t W - I| on DEV (FI):", np.abs(fi.proj.T @ St @ fi.proj - np.eye(m)).max())

print(f"d={d} m={m} n_dev={n} ({C} classes x {s}); 20 reps; baseline m(m+1)/n_dev = {m * (m + 1) / n:.3f}, (d/m)-scaled = {d * (d + 1) / n * (m / d) ** 2:.3f}")
print(f"{'transform':<26} {'E||D||_F^2':>11} {'E||D0||_F^2':>12} {'E tr(D)/m':>10}")
for k, v in rows.items():
    v = np.array(v)
    print(f"{k:<26} {v[:, 0].mean():11.3f} {v[:, 1].mean():12.3f} {v[:, 2].mean():10.3f}")
print("Prop 3 restricted optimum: max |tr(W^T S_b W) - KyFan_m(M)| over reps =", np.max(np.abs(kyfan_gap)))
