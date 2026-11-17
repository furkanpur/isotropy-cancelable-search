"""Check 4 (synthetic only): Proposition 2 by Monte Carlo, m = 32, r = 8.

Bound: E cos(z_hat, z_v) <= c*(8,32) = 0.48846 for every decoder, every left-invariant key,
any O(m)-invariant law (Gaussian not required). Equality: LRP-full + projection decoder.
Also an anisotropic Gaussian law (outside the theorem) to show the bound is then exceeded.
Rows are samples. Seed 3. Data source: synthetic.
"""
import numpy as np
from scipy.special import gammaln

rng = np.random.default_rng(3)
m, r, n_v, R, L = 32, 8, 1000, 200, 8
cstar = np.exp(gammaln((r + 1) / 2) + gammaln(m / 2) - gammaln(r / 2) - gammaln((m + 1) / 2))


def haar(m_, k_):
    q, rr = np.linalg.qr(rng.standard_normal((m_, m_)))
    return (q * np.sign(np.diag(rr)))[:, :k_]


def sample(law, n):
    g = rng.standard_normal((n, m))
    if law == "gauss_iso":
        return g
    if law == "lognormal_radius":  # invariant, far from Gaussian
        return g / np.linalg.norm(g, axis=1, keepdims=True) * rng.lognormal(0, 1.0, (n, 1))
    if law == "student_t3":  # Gaussian scale mixture, invariant
        return g / np.sqrt(rng.chisquare(3, (n, 1)) / 3)
    if law == "gauss_aniso":  # NOT invariant: lambda_j ~ 1/j
        lam = 1.0 / np.arange(1, m + 1)
        return g * np.sqrt(lam * m / lam.sum())
    raise ValueError(law)


def cos_rows(a, b):
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-300)


def onehot(codes):
    n, nb = codes.shape
    f = np.zeros((n, nb * L))
    f[np.arange(n)[:, None], np.arange(nb)[None, :] * L + codes] = 1.0
    return f


def ridge(Fr, Zr, Fv, lam):  # z_hat = Zr^T (Fr Fr^T + lam I)^-1 Fr Fv^T   (rows)
    G = Fr @ Fr.T + lam * np.eye(len(Fr))
    return (Fv @ Fr.T) @ np.linalg.solve(G, Zr)


results = {}


def add(key, vals):
    results.setdefault(key, []).append(vals.mean())


for law in ["gauss_iso", "lognormal_radius", "student_t3", "gauss_aniso"]:
    for _ in range(R):
        Zr, Zv = sample(law, r), sample(law, n_v)
        # LRP-full
        U = haar(m, m)
        Tr, Tv = Zr @ U, Zv @ U
        add((law, "LRP-full", "projection"), cos_rows(ridge(Tr, Zr, Tv, 0.0), Zv))
        add((law, "LRP-full", "ridge lam=1"), cos_rows(ridge(Tr, Zr, Tv, 1.0), Zv))
        zp = ridge(Tr, Zr, Tv, 0.0)
        perp = rng.standard_normal(zp.shape)
        perp -= (perp @ np.linalg.pinv(Zr)) @ Zr  # component in S^perp
        add((law, "LRP-full", "projection + S-perp noise"), cos_rows(zp + 0.3 * perp, Zv))
        if law != "gauss_aniso":
            continue
        # (aniso: only LRP-full shown)
    if law == "gauss_aniso":
        continue
    for _ in range(R):
        Zr, Zv = sample(law, r), sample(law, n_v)
        # BioHashing k = m (Haar columns, sign)
        Q = haar(m, m)
        Br, Bv = np.sign(Zr @ Q), np.sign(Zv @ Q)
        add((law, "BioHash k=m", "ridge lam=1e-6"), cos_rows(ridge(Br, Zr, Bv, 1e-6), Zv))
        # angle decoder: cos(pi * hamming/k) estimates cos(z_v, z_j); solve in span of anchors
        ham = (Bv[:, None, :] != Br[None, :, :]).mean(2)  # n_v x r
        gvec = np.cos(np.pi * ham) * np.linalg.norm(Zr, axis=1)[None, :]
        add((law, "BioHash k=m", "SimHash-angle LS"), cos_rows(gvec @ np.linalg.solve(Zr @ Zr.T, Zr), Zv))
        # GRP-IoM: Gaussian m x 256, blocks of 8 -> 32 codes
        K = rng.standard_normal((m, 256)) / np.sqrt(m)
        Cr = (Zr @ K).reshape(r, -1, L).argmax(2)
        Cv = (Zv @ K).reshape(n_v, -1, L).argmax(2)
        Fr, Fv = onehot(Cr), onehot(Cv)
        add((law, "GRP-IoM 256/8", "one-hot ridge lam=1e-6"), cos_rows(ridge(Fr, Zr, Fv, 1e-6), Zv))
        agree = (Cv[:, None, :] == Cr[None, :, :]).mean(2) - 1.0 / L
        add((law, "GRP-IoM 256/8", "agreement-weighted anchors"), cos_rows(agree @ Zr, Zv))

print(f"c*(r={r}, m={m}) = {cstar:.5f}   sqrt(r/m) = {np.sqrt(r / m):.5f}   trials={R}, victims/trial={n_v}")
print(f"{'law':<18} {'scheme':<15} {'decoder':<28} {'E cos':>8} {'se':>7} {'(Ecos-c*)/se':>12}")
for (law, sch, dec), v in results.items():
    v = np.array(v)
    se = v.std(ddof=1) / np.sqrt(len(v))
    print(f"{law:<18} {sch:<15} {dec:<28} {v.mean():8.4f} {se:7.4f} {(v.mean() - cstar) / se:12.2f}")
