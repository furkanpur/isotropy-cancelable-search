"""Check 2 (synthetic only): Prop 1b step 4-5 on random instances.

For Sigma = I + Delta and Haar U, U': A = U^T Sigma U, B = U'^T Sigma U'.
  note pointwise bound:   KL(N(0,A)||N(0,B)) <= ||Delta||_F^2 / lambda_min
  scale-free pointwise:   KL <= ||Sig0||_F^2 / (lbar * lambda_min),  Sig0 = traceless part, lbar = tr(Sigma)/m
  exact identity:         (KL(A||B) + KL(B||A))/2 = 1/4 tr((A-B) B^-1 (A-B) A^-1)
  sandwich (proposed):    ||Sig0||_F^2/(2 lmax^2) <= E KL <= ||Sig0||_F^2/(2 lmin^2)
  note leading order:     E KL ~ 1/2 ||Delta0||_F^2
Seed 1. Data source: synthetic.
"""
import numpy as np

rng = np.random.default_rng(1)


def haar(m):
    q, r = np.linalg.qr(rng.standard_normal((m, m)))
    return q * np.sign(np.diag(r))


def kl(A, B):
    m = len(A)
    return 0.5 * (np.trace(np.linalg.solve(B, A)) - m + np.linalg.slogdet(B)[1] - np.linalg.slogdet(A)[1])


def goe(m):
    g = rng.standard_normal((m, m))
    return (g + g.T) / np.sqrt(2 * m)  # spectrum ~ semicircle on [-2, 2]


def instances(m):
    out = {}
    out["I+0.05*GOE"] = np.eye(m) + 0.05 * goe(m)
    out["I+0.20*GOE"] = np.eye(m) + 0.20 * goe(m)
    out["I+0.45*GOE"] = np.eye(m) + 0.45 * goe(m)
    for n in [2 * m, 10 * m, 100 * m]:
        w = rng.standard_normal((n, m))
        out[f"whiten n={n}"] = np.linalg.inv(w.T @ w / n)  # eigenvalues of (W/n)^-1
    out["1.3I+0.05*GOE (near-scalar)"] = 1.3 * np.eye(m) + 0.05 * goe(m)
    e = np.eye(m)
    out["spike I+3e1e1'"] = e + 3 * np.outer(e[0], e[0])
    q = haar(m)
    out["logspace(-2,0.5)"] = q @ np.diag(np.logspace(-2, 0.5, m)) @ q.T
    return out


for m in [16, 64]:
    print(f"\n===== m = {m} (2000 Haar pairs per instance) =====")
    print(f"{'instance':<28} {'maxKL/note':>10} {'maxKL/sfree':>11} {'idErr':>8} {'EKL':>9} {'1/2|D0|^2':>9} {'lower':>9} {'upper':>9} {'|D|^2/lmin':>10}")
    for name, S in instances(m).items():
        lam = np.linalg.eigvalsh(S)
        lmin, lmax, lbar = lam.min(), lam.max(), lam.mean()
        D = S - np.eye(m)
        D0 = S - lbar * np.eye(m)
        note_b = np.sum(D ** 2) / lmin
        sfree_b = np.sum(D0 ** 2) / (lbar * lmin)
        kls, id_err = [], 0.0
        for _ in range(2000):
            U, V = haar(m), haar(m)
            A, B = U.T @ S @ U, V.T @ S @ V
            k1, k2 = kl(A, B), kl(B, A)
            sym = 0.25 * np.trace((A - B) @ np.linalg.solve(B, A - B) @ np.linalg.inv(A))
            id_err = max(id_err, abs(0.5 * (k1 + k2) - sym) / max(sym, 1e-300))
            kls.append(k1)
        kls = np.array(kls)
        f0 = np.sum(D0 ** 2)
        print(f"{name:<28} {kls.max() / note_b:10.4f} {kls.max() / sfree_b:11.4f} {id_err:8.1e} {kls.mean():9.4f} {0.5 * f0:9.4f} {f0 / (2 * lmax ** 2):9.4f} {f0 / (2 * lmin ** 2):9.4f} {note_b:10.4f}")
