"""Check 7 (synthetic only).
(a) Prop 2 anisotropic remark: E ||P_S z||^2/||z||^2 for r random anchors vs r/m and the top-r variance fraction.
(b) Mean term missing from Prop 1b: with z ~ N(mu, Sigma), a known mu resolves ALL eigenvector signs
    (D_j = sign(<q_j, mu> <p_j, ybar>)), turning the sign-ambiguous second-order leak into the oracle-sign curve.
Seed 8. Data source: synthetic.
"""
import numpy as np

rng = np.random.default_rng(8)


def haar(m_):
    q, rr = np.linalg.qr(rng.standard_normal((m_, m_)))
    return q * np.sign(np.diag(rr))


print("(a) anisotropic anchors, m=32")
m = 32
for name, lam in [("1/j", 1.0 / np.arange(1, m + 1)), ("1/j^2", 1.0 / np.arange(1, m + 1) ** 2), ("flat", np.ones(m))]:
    lam = lam / lam.sum() * m
    pr = lam.sum() ** 2 / np.sum(lam ** 2)
    for r in [4, 8, 16]:
        vals = []
        for _ in range(4000):
            Z = rng.standard_normal((r, m)) * np.sqrt(lam)
            z = rng.standard_normal(m) * np.sqrt(lam)
            Qs, _ = np.linalg.qr(Z.T)
            vals.append(np.sum((Qs.T @ z) ** 2) / np.sum(z ** 2))
        top = np.sort(lam)[::-1][:r].sum() / lam.sum()
        print(f"  spectrum {name:<6} PR={pr:5.1f} r={r:2d}: E||P_S z||^2/||z||^2={np.mean(vals):.3f}   r/m={r / m:.3f}   top-r fraction={top:.3f}")


def wishart(df, m_):
    if df < m_ + 1:
        X = rng.standard_normal((df, m_))
        return X.T @ X
    A = np.tril(rng.standard_normal((m_, m_)), -1)
    A[np.diag_indices(m_)] = np.sqrt(rng.chisquare(df - np.arange(m_)))
    return A @ A.T


print("\n(b) residual mean resolves signs: m=32, n_dev=1024, attacker knows Sigma and mu (oracle AUX)")
m, n_dev, n_v, reps = 32, 1024, 1000, 10
for mu_norm in [0.0, 0.1, 0.3]:
    line = []
    for ratio in [1, 4, 16, 64]:
        N = ratio * n_dev
        vals = []
        for _ in range(reps):
            lam = np.sort(1.0 / np.linalg.eigvalsh(wishart(n_dev, m) / n_dev))[::-1]
            sh = np.sqrt(lam)
            mu = rng.standard_normal(m)
            mu *= mu_norm / np.linalg.norm(mu)
            U = haar(m)
            # database: N users, y = U^T z, z ~ N(mu, Sigma); sufficient stats: ybar and centred covariance
            ybar = U.T @ (mu + sh * rng.standard_normal(m) / np.sqrt(N))
            Ahat = U.T @ ((sh[:, None] * wishart(N - 1, m) * sh[None, :]) / (N - 1)) @ U
            P = np.linalg.eigh(Ahat)[1][:, ::-1]
            if mu_norm > 0:
                D = np.sign(mu * (P.T @ ybar))  # q_j = e_j
            else:
                D = rng.choice([-1.0, 1.0], m)
            z = mu + rng.standard_normal((n_v, m)) * sh
            zc_hat = (((z - mu) @ U) @ P) * D + mu  # reconstruct centred part, add known mean back
            cs = np.sum(zc_hat * z, 1) / (np.linalg.norm(zc_hat, axis=1) * np.linalg.norm(z, axis=1))
            vals.append(cs.mean())
        line.append(f"N/n_dev={ratio:3d}: {np.mean(vals):.3f}")
    print(f"  ||mu||={mu_norm:.1f}  ", "  ".join(line))
