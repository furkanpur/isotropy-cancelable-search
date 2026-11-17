"""Check 5b (synthetic only): does the N/n_dev at which the oracle-sign eigen-alignment attack reaches
E cos = 0.5 scale with m?  Same model as check 5, n_dev = 32 m (so sqrt(m/n_dev) is fixed).
Seed 6. Data source: synthetic.
"""
import numpy as np

rng = np.random.default_rng(6)
n_v, reps = 1000, 8


def haar(m_):
    q, rr = np.linalg.qr(rng.standard_normal((m_, m_)))
    return q * np.sign(np.diag(rr))


def wishart(df, m_):
    if df < m_ + 1:
        X = rng.standard_normal((df, m_))
        return X.T @ X
    A = np.tril(rng.standard_normal((m_, m_)), -1)
    A[np.diag_indices(m_)] = np.sqrt(rng.chisquare(df - np.arange(m_)))
    return A @ A.T


def ecos_oracle(m, n_dev, N):
    out = []
    for _ in range(reps):
        lam = np.sort(1.0 / np.linalg.eigvalsh(wishart(n_dev, m) / n_dev))[::-1]
        sh = np.sqrt(lam)
        U = haar(m)
        Ahat = U.T @ ((sh[:, None] * wishart(N, m) * sh[None, :]) / N) @ U
        P = np.linalg.eigh(Ahat)[1][:, ::-1]
        D = np.sign(np.diag(U @ P))
        z = rng.standard_normal((n_v, m)) * sh
        zh = ((z @ U) @ P) * D
        out.append(np.mean(np.sum(zh * z, 1) / (np.linalg.norm(zh, axis=1) * np.linalg.norm(z, axis=1))))
    return float(np.mean(out))


ratios = np.array([1, 2, 4, 8, 16, 32, 64, 128])
for m in [16, 32, 64, 128]:
    n_dev = 32 * m
    vals = np.array([ecos_oracle(m, n_dev, int(q * n_dev)) for q in ratios])
    # log-linear interpolation of the 0.5 crossing
    k = np.argmax(vals >= 0.5)
    if vals[k] >= 0.5 and k > 0:
        x0, x1 = np.log(ratios[k - 1]), np.log(ratios[k])
        cross = np.exp(x0 + (0.5 - vals[k - 1]) * (x1 - x0) / (vals[k] - vals[k - 1]))
    else:
        cross = np.nan
    print(f"m={m:4d} n_dev={n_dev:5d}  Ecos at N/n_dev={list(ratios)}: {np.round(vals, 3)}  crossing N/n_dev ~ {cross:.1f}  (m/4 = {m / 4:.0f})")
