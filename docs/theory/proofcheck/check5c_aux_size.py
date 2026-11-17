"""Check 5c (synthetic only): eigen-alignment with Sigma estimated from n_aux public samples (not oracle).
m = 32, n_dev = 1024, N = 1000 n_dev (database effectively exact), oracle signs (upper envelope).
Seed 9. Data source: synthetic.
"""
import numpy as np

rng = np.random.default_rng(9)
m, n_dev, n_v, reps = 32, 1024, 1000, 10


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


N = 1000 * n_dev
for q in [1, 4, 16, 64, 256, 1000]:
    n_aux = q * n_dev
    vals = []
    for _ in range(reps):
        lam = np.sort(1.0 / np.linalg.eigvalsh(wishart(n_dev, m) / n_dev))[::-1]
        sh = np.sqrt(lam)
        U = haar(m)
        Ahat = U.T @ ((sh[:, None] * wishart(N, m) * sh[None, :]) / N) @ U
        Saux = (sh[:, None] * wishart(n_aux, m) * sh[None, :]) / n_aux
        Qa = np.linalg.eigh(Saux)[1][:, ::-1]
        P = np.linalg.eigh(Ahat)[1][:, ::-1]
        D = np.sign(np.diag(Qa.T @ U @ P))  # oracle sign of q_j^T U p_j
        Uhat = Qa @ np.diag(D) @ P.T
        z = rng.standard_normal((n_v, m)) * sh
        zh = (z @ U) @ Uhat.T
        vals.append(np.mean(np.sum(zh * z, 1) / (np.linalg.norm(zh, axis=1) * np.linalg.norm(z, axis=1))))
    print(f"n_aux/n_dev={q:5d}: E cos (oracle sign, N=1000 n_dev) = {np.mean(vals):.3f}")
