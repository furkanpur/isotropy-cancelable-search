"""Check 5 (synthetic only): the design rule's predicted transition N ~ n_dev.

Model: m = 32, Sigma = eigenvalues of (W/n_dev)^-1 (whitening estimated from n_dev Gaussian DEV
samples), z ~ N(0, Sigma), LRP-full key U (Haar). The attacker gets the exact Sigma (oracle,
upper bound) and the database sample covariance A_hat = U^T Sigma_hat_N U (sufficient statistic).
MLE of U given Sigma = rank-ordered eigenvector matching (von Neumann trace inequality).
Second-order data never fix the eigenvector signs, so we report
  - oracle-sign attack (signs set from the truth): upper envelope for any sign-resolving add-on
  - random-sign attack: what second-order information alone gives.
Also the rigorous single-victim bound P(cos >= c) <= (I + log 2) / log(1/p0), with
I <= N ||Sig0||_F^2 / (2 lmin^2) (proposed corrected Prop 1b).
Seed 5. Data source: synthetic.
"""
import numpy as np
from scipy.special import betainc

rng = np.random.default_rng(5)
m, n_v, reps, c = 32, 2000, 12, 0.5
p0 = 0.5 * betainc((m - 1) / 2, 0.5, 1 - c * c)


def haar(m_):
    q, rr = np.linalg.qr(rng.standard_normal((m_, m_)))
    return q * np.sign(np.diag(rr))


def wishart_bartlett(df, m_):
    if df < m_ + 1:  # singular / small df: sample directly
        X = rng.standard_normal((df, m_))
        return X.T @ X
    A = np.tril(rng.standard_normal((m_, m_)), -1)
    A[np.diag_indices(m_)] = np.sqrt(rng.chisquare(df - np.arange(m_)))
    return A @ A.T


print(f"m={m}, c={c}: p0 = P(theta_1 >= c) = {p0:.3e}, log(1/p0) = {-np.log(p0):.2f} nats")
for n_dev in [256, 1024, 4096]:
    print(f"\n--- n_dev = {n_dev}  (m*n_dev = {m * n_dev}) ---")
    print(f"{'N/n_dev':>8} {'N':>9} {'Ecos(oracle sgn)':>16} {'P(cos>=.5)':>10} {'Ecos(rand sgn)':>14} {'E|cos|(rand)':>12} {'rigorous P bnd':>14}")
    for ratio in [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]:
        N = int(ratio * n_dev)
        eo, po, er, ea, bnd = [], [], [], [], []
        for _ in range(reps):
            lam = np.sort(1.0 / np.linalg.eigvalsh(wishart_bartlett(n_dev, m) / n_dev))[::-1]
            S_half = np.sqrt(lam)
            U = haar(m)
            SigN = (S_half[:, None] * wishart_bartlett(N, m) * S_half[None, :]) / N
            Ahat = U.T @ SigN @ U
            w, P = np.linalg.eigh(Ahat)
            P = P[:, ::-1]  # descending, matched to lam descending (Sigma eigvecs = e_j)
            D_or = np.sign(np.diag(U @ P))  # sign of e_j^T U p_j
            D_rd = rng.choice([-1.0, 1.0], m)
            z = rng.standard_normal((n_v, m)) * S_half
            y = z @ U  # rows: y = U^T z
            for D, store in [(D_or, "o"), (D_rd, "r")]:
                zh = (y @ P) * D  # rows of U_hat y with U_hat = diag(D) P^T
                cs = np.sum(zh * z, 1) / (np.linalg.norm(zh, axis=1) * np.linalg.norm(z, axis=1))
                if store == "o":
                    eo.append(cs.mean()); po.append((cs >= c).mean())
                else:
                    er.append(cs.mean()); ea.append(np.abs(cs).mean())
            S0 = np.sum((lam - lam.mean()) ** 2)
            I_bnd = N * S0 / (2 * lam.min() ** 2)
            bnd.append(min(1.0, (I_bnd + np.log(2)) / (-np.log(p0))))
        print(f"{ratio:8g} {N:9d} {np.mean(eo):16.3f} {np.mean(po):10.3f} {np.mean(er):14.3f} {np.mean(ea):12.3f} {np.mean(bnd):14.3g}")
    # N at which the rigorous bound stops being informative (< 1)
    lam = np.sort(1.0 / np.linalg.eigvalsh(wishart_bartlett(n_dev, m) / n_dev))[::-1]
    S0 = np.sum((lam - lam.mean()) ** 2)
    N_star = (-np.log(p0) - np.log(2)) * 2 * lam.min() ** 2 / S0
    print(f"   ||Sig0||_F^2 = {S0:.4f} (m(m+1)/n_dev = {m * (m + 1) / n_dev:.4f});  rigorous bound < 1 only for N < {N_star:.1f} = {N_star / n_dev:.4f} n_dev")
