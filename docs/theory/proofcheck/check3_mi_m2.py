"""Check 3 (synthetic only): how loose is step 3 (convexity) of Prop 1b?

m = 2, Sigma = diag(1+d, 1-d), U Haar on O(2), y | U ~ N(0, U^T Sigma U).
Closed form: p(y) = exp(-a r^2/2) I0(b r^2/2) / (2 pi sqrt(1-d^2)), a = 1/(1-d^2), b = d/(1-d^2), r = |y|.
I(U; y) = d^2/(1-d^2) - E log I0(b r^2 / 2),  r^2 = (1+d) g1^2 + (1-d) g2^2.
Compare with 1/4 ||Delta0||_F^2 = d^2/2 (predicted leading order), the pairwise E KL (the
quantity the note bounds, leading order 1/2 ||Delta0||^2 = d^2) and the note's bound ||Delta||_F^2/lambda_min.
Seed 2. Data source: synthetic.
"""
import numpy as np
from scipy.special import ive

rng = np.random.default_rng(2)
g = rng.standard_normal((4_000_000, 2))
psi = np.linspace(0, np.pi, 20001)[:-1]


def log_i0(x):
    return np.log(ive(0, x)) + x


print(f"{'d':>6} {'I(U;y)':>10} {'MCse':>8} {'d^2/2':>9} {'pairEKL':>9} {'d^2':>9} {'I/pairEKL':>9} {'note_bnd':>9}")
for d in [0.02, 0.05, 0.1, 0.2, 0.4]:
    r2 = (1 + d) * g[:, 0] ** 2 + (1 - d) * g[:, 1] ** 2
    terms = d * d / (1 - d * d) - log_i0(d * r2 / (2 * (1 - d * d)))
    I = terms.mean()
    se = terms.std() / np.sqrt(len(terms))
    # pairwise E KL over the relative axis angle psi (uniform on [0, pi))
    S = np.diag([1 + d, 1 - d])
    ekl = 0.0
    for p in psi:
        c, s = np.cos(p), np.sin(p)
        R = np.array([[c, -s], [s, c]])
        B = R @ S @ R.T
        ekl += 0.5 * (np.trace(np.linalg.solve(B, S)) - 2)
    ekl /= len(psi)
    note = 2 * d * d / (1 - d)
    print(f"{d:6.2f} {I:10.6f} {se:8.1e} {d * d / 2:9.6f} {ekl:9.6f} {d * d:9.6f} {I / ekl:9.4f} {note:9.6f}")
