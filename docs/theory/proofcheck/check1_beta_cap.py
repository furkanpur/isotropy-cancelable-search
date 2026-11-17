"""Check 1 (synthetic only): Beta half-moment c*(r,m), the note's numbers, r50, and the cap bound.

Data source: synthetic Gaussian / uniform-sphere draws. Seed 0.
"""
import numpy as np
from scipy import stats
from scipy.special import betainc, gammaln

rng = np.random.default_rng(0)


def cstar(r, m):
    return np.exp(gammaln((r + 1) / 2) + gammaln(m / 2) - gammaln(r / 2) - gammaln((m + 1) / 2))


print("== 1a. E sqrt(B), B~Beta(r/2,(m-r)/2): closed form vs quadrature vs Monte Carlo ==")
print(f"{'r':>4} {'m':>5} {'closed':>9} {'quad':>9} {'MC':>9} {'MC_se':>8} {'sqrt(r/m)':>9}")
for r, m in [(1, 2), (3, 5), (8, 32), (20, 128), (40, 128), (80, 128)]:
    quad = stats.beta(r / 2, (m - r) / 2).expect(np.sqrt)
    g = rng.standard_normal((100_000, m))  # uniform direction via Gaussian; S = first r coordinates
    frac = np.sqrt((g[:, :r] ** 2).sum(1) / (g ** 2).sum(1))
    print(f"{r:>4} {m:>5} {cstar(r, m):9.5f} {quad:9.5f} {frac.mean():9.5f} {frac.std() / np.sqrt(len(frac)):8.5f} {np.sqrt(r / m):9.5f}")

print("\n== 1b. smallest r with c*(r,m) >= 0.5 ==")
for m in [64, 128, 256, 768, 1024, 1536]:
    r = np.arange(1, m)
    r50 = int(r[np.argmax(cstar(r, m) >= 0.5)])
    print(f"m={m:5d}: r50={r50:4d}  m/4+1={m // 4 + 1:4d}  c*(m/4,m)={cstar(m / 4, m):.6f}  c*(m/4+1,m)={cstar(m / 4 + 1, m):.6f}")

print("\n== 1c. cap: exact P(theta_1 >= c) vs Ball bound exp(-m c^2/2), c in [0,1) ==")
cs = np.linspace(0.0, 0.999, 2000)
for m in [2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    exact = 0.5 * betainc((m - 1) / 2, 0.5, 1 - cs ** 2)
    ratio_m = exact / np.exp(-m * cs ** 2 / 2)
    ratio_m1 = exact / np.exp(-(m + 1) * cs ** 2 / 2)
    print(f"m={m:5d}: max exact/exp(-m c^2/2)={ratio_m.max():.4f}   max exact/exp(-(m+1)c^2/2)={ratio_m1.max():.4f}")

print("\n== 1d. how loose is the cap bound at the paper's scale ==")
for m in [64, 128, 256]:
    for c in [0.3, 0.5, 0.7]:
        exact = 0.5 * betainc((m - 1) / 2, 0.5, 1 - c ** 2)
        print(f"m={m:4d} c={c}: exact={exact:.3e}  bound={np.exp(-m * c * c / 2):.3e}  log(1/exact)={-np.log(exact):7.2f}  m c^2/2={m * c * c / 2:7.2f}")

print("\n== 1e. E|theta_1| (the T0 baseline that is not trivially zero) ==")
for m in [32, 64, 128, 256]:
    e_abs = np.exp(gammaln(m / 2) - gammaln((m + 1) / 2)) / np.sqrt(np.pi)
    print(f"m={m:4d}: E|theta_1|={e_abs:.4f}  sqrt(2/(pi m))={np.sqrt(2 / (np.pi * m)):.4f}  1/sqrt(m)={1 / np.sqrt(m):.4f}")
