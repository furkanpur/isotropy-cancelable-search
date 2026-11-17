"""Check 8 (synthetic only): proposed tail version of Prop 2.
Claim: for every decoder, P(cos(z_hat, z_v) >= c) <= P(B >= c^2) + E[1{B<c^2} exp(-(m-r)(c^2-B)/(2(1-B)))],
B ~ Beta(r/2, (m-r)/2). Tested against LRP-full decoders z_hat = P_S z_v + s * (random unit vector in S^perp) * ||P_perp z_v||
('lottery' decoders that trade mean cosine for tail mass).
Seed 10. Data source: synthetic.
"""
import numpy as np

rng = np.random.default_rng(10)
m, r, n = 32, 8, 400_000
g = rng.standard_normal((n, m))
th = g / np.linalg.norm(g, axis=1, keepdims=True)  # victim direction; S = first r coords
B = np.sum(th[:, :r] ** 2, 1)
perp = th[:, r:]
for c in [0.5, 0.6, 0.7, 0.8]:
    bound = np.mean(B >= c * c) + np.mean(np.where(B < c * c, np.exp(-(m - r) * (c * c - B) / (2 * (1 - np.minimum(B, 1 - 1e-12)))), 0.0))
    best = 0.0
    for s in [0.0, 0.25, 0.5, 1.0, 2.0]:
        u = rng.standard_normal((n, m - r))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        zhat = np.concatenate([th[:, :r], s * np.sqrt(1 - B)[:, None] * u], 1)
        cs = np.sum(zhat * th, 1) / np.linalg.norm(zhat, axis=1)
        best = max(best, np.mean(cs >= c))
        if s == 0.0:
            proj = np.mean(cs >= c)
    print(f"c={c}: P_proj(cos>=c)={proj:.4f}  best lottery={best:.4f}  proposed bound={bound:.4f}  holds={best <= bound + 3e-3}")
