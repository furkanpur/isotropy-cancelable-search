# Proof checks (synthetic data only)

These are numeric checks of the theory in the paper (proofs in its Appendix A). Every script draws from fixed seeds and uses synthetic Gaussian or synthetic invariant data only. No biometric data is involved. Each `.out` file is the recorded output of the matching `.py`.

To run one: `.venv/bin/python docs/theory/proofcheck/<script>.py`, from the repository root.

| Script | Checks |
|---|---|
| check1_beta_cap | Exact cap tail against Ball's bound; E\|theta_1\|; c*(r,m) values and the m/4+1 threshold |
| check2_kl_bound | Gaussian KL bounds; the symmetrised identity; the traceless sandwich |
| check3_mi_m2 | Per-sample information against the pairwise KL bound (m=2) |
| check4_prop2_mc | Prop. 2 Monte Carlo across data laws and schemes |
| check5_design_rule_mc, check5b_crossing_vs_m, check5c_aux_size | Second-order alignment scaling in N/n_dev, m and n_aux (oracle Sigma and signs) |
| check6_fi_selection | FisherIsotropic implementation against the Ky Fan optimum; DEV-selection inflation |
| check7_aniso_and_mean | Anisotropic collusion fractions; the mean as a sign channel |
| check8_prop2_tail | Prop. 2 tail bound against the projection decoder |
