# Isotropy study results (2026-11-17)

Data sources: all results are on real embeddings (D1: SIN 2026 image set, identity labels re-derived from each source's native naming by content matching; D2: development folds of an internal compilation with labels parsed from each source's native naming, finger-disjoint from D1; the shifted AUX of the D1 runs and the real scale-run distractors: the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed), except the table 7 rows of the synthetic-distractor scale run, whose HNSW distractors are synthetic. The main tables use the FMR 0.1% operating point; table 12 uses separate runs at FMR 1%. Values are mean [95% CI]. Intervals on every metric other than E1/E2 relative success are Student-t intervals over the 5 seed means (4 df), mean +/- t(0.975, 4) * sd / sqrt(5): between-split variability over 5 overlapping splits; not an identity-level CI. Lower bounds of nonnegative quantities are truncated at 0; signed cosines are not truncated. E1/E2 relative success carries the wider of two 95% intervals, elementwise: the identity-level interval (the runner's per-row paired-bootstrap standard error, resampling targets, see `src/ijiss_btp/eval/stats.py:63-72`, averaged over keys and seeds; mean +/- 1.96 * SE_bar, which treats the overlapping splits as fully correlated) and the seed-level Student-t interval over the 5 seed means (4 df). The reported interval is their union: lower = min of the two lower bounds, upper = max of the two upper bounds; lower bounds are truncated at 0.

| Encoder | PR | EER none | EER fi128 | r50 E1 none | r50 E1 fi128 | r50 E2 none | r50 E2 fi128 | key-known (T2) level E1, fi128 lrp_full | key-known (T2) level E2, fi128 lrp_full | zero-pair best E1, none (common methods eigen, wp) | zero-pair best E1, fi128 (common methods eigen, wp) | zero-pair ICA best E1, fi128 | zero-effort impostor level (target FMR / e1_genuine) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1_base | 293.975 | 30.853 [30.441, 31.265] % | 29.774 [29.093, 30.455] % (fi128_p512) | 211.3 (proj) | >= 640 (proj); back-map-limited: T2 level 0.036 [0.004, 0.067] < 0.5 | 211.3 (proj) | 40.9 (proj) | 0.036 [0.004, 0.067] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.001 [0.000, 0.003] | 0 successes; CP95 upper <= 0.024 | 0.010 [0.000, 0.028] | 0.0089 |
| d1_large | 445.250 | 31.739 [31.439, 32.039] % | 32.431 [32.057, 32.806] % (fi128_p512) | 233.4 (proj) | >= 640 (proj); back-map-limited: T2 level 0.042 [0.009, 0.075] < 0.5 | 233.4 (proj) | 41.6 (proj) | 0.042 [0.009, 0.075] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.002 [0.000, 0.006] | 0.001 [0.000, 0.006] | 0.010 [0.000, 0.026] | 0.0094 |
| d2_fingervit | 13.348 | 17.053 [15.809, 18.297] % | 6.867 [6.326, 7.408] % (fi128_p512) | 53.0 (proj) | 145.4 (proj) | 53.0 (proj) | 58.8 (proj) | 0.548 [0.446, 0.650] (analytic) | 1.000 [0.999, 1.001] (analytic) | 0.043 [0.000, 0.122] | 0 successes; CP95 upper <= 0.044 | 0.001 [0.000, 0.005] | 0.0082 |
| d2_deepprint | 21.930 | 29.613 [28.528, 30.697] % | 28.421 [27.502, 29.341] % (fi128_p384) | 27.9 (proj) | >= 320 (proj); T2 level 0.488 [0.363, 0.613] | 27.9 (proj) | 44.3 (proj) | 0.488 [0.363, 0.613] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0 successes; CP95 upper <= 0.042 | 0.001 [0.000, 0.005] | 0 successes; CP95 upper <= 0.042 | 0.0074 |
| d2_ctrlvit | 159.810 | 2.691 [2.124, 3.259] % | 3.442 [2.843, 4.040] % (fi128_p512) | 87.2 (proj) | 100.1 (proj) | 87.2 (proj) | 51.4 (proj) | 0.750 [0.717, 0.783] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.017 [0.003, 0.031] | 0.001 [0.000, 0.002] | 0.003 [0.000, 0.006] | 0.0012 |

Headline notes. r50 is the number of malicious enrollees (exact anchors, lrp_full) at which the mean E1 (E2) relative success reaches 0.5, for the best decoder (named). E1 is raw-space acceptance: the proxy, mapped back with the public pre-conditioner inverse, is scored against the victim's fresh captures at the DEV raw threshold (an unprotected or cross-system endpoint). E2 is the in-system endpoint after key renewal: the proxy is re-protected with the renewed key and scored against the victim's fresh captures under that key. `back-map-limited` (lrp_full, exact anchors) means 0.5 was not reached up to r_max and the key-known level (exact pre-key vector, public LMMSE back-map) has its reported upper bound (wider interval) below 0.5; this bounds only attacks that reconstruct the pre-key vector and use that back-map. For other schemes and for fresh anchors the key-known level is a reference, not a bound: evaluated collusion decoders exceed it (summary_fmr1). `>= r_max` means 0.5 was not reached up to r_max. Zero-pair columns use the key-free methods that run for every pre-conditioner (eigen, wp); the ICA column is separate because FastICA ran only for m <= 128. The zero-effort impostor level of E1 relative success is target FMR / e1_genuine (table 3c). Observed values can fall below it, because a reconstructed proxy is not a random impostor capture. Zero observed successes are reported with the one-sided 95% Clopper-Pearson upper bound, never as zero risk. Scope: hashing-scheme collusion numbers (lrp_half, biohash_half, iom_256) hold only for the ridge/proj/mlp decoders evaluated; a stronger IoM decoder exists in the synthetic proof check (docs/theory/proofcheck/check4), so hashing collusion resistance is not claimed.

## Generated tables (scripts/figures/results_tables.py)

### 1. Provenance (all 12 runs)

| run | encoder | dataset / data source | attack target FMR | git_sha | started_utc | runtime_s | data-file hash |
|---|---|---|---|---|---|---|---|
| isotropy_d1_base-20261114-194618 | d1_base | D1: SIN 2026 image set, identity labels re-derived from each source's native naming by content matching (real) | 0.001 | df324d0 | 2026-11-14T16:46:20.227245+00:00 | 3599 | yes |
| isotropy_d1_base_fmr1-20261114-194626 | d1_base_fmr1 | D1: SIN 2026 image set, identity labels re-derived from each source's native naming by content matching (real) | 0.01 | df324d0 | 2026-11-14T16:46:29.673141+00:00 | 3588 | yes |
| isotropy_d1_large-20261114-194603 | d1_large | D1: SIN 2026 image set, identity labels re-derived from each source's native naming by content matching (real) | 0.001 | df324d0 | 2026-11-14T16:46:05.411238+00:00 | 3806 | yes |
| isotropy_d1_large_fmr1-20261114-194611 | d1_large_fmr1 | D1: SIN 2026 image set, identity labels re-derived from each source's native naming by content matching (real) | 0.01 | df324d0 | 2026-11-14T16:46:13.349353+00:00 | 3817 | yes |
| isotropy_d2_ctrlvit-20261115-125450 | d2_ctrlvit | D2: development folds of an internal compilation with labels parsed from each source's native naming, finger-disjoint from D1 (real), ctrl-vit: a control ViT trained by us on the compilation images of the D1 fingers | 0.001 | c679b2c | 2026-11-15T09:54:51.605968+00:00 | 1883 | yes |
| isotropy_d2_ctrlvit_fmr1-20261115-125457 | d2_ctrlvit_fmr1 | D2: development folds of an internal compilation with labels parsed from each source's native naming, finger-disjoint from D1 (real), ctrl-vit: a control ViT trained by us on the compilation images of the D1 fingers | 0.01 | c679b2c | 2026-11-15T09:54:59.275189+00:00 | 1903 | yes |
| isotropy_d2_deepprint-20261114-194634 | d2_deepprint | D2: development folds of an internal compilation with labels parsed from each source's native naming, finger-disjoint from D1 (real) | 0.001 | df324d0 | 2026-11-14T16:46:37.940452+00:00 | 2358 | yes |
| isotropy_d2_deepprint_fmr1-20261114-194642 | d2_deepprint_fmr1 | D2: development folds of an internal compilation with labels parsed from each source's native naming, finger-disjoint from D1 (real) | 0.01 | df324d0 | 2026-11-14T16:46:46.884343+00:00 | 2364 | yes |
| isotropy_d2_fingervit-20261114-194546 | d2_fingervit | D2: development folds of an internal compilation with labels parsed from each source's native naming, finger-disjoint from D1 (real) | 0.001 | df324d0 | 2026-11-14T16:45:47.264123+00:00 | 3566 | yes |
| isotropy_d2_fingervit_fmr1-20261114-194553 | d2_fingervit_fmr1 | D2: development folds of an internal compilation with labels parsed from each source's native naming, finger-disjoint from D1 (real) | 0.01 | df324d0 | 2026-11-14T16:45:55.070422+00:00 | 3597 | yes |
| scale_d1_base_secondary-20261114-213920 | scale_d1_base_secondary | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | n/a | df2253b | 2026-11-14T18:39:21.172086+00:00 | 7 | yes |
| scale_d1_base_synthetic-20261114-205007 | scale_d1_base_synthetic | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | n/a | df324d0 | 2026-11-14T17:50:08.471774+00:00 | 191 | yes |

The ten joint-stage runs of sections 13 and 13b record their provenance in results/joint/*/metadata.json; results/joint/PARITY.md holds their parity check with the anchor stage.

### 1b. Corpora, splits and AUX per isotropy run

| run | encoder | data source (metadata.json) | images | fingers (with >= min_tgt_images images) | split fingers DEV/TGT/AUX | split images DEV/TGT/AUX (min-max over seeds) | shifted AUX rows | shifted AUX removed: fingers shared with D1 | shifted AUX removed: near-duplicates (cos >= 0.95) | shifted AUX source |
|---|---|---|---|---|---|---|---|---|---|---|
| isotropy_d1_base-20261114-194618 | d1_base | sin:primary:base:relabel=d1_relabel.csv:key=basename:dropped=126 | 10233 | 3694 (2362) | 1000/1200/1494 | 1863-1965 / 4135-4171 / 4105-4221 | 3225 | 11387 | 2 | sin:secondary:base:relabel=secondary_relabel.csv:key=path:dropped=249:shared_ids_removed=11387:dedup<0.95 |
| isotropy_d1_base_fmr1-20261114-194626 | d1_base_fmr1 | sin:primary:base:relabel=d1_relabel.csv:key=basename:dropped=126 | 10233 | 3694 (2362) | 1000/1200/1494 | 1863-1965 / 4135-4171 / 4105-4221 | 3225 | 11387 | 2 | sin:secondary:base:relabel=secondary_relabel.csv:key=path:dropped=249:shared_ids_removed=11387:dedup<0.95 |
| isotropy_d1_large-20261114-194603 | d1_large | sin:primary:large:relabel=d1_relabel.csv:key=basename:dropped=126 | 10233 | 3694 (2362) | 1000/1200/1494 | 1863-1965 / 4135-4171 / 4105-4221 | 3224 | 11387 | 3 | sin:secondary:large:relabel=secondary_relabel.csv:key=path:dropped=249:shared_ids_removed=11387:dedup<0.95 |
| isotropy_d1_large_fmr1-20261114-194611 | d1_large_fmr1 | sin:primary:large:relabel=d1_relabel.csv:key=basename:dropped=126 | 10233 | 3694 (2362) | 1000/1200/1494 | 1863-1965 / 4135-4171 / 4105-4221 | 3224 | 11387 | 3 | sin:secondary:large:relabel=secondary_relabel.csv:key=path:dropped=249:shared_ids_removed=11387:dedup<0.95 |
| isotropy_d2_ctrlvit-20261115-125450 | d2_ctrlvit | sin:corpus_d2:ctrlvit | 18493 | 3140 (3140) | 500/600/2040 | 2994-3025 / 3498-3597 / 11901-11997 | n/a | n/a | n/a | n/a |
| isotropy_d2_ctrlvit_fmr1-20261115-125457 | d2_ctrlvit_fmr1 | sin:corpus_d2:ctrlvit | 18493 | 3140 (3140) | 500/600/2040 | 2994-3025 / 3498-3597 / 11901-11997 | n/a | n/a | n/a | n/a |
| isotropy_d2_deepprint-20261114-194634 | d2_deepprint | sin:corpus_d2:deepprint | 18493 | 3140 (3140) | 500/600/2040 | 2912-3041 / 3508-3575 / 11944-12025 | n/a | n/a | n/a | n/a |
| isotropy_d2_deepprint_fmr1-20261114-194642 | d2_deepprint_fmr1 | sin:corpus_d2:deepprint | 18493 | 3140 (3140) | 500/600/2040 | 2912-3041 / 3508-3575 / 11944-12025 | n/a | n/a | n/a | n/a |
| isotropy_d2_fingervit-20261114-194546 | d2_fingervit | sin:corpus_d2:fingervit | 18493 | 3140 (3140) | 500/600/2040 | 2994-3025 / 3498-3597 / 11901-11997 | n/a | n/a | n/a | n/a |
| isotropy_d2_fingervit_fmr1-20261114-194553 | d2_fingervit_fmr1 | sin:corpus_d2:fingervit | 18493 | 3140 (3140) | 500/600/2040 | 2994-3025 / 3498-3597 / 11901-11997 | n/a | n/a | n/a | n/a |

Images and fingers are those of the loaded corpus (after the relabel step for D1); in parentheses, the fingers with at least `min_tgt_images` images, from which TGT is drawn. The shifted AUX of the D1 runs comes from the secondary corpus (FVC set B, SD301a, SD302b/d): the rows of fingers shared with D1 are removed first, then the near-duplicates of any D1 row (uncentered cosine >= 0.95).

### 2. Spectra (raw PR is of the centered, normalized embeddings)

| encoder | dim | raw PR (centered, normalized) | pre | PR after pre | anisotropy_index |
|---|---|---|---|---|---|
| d1_base | 768 | 293.975 | none | 293.975 | 1.270 |
| d1_base | 768 | 293.975 | pca128_a1 | 103.313 | 0.489 |
| d1_base | 768 | 293.975 | fi128_p512 | 109.416 | 0.412 |
| d1_base | 768 | 293.975 | firbig128_p512_i2 | 109.588 | 0.410 |
| d1_large | 1024 | 445.250 | none | 445.250 | 1.140 |
| d1_large | 1024 | 445.250 | pca128_a1 | 110.532 | 0.398 |
| d1_large | 1024 | 445.250 | fi128_p512 | 112.256 | 0.374 |
| d1_large | 1024 | 445.250 | firbig128_p512_i2 | 112.329 | 0.374 |
| d2_fingervit | 1536 | 13.348 | none | 13.348 | 10.681 |
| d2_fingervit | 1536 | 13.348 | pca128_a1 | 111.932 | 0.379 |
| d2_fingervit | 1536 | 13.348 | fi128_p512 | 107.737 | 0.434 |
| d2_fingervit | 1536 | 13.348 | firbig128_p512_i2 | 107.047 | 0.442 |
| d2_deepprint | 512 | 21.930 | none | 21.930 | 4.727 |
| d2_deepprint | 512 | 21.930 | pca128_a1 | 118.489 | 0.283 |
| d2_deepprint | 512 | 21.930 | fi128_p384 | 116.657 | 0.312 |
| d2_deepprint | 512 | 21.930 | firbig128_p384_i2 | 117.091 | 0.305 |
| d2_ctrlvit | 768 | 159.810 | none | 159.810 | 1.951 |
| d2_ctrlvit | 768 | 159.810 | pca128_a1 | 110.808 | 0.394 |
| d2_ctrlvit | 768 | 159.810 | fi128_p512 | 112.553 | 0.370 |
| d2_ctrlvit | 768 | 159.810 | firbig128_p512_i2 | 112.719 | 0.368 |

### 3a. Utility, scheme lrp_full, every pre (FMR 0.1% operating point; threshold from DEV)

| encoder | pre | m | k | EER % [95% CI] | TMR at FMR 0.1% [95% CI] |
|---|---|---|---|---|---|
| d1_base | none | 768 | 768 | 30.853 [30.441, 31.265] | 0.120 [0.114, 0.126] |
| d1_base | pca128_a1 | 128 | 128 | 34.723 [34.277, 35.169] | 0.107 [0.101, 0.113] |
| d1_base | pca256_a1 | 256 | 256 | 32.470 [31.823, 33.116] | 0.146 [0.141, 0.150] |
| d1_base | pca128_a0.5 | 128 | 128 | 34.626 [34.216, 35.036] | 0.097 [0.089, 0.105] |
| d1_base | fi64_p512 | 64 | 64 | 30.637 [29.516, 31.759] | 0.141 [0.129, 0.152] |
| d1_base | fi128_p512 | 128 | 128 | 29.774 [29.093, 30.455] | 0.165 [0.159, 0.171] |
| d1_base | fi256_p512 | 256 | 256 | 29.818 [29.412, 30.225] | 0.186 [0.179, 0.194] |
| d1_base | rbig128_i2 | 128 | 128 | 34.754 [34.231, 35.276] | 0.111 [0.105, 0.117] |
| d1_base | firbig128_p512_i2 | 128 | 128 | 29.877 [29.334, 30.419] | 0.165 [0.159, 0.171] |
| d1_large | none | 1024 | 1024 | 31.739 [31.439, 32.039] | 0.111 [0.103, 0.118] |
| d1_large | pca128_a1 | 128 | 128 | 37.867 [37.634, 38.100] | 0.083 [0.074, 0.092] |
| d1_large | pca256_a1 | 256 | 256 | 35.210 [34.571, 35.849] | 0.115 [0.110, 0.120] |
| d1_large | pca128_a0.5 | 128 | 128 | 37.737 [37.463, 38.011] | 0.077 [0.068, 0.087] |
| d1_large | fi64_p512 | 64 | 64 | 33.604 [32.522, 34.686] | 0.115 [0.106, 0.125] |
| d1_large | fi128_p512 | 128 | 128 | 32.431 [32.057, 32.806] | 0.131 [0.121, 0.141] |
| d1_large | fi256_p512 | 256 | 256 | 32.026 [31.389, 32.663] | 0.147 [0.137, 0.156] |
| d1_large | rbig128_i2 | 128 | 128 | 37.746 [37.375, 38.116] | 0.089 [0.080, 0.097] |
| d1_large | firbig128_p512_i2 | 128 | 128 | 32.436 [32.103, 32.768] | 0.132 [0.121, 0.142] |
| d2_fingervit | none | 1536 | 1536 | 17.053 [15.809, 18.297] | 0.193 [0.178, 0.207] |
| d2_fingervit | pca128_a1 | 128 | 128 | 14.438 [13.312, 15.564] | 0.365 [0.341, 0.388] |
| d2_fingervit | pca256_a1 | 256 | 256 | 14.281 [13.360, 15.202] | 0.428 [0.402, 0.454] |
| d2_fingervit | pca128_a0.5 | 128 | 128 | 14.499 [13.317, 15.682] | 0.292 [0.278, 0.307] |
| d2_fingervit | fi64_p512 | 64 | 64 | 5.907 [5.365, 6.448] | 0.469 [0.442, 0.497] |
| d2_fingervit | fi128_p512 | 128 | 128 | 6.867 [6.326, 7.408] | 0.554 [0.529, 0.579] |
| d2_fingervit | fi256_p512 | 256 | 256 | 8.699 [8.490, 8.908] | 0.600 [0.585, 0.615] |
| d2_fingervit | rbig128_i2 | 128 | 128 | 14.445 [13.278, 15.611] | 0.370 [0.345, 0.394] |
| d2_fingervit | firbig128_p512_i2 | 128 | 128 | 6.682 [6.047, 7.317] | 0.556 [0.536, 0.577] |
| d2_deepprint | none | 512 | 512 | 29.613 [28.528, 30.697] | 0.170 [0.159, 0.180] |
| d2_deepprint | pca128_a1 | 128 | 128 | 31.483 [30.332, 32.635] | 0.226 [0.212, 0.240] |
| d2_deepprint | pca256_a1 | 256 | 256 | 32.122 [30.974, 33.269] | 0.245 [0.229, 0.261] |
| d2_deepprint | pca128_a0.5 | 128 | 128 | 28.389 [27.292, 29.485] | 0.207 [0.193, 0.221] |
| d2_deepprint | fi64_p384 | 64 | 64 | 25.612 [24.632, 26.591] | 0.232 [0.220, 0.244] |
| d2_deepprint | fi128_p384 | 128 | 128 | 28.421 [27.502, 29.341] | 0.259 [0.240, 0.277] |
| d2_deepprint | fi256_p384 | 256 | 256 | 31.035 [29.940, 32.131] | 0.270 [0.251, 0.288] |
| d2_deepprint | rbig128_i2 | 128 | 128 | 31.480 [30.367, 32.593] | 0.226 [0.214, 0.238] |
| d2_deepprint | firbig128_p384_i2 | 128 | 128 | 28.288 [27.418, 29.157] | 0.258 [0.241, 0.276] |
| d2_ctrlvit | none | 768 | 768 | 2.691 [2.124, 3.259] | 0.802 [0.789, 0.816] |
| d2_ctrlvit | pca128_a1 | 128 | 128 | 3.067 [2.429, 3.705] | 0.860 [0.853, 0.868] |
| d2_ctrlvit | pca256_a1 | 256 | 256 | 3.221 [2.521, 3.922] | 0.901 [0.893, 0.909] |
| d2_ctrlvit | pca128_a0.5 | 128 | 128 | 2.772 [2.201, 3.343] | 0.846 [0.838, 0.855] |
| d2_ctrlvit | fi64_p512 | 64 | 64 | 3.687 [3.098, 4.275] | 0.683 [0.661, 0.705] |
| d2_ctrlvit | fi128_p512 | 128 | 128 | 3.442 [2.843, 4.040] | 0.807 [0.793, 0.821] |
| d2_ctrlvit | fi256_p512 | 256 | 256 | 3.475 [2.898, 4.052] | 0.896 [0.889, 0.903] |
| d2_ctrlvit | rbig128_i2 | 128 | 128 | 3.082 [2.490, 3.673] | 0.862 [0.854, 0.871] |
| d2_ctrlvit | firbig128_p512_i2 | 128 | 128 | 3.467 [2.895, 4.040] | 0.810 [0.797, 0.822] |

Intervals on every metric other than E1/E2 relative success are Student-t intervals over the 5 seed means (4 df), mean +/- t(0.975, 4) * sd / sqrt(5): between-split variability over 5 overlapping splits; not an identity-level CI. Lower bounds of nonnegative quantities are truncated at 0; signed cosines are not truncated.

### 3b. Utility, other schemes, pres none and fi128

| encoder | scheme | pre | m | k | EER % [95% CI] | TMR at FMR 0.1% [95% CI] |
|---|---|---|---|---|---|---|
| d1_base | biohash_half | none | 768 | 384 | 35.516 [35.207, 35.825] | 0.118 [0.112, 0.125] |
| d1_base | iom_256 | none | 768 | 256 | 39.948 [39.693, 40.203] | 0.097 [0.091, 0.103] |
| d1_base | lrp_half | none | 768 | 384 | 32.433 [31.851, 33.015] | 0.120 [0.115, 0.124] |
| d1_base | biohash_half | fi128_p512 | 128 | 64 | 37.059 [36.735, 37.383] | 0.097 [0.094, 0.101] |
| d1_base | iom_256 | fi128_p512 | 128 | 256 | 37.735 [37.402, 38.068] | 0.099 [0.093, 0.105] |
| d1_base | lrp_half | fi128_p512 | 128 | 64 | 33.228 [32.517, 33.938] | 0.154 [0.146, 0.161] |
| d1_large | biohash_half | none | 1024 | 512 | 36.878 [36.509, 37.246] | 0.108 [0.100, 0.116] |
| d1_large | iom_256 | none | 1024 | 256 | 41.860 [41.272, 42.447] | 0.087 [0.081, 0.093] |
| d1_large | lrp_half | none | 1024 | 512 | 34.050 [33.278, 34.822] | 0.111 [0.103, 0.119] |
| d1_large | biohash_half | fi128_p512 | 128 | 64 | 38.885 [38.126, 39.643] | 0.085 [0.080, 0.090] |
| d1_large | iom_256 | fi128_p512 | 128 | 256 | 39.420 [38.715, 40.125] | 0.085 [0.078, 0.092] |
| d1_large | lrp_half | fi128_p512 | 128 | 64 | 35.187 [35.004, 35.370] | 0.123 [0.115, 0.132] |
| d2_fingervit | biohash_half | none | 1536 | 768 | 17.302 [15.995, 18.608] | 0.189 [0.179, 0.199] |
| d2_fingervit | iom_256 | none | 1536 | 256 | 19.321 [17.334, 21.308] | 0.139 [0.130, 0.147] |
| d2_fingervit | lrp_half | none | 1536 | 768 | 17.103 [15.785, 18.421] | 0.192 [0.182, 0.201] |
| d2_fingervit | biohash_half | fi128_p512 | 128 | 64 | 15.655 [15.226, 16.084] | 0.348 [0.327, 0.369] |
| d2_fingervit | iom_256 | fi128_p512 | 128 | 256 | 17.139 [16.650, 17.628] | 0.351 [0.328, 0.375] |
| d2_fingervit | lrp_half | fi128_p512 | 128 | 64 | 10.335 [9.620, 11.051] | 0.504 [0.483, 0.526] |
| d2_deepprint | biohash_half | none | 512 | 256 | 30.160 [28.929, 31.391] | 0.151 [0.142, 0.160] |
| d2_deepprint | iom_256 | none | 512 | 256 | 32.073 [30.952, 33.194] | 0.110 [0.105, 0.115] |
| d2_deepprint | lrp_half | none | 512 | 256 | 30.001 [28.664, 31.337] | 0.170 [0.161, 0.178] |
| d2_deepprint | biohash_half | fi128_p384 | 128 | 64 | 35.370 [34.433, 36.307] | 0.102 [0.094, 0.110] |
| d2_deepprint | iom_256 | fi128_p384 | 128 | 256 | 36.118 [35.175, 37.060] | 0.085 [0.080, 0.090] |
| d2_deepprint | lrp_half | fi128_p384 | 128 | 64 | 31.745 [30.660, 32.830] | 0.208 [0.196, 0.219] |
| d2_ctrlvit | biohash_half | none | 768 | 384 | 3.740 [3.387, 4.093] | 0.740 [0.716, 0.764] |
| d2_ctrlvit | iom_256 | none | 768 | 256 | 10.573 [10.091, 11.056] | 0.449 [0.433, 0.465] |
| d2_ctrlvit | lrp_half | none | 768 | 384 | 2.877 [2.441, 3.312] | 0.790 [0.776, 0.804] |
| d2_ctrlvit | biohash_half | fi128_p512 | 128 | 64 | 10.241 [9.555, 10.927] | 0.441 [0.425, 0.458] |
| d2_ctrlvit | iom_256 | fi128_p512 | 128 | 256 | 11.159 [10.463, 11.855] | 0.458 [0.428, 0.487] |
| d2_ctrlvit | lrp_half | fi128_p512 | 128 | 64 | 5.158 [4.478, 5.837] | 0.742 [0.730, 0.753] |

### 3c. Raw genuine acceptance, trials and zero-effort impostor level per encoder (FMR 0.1% attack threshold)

| encoder | target FMR (attack threshold) | E1 trials per zero-pair/T2 row | E1 trials per anchor row (min-max) | e1_genuine: raw-space genuine acceptance at the DEV threshold, mean over zero-pair rows | zero-effort impostor level (target FMR / e1_genuine) | measured FMR (%): target-split impostor pairs at the DEV threshold (fmr_actual, lrp_full, pre none, mean over seeds) | measured zero-effort level (measured FMR / e1_genuine) |
|---|---|---|---|---|---|---|---|
| d1_base | 0.001 | 1200 | 560-560 | 0.112 (sd 0.005, n = 7350 rows) | 0.0089 | 0.057 | 0.0051 |
| d1_large | 0.001 | 1200 | 560-560 | 0.106 (sd 0.006, n = 7350 rows) | 0.0094 | 0.083 | 0.0078 |
| d2_fingervit | 0.001 | 600 | 280-280 | 0.122 (sd 0.008, n = 3675 rows) | 0.0082 | 0.075 | 0.0062 |
| d2_deepprint | 0.001 | 600 | 280-280 | 0.135 (sd 0.009, n = 3675 rows) | 0.0074 | 0.098 | 0.0073 |
| d2_ctrlvit | 0.001 | 600 | 280-280 | 0.815 (sd 0.006, n = 3675 rows) | 0.0012 | 0.068 | 0.0008 |

e1_genuine is the mean over the zero-pair run rows of the raw-space genuine acceptance at the DEV threshold (the proxy acceptance of a perfect attacker would equal it). An attacker that submits a random template passes at about the target FMR, so E1 relative = proxy acceptance / e1_genuine has a zero-effort impostor level of target FMR / e1_genuine. Observed values can fall below it, because a reconstructed proxy is not a random impostor capture. Absolute proxy acceptance is taken from `e1_proxy` in the data (table 5), not converted through TMR. The measured columns replace the target FMR by the impostor acceptance that the target split shows at the DEV threshold (`fmr_actual` of the utility rows of lrp_full without pre-conditioning; lrp_full keeps cosines, so this is the raw-space rate at the threshold of E1), mean over the seeds. n (E1 trials) is read from the run rows (`e1_trials`) and differs per dataset: see table 3c. Zero observed successes are reported with the largest runner-reported one-sided 95% Clopper-Pearson upper bound, never as zero risk.

### 3d. Paired utility differences (lrp_full, FMR 0.1% operating point): FI128 minus PCA128 and minus none

| encoder | comparison | metric | paired difference (percentage points) [95% interval] | per-seed differences (seed order) | seeds |
|---|---|---|---|---|---|
| d1_base | fi128_p512 - pca128_a1 | eer | -4.95 [-5.59, -4.31] | -5.58, -4.21, -5.25, -4.77, -4.95 | 5 |
| d1_base | fi128_p512 - pca128_a1 | tmr | 5.73 [5.33, 6.13] | 5.51, 5.48, 5.62, 5.77, 6.27 | 5 |
| d1_base | fi128_p512 - none | eer | -1.08 [-1.84, -0.32] | -1.58, -0.38, -1.81, -1.02, -0.61 | 5 |
| d1_base | fi128_p512 - none | tmr | 4.48 [3.83, 5.13] | 4.10, 3.91, 4.39, 4.81, 5.18 | 5 |
| d1_large | fi128_p512 - pca128_a1 | eer | -5.44 [-5.82, -5.06] | -5.55, -5.84, -5.50, -5.23, -5.05 | 5 |
| d1_large | fi128_p512 - pca128_a1 | tmr | 4.77 [3.63, 5.90] | 4.75, 4.58, 3.79, 6.26, 4.45 | 5 |
| d1_large | fi128_p512 - none | eer | 0.69 [0.21, 1.17] | 0.50, 0.27, 0.58, 0.83, 1.28 | 5 |
| d1_large | fi128_p512 - none | tmr | 2.03 [1.34, 2.72] | 1.75, 1.87, 2.02, 2.97, 1.53 | 5 |
| d2_fingervit | fi128_p512 - pca128_a1 | eer | -7.57 [-9.09, -6.05] | -9.64, -6.49, -7.32, -6.85, -7.56 | 5 |
| d2_fingervit | fi128_p512 - pca128_a1 | tmr | 18.91 [16.14, 21.68] | 18.94, 16.64, 16.91, 20.06, 21.99 | 5 |
| d2_fingervit | fi128_p512 - none | eer | -10.19 [-11.82, -8.55] | -12.31, -8.94, -9.57, -9.59, -10.53 | 5 |
| d2_fingervit | fi128_p512 - none | tmr | 36.12 [34.67, 37.57] | 34.90, 36.12, 35.64, 38.04, 35.89 | 5 |
| d2_deepprint | fi128_p384 - pca128_a1 | eer | -3.06 [-3.42, -2.70] | -3.30, -3.10, -3.21, -3.14, -2.56 | 5 |
| d2_deepprint | fi128_p384 - pca128_a1 | tmr | 3.28 [2.68, 3.89] | 2.92, 2.70, 3.34, 3.51, 3.94 | 5 |
| d2_deepprint | fi128_p384 - none | eer | -1.19 [-1.61, -0.78] | -1.41, -1.34, -1.23, -0.61, -1.38 | 5 |
| d2_deepprint | fi128_p384 - none | tmr | 8.90 [7.88, 9.92] | 8.04, 7.99, 9.48, 9.28, 9.71 | 5 |
| d2_ctrlvit | fi128_p512 - pca128_a1 | eer | 0.37 [0.01, 0.73] | 0.64, 0.39, 0.60, 0.34, -0.09 | 5 |
| d2_ctrlvit | fi128_p512 - pca128_a1 | tmr | -5.37 [-6.84, -3.91] | -5.22, -7.34, -5.40, -4.48, -4.43 | 5 |
| d2_ctrlvit | fi128_p512 - none | eer | 0.75 [0.38, 1.12] | 0.85, 0.76, 1.11, 0.74, 0.29 | 5 |
| d2_ctrlvit | fi128_p512 - none | tmr | 0.45 [-1.86, 2.76] | 1.73, -2.65, 1.98, 0.94, 0.26 | 5 |

Per seed, the difference of the utility rows of the two pre-conditioners (key 0; both scored on the same split, genuine and impostor pairs and DEV threshold rule), in percentage points; the interval is the Student-t 95% interval over the per-seed differences (4 df). Like the other seed intervals it describes between-split variability over overlapping splits of one corpus; it is not identity-level and does not model subject clustering. The mean equals the difference of the seed means of table 3a.

### 4a. Collusion (T1, exact anchors, proj decoder, lrp_full): rec_cos_z vs r

| encoder | pre | m | r | rec_cos_z (pre-key space, proj decoder) [95% CI] | cstar | sqrt_r_m |
|---|---|---|---|---|---|---|
| d1_base | none | 768 | 20 | 0.253 [0.247, 0.260] | 0.159 | 0.161 |
| d1_base | none | 768 | 40 | 0.359 [0.350, 0.369] | 0.227 | 0.228 |
| d1_base | none | 768 | 80 | 0.490 [0.486, 0.494] | 0.322 | 0.323 |
| d1_base | none | 768 | 160 | 0.645 [0.642, 0.648] | 0.456 | 0.456 |
| d1_base | pca128_a1 | 128 | 20 | 0.453 [0.443, 0.463] | 0.391 | 0.395 |
| d1_base | pca128_a1 | 128 | 40 | 0.633 [0.622, 0.644] | 0.557 | 0.559 |
| d1_base | pca128_a1 | 128 | 80 | 0.845 [0.841, 0.848] | 0.790 | 0.791 |
| d1_base | pca128_a1 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |
| d1_base | fi128_p512 | 128 | 20 | 0.421 [0.417, 0.426] | 0.391 | 0.395 |
| d1_base | fi128_p512 | 128 | 40 | 0.593 [0.587, 0.599] | 0.557 | 0.559 |
| d1_base | fi128_p512 | 128 | 80 | 0.817 [0.814, 0.820] | 0.790 | 0.791 |
| d1_base | fi128_p512 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |
| d1_large | none | 1024 | 20 | 0.200 [0.191, 0.209] | 0.138 | 0.140 |
| d1_large | none | 1024 | 40 | 0.286 [0.279, 0.294] | 0.196 | 0.198 |
| d1_large | none | 1024 | 80 | 0.400 [0.394, 0.406] | 0.279 | 0.280 |
| d1_large | none | 1024 | 160 | 0.545 [0.542, 0.548] | 0.395 | 0.395 |
| d1_large | pca128_a1 | 128 | 20 | 0.436 [0.430, 0.442] | 0.391 | 0.395 |
| d1_large | pca128_a1 | 128 | 40 | 0.613 [0.610, 0.615] | 0.557 | 0.559 |
| d1_large | pca128_a1 | 128 | 80 | 0.831 [0.828, 0.834] | 0.790 | 0.791 |
| d1_large | pca128_a1 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |
| d1_large | fi128_p512 | 128 | 20 | 0.418 [0.413, 0.422] | 0.391 | 0.395 |
| d1_large | fi128_p512 | 128 | 40 | 0.587 [0.582, 0.592] | 0.557 | 0.559 |
| d1_large | fi128_p512 | 128 | 80 | 0.813 [0.811, 0.815] | 0.790 | 0.791 |
| d1_large | fi128_p512 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |
| d2_fingervit | none | 1536 | 20 | 0.836 [0.820, 0.852] | 0.113 | 0.114 |
| d2_fingervit | none | 1536 | 40 | 0.913 [0.906, 0.921] | 0.160 | 0.161 |
| d2_fingervit | none | 1536 | 80 | 0.953 [0.950, 0.956] | 0.228 | 0.228 |
| d2_fingervit | none | 1536 | 160 | 0.974 [0.972, 0.975] | 0.322 | 0.323 |
| d2_fingervit | pca128_a1 | 128 | 20 | 0.469 [0.457, 0.481] | 0.391 | 0.395 |
| d2_fingervit | pca128_a1 | 128 | 40 | 0.650 [0.639, 0.661] | 0.557 | 0.559 |
| d2_fingervit | pca128_a1 | 128 | 80 | 0.868 [0.861, 0.875] | 0.790 | 0.791 |
| d2_fingervit | pca128_a1 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |
| d2_fingervit | fi128_p512 | 128 | 20 | 0.428 [0.420, 0.437] | 0.391 | 0.395 |
| d2_fingervit | fi128_p512 | 128 | 40 | 0.602 [0.595, 0.610] | 0.557 | 0.559 |
| d2_fingervit | fi128_p512 | 128 | 80 | 0.830 [0.823, 0.837] | 0.790 | 0.791 |
| d2_fingervit | fi128_p512 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |
| d2_deepprint | none | 512 | 20 | 0.737 [0.727, 0.748] | 0.195 | 0.198 |
| d2_deepprint | none | 512 | 40 | 0.859 [0.856, 0.862] | 0.278 | 0.280 |
| d2_deepprint | none | 512 | 80 | 0.949 [0.947, 0.950] | 0.394 | 0.395 |
| d2_deepprint | none | 512 | 160 | 0.990 [0.990, 0.991] | 0.558 | 0.559 |
| d2_deepprint | pca128_a1 | 128 | 20 | 0.429 [0.421, 0.438] | 0.391 | 0.395 |
| d2_deepprint | pca128_a1 | 128 | 40 | 0.601 [0.596, 0.606] | 0.557 | 0.559 |
| d2_deepprint | pca128_a1 | 128 | 80 | 0.827 [0.823, 0.831] | 0.790 | 0.791 |
| d2_deepprint | pca128_a1 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |
| d2_deepprint | fi128_p384 | 128 | 20 | 0.413 [0.409, 0.418] | 0.391 | 0.395 |
| d2_deepprint | fi128_p384 | 128 | 40 | 0.583 [0.580, 0.585] | 0.557 | 0.559 |
| d2_deepprint | fi128_p384 | 128 | 80 | 0.811 [0.809, 0.813] | 0.790 | 0.791 |
| d2_deepprint | fi128_p384 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |
| d2_ctrlvit | none | 768 | 20 | 0.324 [0.319, 0.329] | 0.159 | 0.161 |
| d2_ctrlvit | none | 768 | 40 | 0.437 [0.432, 0.442] | 0.227 | 0.228 |
| d2_ctrlvit | none | 768 | 80 | 0.575 [0.570, 0.579] | 0.322 | 0.323 |
| d2_ctrlvit | none | 768 | 160 | 0.719 [0.716, 0.722] | 0.456 | 0.456 |
| d2_ctrlvit | pca128_a1 | 128 | 20 | 0.413 [0.406, 0.420] | 0.391 | 0.395 |
| d2_ctrlvit | pca128_a1 | 128 | 40 | 0.584 [0.580, 0.588] | 0.557 | 0.559 |
| d2_ctrlvit | pca128_a1 | 128 | 80 | 0.813 [0.810, 0.815] | 0.790 | 0.791 |
| d2_ctrlvit | pca128_a1 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |
| d2_ctrlvit | fi128_p512 | 128 | 20 | 0.412 [0.407, 0.417] | 0.391 | 0.395 |
| d2_ctrlvit | fi128_p512 | 128 | 40 | 0.580 [0.575, 0.585] | 0.557 | 0.559 |
| d2_ctrlvit | fi128_p512 | 128 | 80 | 0.810 [0.807, 0.812] | 0.790 | 0.791 |
| d2_ctrlvit | fi128_p512 | 128 | 160 | 1.000 [1.000, 1.000] | 1.000 | n/a |

`rec_cos_z` is a pre-key-space cosine (Proposition 2 and `cstar` refer to it); `rec_cos` (raw space, after the pre-conditioner's inverse map) is in `results/isotropy/summary/summary_anchor.csv`. `sqrt_r_m` is n/a for r > m. Intervals on every metric other than E1/E2 relative success are Student-t intervals over the 5 seed means (4 df), mean +/- t(0.975, 4) * sd / sqrt(5): between-split variability over 5 overlapping splits; not an identity-level CI. Lower bounds of nonnegative quantities are truncated at 0; signed cosines are not truncated.

### 4b. Collusion (T1): r50 of E1 and E2 relative success per pre, scheme and anchor variant

| encoder | pre | m | scheme | anchors | r50 E1 (best decoder) | r50 E1 per-seed range [min-max] | key-known (T2) level E1 [95% interval] (T2 method) | r50 E2 (best decoder) | r50 E2 per-seed range [min-max] | key-known (T2) level E2 [95% interval] (T2 method) |
|---|---|---|---|---|---|---|---|---|---|---|
| d1_base | none | 768 | lrp_full | exact | 211.3 (proj) | [197.4-228.9] | 1.000 [1.000, 1.000] (analytic) | 211.3 (proj) | [197.4-228.9] | 1.000 [1.000, 1.000] (analytic) |
| d1_base | none | 768 | lrp_full | fresh | >= 640 (proj); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) | >= 640 (proj); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_base | none | 768 | lrp_half | exact | 554.5 (ridge) | [534.3-575.0] | 0.781 [0.728, 0.834] (learned) | 552.3 (ridge) | [533.2-574.9] | 0.793 [0.740, 0.846] (learned) |
| d1_base | none | 768 | lrp_half | fresh | >= 640 (ridge); T2 level 0.781 [0.728, 0.834] | all 5 seeds >= 640 | 0.781 [0.728, 0.834] (learned) | >= 640 (ridge); T2 level 0.793 [0.740, 0.846] | all 5 seeds >= 640 | 0.793 [0.740, 0.846] (learned) |
| d1_base | none | 768 | biohash_half | exact | >= 640 (ridge); T2 level 0.416 [0.347, 0.485] | all 5 seeds >= 640 | 0.416 [0.347, 0.485] (learned) | >= 640 (ridge); T2 level 0.389 [0.319, 0.459] | all 5 seeds >= 640 | 0.389 [0.319, 0.459] (learned) |
| d1_base | none | 768 | biohash_half | fresh | >= 640 (ridge); T2 level 0.416 [0.347, 0.485] | all 5 seeds >= 640 | 0.416 [0.347, 0.485] (learned) | >= 640 (ridge); T2 level 0.389 [0.319, 0.459] | all 5 seeds >= 640 | 0.389 [0.319, 0.459] (learned) |
| d1_base | none | 768 | iom_256 | exact | >= 640 (ridge); T2 level 0.026 [0.003, 0.049] | all 5 seeds >= 640 | 0.026 [0.003, 0.049] (learned) | >= 640 (ridge); T2 level 0.063 [0.028, 0.097] | all 5 seeds >= 640 | 0.063 [0.028, 0.097] (learned) |
| d1_base | none | 768 | iom_256 | fresh | >= 640 (mlp); T2 level 0.026 [0.003, 0.049] | all 5 seeds >= 640 | 0.026 [0.003, 0.049] (learned) | >= 640 (ridge); T2 level 0.063 [0.028, 0.097] | all 5 seeds >= 640 | 0.063 [0.028, 0.097] (learned) |
| d1_base | pca128_a1 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.314 [0.248, 0.381] < 0.5 | all 5 seeds >= 640 | 0.314 [0.248, 0.381] (analytic) | 67.2 (proj) | [63.4-71.6] | 1.000 [1.000, 1.000] (analytic) |
| d1_base | pca128_a1 | 128 | lrp_full | fresh | >= 640 (proj); T2 level 0.314 [0.248, 0.381] | all 5 seeds >= 640 | 0.314 [0.248, 0.381] (analytic) | >= 640 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_base | pca128_a1 | 128 | lrp_half | exact | >= 640 (ridge); T2 level 0.042 [0.013, 0.070] | all 5 seeds >= 640 | 0.042 [0.013, 0.070] (learned) | >= 640 (ridge); T2 level 0.320 [0.217, 0.422] | all 5 seeds >= 640 | 0.320 [0.217, 0.422] (learned) |
| d1_base | pca128_a1 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0.042 [0.013, 0.070] | all 5 seeds >= 640 | 0.042 [0.013, 0.070] (learned) | >= 640 (proj); T2 level 0.320 [0.217, 0.422] | all 5 seeds >= 640 | 0.320 [0.217, 0.422] (learned) |
| d1_base | pca128_a1 | 128 | biohash_half | exact | >= 640 (ridge); T2 level 0.001 [0.000, 0.003] | all 5 seeds >= 640 | 0.001 [0.000, 0.003] (learned) | >= 640 (ridge); T2 level 0.061 [0.025, 0.097] | all 5 seeds >= 640 | 0.061 [0.025, 0.097] (learned) |
| d1_base | pca128_a1 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0.001 [0.000, 0.003] | all 5 seeds >= 640 | 0.001 [0.000, 0.003] (learned) | >= 640 (proj); T2 level 0.061 [0.025, 0.097] | all 5 seeds >= 640 | 0.061 [0.025, 0.097] (learned) |
| d1_base | pca128_a1 | 128 | iom_256 | exact | >= 640 (ridge); T2 level 0.032 [0.007, 0.058] | all 5 seeds >= 640 | 0.032 [0.007, 0.058] (learned) | >= 640 (ridge); T2 level 0.169 [0.108, 0.230] | all 5 seeds >= 640 | 0.169 [0.108, 0.230] (learned) |
| d1_base | pca128_a1 | 128 | iom_256 | fresh | >= 640 (ridge); T2 level 0.032 [0.007, 0.058] | all 5 seeds >= 640 | 0.032 [0.007, 0.058] (learned) | >= 640 (ridge); T2 level 0.169 [0.108, 0.230] | all 5 seeds >= 640 | 0.169 [0.108, 0.230] (learned) |
| d1_base | pca256_a1 | 256 | lrp_full | exact | 217.0 (proj) | [195.7-246.7] | 0.721 [0.653, 0.790] (analytic) | 87.7 (proj) | [81.5-96.0] | 1.000 [1.000, 1.000] (analytic) |
| d1_base | pca256_a1 | 256 | lrp_full | fresh | >= 640 (proj); T2 level 0.721 [0.653, 0.790] | all 5 seeds >= 640 | 0.721 [0.653, 0.790] (analytic) | >= 640 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_base | pca256_a1 | 256 | lrp_half | exact | >= 640 (ridge); T2 level 0.164 [0.114, 0.215] | all 5 seeds >= 640 | 0.164 [0.114, 0.215] (learned) | 278.6 (ridge) | [243.0-359.6] | 0.718 [0.653, 0.782] (learned) |
| d1_base | pca256_a1 | 256 | lrp_half | fresh | >= 640 (proj); T2 level 0.164 [0.114, 0.215] | all 5 seeds >= 640 | 0.164 [0.114, 0.215] (learned) | >= 640 (proj); T2 level 0.718 [0.653, 0.782] | all 5 seeds >= 640 | 0.718 [0.653, 0.782] (learned) |
| d1_base | pca256_a1 | 256 | biohash_half | exact | >= 640 (ridge); T2 level 0.013 [0.000, 0.026] | all 5 seeds >= 640 | 0.013 [0.000, 0.026] (learned) | >= 640 (ridge); T2 level 0.358 [0.298, 0.418] | all 5 seeds >= 640 | 0.358 [0.298, 0.418] (learned) |
| d1_base | pca256_a1 | 256 | biohash_half | fresh | >= 640 (proj); T2 level 0.013 [0.000, 0.026] | all 5 seeds >= 640 | 0.013 [0.000, 0.026] (learned) | >= 640 (proj); T2 level 0.358 [0.298, 0.418] | all 5 seeds >= 640 | 0.358 [0.298, 0.418] (learned) |
| d1_base | pca256_a1 | 256 | iom_256 | exact | >= 640 (ridge); T2 level 0.010 [0.000, 0.024] | all 5 seeds >= 640 | 0.010 [0.000, 0.024] (learned) | >= 640 (ridge); T2 level 0.167 [0.115, 0.220] | all 5 seeds >= 640 | 0.167 [0.115, 0.220] (learned) |
| d1_base | pca256_a1 | 256 | iom_256 | fresh | >= 640 (proj); T2 level 0.010 [0.000, 0.024] | all 5 seeds >= 640 | 0.010 [0.000, 0.024] (learned) | >= 640 (ridge); T2 level 0.167 [0.115, 0.220] | all 5 seeds >= 640 | 0.167 [0.115, 0.220] (learned) |
| d1_base | pca128_a0.5 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.314 [0.248, 0.381] < 0.5 | all 5 seeds >= 640 | 0.314 [0.248, 0.381] (analytic) | 81.2 (proj) | [76.8-88.5] | 1.000 [1.000, 1.000] (analytic) |
| d1_base | pca128_a0.5 | 128 | lrp_full | fresh | >= 640 (ridge); T2 level 0.314 [0.248, 0.381] | all 5 seeds >= 640 | 0.314 [0.248, 0.381] (analytic) | >= 640 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_base | pca128_a0.5 | 128 | lrp_half | exact | >= 640 (ridge); T2 level 0.044 [0.015, 0.072] | all 5 seeds >= 640 | 0.044 [0.015, 0.072] (learned) | >= 640 (ridge); T2 level 0.198 [0.130, 0.265] | all 5 seeds >= 640 | 0.198 [0.130, 0.265] (learned) |
| d1_base | pca128_a0.5 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0.044 [0.015, 0.072] | all 5 seeds >= 640 | 0.044 [0.015, 0.072] (learned) | >= 640 (proj); T2 level 0.198 [0.130, 0.265] | all 5 seeds >= 640 | 0.198 [0.130, 0.265] (learned) |
| d1_base | pca128_a0.5 | 128 | biohash_half | exact | >= 640 (ridge); T2 level 0.002 [0.000, 0.008] | all 5 seeds >= 640 | 0.002 [0.000, 0.008] (learned) | >= 640 (ridge); T2 level 0.060 [0.021, 0.098] | all 5 seeds >= 640 | 0.060 [0.021, 0.098] (learned) |
| d1_base | pca128_a0.5 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0.002 [0.000, 0.008] | all 5 seeds >= 640 | 0.002 [0.000, 0.008] (learned) | >= 640 (ridge); T2 level 0.060 [0.021, 0.098] | all 5 seeds >= 640 | 0.060 [0.021, 0.098] (learned) |
| d1_base | pca128_a0.5 | 128 | iom_256 | exact | >= 640 (ridge); T2 level 0.032 [0.007, 0.057] | all 5 seeds >= 640 | 0.032 [0.007, 0.057] (learned) | >= 640 (ridge); T2 level 0.083 [0.037, 0.130] | all 5 seeds >= 640 | 0.083 [0.037, 0.130] (learned) |
| d1_base | pca128_a0.5 | 128 | iom_256 | fresh | >= 640 (proj); T2 level 0.032 [0.007, 0.057] | all 5 seeds >= 640 | 0.032 [0.007, 0.057] (learned) | >= 640 (ridge); T2 level 0.083 [0.037, 0.130] | all 5 seeds >= 640 | 0.083 [0.037, 0.130] (learned) |
| d1_base | fi64_p512 | 64 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.018 [0.000, 0.041] < 0.5 | all 5 seeds >= 640 | 0.018 [0.000, 0.041] (analytic) | 27.5 (proj) | [27.3-27.7] | 1.000 [1.000, 1.000] (analytic) |
| d1_base | fi64_p512 | 64 | lrp_full | fresh | >= 640 (proj); T2 level 0.018 [0.000, 0.041] | all 5 seeds >= 640 | 0.018 [0.000, 0.041] (analytic) | >= 640 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_base | fi64_p512 | 64 | lrp_half | exact | >= 640 (proj); T2 level 0.002 [0.000, 0.009] | all 5 seeds >= 640 | 0.002 [0.000, 0.009] (analytic) | >= 640 (ridge); T2 level 0.493 [0.423, 0.564] | [493.9-(>= 640)] (3 of 5 seeds censored) | 0.493 [0.423, 0.564] (learned) |
| d1_base | fi64_p512 | 64 | lrp_half | fresh | >= 640 (proj); T2 level 0.002 [0.000, 0.009] | all 5 seeds >= 640 | 0.002 [0.000, 0.009] (analytic) | >= 640 (proj); T2 level 0.493 [0.423, 0.564] | all 5 seeds >= 640 | 0.493 [0.423, 0.564] (learned) |
| d1_base | fi64_p512 | 64 | biohash_half | exact | >= 640 (proj); T2 level 0.001 [0.000, 0.003] | all 5 seeds >= 640 | 0.001 [0.000, 0.003] (analytic) | >= 640 (ridge); T2 level 0.117 [0.054, 0.180] | all 5 seeds >= 640 | 0.117 [0.054, 0.180] (learned) |
| d1_base | fi64_p512 | 64 | biohash_half | fresh | >= 640 (proj); T2 level 0.001 [0.000, 0.003] | all 5 seeds >= 640 | 0.001 [0.000, 0.003] (analytic) | >= 640 (ridge); T2 level 0.117 [0.054, 0.180] | all 5 seeds >= 640 | 0.117 [0.054, 0.180] (learned) |
| d1_base | fi64_p512 | 64 | iom_256 | exact | >= 640 (proj); T2 level 0.007 [0.000, 0.022] | all 5 seeds >= 640 | 0.007 [0.000, 0.022] (analytic) | >= 640 (ridge); T2 level 0.496 [0.415, 0.578] | all 5 seeds >= 640 | 0.496 [0.415, 0.578] (learned) |
| d1_base | fi64_p512 | 64 | iom_256 | fresh | >= 640 (proj); T2 level 0.007 [0.000, 0.022] | all 5 seeds >= 640 | 0.007 [0.000, 0.022] (analytic) | >= 640 (ridge); T2 level 0.496 [0.415, 0.578] | all 5 seeds >= 640 | 0.496 [0.415, 0.578] (learned) |
| d1_base | fi128_p512 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.036 [0.004, 0.067] < 0.5 | all 5 seeds >= 640 | 0.036 [0.004, 0.067] (analytic) | 40.9 (proj) | [39.1-43.7] | 1.000 [1.000, 1.000] (analytic) |
| d1_base | fi128_p512 | 128 | lrp_full | fresh | >= 640 (proj); T2 level 0.036 [0.004, 0.067] | all 5 seeds >= 640 | 0.036 [0.004, 0.067] (analytic) | >= 640 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_base | fi128_p512 | 128 | lrp_half | exact | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | 181.5 (ridge) | [167.0-203.3] | 0.682 [0.625, 0.739] (learned) |
| d1_base | fi128_p512 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (proj); T2 level 0.682 [0.625, 0.739] | all 5 seeds >= 640 | 0.682 [0.625, 0.739] (learned) |
| d1_base | fi128_p512 | 128 | biohash_half | exact | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.228 [0.163, 0.292] | all 5 seeds >= 640 | 0.228 [0.163, 0.292] (learned) |
| d1_base | fi128_p512 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.228 [0.163, 0.292] | all 5 seeds >= 640 | 0.228 [0.163, 0.292] (learned) |
| d1_base | fi128_p512 | 128 | iom_256 | exact | >= 640 (proj); T2 level 0.001 [0.000, 0.003] | all 5 seeds >= 640 | 0.001 [0.000, 0.003] (learned) | >= 640 (ridge); T2 level 0.300 [0.230, 0.369] | all 5 seeds >= 640 | 0.300 [0.230, 0.369] (learned) |
| d1_base | fi128_p512 | 128 | iom_256 | fresh | >= 640 (proj); T2 level 0.001 [0.000, 0.003] | all 5 seeds >= 640 | 0.001 [0.000, 0.003] (learned) | >= 640 (ridge); T2 level 0.300 [0.230, 0.369] | all 5 seeds >= 640 | 0.300 [0.230, 0.369] (learned) |
| d1_base | fi256_p512 | 256 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.126 [0.078, 0.173] < 0.5 | all 5 seeds >= 640 | 0.126 [0.078, 0.173] (analytic) | 58.6 (proj) | [57.8-59.6] | 1.000 [1.000, 1.000] (analytic) |
| d1_base | fi256_p512 | 256 | lrp_full | fresh | >= 640 (proj); T2 level 0.126 [0.078, 0.173] | all 5 seeds >= 640 | 0.126 [0.078, 0.173] (analytic) | >= 640 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_base | fi256_p512 | 256 | lrp_half | exact | >= 640 (proj); T2 level 0.005 [0.000, 0.018] | all 5 seeds >= 640 | 0.005 [0.000, 0.018] (analytic) | 153.7 (ridge) | [150.5-156.8] | 0.785 [0.740, 0.830] (learned) |
| d1_base | fi256_p512 | 256 | lrp_half | fresh | >= 640 (proj); T2 level 0.005 [0.000, 0.018] | all 5 seeds >= 640 | 0.005 [0.000, 0.018] (analytic) | >= 640 (ridge); T2 level 0.785 [0.740, 0.830] | all 5 seeds >= 640 | 0.785 [0.740, 0.830] (learned) |
| d1_base | fi256_p512 | 256 | biohash_half | exact | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.446 [0.384, 0.508] | all 5 seeds >= 640 | 0.446 [0.384, 0.508] (learned) |
| d1_base | fi256_p512 | 256 | biohash_half | fresh | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.446 [0.384, 0.508] | all 5 seeds >= 640 | 0.446 [0.384, 0.508] (learned) |
| d1_base | fi256_p512 | 256 | iom_256 | exact | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.240 [0.174, 0.306] | all 5 seeds >= 640 | 0.240 [0.174, 0.306] (learned) |
| d1_base | fi256_p512 | 256 | iom_256 | fresh | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.240 [0.174, 0.306] | all 5 seeds >= 640 | 0.240 [0.174, 0.306] (learned) |
| d1_base | rbig128_i2 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.314 [0.249, 0.380] < 0.5 | all 5 seeds >= 640 | 0.314 [0.249, 0.380] (analytic) | 67.3 (proj) | [62.8-71.7] | 0.989 [0.973, 1.004] (analytic) |
| d1_base | rbig128_i2 | 128 | lrp_full | fresh | >= 640 (proj); T2 level 0.314 [0.249, 0.380] | all 5 seeds >= 640 | 0.314 [0.249, 0.380] (analytic) | >= 640 (proj); T2 level 0.989 [0.973, 1.004] | all 5 seeds >= 640 | 0.989 [0.973, 1.004] (analytic) |
| d1_base | rbig128_i2 | 128 | lrp_half | exact | >= 640 (ridge); T2 level 0.047 [0.018, 0.075] | all 5 seeds >= 640 | 0.047 [0.018, 0.075] (learned) | >= 640 (proj); T2 level 0.336 [0.268, 0.404] | all 5 seeds >= 640 | 0.336 [0.268, 0.404] (learned) |
| d1_base | rbig128_i2 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0.047 [0.018, 0.075] | all 5 seeds >= 640 | 0.047 [0.018, 0.075] (learned) | >= 640 (proj); T2 level 0.336 [0.268, 0.404] | all 5 seeds >= 640 | 0.336 [0.268, 0.404] (learned) |
| d1_base | rbig128_i2 | 128 | biohash_half | exact | >= 640 (ridge); T2 level 0.002 [0.000, 0.008] | all 5 seeds >= 640 | 0.002 [0.000, 0.008] (learned) | >= 640 (proj); T2 level 0.095 [0.050, 0.139] | all 5 seeds >= 640 | 0.095 [0.050, 0.139] (learned) |
| d1_base | rbig128_i2 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0.002 [0.000, 0.008] | all 5 seeds >= 640 | 0.002 [0.000, 0.008] (learned) | >= 640 (ridge); T2 level 0.095 [0.050, 0.139] | all 5 seeds >= 640 | 0.095 [0.050, 0.139] (learned) |
| d1_base | rbig128_i2 | 128 | iom_256 | exact | >= 640 (ridge); T2 level 0.027 [0.005, 0.049] | all 5 seeds >= 640 | 0.027 [0.005, 0.049] (learned) | >= 640 (ridge); T2 level 0.188 [0.128, 0.249] | all 5 seeds >= 640 | 0.188 [0.128, 0.249] (learned) |
| d1_base | rbig128_i2 | 128 | iom_256 | fresh | >= 640 (ridge); T2 level 0.027 [0.005, 0.049] | all 5 seeds >= 640 | 0.027 [0.005, 0.049] (learned) | >= 640 (ridge); T2 level 0.188 [0.128, 0.249] | all 5 seeds >= 640 | 0.188 [0.128, 0.249] (learned) |
| d1_base | firbig128_p512_i2 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.037 [0.006, 0.067] < 0.5 | all 5 seeds >= 640 | 0.037 [0.006, 0.067] (analytic) | 41.9 (proj) | [40.1-44.1] | 0.988 [0.970, 1.006] (analytic) |
| d1_base | firbig128_p512_i2 | 128 | lrp_full | fresh | >= 640 (proj); T2 level 0.037 [0.006, 0.067] | all 5 seeds >= 640 | 0.037 [0.006, 0.067] (analytic) | >= 640 (ridge); T2 level 0.988 [0.970, 1.006] | all 5 seeds >= 640 | 0.988 [0.970, 1.006] (analytic) |
| d1_base | firbig128_p512_i2 | 128 | lrp_half | exact | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | 193.8 (ridge) | [162.4-215.1] | 0.667 [0.609, 0.725] (learned) |
| d1_base | firbig128_p512_i2 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (proj); T2 level 0.667 [0.609, 0.725] | all 5 seeds >= 640 | 0.667 [0.609, 0.725] (learned) |
| d1_base | firbig128_p512_i2 | 128 | biohash_half | exact | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.199 [0.140, 0.259] | all 5 seeds >= 640 | 0.199 [0.140, 0.259] (learned) |
| d1_base | firbig128_p512_i2 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.199 [0.140, 0.259] | all 5 seeds >= 640 | 0.199 [0.140, 0.259] (learned) |
| d1_base | firbig128_p512_i2 | 128 | iom_256 | exact | >= 640 (ridge); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.300 [0.231, 0.370] | all 5 seeds >= 640 | 0.300 [0.231, 0.370] (learned) |
| d1_base | firbig128_p512_i2 | 128 | iom_256 | fresh | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.024 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.024 (rel.) (analytic) | >= 640 (ridge); T2 level 0.300 [0.231, 0.370] | all 5 seeds >= 640 | 0.300 [0.231, 0.370] (learned) |
| d1_large | none | 1024 | lrp_full | exact | 233.4 (proj) | [224.2-243.6] | 1.000 [1.000, 1.000] (analytic) | 233.4 (proj) | [224.2-243.6] | 1.000 [1.000, 1.000] (analytic) |
| d1_large | none | 1024 | lrp_full | fresh | >= 640 (proj); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) | >= 640 (proj); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_large | none | 1024 | lrp_half | exact | 550.0 (ridge) | [515.7-591.7] | 0.832 [0.782, 0.882] (learned) | 549.0 (ridge) | [519.0-579.7] | 0.829 [0.777, 0.880] (learned) |
| d1_large | none | 1024 | lrp_half | fresh | >= 640 (ridge); T2 level 0.832 [0.782, 0.882] | all 5 seeds >= 640 | 0.832 [0.782, 0.882] (learned) | >= 640 (ridge); T2 level 0.829 [0.777, 0.880] | all 5 seeds >= 640 | 0.829 [0.777, 0.880] (learned) |
| d1_large | none | 1024 | biohash_half | exact | >= 640 (ridge); T2 level 0.525 [0.452, 0.598] | all 5 seeds >= 640 | 0.525 [0.452, 0.598] (learned) | >= 640 (ridge); T2 level 0.507 [0.435, 0.579] | all 5 seeds >= 640 | 0.507 [0.435, 0.579] (learned) |
| d1_large | none | 1024 | biohash_half | fresh | >= 640 (ridge); T2 level 0.525 [0.452, 0.598] | all 5 seeds >= 640 | 0.525 [0.452, 0.598] (learned) | >= 640 (ridge); T2 level 0.507 [0.435, 0.579] | all 5 seeds >= 640 | 0.507 [0.435, 0.579] (learned) |
| d1_large | none | 1024 | iom_256 | exact | >= 640 (ridge); T2 level 0.041 [0.011, 0.070] | all 5 seeds >= 640 | 0.041 [0.011, 0.070] (learned) | >= 640 (ridge); T2 level 0.047 [0.016, 0.078] | all 5 seeds >= 640 | 0.047 [0.016, 0.078] (learned) |
| d1_large | none | 1024 | iom_256 | fresh | >= 640 (proj); T2 level 0.041 [0.011, 0.070] | all 5 seeds >= 640 | 0.041 [0.011, 0.070] (learned) | >= 640 (ridge); T2 level 0.047 [0.016, 0.078] | all 5 seeds >= 640 | 0.047 [0.016, 0.078] (learned) |
| d1_large | pca128_a1 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.273 [0.204, 0.341] < 0.5 | all 5 seeds >= 640 | 0.273 [0.204, 0.341] (analytic) | 84.5 (proj) | [78.3-94.5] | 1.000 [1.000, 1.000] (analytic) |
| d1_large | pca128_a1 | 128 | lrp_full | fresh | >= 640 (proj); T2 level 0.273 [0.204, 0.341] | all 5 seeds >= 640 | 0.273 [0.204, 0.341] (analytic) | >= 640 (proj); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_large | pca128_a1 | 128 | lrp_half | exact | >= 640 (ridge); T2 level 0.052 [0.020, 0.084] | all 5 seeds >= 640 | 0.052 [0.020, 0.084] (learned) | >= 640 (proj); T2 level 0.189 [0.123, 0.255] | all 5 seeds >= 640 | 0.189 [0.123, 0.255] (learned) |
| d1_large | pca128_a1 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0.052 [0.020, 0.084] | all 5 seeds >= 640 | 0.052 [0.020, 0.084] (learned) | >= 640 (proj); T2 level 0.189 [0.123, 0.255] | all 5 seeds >= 640 | 0.189 [0.123, 0.255] (learned) |
| d1_large | pca128_a1 | 128 | biohash_half | exact | >= 640 (proj); T2 level 0.003 [0.000, 0.008] | all 5 seeds >= 640 | 0.003 [0.000, 0.008] (analytic) | >= 640 (ridge); T2 level 0.075 [0.031, 0.118] | all 5 seeds >= 640 | 0.075 [0.031, 0.118] (learned) |
| d1_large | pca128_a1 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0.003 [0.000, 0.008] | all 5 seeds >= 640 | 0.003 [0.000, 0.008] (analytic) | >= 640 (proj); T2 level 0.075 [0.031, 0.118] | all 5 seeds >= 640 | 0.075 [0.031, 0.118] (learned) |
| d1_large | pca128_a1 | 128 | iom_256 | exact | >= 640 (ridge); T2 level 0.045 [0.014, 0.076] | all 5 seeds >= 640 | 0.045 [0.014, 0.076] (learned) | >= 640 (ridge); T2 level 0.131 [0.073, 0.189] | all 5 seeds >= 640 | 0.131 [0.073, 0.189] (learned) |
| d1_large | pca128_a1 | 128 | iom_256 | fresh | >= 640 (proj); T2 level 0.045 [0.014, 0.076] | all 5 seeds >= 640 | 0.045 [0.014, 0.076] (learned) | >= 640 (proj); T2 level 0.131 [0.073, 0.189] | all 5 seeds >= 640 | 0.131 [0.073, 0.189] (learned) |
| d1_large | pca256_a1 | 256 | lrp_full | exact | 249.2 (proj) | [200.9-308.9] | 0.604 [0.512, 0.695] (analytic) | 102.6 (proj) | [98.2-108.9] | 1.000 [1.000, 1.000] (analytic) |
| d1_large | pca256_a1 | 256 | lrp_full | fresh | >= 640 (proj); T2 level 0.604 [0.512, 0.695] | all 5 seeds >= 640 | 0.604 [0.512, 0.695] (analytic) | >= 640 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_large | pca256_a1 | 256 | lrp_half | exact | >= 640 (ridge); T2 level 0.175 [0.121, 0.229] | all 5 seeds >= 640 | 0.175 [0.121, 0.229] (learned) | 498.7 (ridge) | [452.5-638.3] | 0.670 [0.575, 0.765] (learned) |
| d1_large | pca256_a1 | 256 | lrp_half | fresh | >= 640 (proj); T2 level 0.175 [0.121, 0.229] | all 5 seeds >= 640 | 0.175 [0.121, 0.229] (learned) | >= 640 (proj); T2 level 0.670 [0.575, 0.765] | all 5 seeds >= 640 | 0.670 [0.575, 0.765] (learned) |
| d1_large | pca256_a1 | 256 | biohash_half | exact | >= 640 (ridge); T2 level 0.050 [0.018, 0.082] | all 5 seeds >= 640 | 0.050 [0.018, 0.082] (learned) | >= 640 (ridge); T2 level 0.286 [0.198, 0.374] | all 5 seeds >= 640 | 0.286 [0.198, 0.374] (learned) |
| d1_large | pca256_a1 | 256 | biohash_half | fresh | >= 640 (proj); T2 level 0.050 [0.018, 0.082] | all 5 seeds >= 640 | 0.050 [0.018, 0.082] (learned) | >= 640 (proj); T2 level 0.286 [0.198, 0.374] | all 5 seeds >= 640 | 0.286 [0.198, 0.374] (learned) |
| d1_large | pca256_a1 | 256 | iom_256 | exact | >= 640 (ridge); T2 level 0.043 [0.012, 0.075] | all 5 seeds >= 640 | 0.043 [0.012, 0.075] (learned) | >= 640 (ridge); T2 level 0.189 [0.130, 0.247] | all 5 seeds >= 640 | 0.189 [0.130, 0.247] (learned) |
| d1_large | pca256_a1 | 256 | iom_256 | fresh | >= 640 (proj); T2 level 0.043 [0.012, 0.075] | all 5 seeds >= 640 | 0.043 [0.012, 0.075] (learned) | >= 640 (ridge); T2 level 0.189 [0.130, 0.247] | all 5 seeds >= 640 | 0.189 [0.130, 0.247] (learned) |
| d1_large | pca128_a0.5 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.273 [0.203, 0.343] < 0.5 | all 5 seeds >= 640 | 0.273 [0.203, 0.343] (analytic) | 93.3 (proj) | [89.8-100.4] | 1.000 [1.000, 1.000] (analytic) |
| d1_large | pca128_a0.5 | 128 | lrp_full | fresh | >= 640 (proj); T2 level 0.273 [0.203, 0.343] | all 5 seeds >= 640 | 0.273 [0.203, 0.343] (analytic) | >= 640 (proj); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_large | pca128_a0.5 | 128 | lrp_half | exact | >= 640 (ridge); T2 level 0.053 [0.021, 0.085] | all 5 seeds >= 640 | 0.053 [0.021, 0.085] (analytic) | >= 640 (ridge); T2 level 0.085 [0.039, 0.131] | all 5 seeds >= 640 | 0.085 [0.039, 0.131] (learned) |
| d1_large | pca128_a0.5 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0.053 [0.021, 0.085] | all 5 seeds >= 640 | 0.053 [0.021, 0.085] (analytic) | >= 640 (proj); T2 level 0.085 [0.039, 0.131] | all 5 seeds >= 640 | 0.085 [0.039, 0.131] (learned) |
| d1_large | pca128_a0.5 | 128 | biohash_half | exact | >= 640 (proj); T2 level 0.003 [0.000, 0.008] | all 5 seeds >= 640 | 0.003 [0.000, 0.008] (learned) | >= 640 (ridge); T2 level 0.038 [0.008, 0.069] | all 5 seeds >= 640 | 0.038 [0.008, 0.069] (learned) |
| d1_large | pca128_a0.5 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0.003 [0.000, 0.008] | all 5 seeds >= 640 | 0.003 [0.000, 0.008] (learned) | >= 640 (proj); T2 level 0.038 [0.008, 0.069] | all 5 seeds >= 640 | 0.038 [0.008, 0.069] (learned) |
| d1_large | pca128_a0.5 | 128 | iom_256 | exact | >= 640 (ridge); T2 level 0.050 [0.017, 0.083] | all 5 seeds >= 640 | 0.050 [0.017, 0.083] (learned) | >= 640 (ridge); T2 level 0.094 [0.040, 0.147] | all 5 seeds >= 640 | 0.094 [0.040, 0.147] (learned) |
| d1_large | pca128_a0.5 | 128 | iom_256 | fresh | >= 640 (proj); T2 level 0.050 [0.017, 0.083] | all 5 seeds >= 640 | 0.050 [0.017, 0.083] (learned) | >= 640 (proj); T2 level 0.094 [0.040, 0.147] | all 5 seeds >= 640 | 0.094 [0.040, 0.147] (learned) |
| d1_large | fi64_p512 | 64 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.025 [0.000, 0.056] < 0.5 | all 5 seeds >= 640 | 0.025 [0.000, 0.056] (analytic) | 27.6 (proj) | [26.9-28.6] | 1.000 [1.000, 1.000] (analytic) |
| d1_large | fi64_p512 | 64 | lrp_full | fresh | >= 640 (proj); T2 level 0.025 [0.000, 0.056] | all 5 seeds >= 640 | 0.025 [0.000, 0.056] (analytic) | >= 640 (proj); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_large | fi64_p512 | 64 | lrp_half | exact | >= 640 (proj); T2 level 0.001 [0.000, 0.006] | all 5 seeds >= 640 | 0.001 [0.000, 0.006] (analytic) | >= 640 (proj); T2 level 0.538 [0.452, 0.624] | [242.3-(>= 640)] (4 of 5 seeds censored) | 0.538 [0.452, 0.624] (learned) |
| d1_large | fi64_p512 | 64 | lrp_half | fresh | >= 640 (proj); T2 level 0.001 [0.000, 0.006] | all 5 seeds >= 640 | 0.001 [0.000, 0.006] (analytic) | >= 640 (proj); T2 level 0.538 [0.452, 0.624] | all 5 seeds >= 640 | 0.538 [0.452, 0.624] (learned) |
| d1_large | fi64_p512 | 64 | biohash_half | exact | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.026 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.026 (rel.) (analytic) | >= 640 (proj); T2 level 0.122 [0.057, 0.187] | all 5 seeds >= 640 | 0.122 [0.057, 0.187] (learned) |
| d1_large | fi64_p512 | 64 | biohash_half | fresh | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.026 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.026 (rel.) (analytic) | >= 640 (proj); T2 level 0.122 [0.057, 0.187] | all 5 seeds >= 640 | 0.122 [0.057, 0.187] (learned) |
| d1_large | fi64_p512 | 64 | iom_256 | exact | >= 640 (proj); T2 level 0.008 [0.000, 0.020] | all 5 seeds >= 640 | 0.008 [0.000, 0.020] (analytic) | >= 640 (ridge); T2 level 0.509 [0.425, 0.594] | all 5 seeds >= 640 | 0.509 [0.425, 0.594] (learned) |
| d1_large | fi64_p512 | 64 | iom_256 | fresh | >= 640 (proj); T2 level 0.008 [0.000, 0.020] | all 5 seeds >= 640 | 0.008 [0.000, 0.020] (analytic) | >= 640 (ridge); T2 level 0.509 [0.425, 0.594] | all 5 seeds >= 640 | 0.509 [0.425, 0.594] (learned) |
| d1_large | fi128_p512 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.042 [0.009, 0.075] < 0.5 | all 5 seeds >= 640 | 0.042 [0.009, 0.075] (analytic) | 41.6 (proj) | [38.9-44.5] | 1.000 [1.000, 1.000] (analytic) |
| d1_large | fi128_p512 | 128 | lrp_full | fresh | >= 640 (proj); T2 level 0.042 [0.009, 0.075] | all 5 seeds >= 640 | 0.042 [0.009, 0.075] (analytic) | >= 640 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_large | fi128_p512 | 128 | lrp_half | exact | >= 640 (proj); T2 level 0.006 [0.000, 0.019] | all 5 seeds >= 640 | 0.006 [0.000, 0.019] (analytic) | 194.1 (ridge) | [177.5-223.2] | 0.701 [0.642, 0.760] (learned) |
| d1_large | fi128_p512 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0.006 [0.000, 0.019] | all 5 seeds >= 640 | 0.006 [0.000, 0.019] (analytic) | >= 640 (proj); T2 level 0.701 [0.642, 0.760] | all 5 seeds >= 640 | 0.701 [0.642, 0.760] (learned) |
| d1_large | fi128_p512 | 128 | biohash_half | exact | >= 640 (proj); T2 level 0.002 [0.000, 0.006] | all 5 seeds >= 640 | 0.002 [0.000, 0.006] (analytic) | >= 640 (proj); T2 level 0.204 [0.141, 0.267] | all 5 seeds >= 640 | 0.204 [0.141, 0.267] (learned) |
| d1_large | fi128_p512 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0.002 [0.000, 0.006] | all 5 seeds >= 640 | 0.002 [0.000, 0.006] (analytic) | >= 640 (proj); T2 level 0.204 [0.141, 0.267] | all 5 seeds >= 640 | 0.204 [0.141, 0.267] (learned) |
| d1_large | fi128_p512 | 128 | iom_256 | exact | >= 640 (proj); T2 level 0.008 [0.000, 0.020] | all 5 seeds >= 640 | 0.008 [0.000, 0.020] (analytic) | >= 640 (ridge); T2 level 0.314 [0.241, 0.387] | all 5 seeds >= 640 | 0.314 [0.241, 0.387] (learned) |
| d1_large | fi128_p512 | 128 | iom_256 | fresh | >= 640 (proj); T2 level 0.008 [0.000, 0.020] | all 5 seeds >= 640 | 0.008 [0.000, 0.020] (analytic) | >= 640 (ridge); T2 level 0.314 [0.241, 0.387] | all 5 seeds >= 640 | 0.314 [0.241, 0.387] (learned) |
| d1_large | fi256_p512 | 256 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.156 [0.092, 0.219] < 0.5 | all 5 seeds >= 640 | 0.156 [0.092, 0.219] (analytic) | 59.8 (proj) | [53.7-62.6] | 1.000 [1.000, 1.000] (analytic) |
| d1_large | fi256_p512 | 256 | lrp_full | fresh | >= 640 (proj); T2 level 0.156 [0.092, 0.219] | all 5 seeds >= 640 | 0.156 [0.092, 0.219] (analytic) | >= 640 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 640 | 1.000 [1.000, 1.000] (analytic) |
| d1_large | fi256_p512 | 256 | lrp_half | exact | >= 640 (ridge); T2 level 0.011 [0.000, 0.025] | all 5 seeds >= 640 | 0.011 [0.000, 0.025] (analytic) | 155.1 (ridge) | [145.8-166.8] | 0.789 [0.741, 0.837] (learned) |
| d1_large | fi256_p512 | 256 | lrp_half | fresh | >= 640 (proj); T2 level 0.011 [0.000, 0.025] | all 5 seeds >= 640 | 0.011 [0.000, 0.025] (analytic) | >= 640 (proj); T2 level 0.789 [0.741, 0.837] | all 5 seeds >= 640 | 0.789 [0.741, 0.837] (learned) |
| d1_large | fi256_p512 | 256 | biohash_half | exact | >= 640 (proj); T2 level 0.004 [0.000, 0.010] | all 5 seeds >= 640 | 0.004 [0.000, 0.010] (analytic) | >= 640 (ridge); T2 level 0.451 [0.377, 0.525] | all 5 seeds >= 640 | 0.451 [0.377, 0.525] (learned) |
| d1_large | fi256_p512 | 256 | biohash_half | fresh | >= 640 (proj); T2 level 0.004 [0.000, 0.010] | all 5 seeds >= 640 | 0.004 [0.000, 0.010] (analytic) | >= 640 (ridge); T2 level 0.451 [0.377, 0.525] | all 5 seeds >= 640 | 0.451 [0.377, 0.525] (learned) |
| d1_large | fi256_p512 | 256 | iom_256 | exact | >= 640 (proj); T2 level 0.001 [0.000, 0.003] | all 5 seeds >= 640 | 0.001 [0.000, 0.003] (learned) | >= 640 (ridge); T2 level 0.201 [0.142, 0.260] | all 5 seeds >= 640 | 0.201 [0.142, 0.260] (learned) |
| d1_large | fi256_p512 | 256 | iom_256 | fresh | >= 640 (proj); T2 level 0.001 [0.000, 0.003] | all 5 seeds >= 640 | 0.001 [0.000, 0.003] (learned) | >= 640 (ridge); T2 level 0.201 [0.142, 0.260] | all 5 seeds >= 640 | 0.201 [0.142, 0.260] (learned) |
| d1_large | rbig128_i2 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.273 [0.207, 0.339] < 0.5 | all 5 seeds >= 640 | 0.273 [0.207, 0.339] (analytic) | 82.7 (proj) | [76.7-92.1] | 0.983 [0.960, 1.006] (analytic) |
| d1_large | rbig128_i2 | 128 | lrp_full | fresh | >= 640 (proj); T2 level 0.273 [0.207, 0.339] | all 5 seeds >= 640 | 0.273 [0.207, 0.339] (analytic) | >= 640 (proj); T2 level 0.983 [0.960, 1.006] | all 5 seeds >= 640 | 0.983 [0.960, 1.006] (analytic) |
| d1_large | rbig128_i2 | 128 | lrp_half | exact | >= 640 (ridge); T2 level 0.043 [0.014, 0.072] | all 5 seeds >= 640 | 0.043 [0.014, 0.072] (analytic) | >= 640 (ridge); T2 level 0.211 [0.146, 0.277] | all 5 seeds >= 640 | 0.211 [0.146, 0.277] (learned) |
| d1_large | rbig128_i2 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0.043 [0.014, 0.072] | all 5 seeds >= 640 | 0.043 [0.014, 0.072] (analytic) | >= 640 (proj); T2 level 0.211 [0.146, 0.277] | all 5 seeds >= 640 | 0.211 [0.146, 0.277] (learned) |
| d1_large | rbig128_i2 | 128 | biohash_half | exact | >= 640 (proj); T2 level 0.003 [0.000, 0.008] | all 5 seeds >= 640 | 0.003 [0.000, 0.008] (learned) | >= 640 (proj); T2 level 0.069 [0.027, 0.110] | all 5 seeds >= 640 | 0.069 [0.027, 0.110] (learned) |
| d1_large | rbig128_i2 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0.003 [0.000, 0.008] | all 5 seeds >= 640 | 0.003 [0.000, 0.008] (learned) | >= 640 (proj); T2 level 0.069 [0.027, 0.110] | all 5 seeds >= 640 | 0.069 [0.027, 0.110] (learned) |
| d1_large | rbig128_i2 | 128 | iom_256 | exact | >= 640 (ridge); T2 level 0.045 [0.014, 0.076] | all 5 seeds >= 640 | 0.045 [0.014, 0.076] (learned) | >= 640 (ridge); T2 level 0.138 [0.057, 0.219] | all 5 seeds >= 640 | 0.138 [0.057, 0.219] (learned) |
| d1_large | rbig128_i2 | 128 | iom_256 | fresh | >= 640 (proj); T2 level 0.045 [0.014, 0.076] | all 5 seeds >= 640 | 0.045 [0.014, 0.076] (learned) | >= 640 (proj); T2 level 0.138 [0.057, 0.219] | all 5 seeds >= 640 | 0.138 [0.057, 0.219] (learned) |
| d1_large | firbig128_p512_i2 | 128 | lrp_full | exact | >= 640 (proj); back-map-limited: T2 level 0.042 [0.009, 0.075] < 0.5 | all 5 seeds >= 640 | 0.042 [0.009, 0.075] (analytic) | 43.2 (proj) | [40.5-46.0] | 0.977 [0.954, 0.999] (analytic) |
| d1_large | firbig128_p512_i2 | 128 | lrp_full | fresh | >= 640 (proj); T2 level 0.042 [0.009, 0.075] | all 5 seeds >= 640 | 0.042 [0.009, 0.075] (analytic) | >= 640 (proj); T2 level 0.977 [0.954, 0.999] | all 5 seeds >= 640 | 0.977 [0.954, 0.999] (analytic) |
| d1_large | firbig128_p512_i2 | 128 | lrp_half | exact | >= 640 (proj); T2 level 0.002 [0.000, 0.007] | all 5 seeds >= 640 | 0.002 [0.000, 0.007] (analytic) | 233.1 (ridge) | [183.8-304.7] | 0.655 [0.591, 0.719] (learned) |
| d1_large | firbig128_p512_i2 | 128 | lrp_half | fresh | >= 640 (proj); T2 level 0.002 [0.000, 0.007] | all 5 seeds >= 640 | 0.002 [0.000, 0.007] (analytic) | >= 640 (proj); T2 level 0.655 [0.591, 0.719] | all 5 seeds >= 640 | 0.655 [0.591, 0.719] (learned) |
| d1_large | firbig128_p512_i2 | 128 | biohash_half | exact | >= 640 (ridge); T2 level 0 successes; CP95 upper <= 0.026 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.026 (rel.) (analytic) | >= 640 (ridge); T2 level 0.188 [0.128, 0.248] | all 5 seeds >= 640 | 0.188 [0.128, 0.248] (learned) |
| d1_large | firbig128_p512_i2 | 128 | biohash_half | fresh | >= 640 (proj); T2 level 0 successes; CP95 upper <= 0.026 (rel.) | all 5 seeds >= 640 | 0 successes; CP95 upper <= 0.026 (rel.) (analytic) | >= 640 (ridge); T2 level 0.188 [0.128, 0.248] | all 5 seeds >= 640 | 0.188 [0.128, 0.248] (learned) |
| d1_large | firbig128_p512_i2 | 128 | iom_256 | exact | >= 640 (ridge); T2 level 0.002 [0.000, 0.007] | all 5 seeds >= 640 | 0.002 [0.000, 0.007] (analytic) | >= 640 (ridge); T2 level 0.284 [0.213, 0.354] | all 5 seeds >= 640 | 0.284 [0.213, 0.354] (learned) |
| d1_large | firbig128_p512_i2 | 128 | iom_256 | fresh | >= 640 (proj); T2 level 0.002 [0.000, 0.007] | all 5 seeds >= 640 | 0.002 [0.000, 0.007] (analytic) | >= 640 (ridge); T2 level 0.284 [0.213, 0.354] | all 5 seeds >= 640 | 0.284 [0.213, 0.354] (learned) |
| d2_fingervit | none | 1536 | lrp_full | exact | 53.0 (proj) | [42.8-69.7] | 1.001 [0.998, 1.005] (learned) | 53.0 (proj) | [42.8-69.7] | 1.001 [0.998, 1.005] (learned) |
| d2_fingervit | none | 1536 | lrp_full | fresh | >= 320 (mlp); T2 level 1.001 [0.998, 1.005] | all 5 seeds >= 320 | 1.001 [0.998, 1.005] (learned) | >= 320 (mlp); T2 level 1.001 [0.998, 1.005] | all 5 seeds >= 320 | 1.001 [0.998, 1.005] (learned) |
| d2_fingervit | none | 1536 | lrp_half | exact | 58.5 (proj) | [46.9-81.3] | 0.991 [0.964, 1.018] (learned) | 57.3 (proj) | [48.2-79.0] | 0.989 [0.964, 1.014] (learned) |
| d2_fingervit | none | 1536 | lrp_half | fresh | >= 320 (ridge); T2 level 0.991 [0.964, 1.018] | all 5 seeds >= 320 | 0.991 [0.964, 1.018] (learned) | >= 320 (ridge); T2 level 0.989 [0.964, 1.014] | all 5 seeds >= 320 | 0.989 [0.964, 1.014] (learned) |
| d2_fingervit | none | 1536 | biohash_half | exact | 152.9 (proj) | [82.2-(>= 320)] (3 of 5 seeds censored) | 0.765 [0.685, 0.844] (learned) | 156.1 (proj) | [84.4-(>= 320)] (1 of 5 seeds censored) | 0.789 [0.708, 0.869] (learned) |
| d2_fingervit | none | 1536 | biohash_half | fresh | >= 320 (proj); T2 level 0.765 [0.685, 0.844] | all 5 seeds >= 320 | 0.765 [0.685, 0.844] (learned) | >= 320 (proj); T2 level 0.789 [0.708, 0.869] | all 5 seeds >= 320 | 0.789 [0.708, 0.869] (learned) |
| d2_fingervit | none | 1536 | iom_256 | exact | >= 320 (ridge); T2 level 0.381 [0.254, 0.508] | all 5 seeds >= 320 | 0.381 [0.254, 0.508] (learned) | >= 320 (ridge); T2 level 0.352 [0.204, 0.500] | all 5 seeds >= 320 | 0.352 [0.204, 0.500] (learned) |
| d2_fingervit | none | 1536 | iom_256 | fresh | >= 320 (ridge); T2 level 0.381 [0.254, 0.508] | all 5 seeds >= 320 | 0.381 [0.254, 0.508] (learned) | >= 320 (ridge); T2 level 0.352 [0.204, 0.500] | all 5 seeds >= 320 | 0.352 [0.204, 0.500] (learned) |
| d2_fingervit | pca128_a1 | 128 | lrp_full | exact | 106.8 (ridge) | [98.2-112.4] | 0.928 [0.877, 0.978] (analytic) | 55.3 (proj) | [53.5-56.4] | 1.000 [1.000, 1.000] (analytic) |
| d2_fingervit | pca128_a1 | 128 | lrp_full | fresh | >= 320 (proj); T2 level 0.928 [0.877, 0.978] | all 5 seeds >= 320 | 0.928 [0.877, 0.978] (analytic) | 236.0 (ridge) | [212.6-290.4] | 1.000 [1.000, 1.000] (analytic) |
| d2_fingervit | pca128_a1 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.004 [0.000, 0.012] | all 5 seeds >= 320 | 0.004 [0.000, 0.012] (learned) | >= 320 (ridge); T2 level 0.452 [0.387, 0.517] | all 5 seeds >= 320 | 0.452 [0.387, 0.517] (analytic) |
| d2_fingervit | pca128_a1 | 128 | lrp_half | fresh | >= 320 (proj); T2 level 0.004 [0.000, 0.012] | all 5 seeds >= 320 | 0.004 [0.000, 0.012] (learned) | >= 320 (ridge); T2 level 0.452 [0.387, 0.517] | all 5 seeds >= 320 | 0.452 [0.387, 0.517] (analytic) |
| d2_fingervit | pca128_a1 | 128 | biohash_half | exact | >= 320 (proj); T2 level 0.004 [0.000, 0.016] | all 5 seeds >= 320 | 0.004 [0.000, 0.016] (learned) | >= 320 (ridge); T2 level 0.150 [0.100, 0.201] | all 5 seeds >= 320 | 0.150 [0.100, 0.201] (learned) |
| d2_fingervit | pca128_a1 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.004 [0.000, 0.016] | all 5 seeds >= 320 | 0.004 [0.000, 0.016] (learned) | >= 320 (ridge); T2 level 0.150 [0.100, 0.201] | all 5 seeds >= 320 | 0.150 [0.100, 0.201] (learned) |
| d2_fingervit | pca128_a1 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.005 [0.000, 0.015] | all 5 seeds >= 320 | 0.005 [0.000, 0.015] (learned) | >= 320 (ridge); T2 level 0.267 [0.203, 0.331] | all 5 seeds >= 320 | 0.267 [0.203, 0.331] (learned) |
| d2_fingervit | pca128_a1 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.005 [0.000, 0.015] | all 5 seeds >= 320 | 0.005 [0.000, 0.015] (learned) | >= 320 (ridge); T2 level 0.267 [0.203, 0.331] | all 5 seeds >= 320 | 0.267 [0.203, 0.331] (learned) |
| d2_fingervit | pca256_a1 | 256 | lrp_full | exact | 207.6 (ridge) | [195.4-218.0] | 0.964 [0.927, 1.002] (analytic) | 79.4 (proj) | [73.7-85.4] | 1.000 [1.000, 1.000] (analytic) |
| d2_fingervit | pca256_a1 | 256 | lrp_full | fresh | >= 320 (ridge); T2 level 0.964 [0.927, 1.002] | all 5 seeds >= 320 | 0.964 [0.927, 1.002] (analytic) | 306.0 (ridge) | [285.1-(>= 320)] (2 of 5 seeds censored) | 1.000 [1.000, 1.000] (analytic) |
| d2_fingervit | pca256_a1 | 256 | lrp_half | exact | >= 320 (ridge); T2 level 0.029 [0.000, 0.060] | all 5 seeds >= 320 | 0.029 [0.000, 0.060] (analytic) | 301.2 (ridge) | [265.4-(>= 320)] (1 of 5 seeds censored) | 0.662 [0.609, 0.716] (learned) |
| d2_fingervit | pca256_a1 | 256 | lrp_half | fresh | >= 320 (ridge); T2 level 0.029 [0.000, 0.060] | all 5 seeds >= 320 | 0.029 [0.000, 0.060] (analytic) | >= 320 (ridge); T2 level 0.662 [0.609, 0.716] | all 5 seeds >= 320 | 0.662 [0.609, 0.716] (learned) |
| d2_fingervit | pca256_a1 | 256 | biohash_half | exact | >= 320 (proj); T2 level 0.001 [0.000, 0.005] | all 5 seeds >= 320 | 0.001 [0.000, 0.005] (learned) | >= 320 (ridge); T2 level 0.327 [0.268, 0.387] | all 5 seeds >= 320 | 0.327 [0.268, 0.387] (learned) |
| d2_fingervit | pca256_a1 | 256 | biohash_half | fresh | >= 320 (proj); T2 level 0.001 [0.000, 0.005] | all 5 seeds >= 320 | 0.001 [0.000, 0.005] (learned) | >= 320 (ridge); T2 level 0.327 [0.268, 0.387] | all 5 seeds >= 320 | 0.327 [0.268, 0.387] (learned) |
| d2_fingervit | pca256_a1 | 256 | iom_256 | exact | >= 320 (proj); T2 level 0 successes; CP95 upper <= 0.044 (rel.) | all 5 seeds >= 320 | 0 successes; CP95 upper <= 0.044 (rel.) (analytic) | >= 320 (ridge); T2 level 0.166 [0.113, 0.219] | all 5 seeds >= 320 | 0.166 [0.113, 0.219] (learned) |
| d2_fingervit | pca256_a1 | 256 | iom_256 | fresh | >= 320 (proj); T2 level 0 successes; CP95 upper <= 0.044 (rel.) | all 5 seeds >= 320 | 0 successes; CP95 upper <= 0.044 (rel.) (analytic) | >= 320 (ridge); T2 level 0.166 [0.113, 0.219] | all 5 seeds >= 320 | 0.166 [0.113, 0.219] (learned) |
| d2_fingervit | pca128_a0.5 | 128 | lrp_full | exact | 46.4 (proj) | [39.3-58.7] | 0.928 [0.878, 0.977] (analytic) | 53.9 (proj) | [49.6-57.2] | 1.000 [1.000, 1.000] (analytic) |
| d2_fingervit | pca128_a0.5 | 128 | lrp_full | fresh | 278.3 (ridge) | [147.7-(>= 320)] (2 of 5 seeds censored) | 0.928 [0.878, 0.977] (analytic) | 169.5 (ridge) | [145.2-232.5] | 1.000 [1.000, 1.000] (analytic) |
| d2_fingervit | pca128_a0.5 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.374 [0.254, 0.494] | all 5 seeds >= 320 | 0.374 [0.254, 0.494] (analytic) | >= 320 (ridge); T2 level 0.484 [0.391, 0.577] | all 5 seeds >= 320 | 0.484 [0.391, 0.577] (learned) |
| d2_fingervit | pca128_a0.5 | 128 | lrp_half | fresh | >= 320 (ridge); T2 level 0.374 [0.254, 0.494] | all 5 seeds >= 320 | 0.374 [0.254, 0.494] (analytic) | >= 320 (ridge); T2 level 0.484 [0.391, 0.577] | [315.7-(>= 320)] (4 of 5 seeds censored) | 0.484 [0.391, 0.577] (learned) |
| d2_fingervit | pca128_a0.5 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.080 [0.024, 0.136] | all 5 seeds >= 320 | 0.080 [0.024, 0.136] (learned) | >= 320 (ridge); T2 level 0.117 [0.058, 0.175] | all 5 seeds >= 320 | 0.117 [0.058, 0.175] (learned) |
| d2_fingervit | pca128_a0.5 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.080 [0.024, 0.136] | all 5 seeds >= 320 | 0.080 [0.024, 0.136] (learned) | >= 320 (ridge); T2 level 0.117 [0.058, 0.175] | all 5 seeds >= 320 | 0.117 [0.058, 0.175] (learned) |
| d2_fingervit | pca128_a0.5 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.187 [0.093, 0.281] | all 5 seeds >= 320 | 0.187 [0.093, 0.281] (learned) | >= 320 (ridge); T2 level 0.239 [0.162, 0.316] | all 5 seeds >= 320 | 0.239 [0.162, 0.316] (learned) |
| d2_fingervit | pca128_a0.5 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.187 [0.093, 0.281] | all 5 seeds >= 320 | 0.187 [0.093, 0.281] (learned) | >= 320 (ridge); T2 level 0.239 [0.162, 0.316] | all 5 seeds >= 320 | 0.239 [0.162, 0.316] (learned) |
| d2_fingervit | fi64_p512 | 64 | lrp_full | exact | >= 320 (proj); T2 level 0.481 [0.382, 0.580] | [68.4-(>= 320)] (2 of 5 seeds censored) | 0.481 [0.382, 0.580] (analytic) | 40.1 (proj) | [37.4-43.6] | 1.000 [0.998, 1.002] (analytic) |
| d2_fingervit | fi64_p512 | 64 | lrp_full | fresh | >= 320 (proj); T2 level 0.481 [0.382, 0.580] | all 5 seeds >= 320 | 0.481 [0.382, 0.580] (analytic) | 188.1 (ridge) | [173.1-206.0] | 1.000 [0.998, 1.002] (analytic) |
| d2_fingervit | fi64_p512 | 64 | lrp_half | exact | >= 320 (proj); T2 level 0.033 [0.000, 0.071] | all 5 seeds >= 320 | 0.033 [0.000, 0.071] (analytic) | >= 320 (ridge); T2 level 0.225 [0.178, 0.273] | all 5 seeds >= 320 | 0.225 [0.178, 0.273] (learned) |
| d2_fingervit | fi64_p512 | 64 | lrp_half | fresh | >= 320 (proj); T2 level 0.033 [0.000, 0.071] | all 5 seeds >= 320 | 0.033 [0.000, 0.071] (analytic) | >= 320 (ridge); T2 level 0.225 [0.178, 0.273] | all 5 seeds >= 320 | 0.225 [0.178, 0.273] (learned) |
| d2_fingervit | fi64_p512 | 64 | biohash_half | exact | >= 320 (proj); T2 level 0.008 [0.000, 0.030] | all 5 seeds >= 320 | 0.008 [0.000, 0.030] (learned) | >= 320 (ridge); T2 level 0.080 [0.031, 0.129] | all 5 seeds >= 320 | 0.080 [0.031, 0.129] (learned) |
| d2_fingervit | fi64_p512 | 64 | biohash_half | fresh | >= 320 (proj); T2 level 0.008 [0.000, 0.030] | all 5 seeds >= 320 | 0.008 [0.000, 0.030] (learned) | >= 320 (ridge); T2 level 0.080 [0.031, 0.129] | all 5 seeds >= 320 | 0.080 [0.031, 0.129] (learned) |
| d2_fingervit | fi64_p512 | 64 | iom_256 | exact | >= 320 (ridge); T2 level 0.084 [0.018, 0.149] | all 5 seeds >= 320 | 0.084 [0.018, 0.149] (learned) | >= 320 (ridge); T2 level 0.405 [0.329, 0.481] | all 5 seeds >= 320 | 0.405 [0.329, 0.481] (learned) |
| d2_fingervit | fi64_p512 | 64 | iom_256 | fresh | >= 320 (ridge); T2 level 0.084 [0.018, 0.149] | all 5 seeds >= 320 | 0.084 [0.018, 0.149] (learned) | >= 320 (ridge); T2 level 0.405 [0.329, 0.481] | all 5 seeds >= 320 | 0.405 [0.329, 0.481] (learned) |
| d2_fingervit | fi128_p512 | 128 | lrp_full | exact | 145.4 (proj) | [134.0-(>= 320)] (1 of 5 seeds censored) | 0.548 [0.446, 0.650] (analytic) | 58.8 (proj) | [53.8-60.8] | 1.000 [0.999, 1.001] (analytic) |
| d2_fingervit | fi128_p512 | 128 | lrp_full | fresh | >= 320 (proj); T2 level 0.548 [0.446, 0.650] | all 5 seeds >= 320 | 0.548 [0.446, 0.650] (analytic) | 275.6 (ridge) | [252.8-(>= 320)] (1 of 5 seeds censored) | 1.000 [0.999, 1.001] (analytic) |
| d2_fingervit | fi128_p512 | 128 | lrp_half | exact | >= 320 (proj); T2 level 0.026 [0.000, 0.067] | all 5 seeds >= 320 | 0.026 [0.000, 0.067] (learned) | >= 320 (ridge); T2 level 0.491 [0.442, 0.539] | all 5 seeds >= 320 | 0.491 [0.442, 0.539] (learned) |
| d2_fingervit | fi128_p512 | 128 | lrp_half | fresh | >= 320 (ridge); T2 level 0.026 [0.000, 0.067] | all 5 seeds >= 320 | 0.026 [0.000, 0.067] (learned) | >= 320 (ridge); T2 level 0.491 [0.442, 0.539] | all 5 seeds >= 320 | 0.491 [0.442, 0.539] (learned) |
| d2_fingervit | fi128_p512 | 128 | biohash_half | exact | >= 320 (proj); T2 level 0.003 [0.000, 0.011] | all 5 seeds >= 320 | 0.003 [0.000, 0.011] (analytic) | >= 320 (ridge); T2 level 0.180 [0.134, 0.226] | all 5 seeds >= 320 | 0.180 [0.134, 0.226] (learned) |
| d2_fingervit | fi128_p512 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.003 [0.000, 0.011] | all 5 seeds >= 320 | 0.003 [0.000, 0.011] (analytic) | >= 320 (ridge); T2 level 0.180 [0.134, 0.226] | all 5 seeds >= 320 | 0.180 [0.134, 0.226] (learned) |
| d2_fingervit | fi128_p512 | 128 | iom_256 | exact | >= 320 (mlp); T2 level 0.017 [0.000, 0.039] | all 5 seeds >= 320 | 0.017 [0.000, 0.039] (learned) | >= 320 (ridge); T2 level 0.280 [0.224, 0.335] | all 5 seeds >= 320 | 0.280 [0.224, 0.335] (learned) |
| d2_fingervit | fi128_p512 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.017 [0.000, 0.039] | all 5 seeds >= 320 | 0.017 [0.000, 0.039] (learned) | >= 320 (ridge); T2 level 0.280 [0.224, 0.335] | all 5 seeds >= 320 | 0.280 [0.224, 0.335] (learned) |
| d2_fingervit | fi256_p512 | 256 | lrp_full | exact | 270.2 (proj) | [250.7-318.3] | 0.618 [0.517, 0.719] (analytic) | 80.2 (proj) | [71.6-86.7] | 1.000 [1.000, 1.000] (analytic) |
| d2_fingervit | fi256_p512 | 256 | lrp_full | fresh | >= 320 (ridge); T2 level 0.618 [0.517, 0.719] | all 5 seeds >= 320 | 0.618 [0.517, 0.719] (analytic) | >= 320 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) |
| d2_fingervit | fi256_p512 | 256 | lrp_half | exact | >= 320 (ridge); T2 level 0.022 [0.000, 0.048] | all 5 seeds >= 320 | 0.022 [0.000, 0.048] (analytic) | >= 320 (ridge); T2 level 0.655 [0.611, 0.698] | [263.0-(>= 320)] (4 of 5 seeds censored) | 0.655 [0.611, 0.698] (learned) |
| d2_fingervit | fi256_p512 | 256 | lrp_half | fresh | >= 320 (proj); T2 level 0.022 [0.000, 0.048] | all 5 seeds >= 320 | 0.022 [0.000, 0.048] (analytic) | >= 320 (ridge); T2 level 0.655 [0.611, 0.698] | all 5 seeds >= 320 | 0.655 [0.611, 0.698] (learned) |
| d2_fingervit | fi256_p512 | 256 | biohash_half | exact | >= 320 (ridge); T2 level 0.008 [0.000, 0.021] | all 5 seeds >= 320 | 0.008 [0.000, 0.021] (learned) | >= 320 (ridge); T2 level 0.338 [0.283, 0.394] | all 5 seeds >= 320 | 0.338 [0.283, 0.394] (learned) |
| d2_fingervit | fi256_p512 | 256 | biohash_half | fresh | >= 320 (proj); T2 level 0.008 [0.000, 0.021] | all 5 seeds >= 320 | 0.008 [0.000, 0.021] (learned) | >= 320 (ridge); T2 level 0.338 [0.283, 0.394] | all 5 seeds >= 320 | 0.338 [0.283, 0.394] (learned) |
| d2_fingervit | fi256_p512 | 256 | iom_256 | exact | >= 320 (proj); T2 level 0.003 [0.000, 0.008] | all 5 seeds >= 320 | 0.003 [0.000, 0.008] (learned) | >= 320 (ridge); T2 level 0.177 [0.125, 0.229] | all 5 seeds >= 320 | 0.177 [0.125, 0.229] (learned) |
| d2_fingervit | fi256_p512 | 256 | iom_256 | fresh | >= 320 (proj); T2 level 0.003 [0.000, 0.008] | all 5 seeds >= 320 | 0.003 [0.000, 0.008] (learned) | >= 320 (ridge); T2 level 0.177 [0.125, 0.229] | all 5 seeds >= 320 | 0.177 [0.125, 0.229] (learned) |
| d2_fingervit | rbig128_i2 | 128 | lrp_full | exact | 106.6 (ridge) | [98.0-112.0] | 0.926 [0.875, 0.978] (analytic) | 54.7 (proj) | [52.8-57.0] | 1.001 [0.997, 1.006] (analytic) |
| d2_fingervit | rbig128_i2 | 128 | lrp_full | fresh | >= 320 (proj); T2 level 0.926 [0.875, 0.978] | all 5 seeds >= 320 | 0.926 [0.875, 0.978] (analytic) | 244.3 (ridge) | [207.9-308.5] | 1.001 [0.997, 1.006] (analytic) |
| d2_fingervit | rbig128_i2 | 128 | lrp_half | exact | >= 320 (proj); T2 level 0.034 [0.000, 0.070] | all 5 seeds >= 320 | 0.034 [0.000, 0.070] (analytic) | >= 320 (ridge); T2 level 0.478 [0.413, 0.542] | all 5 seeds >= 320 | 0.478 [0.413, 0.542] (learned) |
| d2_fingervit | rbig128_i2 | 128 | lrp_half | fresh | >= 320 (ridge); T2 level 0.034 [0.000, 0.070] | all 5 seeds >= 320 | 0.034 [0.000, 0.070] (analytic) | >= 320 (ridge); T2 level 0.478 [0.413, 0.542] | all 5 seeds >= 320 | 0.478 [0.413, 0.542] (learned) |
| d2_fingervit | rbig128_i2 | 128 | biohash_half | exact | >= 320 (proj); T2 level 0.007 [0.000, 0.021] | all 5 seeds >= 320 | 0.007 [0.000, 0.021] (learned) | >= 320 (ridge); T2 level 0.136 [0.087, 0.184] | all 5 seeds >= 320 | 0.136 [0.087, 0.184] (learned) |
| d2_fingervit | rbig128_i2 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.007 [0.000, 0.021] | all 5 seeds >= 320 | 0.007 [0.000, 0.021] (learned) | >= 320 (ridge); T2 level 0.136 [0.087, 0.184] | all 5 seeds >= 320 | 0.136 [0.087, 0.184] (learned) |
| d2_fingervit | rbig128_i2 | 128 | iom_256 | exact | >= 320 (proj); T2 level 0.009 [0.000, 0.023] | all 5 seeds >= 320 | 0.009 [0.000, 0.023] (learned) | >= 320 (ridge); T2 level 0.255 [0.192, 0.319] | all 5 seeds >= 320 | 0.255 [0.192, 0.319] (learned) |
| d2_fingervit | rbig128_i2 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.009 [0.000, 0.023] | all 5 seeds >= 320 | 0.009 [0.000, 0.023] (learned) | >= 320 (ridge); T2 level 0.255 [0.192, 0.319] | all 5 seeds >= 320 | 0.255 [0.192, 0.319] (learned) |
| d2_fingervit | firbig128_p512_i2 | 128 | lrp_full | exact | 144.7 (proj) | [131.8-(>= 320)] (1 of 5 seeds censored) | 0.548 [0.447, 0.648] (analytic) | 61.8 (proj) | [56.9-65.3] | 0.994 [0.976, 1.012] (analytic) |
| d2_fingervit | firbig128_p512_i2 | 128 | lrp_full | fresh | >= 320 (proj); T2 level 0.548 [0.447, 0.648] | all 5 seeds >= 320 | 0.548 [0.447, 0.648] (analytic) | >= 320 (ridge); T2 level 0.994 [0.976, 1.012] | all 5 seeds >= 320 | 0.994 [0.976, 1.012] (analytic) |
| d2_fingervit | firbig128_p512_i2 | 128 | lrp_half | exact | >= 320 (proj); T2 level 0.029 [0.000, 0.063] | all 5 seeds >= 320 | 0.029 [0.000, 0.063] (analytic) | >= 320 (proj); T2 level 0.452 [0.398, 0.506] | all 5 seeds >= 320 | 0.452 [0.398, 0.506] (learned) |
| d2_fingervit | firbig128_p512_i2 | 128 | lrp_half | fresh | >= 320 (proj); T2 level 0.029 [0.000, 0.063] | all 5 seeds >= 320 | 0.029 [0.000, 0.063] (analytic) | >= 320 (proj); T2 level 0.452 [0.398, 0.506] | all 5 seeds >= 320 | 0.452 [0.398, 0.506] (learned) |
| d2_fingervit | firbig128_p512_i2 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.010 [0.000, 0.025] | all 5 seeds >= 320 | 0.010 [0.000, 0.025] (learned) | >= 320 (proj); T2 level 0.140 [0.098, 0.181] | all 5 seeds >= 320 | 0.140 [0.098, 0.181] (learned) |
| d2_fingervit | firbig128_p512_i2 | 128 | biohash_half | fresh | >= 320 (proj); T2 level 0.010 [0.000, 0.025] | all 5 seeds >= 320 | 0.010 [0.000, 0.025] (learned) | >= 320 (proj); T2 level 0.140 [0.098, 0.181] | all 5 seeds >= 320 | 0.140 [0.098, 0.181] (learned) |
| d2_fingervit | firbig128_p512_i2 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.033 [0.000, 0.065] | all 5 seeds >= 320 | 0.033 [0.000, 0.065] (learned) | >= 320 (ridge); T2 level 0.238 [0.187, 0.289] | all 5 seeds >= 320 | 0.238 [0.187, 0.289] (learned) |
| d2_fingervit | firbig128_p512_i2 | 128 | iom_256 | fresh | >= 320 (proj); T2 level 0.033 [0.000, 0.065] | all 5 seeds >= 320 | 0.033 [0.000, 0.065] (learned) | >= 320 (ridge); T2 level 0.238 [0.187, 0.289] | all 5 seeds >= 320 | 0.238 [0.187, 0.289] (learned) |
| d2_deepprint | none | 512 | lrp_full | exact | 27.9 (proj) | [27.2-29.7] | 1.000 [1.000, 1.000] (analytic) | 27.9 (proj) | [27.2-29.7] | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | none | 512 | lrp_full | fresh | >= 320 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) | >= 320 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | none | 512 | lrp_half | exact | 31.2 (proj) | [30.3-34.1] | 1.001 [0.992, 1.011] (learned) | 31.7 (proj) | [29.8-34.1] | 1.001 [0.994, 1.008] (analytic) |
| d2_deepprint | none | 512 | lrp_half | fresh | >= 320 (ridge); T2 level 1.001 [0.992, 1.011] | all 5 seeds >= 320 | 1.001 [0.992, 1.011] (learned) | >= 320 (ridge); T2 level 1.001 [0.994, 1.008] | all 5 seeds >= 320 | 1.001 [0.994, 1.008] (analytic) |
| d2_deepprint | none | 512 | biohash_half | exact | 76.6 (ridge) | [60.8-221.0] | 0.900 [0.836, 0.964] (learned) | 223.7 (ridge) | [212.8-242.0] | 0.829 [0.741, 0.917] (learned) |
| d2_deepprint | none | 512 | biohash_half | fresh | >= 320 (ridge); T2 level 0.900 [0.836, 0.964] | all 5 seeds >= 320 | 0.900 [0.836, 0.964] (learned) | >= 320 (ridge); T2 level 0.829 [0.741, 0.917] | all 5 seeds >= 320 | 0.829 [0.741, 0.917] (learned) |
| d2_deepprint | none | 512 | iom_256 | exact | 313.4 (ridge) | [285.7-(>= 320)] (2 of 5 seeds censored) | 0.699 [0.614, 0.783] (learned) | >= 320 (ridge); T2 level 0.545 [0.405, 0.686] | all 5 seeds >= 320 | 0.545 [0.405, 0.686] (learned) |
| d2_deepprint | none | 512 | iom_256 | fresh | >= 320 (ridge); T2 level 0.699 [0.614, 0.783] | all 5 seeds >= 320 | 0.699 [0.614, 0.783] (learned) | >= 320 (ridge); T2 level 0.545 [0.405, 0.686] | all 5 seeds >= 320 | 0.545 [0.405, 0.686] (learned) |
| d2_deepprint | pca128_a1 | 128 | lrp_full | exact | 86.3 (ridge) | [81.2-89.9] | 1.002 [0.992, 1.011] (analytic) | 48.2 (proj) | [46.3-50.3] | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | pca128_a1 | 128 | lrp_full | fresh | >= 320 (ridge); T2 level 1.002 [0.992, 1.011] | all 5 seeds >= 320 | 1.002 [0.992, 1.011] (analytic) | >= 320 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | pca128_a1 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.195 [0.115, 0.275] | all 5 seeds >= 320 | 0.195 [0.115, 0.275] (analytic) | >= 320 (ridge); T2 level 0.525 [0.445, 0.605] | all 5 seeds >= 320 | 0.525 [0.445, 0.605] (analytic) |
| d2_deepprint | pca128_a1 | 128 | lrp_half | fresh | >= 320 (ridge); T2 level 0.195 [0.115, 0.275] | all 5 seeds >= 320 | 0.195 [0.115, 0.275] (analytic) | >= 320 (ridge); T2 level 0.525 [0.445, 0.605] | all 5 seeds >= 320 | 0.525 [0.445, 0.605] (analytic) |
| d2_deepprint | pca128_a1 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.066 [0.021, 0.112] | all 5 seeds >= 320 | 0.066 [0.021, 0.112] (learned) | >= 320 (ridge); T2 level 0.146 [0.074, 0.218] | all 5 seeds >= 320 | 0.146 [0.074, 0.218] (learned) |
| d2_deepprint | pca128_a1 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.066 [0.021, 0.112] | all 5 seeds >= 320 | 0.066 [0.021, 0.112] (learned) | >= 320 (ridge); T2 level 0.146 [0.074, 0.218] | all 5 seeds >= 320 | 0.146 [0.074, 0.218] (learned) |
| d2_deepprint | pca128_a1 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.096 [0.045, 0.148] | all 5 seeds >= 320 | 0.096 [0.045, 0.148] (learned) | >= 320 (ridge); T2 level 0.235 [0.146, 0.324] | all 5 seeds >= 320 | 0.235 [0.146, 0.324] (learned) |
| d2_deepprint | pca128_a1 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.096 [0.045, 0.148] | all 5 seeds >= 320 | 0.096 [0.045, 0.148] (learned) | >= 320 (ridge); T2 level 0.235 [0.146, 0.324] | all 5 seeds >= 320 | 0.235 [0.146, 0.324] (learned) |
| d2_deepprint | pca256_a1 | 256 | lrp_full | exact | 169.7 (ridge) | [164.3-177.2] | 1.000 [1.000, 1.000] (analytic) | 68.9 (proj) | [66.7-71.5] | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | pca256_a1 | 256 | lrp_full | fresh | >= 320 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) | >= 320 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | pca256_a1 | 256 | lrp_half | exact | >= 320 (ridge); T2 level 0.229 [0.157, 0.300] | all 5 seeds >= 320 | 0.229 [0.157, 0.300] (analytic) | 271.2 (ridge) | [238.5-(>= 320)] (1 of 5 seeds censored) | 0.669 [0.600, 0.738] (analytic) |
| d2_deepprint | pca256_a1 | 256 | lrp_half | fresh | >= 320 (ridge); T2 level 0.229 [0.157, 0.300] | all 5 seeds >= 320 | 0.229 [0.157, 0.300] (analytic) | >= 320 (ridge); T2 level 0.669 [0.600, 0.738] | all 5 seeds >= 320 | 0.669 [0.600, 0.738] (analytic) |
| d2_deepprint | pca256_a1 | 256 | biohash_half | exact | >= 320 (ridge); T2 level 0.064 [0.022, 0.105] | all 5 seeds >= 320 | 0.064 [0.022, 0.105] (learned) | >= 320 (ridge); T2 level 0.286 [0.205, 0.367] | all 5 seeds >= 320 | 0.286 [0.205, 0.367] (learned) |
| d2_deepprint | pca256_a1 | 256 | biohash_half | fresh | >= 320 (ridge); T2 level 0.064 [0.022, 0.105] | all 5 seeds >= 320 | 0.064 [0.022, 0.105] (learned) | >= 320 (ridge); T2 level 0.286 [0.205, 0.367] | all 5 seeds >= 320 | 0.286 [0.205, 0.367] (learned) |
| d2_deepprint | pca256_a1 | 256 | iom_256 | exact | >= 320 (ridge); T2 level 0.020 [0.000, 0.041] | all 5 seeds >= 320 | 0.020 [0.000, 0.041] (learned) | >= 320 (ridge); T2 level 0.140 [0.061, 0.220] | all 5 seeds >= 320 | 0.140 [0.061, 0.220] (learned) |
| d2_deepprint | pca256_a1 | 256 | iom_256 | fresh | >= 320 (ridge); T2 level 0.020 [0.000, 0.041] | all 5 seeds >= 320 | 0.020 [0.000, 0.041] (learned) | >= 320 (ridge); T2 level 0.140 [0.061, 0.220] | all 5 seeds >= 320 | 0.140 [0.061, 0.220] (learned) |
| d2_deepprint | pca128_a0.5 | 128 | lrp_full | exact | 32.9 (proj) | [30.9-34.9] | 1.002 [0.992, 1.011] (analytic) | 40.3 (proj) | [39.2-40.9] | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | pca128_a0.5 | 128 | lrp_full | fresh | >= 320 (ridge); T2 level 1.002 [0.992, 1.011] | all 5 seeds >= 320 | 1.002 [0.992, 1.011] (analytic) | >= 320 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | pca128_a0.5 | 128 | lrp_half | exact | 128.6 (ridge) | [109.6-147.9] | 0.649 [0.566, 0.733] (learned) | 147.4 (ridge) | [117.9-168.9] | 0.666 [0.591, 0.741] (learned) |
| d2_deepprint | pca128_a0.5 | 128 | lrp_half | fresh | >= 320 (ridge); T2 level 0.649 [0.566, 0.733] | all 5 seeds >= 320 | 0.649 [0.566, 0.733] (learned) | >= 320 (ridge); T2 level 0.666 [0.591, 0.741] | all 5 seeds >= 320 | 0.666 [0.591, 0.741] (learned) |
| d2_deepprint | pca128_a0.5 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.315 [0.231, 0.398] | all 5 seeds >= 320 | 0.315 [0.231, 0.398] (learned) | >= 320 (ridge); T2 level 0.234 [0.151, 0.317] | all 5 seeds >= 320 | 0.234 [0.151, 0.317] (learned) |
| d2_deepprint | pca128_a0.5 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.315 [0.231, 0.398] | all 5 seeds >= 320 | 0.315 [0.231, 0.398] (learned) | >= 320 (ridge); T2 level 0.234 [0.151, 0.317] | all 5 seeds >= 320 | 0.234 [0.151, 0.317] (learned) |
| d2_deepprint | pca128_a0.5 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.463 [0.376, 0.550] | all 5 seeds >= 320 | 0.463 [0.376, 0.550] (learned) | >= 320 (ridge); T2 level 0.374 [0.264, 0.484] | all 5 seeds >= 320 | 0.374 [0.264, 0.484] (learned) |
| d2_deepprint | pca128_a0.5 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.463 [0.376, 0.550] | all 5 seeds >= 320 | 0.463 [0.376, 0.550] (learned) | >= 320 (ridge); T2 level 0.374 [0.264, 0.484] | all 5 seeds >= 320 | 0.374 [0.264, 0.484] (learned) |
| d2_deepprint | fi64_p384 | 64 | lrp_full | exact | >= 320 (proj); back-map-limited: T2 level 0.372 [0.284, 0.461] < 0.5 | all 5 seeds >= 320 | 0.372 [0.284, 0.461] (analytic) | 30.4 (proj) | [28.8-32.8] | 0.999 [0.996, 1.002] (analytic) |
| d2_deepprint | fi64_p384 | 64 | lrp_full | fresh | >= 320 (proj); T2 level 0.372 [0.284, 0.461] | all 5 seeds >= 320 | 0.372 [0.284, 0.461] (analytic) | >= 320 (ridge); T2 level 0.999 [0.996, 1.002] | all 5 seeds >= 320 | 0.999 [0.996, 1.002] (analytic) |
| d2_deepprint | fi64_p384 | 64 | lrp_half | exact | >= 320 (ridge); T2 level 0.053 [0.011, 0.094] | all 5 seeds >= 320 | 0.053 [0.011, 0.094] (learned) | >= 320 (ridge); T2 level 0.369 [0.277, 0.462] | all 5 seeds >= 320 | 0.369 [0.277, 0.462] (learned) |
| d2_deepprint | fi64_p384 | 64 | lrp_half | fresh | >= 320 (proj); T2 level 0.053 [0.011, 0.094] | all 5 seeds >= 320 | 0.053 [0.011, 0.094] (learned) | >= 320 (proj); T2 level 0.369 [0.277, 0.462] | all 5 seeds >= 320 | 0.369 [0.277, 0.462] (learned) |
| d2_deepprint | fi64_p384 | 64 | biohash_half | exact | >= 320 (ridge); T2 level 0.013 [0.000, 0.029] | all 5 seeds >= 320 | 0.013 [0.000, 0.029] (learned) | >= 320 (ridge); T2 level 0.125 [0.010, 0.240] | all 5 seeds >= 320 | 0.125 [0.010, 0.240] (learned) |
| d2_deepprint | fi64_p384 | 64 | biohash_half | fresh | >= 320 (ridge); T2 level 0.013 [0.000, 0.029] | all 5 seeds >= 320 | 0.013 [0.000, 0.029] (learned) | >= 320 (proj); T2 level 0.125 [0.010, 0.240] | all 5 seeds >= 320 | 0.125 [0.010, 0.240] (learned) |
| d2_deepprint | fi64_p384 | 64 | iom_256 | exact | >= 320 (ridge); T2 level 0.119 [0.060, 0.178] | all 5 seeds >= 320 | 0.119 [0.060, 0.178] (learned) | >= 320 (ridge); T2 level 0.421 [0.293, 0.549] | all 5 seeds >= 320 | 0.421 [0.293, 0.549] (learned) |
| d2_deepprint | fi64_p384 | 64 | iom_256 | fresh | >= 320 (ridge); T2 level 0.119 [0.060, 0.178] | all 5 seeds >= 320 | 0.119 [0.060, 0.178] (learned) | >= 320 (ridge); T2 level 0.421 [0.293, 0.549] | all 5 seeds >= 320 | 0.421 [0.293, 0.549] (learned) |
| d2_deepprint | fi128_p384 | 128 | lrp_full | exact | >= 320 (proj); T2 level 0.488 [0.363, 0.613] | [141.7-(>= 320)] (2 of 5 seeds censored) | 0.488 [0.363, 0.613] (analytic) | 44.3 (proj) | [42.3-46.5] | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | fi128_p384 | 128 | lrp_full | fresh | >= 320 (ridge); T2 level 0.488 [0.363, 0.613] | all 5 seeds >= 320 | 0.488 [0.363, 0.613] (analytic) | >= 320 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | fi128_p384 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.059 [0.019, 0.098] | all 5 seeds >= 320 | 0.059 [0.019, 0.098] (analytic) | >= 320 (ridge); T2 level 0.535 [0.452, 0.617] | [274.1-(>= 320)] (4 of 5 seeds censored) | 0.535 [0.452, 0.617] (analytic) |
| d2_deepprint | fi128_p384 | 128 | lrp_half | fresh | >= 320 (proj); T2 level 0.059 [0.019, 0.098] | all 5 seeds >= 320 | 0.059 [0.019, 0.098] (analytic) | >= 320 (proj); T2 level 0.535 [0.452, 0.617] | all 5 seeds >= 320 | 0.535 [0.452, 0.617] (analytic) |
| d2_deepprint | fi128_p384 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.015 [0.000, 0.032] | all 5 seeds >= 320 | 0.015 [0.000, 0.032] (learned) | >= 320 (proj); T2 level 0.190 [0.098, 0.281] | all 5 seeds >= 320 | 0.190 [0.098, 0.281] (learned) |
| d2_deepprint | fi128_p384 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.015 [0.000, 0.032] | all 5 seeds >= 320 | 0.015 [0.000, 0.032] (learned) | >= 320 (proj); T2 level 0.190 [0.098, 0.281] | all 5 seeds >= 320 | 0.190 [0.098, 0.281] (learned) |
| d2_deepprint | fi128_p384 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.049 [0.012, 0.086] | all 5 seeds >= 320 | 0.049 [0.012, 0.086] (learned) | >= 320 (ridge); T2 level 0.254 [0.143, 0.364] | all 5 seeds >= 320 | 0.254 [0.143, 0.364] (learned) |
| d2_deepprint | fi128_p384 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.049 [0.012, 0.086] | all 5 seeds >= 320 | 0.049 [0.012, 0.086] (learned) | >= 320 (ridge); T2 level 0.254 [0.143, 0.364] | all 5 seeds >= 320 | 0.254 [0.143, 0.364] (learned) |
| d2_deepprint | fi256_p384 | 256 | lrp_full | exact | 229.1 (ridge) | [198.9-272.5] | 0.720 [0.638, 0.802] (analytic) | 61.7 (proj) | [60.1-65.3] | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | fi256_p384 | 256 | lrp_full | fresh | >= 320 (ridge); T2 level 0.720 [0.638, 0.802] | all 5 seeds >= 320 | 0.720 [0.638, 0.802] (analytic) | >= 320 (ridge); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) |
| d2_deepprint | fi256_p384 | 256 | lrp_half | exact | >= 320 (ridge); T2 level 0.111 [0.055, 0.167] | all 5 seeds >= 320 | 0.111 [0.055, 0.167] (learned) | 264.4 (ridge) | [231.3-295.6] | 0.670 [0.601, 0.739] (learned) |
| d2_deepprint | fi256_p384 | 256 | lrp_half | fresh | >= 320 (ridge); T2 level 0.111 [0.055, 0.167] | all 5 seeds >= 320 | 0.111 [0.055, 0.167] (learned) | >= 320 (ridge); T2 level 0.670 [0.601, 0.739] | all 5 seeds >= 320 | 0.670 [0.601, 0.739] (learned) |
| d2_deepprint | fi256_p384 | 256 | biohash_half | exact | >= 320 (ridge); T2 level 0.039 [0.005, 0.072] | all 5 seeds >= 320 | 0.039 [0.005, 0.072] (learned) | >= 320 (ridge); T2 level 0.295 [0.206, 0.385] | all 5 seeds >= 320 | 0.295 [0.206, 0.385] (learned) |
| d2_deepprint | fi256_p384 | 256 | biohash_half | fresh | >= 320 (ridge); T2 level 0.039 [0.005, 0.072] | all 5 seeds >= 320 | 0.039 [0.005, 0.072] (learned) | >= 320 (ridge); T2 level 0.295 [0.206, 0.385] | all 5 seeds >= 320 | 0.295 [0.206, 0.385] (learned) |
| d2_deepprint | fi256_p384 | 256 | iom_256 | exact | >= 320 (ridge); T2 level 0.007 [0.000, 0.019] | all 5 seeds >= 320 | 0.007 [0.000, 0.019] (learned) | >= 320 (ridge); T2 level 0.135 [0.053, 0.217] | all 5 seeds >= 320 | 0.135 [0.053, 0.217] (learned) |
| d2_deepprint | fi256_p384 | 256 | iom_256 | fresh | >= 320 (ridge); T2 level 0.007 [0.000, 0.019] | all 5 seeds >= 320 | 0.007 [0.000, 0.019] (learned) | >= 320 (proj); T2 level 0.135 [0.053, 0.217] | all 5 seeds >= 320 | 0.135 [0.053, 0.217] (learned) |
| d2_deepprint | rbig128_i2 | 128 | lrp_full | exact | 85.4 (ridge) | [80.3-89.4] | 1.002 [0.992, 1.011] (analytic) | 47.8 (proj) | [46.1-49.7] | 0.999 [0.997, 1.002] (analytic) |
| d2_deepprint | rbig128_i2 | 128 | lrp_full | fresh | >= 320 (ridge); T2 level 1.002 [0.992, 1.011] | all 5 seeds >= 320 | 1.002 [0.992, 1.011] (analytic) | >= 320 (ridge); T2 level 0.999 [0.997, 1.002] | all 5 seeds >= 320 | 0.999 [0.997, 1.002] (analytic) |
| d2_deepprint | rbig128_i2 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.189 [0.119, 0.258] | all 5 seeds >= 320 | 0.189 [0.119, 0.258] (analytic) | >= 320 (ridge); T2 level 0.534 [0.452, 0.616] | [305.2-(>= 320)] (3 of 5 seeds censored) | 0.534 [0.452, 0.616] (learned) |
| d2_deepprint | rbig128_i2 | 128 | lrp_half | fresh | >= 320 (ridge); T2 level 0.189 [0.119, 0.258] | all 5 seeds >= 320 | 0.189 [0.119, 0.258] (analytic) | >= 320 (ridge); T2 level 0.534 [0.452, 0.616] | all 5 seeds >= 320 | 0.534 [0.452, 0.616] (learned) |
| d2_deepprint | rbig128_i2 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.065 [0.024, 0.106] | all 5 seeds >= 320 | 0.065 [0.024, 0.106] (learned) | >= 320 (ridge); T2 level 0.179 [0.095, 0.263] | all 5 seeds >= 320 | 0.179 [0.095, 0.263] (learned) |
| d2_deepprint | rbig128_i2 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.065 [0.024, 0.106] | all 5 seeds >= 320 | 0.065 [0.024, 0.106] (learned) | >= 320 (ridge); T2 level 0.179 [0.095, 0.263] | all 5 seeds >= 320 | 0.179 [0.095, 0.263] (learned) |
| d2_deepprint | rbig128_i2 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.137 [0.079, 0.196] | all 5 seeds >= 320 | 0.137 [0.079, 0.196] (learned) | >= 320 (ridge); T2 level 0.204 [0.118, 0.289] | all 5 seeds >= 320 | 0.204 [0.118, 0.289] (learned) |
| d2_deepprint | rbig128_i2 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.137 [0.079, 0.196] | all 5 seeds >= 320 | 0.137 [0.079, 0.196] (learned) | >= 320 (ridge); T2 level 0.204 [0.118, 0.289] | all 5 seeds >= 320 | 0.204 [0.118, 0.289] (learned) |
| d2_deepprint | firbig128_p384_i2 | 128 | lrp_full | exact | >= 320 (proj); T2 level 0.487 [0.360, 0.613] | [142.4-(>= 320)] (2 of 5 seeds censored) | 0.487 [0.360, 0.613] (analytic) | 45.4 (proj) | [43.0-48.0] | 1.002 [0.987, 1.017] (analytic) |
| d2_deepprint | firbig128_p384_i2 | 128 | lrp_full | fresh | >= 320 (proj); T2 level 0.487 [0.360, 0.613] | all 5 seeds >= 320 | 0.487 [0.360, 0.613] (analytic) | >= 320 (ridge); T2 level 1.002 [0.987, 1.017] | all 5 seeds >= 320 | 1.002 [0.987, 1.017] (analytic) |
| d2_deepprint | firbig128_p384_i2 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.038 [0.007, 0.069] | all 5 seeds >= 320 | 0.038 [0.007, 0.069] (analytic) | >= 320 (proj); T2 level 0.556 [0.475, 0.638] | all 5 seeds >= 320 | 0.556 [0.475, 0.638] (learned) |
| d2_deepprint | firbig128_p384_i2 | 128 | lrp_half | fresh | >= 320 (proj); T2 level 0.038 [0.007, 0.069] | all 5 seeds >= 320 | 0.038 [0.007, 0.069] (analytic) | >= 320 (proj); T2 level 0.556 [0.475, 0.638] | all 5 seeds >= 320 | 0.556 [0.475, 0.638] (learned) |
| d2_deepprint | firbig128_p384_i2 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.018 [0.000, 0.044] | all 5 seeds >= 320 | 0.018 [0.000, 0.044] (learned) | >= 320 (proj); T2 level 0.144 [0.066, 0.222] | all 5 seeds >= 320 | 0.144 [0.066, 0.222] (learned) |
| d2_deepprint | firbig128_p384_i2 | 128 | biohash_half | fresh | >= 320 (proj); T2 level 0.018 [0.000, 0.044] | all 5 seeds >= 320 | 0.018 [0.000, 0.044] (learned) | >= 320 (ridge); T2 level 0.144 [0.066, 0.222] | all 5 seeds >= 320 | 0.144 [0.066, 0.222] (learned) |
| d2_deepprint | firbig128_p384_i2 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.031 [0.003, 0.059] | all 5 seeds >= 320 | 0.031 [0.003, 0.059] (learned) | >= 320 (ridge); T2 level 0.186 [0.087, 0.284] | all 5 seeds >= 320 | 0.186 [0.087, 0.284] (learned) |
| d2_deepprint | firbig128_p384_i2 | 128 | iom_256 | fresh | >= 320 (proj); T2 level 0.031 [0.003, 0.059] | all 5 seeds >= 320 | 0.031 [0.003, 0.059] (learned) | >= 320 (ridge); T2 level 0.186 [0.087, 0.284] | all 5 seeds >= 320 | 0.186 [0.087, 0.284] (learned) |
| d2_ctrlvit | none | 768 | lrp_full | exact | 87.2 (proj) | [75.7-98.9] | 1.000 [1.000, 1.000] (analytic) | 87.2 (proj) | [75.7-98.9] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | none | 768 | lrp_full | fresh | >= 320 (proj); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) | >= 320 (proj); T2 level 1.000 [1.000, 1.000] | all 5 seeds >= 320 | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | none | 768 | lrp_half | exact | >= 320 (ridge); T2 level 0.961 [0.946, 0.976] | [146.9-(>= 320)] (2 of 5 seeds censored) | 0.961 [0.946, 0.976] (learned) | >= 320 (ridge); T2 level 0.951 [0.931, 0.971] | [145.5-(>= 320)] (4 of 5 seeds censored) | 0.951 [0.931, 0.971] (learned) |
| d2_ctrlvit | none | 768 | lrp_half | fresh | >= 320 (ridge); T2 level 0.961 [0.946, 0.976] | all 5 seeds >= 320 | 0.961 [0.946, 0.976] (learned) | >= 320 (ridge); T2 level 0.951 [0.931, 0.971] | all 5 seeds >= 320 | 0.951 [0.931, 0.971] (learned) |
| d2_ctrlvit | none | 768 | biohash_half | exact | >= 320 (proj); T2 level 0.834 [0.806, 0.863] | all 5 seeds >= 320 | 0.834 [0.806, 0.863] (learned) | >= 320 (ridge); T2 level 0.767 [0.734, 0.801] | all 5 seeds >= 320 | 0.767 [0.734, 0.801] (learned) |
| d2_ctrlvit | none | 768 | biohash_half | fresh | >= 320 (proj); T2 level 0.834 [0.806, 0.863] | all 5 seeds >= 320 | 0.834 [0.806, 0.863] (learned) | >= 320 (proj); T2 level 0.767 [0.734, 0.801] | all 5 seeds >= 320 | 0.767 [0.734, 0.801] (learned) |
| d2_ctrlvit | none | 768 | iom_256 | exact | >= 320 (ridge); T2 level 0.294 [0.246, 0.343] | all 5 seeds >= 320 | 0.294 [0.246, 0.343] (learned) | >= 320 (ridge); T2 level 0.163 [0.126, 0.201] | all 5 seeds >= 320 | 0.163 [0.126, 0.201] (learned) |
| d2_ctrlvit | none | 768 | iom_256 | fresh | >= 320 (ridge); T2 level 0.294 [0.246, 0.343] | all 5 seeds >= 320 | 0.294 [0.246, 0.343] (learned) | >= 320 (ridge); T2 level 0.163 [0.126, 0.201] | all 5 seeds >= 320 | 0.163 [0.126, 0.201] (learned) |
| d2_ctrlvit | pca128_a1 | 128 | lrp_full | exact | 65.6 (proj) | [61.6-71.7] | 0.972 [0.956, 0.988] (analytic) | 44.8 (proj) | [41.8-48.0] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | pca128_a1 | 128 | lrp_full | fresh | 223.8 (ridge) | [204.6-258.7] | 0.972 [0.956, 0.988] (analytic) | 150.8 (ridge) | [141.5-158.1] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | pca128_a1 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.430 [0.385, 0.475] | all 5 seeds >= 320 | 0.430 [0.385, 0.475] (learned) | 290.4 (ridge) | [244.3-(>= 320)] (1 of 5 seeds censored) | 0.636 [0.602, 0.670] (learned) |
| d2_ctrlvit | pca128_a1 | 128 | lrp_half | fresh | >= 320 (ridge); T2 level 0.430 [0.385, 0.475] | all 5 seeds >= 320 | 0.430 [0.385, 0.475] (learned) | >= 320 (ridge); T2 level 0.636 [0.602, 0.670] | all 5 seeds >= 320 | 0.636 [0.602, 0.670] (learned) |
| d2_ctrlvit | pca128_a1 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.131 [0.091, 0.170] | all 5 seeds >= 320 | 0.131 [0.091, 0.170] (learned) | >= 320 (proj); T2 level 0.151 [0.119, 0.182] | all 5 seeds >= 320 | 0.151 [0.119, 0.182] (learned) |
| d2_ctrlvit | pca128_a1 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.131 [0.091, 0.170] | all 5 seeds >= 320 | 0.131 [0.091, 0.170] (learned) | >= 320 (proj); T2 level 0.151 [0.119, 0.182] | all 5 seeds >= 320 | 0.151 [0.119, 0.182] (learned) |
| d2_ctrlvit | pca128_a1 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.286 [0.245, 0.326] | all 5 seeds >= 320 | 0.286 [0.245, 0.326] (learned) | >= 320 (ridge); T2 level 0.238 [0.200, 0.277] | all 5 seeds >= 320 | 0.238 [0.200, 0.277] (learned) |
| d2_ctrlvit | pca128_a1 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.286 [0.245, 0.326] | all 5 seeds >= 320 | 0.286 [0.245, 0.326] (learned) | >= 320 (ridge); T2 level 0.238 [0.200, 0.277] | all 5 seeds >= 320 | 0.238 [0.200, 0.277] (learned) |
| d2_ctrlvit | pca256_a1 | 256 | lrp_full | exact | 118.8 (proj) | [113.4-125.6] | 0.999 [0.988, 1.011] (analytic) | 51.0 (proj) | [46.5-53.5] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | pca256_a1 | 256 | lrp_full | fresh | >= 320 (ridge); T2 level 0.999 [0.988, 1.011] | all 5 seeds >= 320 | 0.999 [0.988, 1.011] (analytic) | 236.4 (ridge) | [225.1-244.4] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | pca256_a1 | 256 | lrp_half | exact | >= 320 (ridge); T2 level 0.527 [0.479, 0.574] | all 5 seeds >= 320 | 0.527 [0.479, 0.574] (learned) | 166.6 (ridge) | [159.6-179.2] | 0.822 [0.797, 0.848] (learned) |
| d2_ctrlvit | pca256_a1 | 256 | lrp_half | fresh | >= 320 (ridge); T2 level 0.527 [0.479, 0.574] | all 5 seeds >= 320 | 0.527 [0.479, 0.574] (learned) | >= 320 (ridge); T2 level 0.822 [0.797, 0.848] | all 5 seeds >= 320 | 0.822 [0.797, 0.848] (learned) |
| d2_ctrlvit | pca256_a1 | 256 | biohash_half | exact | >= 320 (ridge); T2 level 0.185 [0.154, 0.216] | all 5 seeds >= 320 | 0.185 [0.154, 0.216] (learned) | >= 320 (ridge); T2 level 0.333 [0.292, 0.375] | all 5 seeds >= 320 | 0.333 [0.292, 0.375] (learned) |
| d2_ctrlvit | pca256_a1 | 256 | biohash_half | fresh | >= 320 (ridge); T2 level 0.185 [0.154, 0.216] | all 5 seeds >= 320 | 0.185 [0.154, 0.216] (learned) | >= 320 (ridge); T2 level 0.333 [0.292, 0.375] | all 5 seeds >= 320 | 0.333 [0.292, 0.375] (learned) |
| d2_ctrlvit | pca256_a1 | 256 | iom_256 | exact | >= 320 (ridge); T2 level 0.096 [0.068, 0.124] | all 5 seeds >= 320 | 0.096 [0.068, 0.124] (learned) | >= 320 (ridge); T2 level 0.128 [0.097, 0.160] | all 5 seeds >= 320 | 0.128 [0.097, 0.160] (learned) |
| d2_ctrlvit | pca256_a1 | 256 | iom_256 | fresh | >= 320 (ridge); T2 level 0.096 [0.068, 0.124] | all 5 seeds >= 320 | 0.096 [0.068, 0.124] (learned) | >= 320 (ridge); T2 level 0.128 [0.097, 0.160] | all 5 seeds >= 320 | 0.128 [0.097, 0.160] (learned) |
| d2_ctrlvit | pca128_a0.5 | 128 | lrp_full | exact | 54.3 (proj) | [50.9-58.9] | 0.972 [0.956, 0.988] (analytic) | 47.7 (proj) | [44.0-49.6] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | pca128_a0.5 | 128 | lrp_full | fresh | 165.2 (ridge) | [151.4-194.5] | 0.972 [0.956, 0.988] (analytic) | 146.9 (ridge) | [137.6-150.9] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | pca128_a0.5 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.556 [0.510, 0.602] | [312.8-(>= 320)] (4 of 5 seeds censored) | 0.556 [0.510, 0.602] (learned) | 307.4 (ridge) | [277.2-(>= 320)] (1 of 5 seeds censored) | 0.639 [0.603, 0.674] (learned) |
| d2_ctrlvit | pca128_a0.5 | 128 | lrp_half | fresh | >= 320 (ridge); T2 level 0.556 [0.510, 0.602] | all 5 seeds >= 320 | 0.556 [0.510, 0.602] (learned) | >= 320 (ridge); T2 level 0.639 [0.603, 0.674] | all 5 seeds >= 320 | 0.639 [0.603, 0.674] (learned) |
| d2_ctrlvit | pca128_a0.5 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.243 [0.204, 0.282] | all 5 seeds >= 320 | 0.243 [0.204, 0.282] (learned) | >= 320 (ridge); T2 level 0.190 [0.156, 0.223] | all 5 seeds >= 320 | 0.190 [0.156, 0.223] (learned) |
| d2_ctrlvit | pca128_a0.5 | 128 | biohash_half | fresh | >= 320 (ridge); T2 level 0.243 [0.204, 0.282] | all 5 seeds >= 320 | 0.243 [0.204, 0.282] (learned) | >= 320 (ridge); T2 level 0.190 [0.156, 0.223] | all 5 seeds >= 320 | 0.190 [0.156, 0.223] (learned) |
| d2_ctrlvit | pca128_a0.5 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.407 [0.366, 0.447] | all 5 seeds >= 320 | 0.407 [0.366, 0.447] (learned) | >= 320 (ridge); T2 level 0.313 [0.272, 0.353] | all 5 seeds >= 320 | 0.313 [0.272, 0.353] (learned) |
| d2_ctrlvit | pca128_a0.5 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.407 [0.366, 0.447] | all 5 seeds >= 320 | 0.407 [0.366, 0.447] (learned) | >= 320 (ridge); T2 level 0.313 [0.272, 0.353] | all 5 seeds >= 320 | 0.313 [0.272, 0.353] (learned) |
| d2_ctrlvit | fi64_p512 | 64 | lrp_full | exact | 75.5 (proj) | [72.9-(>= 320)] (1 of 5 seeds censored) | 0.522 [0.484, 0.561] (analytic) | 37.2 (proj) | [35.2-40.9] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | fi64_p512 | 64 | lrp_full | fresh | >= 320 (ridge); T2 level 0.522 [0.484, 0.561] | all 5 seeds >= 320 | 0.522 [0.484, 0.561] (analytic) | 148.3 (ridge) | [140.3-156.6] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | fi64_p512 | 64 | lrp_half | exact | >= 320 (proj); T2 level 0.134 [0.107, 0.161] | all 5 seeds >= 320 | 0.134 [0.107, 0.161] (learned) | >= 320 (proj); T2 level 0.292 [0.250, 0.334] | all 5 seeds >= 320 | 0.292 [0.250, 0.334] (learned) |
| d2_ctrlvit | fi64_p512 | 64 | lrp_half | fresh | >= 320 (proj); T2 level 0.134 [0.107, 0.161] | all 5 seeds >= 320 | 0.134 [0.107, 0.161] (learned) | >= 320 (ridge); T2 level 0.292 [0.250, 0.334] | all 5 seeds >= 320 | 0.292 [0.250, 0.334] (learned) |
| d2_ctrlvit | fi64_p512 | 64 | biohash_half | exact | >= 320 (proj); T2 level 0.036 [0.020, 0.051] | all 5 seeds >= 320 | 0.036 [0.020, 0.051] (learned) | >= 320 (ridge); T2 level 0.092 [0.057, 0.128] | all 5 seeds >= 320 | 0.092 [0.057, 0.128] (learned) |
| d2_ctrlvit | fi64_p512 | 64 | biohash_half | fresh | >= 320 (proj); T2 level 0.036 [0.020, 0.051] | all 5 seeds >= 320 | 0.036 [0.020, 0.051] (learned) | >= 320 (ridge); T2 level 0.092 [0.057, 0.128] | all 5 seeds >= 320 | 0.092 [0.057, 0.128] (learned) |
| d2_ctrlvit | fi64_p512 | 64 | iom_256 | exact | >= 320 (ridge); T2 level 0.199 [0.166, 0.232] | all 5 seeds >= 320 | 0.199 [0.166, 0.232] (learned) | >= 320 (ridge); T2 level 0.405 [0.353, 0.457] | all 5 seeds >= 320 | 0.405 [0.353, 0.457] (learned) |
| d2_ctrlvit | fi64_p512 | 64 | iom_256 | fresh | >= 320 (ridge); T2 level 0.199 [0.166, 0.232] | all 5 seeds >= 320 | 0.199 [0.166, 0.232] (learned) | >= 320 (ridge); T2 level 0.405 [0.353, 0.457] | all 5 seeds >= 320 | 0.405 [0.353, 0.457] (learned) |
| d2_ctrlvit | fi128_p512 | 128 | lrp_full | exact | 100.1 (proj) | [92.0-109.4] | 0.750 [0.717, 0.783] (analytic) | 51.4 (proj) | [48.1-55.1] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | fi128_p512 | 128 | lrp_full | fresh | >= 320 (ridge); T2 level 0.750 [0.717, 0.783] | all 5 seeds >= 320 | 0.750 [0.717, 0.783] (analytic) | 199.8 (ridge) | [178.3-217.5] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | fi128_p512 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.221 [0.175, 0.268] | all 5 seeds >= 320 | 0.221 [0.175, 0.268] (learned) | >= 320 (ridge); T2 level 0.571 [0.534, 0.607] | all 5 seeds >= 320 | 0.571 [0.534, 0.607] (learned) |
| d2_ctrlvit | fi128_p512 | 128 | lrp_half | fresh | >= 320 (ridge); T2 level 0.221 [0.175, 0.268] | all 5 seeds >= 320 | 0.221 [0.175, 0.268] (learned) | >= 320 (ridge); T2 level 0.571 [0.534, 0.607] | all 5 seeds >= 320 | 0.571 [0.534, 0.607] (learned) |
| d2_ctrlvit | fi128_p512 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.062 [0.043, 0.080] | all 5 seeds >= 320 | 0.062 [0.043, 0.080] (learned) | >= 320 (ridge); T2 level 0.138 [0.105, 0.171] | all 5 seeds >= 320 | 0.138 [0.105, 0.171] (learned) |
| d2_ctrlvit | fi128_p512 | 128 | biohash_half | fresh | >= 320 (proj); T2 level 0.062 [0.043, 0.080] | all 5 seeds >= 320 | 0.062 [0.043, 0.080] (learned) | >= 320 (ridge); T2 level 0.138 [0.105, 0.171] | all 5 seeds >= 320 | 0.138 [0.105, 0.171] (learned) |
| d2_ctrlvit | fi128_p512 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.134 [0.100, 0.168] | all 5 seeds >= 320 | 0.134 [0.100, 0.168] (learned) | >= 320 (ridge); T2 level 0.246 [0.205, 0.288] | all 5 seeds >= 320 | 0.246 [0.205, 0.288] (learned) |
| d2_ctrlvit | fi128_p512 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.134 [0.100, 0.168] | all 5 seeds >= 320 | 0.134 [0.100, 0.168] (learned) | >= 320 (ridge); T2 level 0.246 [0.205, 0.288] | all 5 seeds >= 320 | 0.246 [0.205, 0.288] (learned) |
| d2_ctrlvit | fi256_p512 | 256 | lrp_full | exact | 150.6 (proj) | [138.6-164.2] | 0.905 [0.881, 0.929] (analytic) | 51.2 (proj) | [47.3-54.2] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | fi256_p512 | 256 | lrp_full | fresh | >= 320 (ridge); T2 level 0.905 [0.881, 0.929] | all 5 seeds >= 320 | 0.905 [0.881, 0.929] (analytic) | 246.2 (ridge) | [232.2-251.8] | 1.000 [1.000, 1.000] (analytic) |
| d2_ctrlvit | fi256_p512 | 256 | lrp_half | exact | >= 320 (ridge); T2 level 0.368 [0.326, 0.410] | all 5 seeds >= 320 | 0.368 [0.326, 0.410] (learned) | 173.2 (ridge) | [154.9-199.6] | 0.808 [0.776, 0.839] (learned) |
| d2_ctrlvit | fi256_p512 | 256 | lrp_half | fresh | >= 320 (ridge); T2 level 0.368 [0.326, 0.410] | all 5 seeds >= 320 | 0.368 [0.326, 0.410] (learned) | >= 320 (ridge); T2 level 0.808 [0.776, 0.839] | all 5 seeds >= 320 | 0.808 [0.776, 0.839] (learned) |
| d2_ctrlvit | fi256_p512 | 256 | biohash_half | exact | >= 320 (ridge); T2 level 0.104 [0.082, 0.127] | all 5 seeds >= 320 | 0.104 [0.082, 0.127] (learned) | >= 320 (ridge); T2 level 0.349 [0.312, 0.387] | all 5 seeds >= 320 | 0.349 [0.312, 0.387] (learned) |
| d2_ctrlvit | fi256_p512 | 256 | biohash_half | fresh | >= 320 (ridge); T2 level 0.104 [0.082, 0.127] | all 5 seeds >= 320 | 0.104 [0.082, 0.127] (learned) | >= 320 (ridge); T2 level 0.349 [0.312, 0.387] | all 5 seeds >= 320 | 0.349 [0.312, 0.387] (learned) |
| d2_ctrlvit | fi256_p512 | 256 | iom_256 | exact | >= 320 (ridge); T2 level 0.051 [0.032, 0.071] | all 5 seeds >= 320 | 0.051 [0.032, 0.071] (learned) | >= 320 (ridge); T2 level 0.137 [0.104, 0.170] | all 5 seeds >= 320 | 0.137 [0.104, 0.170] (learned) |
| d2_ctrlvit | fi256_p512 | 256 | iom_256 | fresh | >= 320 (ridge); T2 level 0.051 [0.032, 0.071] | all 5 seeds >= 320 | 0.051 [0.032, 0.071] (learned) | >= 320 (ridge); T2 level 0.137 [0.104, 0.170] | all 5 seeds >= 320 | 0.137 [0.104, 0.170] (learned) |
| d2_ctrlvit | rbig128_i2 | 128 | lrp_full | exact | 66.9 (proj) | [63.4-72.8] | 0.972 [0.956, 0.989] (analytic) | 44.5 (proj) | [41.3-48.0] | 1.001 [0.998, 1.005] (analytic) |
| d2_ctrlvit | rbig128_i2 | 128 | lrp_full | fresh | 241.1 (ridge) | [221.4-280.6] | 0.972 [0.956, 0.989] (analytic) | 153.9 (ridge) | [144.2-164.2] | 1.001 [0.998, 1.005] (analytic) |
| d2_ctrlvit | rbig128_i2 | 128 | lrp_half | exact | >= 320 (ridge); T2 level 0.424 [0.387, 0.461] | all 5 seeds >= 320 | 0.424 [0.387, 0.461] (learned) | 292.7 (ridge) | [231.8-(>= 320)] (2 of 5 seeds censored) | 0.635 [0.601, 0.668] (analytic) |
| d2_ctrlvit | rbig128_i2 | 128 | lrp_half | fresh | >= 320 (proj); T2 level 0.424 [0.387, 0.461] | all 5 seeds >= 320 | 0.424 [0.387, 0.461] (learned) | >= 320 (ridge); T2 level 0.635 [0.601, 0.668] | all 5 seeds >= 320 | 0.635 [0.601, 0.668] (analytic) |
| d2_ctrlvit | rbig128_i2 | 128 | biohash_half | exact | >= 320 (ridge); T2 level 0.129 [0.103, 0.155] | all 5 seeds >= 320 | 0.129 [0.103, 0.155] (learned) | >= 320 (proj); T2 level 0.172 [0.128, 0.216] | all 5 seeds >= 320 | 0.172 [0.128, 0.216] (learned) |
| d2_ctrlvit | rbig128_i2 | 128 | biohash_half | fresh | >= 320 (proj); T2 level 0.129 [0.103, 0.155] | all 5 seeds >= 320 | 0.129 [0.103, 0.155] (learned) | >= 320 (proj); T2 level 0.172 [0.128, 0.216] | all 5 seeds >= 320 | 0.172 [0.128, 0.216] (learned) |
| d2_ctrlvit | rbig128_i2 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.251 [0.210, 0.291] | all 5 seeds >= 320 | 0.251 [0.210, 0.291] (learned) | >= 320 (ridge); T2 level 0.244 [0.204, 0.285] | all 5 seeds >= 320 | 0.244 [0.204, 0.285] (learned) |
| d2_ctrlvit | rbig128_i2 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.251 [0.210, 0.291] | all 5 seeds >= 320 | 0.251 [0.210, 0.291] (learned) | >= 320 (ridge); T2 level 0.244 [0.204, 0.285] | all 5 seeds >= 320 | 0.244 [0.204, 0.285] (learned) |
| d2_ctrlvit | firbig128_p512_i2 | 128 | lrp_full | exact | 101.9 (proj) | [93.6-110.1] | 0.750 [0.718, 0.782] (analytic) | 51.7 (proj) | [48.6-55.2] | 0.998 [0.991, 1.006] (analytic) |
| d2_ctrlvit | firbig128_p512_i2 | 128 | lrp_full | fresh | >= 320 (ridge); T2 level 0.750 [0.718, 0.782] | all 5 seeds >= 320 | 0.750 [0.718, 0.782] (analytic) | 218.4 (ridge) | [196.3-236.5] | 0.998 [0.991, 1.006] (analytic) |
| d2_ctrlvit | firbig128_p512_i2 | 128 | lrp_half | exact | >= 320 (proj); T2 level 0.201 [0.170, 0.232] | all 5 seeds >= 320 | 0.201 [0.170, 0.232] (learned) | >= 320 (ridge); T2 level 0.552 [0.515, 0.589] | all 5 seeds >= 320 | 0.552 [0.515, 0.589] (analytic) |
| d2_ctrlvit | firbig128_p512_i2 | 128 | lrp_half | fresh | >= 320 (proj); T2 level 0.201 [0.170, 0.232] | all 5 seeds >= 320 | 0.201 [0.170, 0.232] (learned) | >= 320 (ridge); T2 level 0.552 [0.515, 0.589] | all 5 seeds >= 320 | 0.552 [0.515, 0.589] (analytic) |
| d2_ctrlvit | firbig128_p512_i2 | 128 | biohash_half | exact | >= 320 (proj); T2 level 0.056 [0.037, 0.075] | all 5 seeds >= 320 | 0.056 [0.037, 0.075] (learned) | >= 320 (proj); T2 level 0.142 [0.110, 0.175] | all 5 seeds >= 320 | 0.142 [0.110, 0.175] (learned) |
| d2_ctrlvit | firbig128_p512_i2 | 128 | biohash_half | fresh | >= 320 (proj); T2 level 0.056 [0.037, 0.075] | all 5 seeds >= 320 | 0.056 [0.037, 0.075] (learned) | >= 320 (proj); T2 level 0.142 [0.110, 0.175] | all 5 seeds >= 320 | 0.142 [0.110, 0.175] (learned) |
| d2_ctrlvit | firbig128_p512_i2 | 128 | iom_256 | exact | >= 320 (ridge); T2 level 0.119 [0.094, 0.144] | all 5 seeds >= 320 | 0.119 [0.094, 0.144] (learned) | >= 320 (ridge); T2 level 0.223 [0.183, 0.263] | all 5 seeds >= 320 | 0.223 [0.183, 0.263] (learned) |
| d2_ctrlvit | firbig128_p512_i2 | 128 | iom_256 | fresh | >= 320 (ridge); T2 level 0.119 [0.094, 0.144] | all 5 seeds >= 320 | 0.119 [0.094, 0.144] (learned) | >= 320 (ridge); T2 level 0.223 [0.183, 0.263] | all 5 seeds >= 320 | 0.223 [0.183, 0.263] (learned) |

r50 is the smallest r at which the mean relative success reaches 0.5 (log-linear interpolation); the decoder is the best (smallest r50) over ridge, proj and mlp, named in parentheses; `>= r_max` means 0.5 was not reached up to r_max (censored). The per-seed range is computed separately for each of the 5 split seeds: the endpoint relative success is averaged over keys within the seed, r50 is interpolated per decoder (same log-linear rule), the best decoder is the smallest r50 of that seed, and a seed in which no decoder reaches 0.5 is censored (counted as r_max, shown with `>=`; the range is marked censored if any seed is). The headline r50 is computed on the seed-mean curve, so it need not equal the midpoint of the range. The mlp decoder ran only on a subset (pres none and fi128; schemes lrp_full and iom_256; r in {80, 320, 640} for D1 and {80, 320} for D2, from the configs' `mlp_subset`). The key-known (T2) level is the mean E1 (E2) relative success with the key known, of the T2 method (analytic, learned) with the highest mean, shown as mean [95% interval (wider of the identity-level and seed-level intervals)]; if every raw T2 row of that method has zero successes, the cell reads `0 successes; CP95 upper <= x (rel.)`. `back-map-limited` (lrp_full, exact anchors) means 0.5 was not reached up to r_max and the key-known level (exact pre-key vector, public LMMSE back-map) has its reported upper bound (wider interval) below 0.5; this bounds only attacks that reconstruct the pre-key vector and use that back-map. For other schemes and for fresh anchors the key-known level is a reference, not a bound: evaluated collusion decoders exceed it (summary_fmr1). If an r50 was observed although the T2 level is below 0.5 (evidence that the key-known level is not an upper bound), the r50 is shown with a note. Relative success is not capped at 1. E1/E2 relative success carries the wider of two 95% intervals, elementwise: the identity-level interval (the runner's per-row paired-bootstrap standard error, resampling targets, see `src/ijiss_btp/eval/stats.py:63-72`, averaged over keys and seeds; mean +/- 1.96 * SE_bar, which treats the overlapping splits as fully correlated) and the seed-level Student-t interval over the 5 seed means (4 df). The reported interval is their union: lower = min of the two lower bounds, upper = max of the two upper bounds; lower bounds are truncated at 0. Scope: hashing-scheme collusion numbers (lrp_half, biohash_half, iom_256) hold only for the ridge/proj/mlp decoders evaluated; a stronger IoM decoder exists in the synthetic proof check (docs/theory/proofcheck/check4), so hashing collusion resistance is not claimed.

### 4c. Genuine acceptance of the T1 victims (lrp_full, exact anchors)

| encoder | pre | victims per seed and key (n_targets) | E1 genuine acceptance (raw space, DEV raw threshold) | E2 genuine acceptance (renewed key, DEV threshold under that key) |
|---|---|---|---|---|
| d1_base | none | 560 | 0.109 | 0.109 |
| d1_base | pca128_a1 | 560 | 0.109 | 0.094 |
| d1_base | pca256_a1 | 560 | 0.109 | 0.131 |
| d1_base | pca128_a0.5 | 560 | 0.109 | 0.086 |
| d1_base | fi64_p512 | 560 | 0.109 | 0.127 |
| d1_base | fi128_p512 | 560 | 0.109 | 0.151 |
| d1_base | fi256_p512 | 560 | 0.109 | 0.167 |
| d1_base | rbig128_i2 | 560 | 0.109 | 0.098 |
| d1_base | firbig128_p512_i2 | 560 | 0.109 | 0.150 |
| d1_large | none | 560 | 0.105 | 0.105 |
| d1_large | pca128_a1 | 560 | 0.105 | 0.079 |
| d1_large | pca256_a1 | 560 | 0.105 | 0.107 |
| d1_large | pca128_a0.5 | 560 | 0.105 | 0.072 |
| d1_large | fi64_p512 | 560 | 0.105 | 0.111 |
| d1_large | fi128_p512 | 560 | 0.105 | 0.126 |
| d1_large | fi256_p512 | 560 | 0.105 | 0.141 |
| d1_large | rbig128_i2 | 560 | 0.105 | 0.084 |
| d1_large | firbig128_p512_i2 | 560 | 0.105 | 0.126 |
| d2_fingervit | none | 280 | 0.125 | 0.125 |
| d2_fingervit | pca128_a1 | 280 | 0.125 | 0.277 |
| d2_fingervit | pca256_a1 | 280 | 0.125 | 0.347 |
| d2_fingervit | pca128_a0.5 | 280 | 0.125 | 0.204 |
| d2_fingervit | fi64_p512 | 280 | 0.125 | 0.408 |
| d2_fingervit | fi128_p512 | 280 | 0.125 | 0.514 |
| d2_fingervit | fi256_p512 | 280 | 0.125 | 0.561 |
| d2_fingervit | rbig128_i2 | 280 | 0.125 | 0.283 |
| d2_fingervit | firbig128_p512_i2 | 280 | 0.125 | 0.519 |
| d2_deepprint | none | 280 | 0.134 | 0.134 |
| d2_deepprint | pca128_a1 | 280 | 0.134 | 0.170 |
| d2_deepprint | pca256_a1 | 280 | 0.134 | 0.179 |
| d2_deepprint | pca128_a0.5 | 280 | 0.134 | 0.159 |
| d2_deepprint | fi64_p384 | 280 | 0.134 | 0.168 |
| d2_deepprint | fi128_p384 | 280 | 0.134 | 0.189 |
| d2_deepprint | fi256_p384 | 280 | 0.134 | 0.195 |
| d2_deepprint | rbig128_i2 | 280 | 0.134 | 0.170 |
| d2_deepprint | firbig128_p384_i2 | 280 | 0.134 | 0.188 |
| d2_ctrlvit | none | 280 | 0.818 | 0.818 |
| d2_ctrlvit | pca128_a1 | 280 | 0.818 | 0.877 |
| d2_ctrlvit | pca256_a1 | 280 | 0.818 | 0.915 |
| d2_ctrlvit | pca128_a0.5 | 280 | 0.818 | 0.863 |
| d2_ctrlvit | fi64_p512 | 280 | 0.818 | 0.671 |
| d2_ctrlvit | fi128_p512 | 280 | 0.818 | 0.814 |
| d2_ctrlvit | fi256_p512 | 280 | 0.818 | 0.913 |
| d2_ctrlvit | rbig128_i2 | 280 | 0.818 | 0.878 |
| d2_ctrlvit | firbig128_p512_i2 | 280 | 0.818 | 0.821 |

Per seed and key, the targets after the largest r of the grid are the victims (fixed as r grows; the colluders are the first r targets of the same permutation, enrolled users of the stolen database). The genuine acceptance is the mean over seeds of `e1_genuine` and `e2_genuine` of the anchor rows (enrolment capture against the two probe captures of each victim); at r50 the best decoder's proxies are accepted at about half of it.

### 5. Zero-pair attacks (T0, lrp_full): best key-free configuration

| encoder | pre | best E1 relative [95% CI] (CP95 upper if 0 successes) | rec_cos_z (pre-key space) [95% CI] | method | p (argmax of mean E1) | AUX | CP95 upper, absolute proxy acceptance | absolute proxy acceptance (mean e1_proxy at the selected method, AUX and p) | e1_genuine (raw-space genuine acceptance) | oracle gap (eigen_oracle - eigen E1 at the best eigen p and AUX, pre none) | E|cos| pre-key at the best common-method configuration (eigen, wp) / uniform reference |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d1_base | none | 0.001 [0.000, 0.003] | -0.000 [-0.004, 0.004] | wp | 768 | shifted |  | 0.0001 | 0.112 | 0.000 (p 8, same_dist) | 0.031 / 0.029 |
| d1_base | pca128_a1 | 0.010 [0.000, 0.031] | 0.013 [-0.014, 0.039] | wp | 128 | same_dist |  | 0.0011 | 0.112 |  | 0.077 / 0.071 |
| d1_base | pca128_a0.5 | 0.005 [0.000, 0.011] | 0.019 [0.001, 0.038] | wp | 128 | same_dist |  | 0.0005 | 0.112 |  | 0.083 / 0.071 |
| d1_base | pca256_a1 | 0.001 [0.000, 0.004] | 0.013 [0.002, 0.024] | wp | 256 | same_dist |  | 0.0002 | 0.112 |  | 0.058 / 0.050 |
| d1_base | fi64_p512 | 0.005 [0.000, 0.019] | 0.003 [-0.018, 0.024] | ica | 64 | shifted |  | 0.0006 | 0.112 |  | 0.135 / 0.100 |
| d1_base | fi128_p512 | 0.010 [0.000, 0.028] | -0.000 [-0.011, 0.010] | ica | 128 | shifted |  | 0.0012 | 0.112 |  | 0.100 / 0.071 |
| d1_base | fi256_p512 | 0 successes; CP95 upper <= 0.024 | 0.030 [0.007, 0.053] | n/a | n/a | n/a | 0.002 | 0.0000 | 0.112 |  | 0.078 / 0.050 |
| d1_base | rbig128_i2 | 0.011 [0.000, 0.029] | 0.047 [0.023, 0.070] | ica | 128 | same_dist |  | 0.0012 | 0.112 |  | 0.082 / 0.071 |
| d1_base | firbig128_p512_i2 | 0.004 [0.000, 0.017] | -0.004 [-0.018, 0.011] | ica | 128 | shifted |  | 0.0005 | 0.112 |  | 0.100 / 0.071 |
| d1_large | none | 0.002 [0.000, 0.006] | 0.001 [-0.002, 0.005] | wp | 1024 | shifted |  | 0.0002 | 0.106 | -0.001 (p 8, same_dist) | 0.026 / 0.025 |
| d1_large | pca128_a1 | 0.007 [0.000, 0.016] | 0.030 [0.013, 0.048] | ica | 128 | same_dist |  | 0.0008 | 0.106 |  | 0.078 / 0.071 |
| d1_large | pca128_a0.5 | 0.009 [0.000, 0.023] | 0.024 [-0.001, 0.050] | ica | 128 | same_dist |  | 0.0009 | 0.106 |  | 0.076 / 0.071 |
| d1_large | pca256_a1 | 0.002 [0.000, 0.006] | 0.009 [-0.003, 0.020] | wp | 256 | same_dist |  | 0.0003 | 0.106 |  | 0.053 / 0.050 |
| d1_large | fi64_p512 | 0.010 [0.000, 0.026] | 0.017 [-0.020, 0.054] | ica | 64 | shifted |  | 0.0010 | 0.106 |  | 0.116 / 0.100 |
| d1_large | fi128_p512 | 0.010 [0.000, 0.026] | 0.001 [-0.005, 0.008] | ica | 128 | shifted |  | 0.0010 | 0.106 |  | 0.076 / 0.071 |
| d1_large | fi256_p512 | 0.002 [0.000, 0.006] | -0.003 [-0.010, 0.004] | eigen | 8 | shifted |  | 0.0002 | 0.106 |  | 0.049 / 0.050 |
| d1_large | rbig128_i2 | 0.011 [0.000, 0.025] | 0.020 [0.002, 0.038] | ica | 128 | same_dist |  | 0.0012 | 0.106 |  | 0.071 / 0.071 |
| d1_large | firbig128_p512_i2 | 0.010 [0.000, 0.026] | -0.002 [-0.011, 0.007] | ica | 128 | shifted |  | 0.0010 | 0.106 |  | 0.094 / 0.071 |
| d2_fingervit | none | 0.043 [0.000, 0.122] | 0.470 [0.274, 0.665] | wp | 1536 | same_dist |  | 0.0052 | 0.122 | 0.000 (p 8, same_dist) | 0.491 / 0.020 |
| d2_fingervit | pca128_a1 | 0 successes; CP95 upper <= 0.044 | 0.025 [0.013, 0.038] | n/a | n/a | n/a | 0.005 | 0.0000 | 0.122 |  | 0.082 / 0.071 |
| d2_fingervit | pca128_a0.5 | 0.020 [0.000, 0.074] | 0.101 [-0.048, 0.249] | wp | 128 | same_dist |  | 0.0027 | 0.122 |  | 0.177 / 0.071 |
| d2_fingervit | pca256_a1 | 0 successes; CP95 upper <= 0.044 | 0.017 [0.009, 0.025] | n/a | n/a | n/a | 0.005 | 0.0000 | 0.122 |  | 0.059 / 0.050 |
| d2_fingervit | fi64_p512 | 0.018 [0.000, 0.047] | 0.065 [0.038, 0.092] | ica | 64 | same_dist |  | 0.0022 | 0.122 |  | 0.122 / 0.100 |
| d2_fingervit | fi128_p512 | 0.001 [0.000, 0.005] | 0.029 [0.011, 0.048] | ica | 128 | same_dist |  | 0.0002 | 0.122 |  | 0.087 / 0.071 |
| d2_fingervit | fi256_p512 | 0 successes; CP95 upper <= 0.044 | 0.001 [-0.008, 0.010] | n/a | n/a | n/a | 0.005 | 0.0000 | 0.122 |  | 0.055 / 0.050 |
| d2_fingervit | rbig128_i2 | 0 successes; CP95 upper <= 0.044 | 0.016 [0.006, 0.027] | n/a | n/a | n/a | 0.005 | 0.0000 | 0.122 |  | 0.081 / 0.071 |
| d2_fingervit | firbig128_p512_i2 | 0 successes; CP95 upper <= 0.044 | 0.024 [0.001, 0.046] | n/a | n/a | n/a | 0.005 | 0.0000 | 0.122 |  | 0.083 / 0.071 |
| d2_deepprint | none | 0 successes; CP95 upper <= 0.042 | -0.096 [-0.130, -0.062] | n/a | n/a | n/a | 0.005 | 0.0000 | 0.135 | 0.019 (p 8, same_dist) | 0.167 / 0.035 |
| d2_deepprint | pca128_a1 | 0.002 [0.000, 0.007] | 0.022 [0.010, 0.033] | ica | 128 | same_dist |  | 0.0003 | 0.135 |  | 0.074 / 0.071 |
| d2_deepprint | pca128_a0.5 | 0.015 [0.000, 0.053] | 0.021 [-0.075, 0.117] | eigen | 8 | same_dist |  | 0.0018 | 0.135 |  | 0.146 / 0.071 |
| d2_deepprint | pca256_a1 | 0 successes; CP95 upper <= 0.042 | 0.007 [-0.001, 0.016] | n/a | n/a | n/a | 0.005 | 0.0000 | 0.135 |  | 0.055 / 0.050 |
| d2_deepprint | fi64_p384 | 0.001 [0.000, 0.004] | 0.014 [-0.009, 0.037] | ica | 64 | same_dist |  | 0.0002 | 0.135 |  | 0.115 / 0.100 |
| d2_deepprint | fi128_p384 | 0.001 [0.000, 0.005] | -0.004 [-0.013, 0.006] | wp | 128 | same_dist |  | 0.0002 | 0.135 |  | 0.074 / 0.071 |
| d2_deepprint | fi256_p384 | 0.001 [0.000, 0.004] | 0.010 [-0.005, 0.025] | eigen | 32 | same_dist |  | 0.0002 | 0.135 |  | 0.065 / 0.050 |
| d2_deepprint | rbig128_i2 | 0.004 [0.000, 0.014] | 0.023 [0.010, 0.036] | ica | 128 | same_dist |  | 0.0005 | 0.135 |  | 0.074 / 0.071 |
| d2_deepprint | firbig128_p384_i2 | 0.001 [0.000, 0.005] | 0.005 [-0.017, 0.027] | eigen | 32 | same_dist |  | 0.0002 | 0.135 |  | 0.084 / 0.071 |
| d2_ctrlvit | none | 0.017 [0.003, 0.031] | 0.105 [0.057, 0.153] | eigen | 8 | same_dist |  | 0.0137 | 0.815 | 0.003 (p 8, same_dist) | 0.143 / 0.029 |
| d2_ctrlvit | pca128_a1 | 0.001 [0.000, 0.003] | 0.007 [0.001, 0.014] | ica | 128 | same_dist |  | 0.0008 | 0.815 |  | 0.075 / 0.071 |
| d2_ctrlvit | pca128_a0.5 | 0.003 [0.000, 0.007] | 0.015 [0.006, 0.024] | ica | 128 | same_dist |  | 0.0028 | 0.815 |  | 0.077 / 0.071 |
| d2_ctrlvit | pca256_a1 | 0.000 [0.000, 0.001] | 0.007 [-0.006, 0.020] | eigen | 32 | same_dist |  | 0.0002 | 0.815 |  | 0.068 / 0.050 |
| d2_ctrlvit | fi64_p512 | 0.008 [0.001, 0.015] | 0.040 [0.021, 0.058] | ica | 64 | same_dist |  | 0.0065 | 0.815 |  | 0.112 / 0.100 |
| d2_ctrlvit | fi128_p512 | 0.003 [0.000, 0.006] | 0.003 [-0.014, 0.021] | ica | 128 | same_dist |  | 0.0022 | 0.815 |  | 0.074 / 0.071 |
| d2_ctrlvit | fi256_p512 | 0 successes; CP95 upper <= 0.006 | 0.016 [0.002, 0.030] | n/a | n/a | n/a | 0.005 | 0.0000 | 0.815 |  | 0.068 / 0.050 |
| d2_ctrlvit | rbig128_i2 | 0.002 [0.000, 0.006] | 0.007 [-0.012, 0.026] | ica | 128 | same_dist |  | 0.0013 | 0.815 |  | 0.073 / 0.071 |
| d2_ctrlvit | firbig128_p512_i2 | 0.003 [0.000, 0.007] | 0.007 [-0.009, 0.023] | ica | 128 | same_dist |  | 0.0027 | 0.815 |  | 0.076 / 0.071 |

Notes. Key-free methods are eigen, ica and wp over both AUX types. `eigen_oracle` uses the key to resolve signs; it is a diagnostic, never a key-free attack, and appears only in the oracle-gap column. "Best" is the maximum over p, method and AUX, chosen with ground truth; it is an optimistic upper envelope from the attacker's side and therefore errs towards overstating risk. The E1 value and `rec_cos_z` come from `summary_zeropair.csv`. E1 and rec_cos_z come from the same argmax-E1 row per seed and key. The last column is the mean absolute pre-key cosine (`best_abs_cos_z`) at the best common-method (eigen, wp) configuration, next to the uniform-direction reference; it is the value that the manuscript prints as E|cos| (`zp.<enc>.<pre>.abscos`). The p column is the p with the highest mean E1 for the selected method and AUX. FastICA alignment ran only where m <= 128 (`ica_max_dim`), and Wasserstein-Procrustes only for lrp_full with k = m, so pre `none` rows have no ICA result and the none and fi rows compare different attack sets; the headline table therefore uses the common methods eigen and wp for the none-vs-fi128 comparison. CP95 upper is the largest runner-reported one-sided 95% Clopper-Pearson upper bound over seed/key runs of the selected method and AUX (n = `e1_trials` per row, table 3c); the relative form divides by the run's genuine acceptance. n (E1 trials) is read from the run rows (`e1_trials`) and differs per dataset: see table 3c. Zero observed successes are reported with the largest runner-reported one-sided 95% Clopper-Pearson upper bound, never as zero risk. Intervals on every metric other than E1/E2 relative success are Student-t intervals over the 5 seed means (4 df), mean +/- t(0.975, 4) * sd / sqrt(5): between-split variability over 5 overlapping splits; not an identity-level CI. Lower bounds of nonnegative quantities are truncated at 0; signed cosines are not truncated. E1/E2 relative success carries the wider of two 95% intervals, elementwise: the identity-level interval (the runner's per-row paired-bootstrap standard error, resampling targets, see `src/ijiss_btp/eval/stats.py:63-72`, averaged over keys and seeds; mean +/- 1.96 * SE_bar, which treats the overlapping splits as fully correlated) and the seed-level Student-t interval over the 5 seed means (4 df). The reported interval is their union: lower = min of the two lower bounds, upper = max of the two upper bounds; lower bounds are truncated at 0.

### 5b. Sign-resolution reference (eigen_oracle, pre none, lrp_full)

| encoder | max E1 relative of eigen_oracle over AUX and p (pre none) | at p, AUX | absolute E1 acceptance at that entry (mean e1_proxy) | max rec_cos_z (pre-key) of eigen_oracle | at p, AUX | oracle gap (eigen_oracle - eigen E1) at the key-free eigen selection | selected p, AUX |
|---|---|---|---|---|---|---|---|
| d1_base | 0 successes; CP95 upper <= 0.024 | n/a (no success) | n/a | 0.090 | 32, same_dist | 0.000 | 8, same_dist |
| d1_large | 0.002 | 8, shifted | <0.001 | 0.056 | 128, same_dist | -0.001 | 8, same_dist |
| d2_fingervit | 0 successes; CP95 upper <= 0.044 | n/a (no success) | n/a | 0.526 | 8, same_dist | 0.000 | 8, same_dist |
| d2_deepprint | 0.019 | 8, same_dist | 0.003 | 0.327 | 64, same_dist | 0.019 | 8, same_dist |
| d2_ctrlvit | 0.020 | 8, same_dist | 0.016 | 0.183 | 16, same_dist | 0.003 | 8, same_dist |

`eigen_oracle` is eigen alignment of the stolen templates with AUX (leading p principal axes matched by rank, variances rescaled) whose per-component signs come from the key: alignment followed by exact sign resolution, which an attacker could at best obtain from anchors. It is a diagnostic, never a key-free attack. Values are means over keys and seeds; the maximum over AUX type and p (grid 8 to 128) is selected with ground truth and has no interval. A maximum with zero successes in every row gives the largest relative one-sided 95% Clopper-Pearson bound. The joint alignment-plus-anchor attack, with anchor-voted signs, is in sections 13 and 13b and in the paper's Table S5.

### 5c. Sign-resolution reference after whitening (eigen_oracle, pca128, pca256, fi64, fi128, fi256, lrp_full)

| encoder | pre | max E1 relative of eigen_oracle over AUX and p | max E1 absolute (mean e1_proxy) | max E2 relative | max E2 absolute (mean e2_proxy) | max rec_cos_z (pre-key) |
|---|---|---|---|---|---|---|
| d1_base | pca128 | 0 successes; CP95 upper <= 0.024 | 0.000 | 0 successes; CP95 upper <= 0.027 | 0.000 | 0.123 |
| d1_base | pca256 | 0 successes; CP95 upper <= 0.024 | 0.000 | 0.003 | <0.001 | 0.088 |
| d1_base | fi64 | 0 successes; CP95 upper <= 0.024 | 0.000 | 0.022 | 0.003 | 0.191 |
| d1_base | fi128 | 0.001 | <0.001 | 0.013 | 0.002 | 0.124 |
| d1_base | fi256 | 0.004 | <0.001 | 0.020 | 0.003 | 0.081 |
| d1_large | pca128 | 0.001 | <0.001 | 0 successes; CP95 upper <= 0.034 | 0.000 | 0.110 |
| d1_large | pca256 | 0.001 | <0.001 | 0.002 | <0.001 | 0.080 |
| d1_large | fi64 | 0 successes; CP95 upper <= 0.026 | 0.000 | 0.013 | 0.002 | 0.159 |
| d1_large | fi128 | 0.001 | <0.001 | 0.009 | 0.001 | 0.117 |
| d1_large | fi256 | 0.006 | 0.001 | 0.014 | 0.002 | 0.077 |
| d2_fingervit | pca128 | 0 successes; CP95 upper <= 0.044 | 0.000 | 0.004 | 0.001 | 0.085 |
| d2_fingervit | pca256 | 0 successes; CP95 upper <= 0.044 | 0.000 | 0.009 | 0.003 | 0.061 |
| d2_fingervit | fi64 | 0 successes; CP95 upper <= 0.044 | 0.000 | 0.004 | 0.002 | 0.152 |
| d2_fingervit | fi128 | 0 successes; CP95 upper <= 0.044 | 0.000 | 0.005 | 0.003 | 0.100 |
| d2_fingervit | fi256 | 0 successes; CP95 upper <= 0.044 | 0.000 | 0.013 | 0.007 | 0.069 |
| d2_deepprint | pca128 | 0.002 | <0.001 | 0.002 | <0.001 | 0.091 |
| d2_deepprint | pca256 | 0 successes; CP95 upper <= 0.042 | 0.000 | 0.006 | 0.001 | 0.065 |
| d2_deepprint | fi64 | 0.001 | <0.001 | 0.006 | 0.001 | 0.142 |
| d2_deepprint | fi128 | 0 successes; CP95 upper <= 0.042 | 0.000 | 0.006 | 0.001 | 0.089 |
| d2_deepprint | fi256 | 0.001 | <0.001 | 0.011 | 0.002 | 0.060 |
| d2_ctrlvit | pca128 | <0.001 | <0.001 | 0.002 | 0.002 | 0.105 |
| d2_ctrlvit | pca256 | 0 successes; CP95 upper <= 0.006 | 0.000 | 0.007 | 0.006 | 0.070 |
| d2_ctrlvit | fi64 | 0.001 | 0.001 | 0.001 | 0.001 | 0.148 |
| d2_ctrlvit | fi128 | 0.001 | 0.001 | <0.001 | <0.001 | 0.102 |
| d2_ctrlvit | fi256 | 0 successes; CP95 upper <= 0.006 | 0.000 | 0.007 | 0.006 | 0.066 |

`eigen_oracle` is eigen alignment of the stolen templates with AUX (leading p principal axes matched by rank, variances rescaled) whose per-component signs come from the key: alignment followed by exact sign resolution, which an attacker could at best obtain from anchors. It is a diagnostic, never a key-free attack. Values are means over keys and seeds; the maximum over AUX type and p (grid 8 to 128) is selected with ground truth and has no interval. A maximum with zero successes in every row gives the largest relative one-sided 95% Clopper-Pearson bound. The joint alignment-plus-anchor attack, with anchor-voted signs, is in sections 13 and 13b and in the paper's Table S5. After whitening, E1 maps the proxy back with the public back-map and E2 re-protects it with the renewed key (the round trip of the protocol); each maximum (relative, absolute, cosine) is taken on its own and can come from a different entry.

### 6. Key compromise (T2): in-system acceptance and E2 after renewal

| encoder | scheme | pre | attack | in-system acceptance [95% CI] | E2 relative, post-renewal [95% interval (wider of the identity-level and seed-level intervals)] |
|---|---|---|---|---|---|
| d1_base | biohash_half | none | analytic | 1.000 [1.000, 1.000] | 0.079 [0.045, 0.113] |
| d1_base | biohash_half | none | learned | 1.000 [1.000, 1.000] | 0.389 [0.319, 0.459] |
| d1_base | iom_256 | none | analytic | 1.000 [1.000, 1.000] | 0.001 [0.000, 0.003] |
| d1_base | iom_256 | none | learned | 1.000 [1.000, 1.000] | 0.063 [0.028, 0.097] |
| d1_base | lrp_full | none | analytic | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d1_base | lrp_full | none | learned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d1_base | lrp_half | none | analytic | 1.000 [1.000, 1.000] | 0.783 [0.727, 0.838] |
| d1_base | lrp_half | none | learned | 1.000 [1.000, 1.000] | 0.793 [0.740, 0.846] |
| d1_base | biohash_half | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.182 [0.125, 0.238] |
| d1_base | biohash_half | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.228 [0.163, 0.292] |
| d1_base | iom_256 | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.268 [0.201, 0.336] |
| d1_base | iom_256 | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.300 [0.230, 0.369] |
| d1_base | lrp_full | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d1_base | lrp_full | fi128_p512 | learned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d1_base | lrp_half | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.676 [0.619, 0.732] |
| d1_base | lrp_half | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.682 [0.625, 0.739] |
| d1_large | biohash_half | none | analytic | 1.000 [1.000, 1.000] | 0.244 [0.185, 0.303] |
| d1_large | biohash_half | none | learned | 1.000 [1.000, 1.000] | 0.507 [0.435, 0.579] |
| d1_large | iom_256 | none | analytic | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.012] |
| d1_large | iom_256 | none | learned | 1.000 [1.000, 1.000] | 0.047 [0.016, 0.078] |
| d1_large | lrp_full | none | analytic | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d1_large | lrp_full | none | learned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d1_large | lrp_half | none | analytic | 1.000 [1.000, 1.000] | 0.818 [0.765, 0.870] |
| d1_large | lrp_half | none | learned | 1.000 [1.000, 1.000] | 0.829 [0.777, 0.880] |
| d1_large | biohash_half | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.184 [0.125, 0.243] |
| d1_large | biohash_half | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.204 [0.141, 0.267] |
| d1_large | iom_256 | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.252 [0.185, 0.319] |
| d1_large | iom_256 | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.314 [0.241, 0.387] |
| d1_large | lrp_full | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d1_large | lrp_full | fi128_p512 | learned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d1_large | lrp_half | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.689 [0.630, 0.748] |
| d1_large | lrp_half | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.701 [0.642, 0.760] |
| d2_fingervit | biohash_half | none | analytic | 1.000 [1.000, 1.000] | 0 successes; CP95 upper <= 0.049 (rel.) |
| d2_fingervit | biohash_half | none | learned | 1.000 [1.000, 1.000] | 0.789 [0.708, 0.869] |
| d2_fingervit | iom_256 | none | analytic | 1.000 [1.000, 1.000] | 0 successes; CP95 upper <= 0.073 (rel.) |
| d2_fingervit | iom_256 | none | learned | 0.990 [0.977, 1.003] | 0.352 [0.204, 0.500] |
| d2_fingervit | lrp_full | none | analytic | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d2_fingervit | lrp_full | none | learned | 1.000 [1.000, 1.000] | 1.001 [0.998, 1.005] |
| d2_fingervit | lrp_half | none | analytic | 1.000 [1.000, 1.000] | 0.987 [0.962, 1.013] |
| d2_fingervit | lrp_half | none | learned | 1.000 [1.000, 1.000] | 0.989 [0.964, 1.014] |
| d2_fingervit | biohash_half | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.129 [0.090, 0.168] |
| d2_fingervit | biohash_half | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.180 [0.134, 0.226] |
| d2_fingervit | iom_256 | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.201 [0.151, 0.252] |
| d2_fingervit | iom_256 | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.280 [0.224, 0.335] |
| d2_fingervit | lrp_full | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.001] |
| d2_fingervit | lrp_full | fi128_p512 | learned | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.001] |
| d2_fingervit | lrp_half | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.486 [0.439, 0.533] |
| d2_fingervit | lrp_half | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.491 [0.442, 0.539] |
| d2_deepprint | biohash_half | none | analytic | 1.000 [1.000, 1.000] | 0 successes; CP95 upper <= 0.047 (rel.) |
| d2_deepprint | biohash_half | none | learned | 1.000 [1.000, 1.000] | 0.829 [0.741, 0.917] |
| d2_deepprint | iom_256 | none | analytic | 1.000 [1.000, 1.000] | 0 successes; CP95 upper <= 0.070 (rel.) |
| d2_deepprint | iom_256 | none | learned | 1.000 [1.000, 1.000] | 0.545 [0.405, 0.686] |
| d2_deepprint | lrp_full | none | analytic | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d2_deepprint | lrp_full | none | learned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d2_deepprint | lrp_half | none | analytic | 1.000 [1.000, 1.000] | 1.001 [0.994, 1.008] |
| d2_deepprint | lrp_half | none | learned | 1.000 [1.000, 1.000] | 1.001 [0.994, 1.008] |
| d2_deepprint | biohash_half | fi128_p384 | analytic | 1.000 [1.000, 1.000] | 0.166 [0.081, 0.250] |
| d2_deepprint | biohash_half | fi128_p384 | learned | 1.000 [1.000, 1.000] | 0.190 [0.098, 0.281] |
| d2_deepprint | iom_256 | fi128_p384 | analytic | 1.000 [1.000, 1.000] | 0.196 [0.099, 0.294] |
| d2_deepprint | iom_256 | fi128_p384 | learned | 1.000 [1.000, 1.000] | 0.254 [0.143, 0.364] |
| d2_deepprint | lrp_full | fi128_p384 | analytic | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d2_deepprint | lrp_full | fi128_p384 | learned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d2_deepprint | lrp_half | fi128_p384 | analytic | 1.000 [1.000, 1.000] | 0.535 [0.452, 0.617] |
| d2_deepprint | lrp_half | fi128_p384 | learned | 1.000 [1.000, 1.000] | 0.529 [0.448, 0.610] |
| d2_ctrlvit | biohash_half | none | analytic | 1.000 [1.000, 1.000] | 0.156 [0.114, 0.198] |
| d2_ctrlvit | biohash_half | none | learned | 1.000 [1.000, 1.000] | 0.767 [0.734, 0.801] |
| d2_ctrlvit | iom_256 | none | analytic | 1.000 [1.000, 1.000] | 0.009 [0.001, 0.017] |
| d2_ctrlvit | iom_256 | none | learned | 1.000 [1.000, 1.000] | 0.163 [0.126, 0.201] |
| d2_ctrlvit | lrp_full | none | analytic | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d2_ctrlvit | lrp_full | none | learned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d2_ctrlvit | lrp_half | none | analytic | 1.000 [1.000, 1.000] | 0.949 [0.929, 0.969] |
| d2_ctrlvit | lrp_half | none | learned | 1.000 [1.000, 1.000] | 0.951 [0.931, 0.971] |
| d2_ctrlvit | biohash_half | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.117 [0.087, 0.148] |
| d2_ctrlvit | biohash_half | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.138 [0.105, 0.171] |
| d2_ctrlvit | iom_256 | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.181 [0.143, 0.218] |
| d2_ctrlvit | iom_256 | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.246 [0.205, 0.288] |
| d2_ctrlvit | lrp_full | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d2_ctrlvit | lrp_full | fi128_p512 | learned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| d2_ctrlvit | lrp_half | fi128_p512 | analytic | 1.000 [1.000, 1.000] | 0.569 [0.533, 0.604] |
| d2_ctrlvit | lrp_half | fi128_p512 | learned | 1.000 [1.000, 1.000] | 0.571 [0.534, 0.607] |

Intervals on every metric other than E1/E2 relative success are Student-t intervals over the 5 seed means (4 df), mean +/- t(0.975, 4) * sd / sqrt(5): between-split variability over 5 overlapping splits; not an identity-level CI. Lower bounds of nonnegative quantities are truncated at 0; signed cosines are not truncated. E1/E2 relative success carries the wider of two 95% intervals, elementwise: the identity-level interval (the runner's per-row paired-bootstrap standard error, resampling targets, see `src/ijiss_btp/eval/stats.py:63-72`, averaged over keys and seeds; mean +/- 1.96 * SE_bar, which treats the overlapping splits as fully correlated) and the seed-level Student-t interval over the 5 seed means (4 df). The reported interval is their union: lower = min of the two lower bounds, upper = max of the two upper bounds; lower bounds are truncated at 0.

### 6b. Key compromise (T2), lrp_full: key-known level with absolute acceptance

| encoder | pre | endpoint | key-known (T2) level, relative [95% interval] | absolute proxy acceptance (mean e*_proxy) | genuine acceptance of the same targets (mean e*_genuine) | T2 method |
|---|---|---|---|---|---|---|
| d1_base | none | E1 | 1.000 [1.000, 1.000] | 0.112 | 0.112 | analytic |
| d1_base | none | E2 | 1.000 [1.000, 1.000] | 0.112 | 0.112 | analytic |
| d1_base | fi128_p512 | E1 | 0.036 [0.004, 0.067] | 0.004 | 0.112 | analytic |
| d1_base | fi128_p512 | E2 | 1.000 [1.000, 1.000] | 0.153 | 0.153 | analytic |
| d1_large | none | E1 | 1.000 [1.000, 1.000] | 0.106 | 0.106 | analytic |
| d1_large | none | E2 | 1.000 [1.000, 1.000] | 0.106 | 0.106 | analytic |
| d1_large | fi128_p512 | E1 | 0.042 [0.009, 0.075] | 0.004 | 0.106 | analytic |
| d1_large | fi128_p512 | E2 | 1.000 [1.000, 1.000] | 0.125 | 0.125 | analytic |
| d2_fingervit | none | E1 | 1.001 [0.998, 1.005] | 0.122 | 0.122 | learned |
| d2_fingervit | none | E2 | 1.001 [0.998, 1.005] | 0.122 | 0.122 | learned |
| d2_fingervit | fi128_p512 | E1 | 0.548 [0.446, 0.650] | 0.067 | 0.122 | analytic |
| d2_fingervit | fi128_p512 | E2 | 1.000 [0.999, 1.001] | 0.508 | 0.508 | analytic |
| d2_deepprint | none | E1 | 1.000 [1.000, 1.000] | 0.135 | 0.135 | analytic |
| d2_deepprint | none | E2 | 1.000 [1.000, 1.000] | 0.135 | 0.135 | analytic |
| d2_deepprint | fi128_p384 | E1 | 0.488 [0.363, 0.613] | 0.066 | 0.135 | analytic |
| d2_deepprint | fi128_p384 | E2 | 1.000 [1.000, 1.000] | 0.192 | 0.192 | analytic |
| d2_ctrlvit | none | E1 | 1.000 [1.000, 1.000] | 0.815 | 0.815 | analytic |
| d2_ctrlvit | none | E2 | 1.000 [1.000, 1.000] | 0.815 | 0.815 | analytic |
| d2_ctrlvit | fi128_p512 | E1 | 0.750 [0.717, 0.783] | 0.611 | 0.815 | analytic |
| d2_ctrlvit | fi128_p512 | E2 | 1.000 [1.000, 1.000] | 0.814 | 0.814 | analytic |

The relative level is the key-known level of table 4b (T2 method with the highest mean); the absolute acceptance is the mean over seeds of `e1_proxy` (`e2_proxy`) of that method, next to the genuine acceptance of the same targets. Renewal re-enrols every target from the same enrolment capture under the second key (no fresh capture). E2 back-maps, normalises and pre-conditions the proxy before the second key, so a pre-key vector recovered exactly with the old key returns as a positive multiple of itself plus a small offset (G(B(z)/q) = z/q + (1/q - 1) W^T x_bar with q = ||B(z)||); acceptance compares cosines, so the E2 level checks that this round trip preserves acceptance to the reported precision, single borderline decisions can flip, and it is no equality of templates. Intervals on every metric other than E1/E2 relative success are Student-t intervals over the 5 seed means (4 df), mean +/- t(0.975, 4) * sd / sqrt(5): between-split variability over 5 overlapping splits; not an identity-level CI. Lower bounds of nonnegative quantities are truncated at 0; signed cosines are not truncated. E1/E2 relative success carries the wider of two 95% intervals, elementwise: the identity-level interval (the runner's per-row paired-bootstrap standard error, resampling targets, see `src/ijiss_btp/eval/stats.py:63-72`, averaged over keys and seeds; mean +/- 1.96 * SE_bar, which treats the overlapping splits as fully correlated) and the seed-level Student-t interval over the 5 seed means (4 df). The reported interval is their union: lower = min of the two lower bounds, upper = max of the two upper bounds; lower bounds are truncated at 0.

### 7a. Scale runs: distractor rows and environment

| run | distractor data source | distractor source | rows loaded (after the relabel step, if any) | removed: fingers shared with D1 | drawn for the largest size | removed: near-duplicates of a D1 row (cos >= 0.95) | available after the near-duplicate filter | requested -> used | CPU model | OMP threads |
|---|---|---|---|---|---|---|---|---|---|---|
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | sin:secondary:base:relabel=secondary_relabel.csv:key=path:dropped=249:shared_ids_removed=11387:dedup<0.95 | 14614 | 11387 | 3000 | 2 | 2998 | 1000 -> 1000, 2000 -> 2000, 3000 -> 2998 | AMD Ryzen 9 7950X 16-Core Processor | 2 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | features_by_id:synthetic_100000_base_features.npz:dedup<0.95 | 100000 | n/a (no exclusion) | 100000 | 0 | 100000 | 10000 -> 10000, 50000 -> 50000, 100000 -> 100000 | AMD Ryzen 9 7950X 16-Core Processor | 8 |

Order of the steps: rows loaded (after the relabel step for the secondary corpus), rows of fingers shared with D1 removed, seeded draw of the largest requested size, near-duplicates of any D1 row removed (uncentered cosine >= 0.95). A requested size above the rows available after the filter is capped (requested -> used); table 7b lists both sizes, and the keys of scripts/manuscript/make_numbers.py follow the size used.

### 7b. Scale (HNSW)

| run | distractor data source | scheme | n_distractors requested | n_distractors used | ef_search | rank1_exact | rank1_hnsw | recall10_tie_aware | latency_ms_per_query | serialized index size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | lrp_full | 1000 | 1000 | 64 | 0.132 | 0.132 | 0.995 | 0.030 | 1.724 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | lrp_full | 1000 | 1000 | 128 | 0.132 | 0.132 | 1.000 | 0.049 | 1.724 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | lrp_full | 2000 | 2000 | 64 | 0.128 | 0.128 | 0.989 | 0.034 | 2.507 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | lrp_full | 2000 | 2000 | 128 | 0.128 | 0.128 | 0.999 | 0.061 | 2.507 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | lrp_full | 3000 | 2998 | 64 | 0.127 | 0.127 | 0.985 | 0.036 | 3.289 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | lrp_full | 3000 | 2998 | 128 | 0.127 | 0.127 | 0.998 | 0.065 | 3.289 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | biohash_half | 1000 | 1000 | 64 | 0.080 | 0.080 | 1.000 | 0.024 | 0.615 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | biohash_half | 1000 | 1000 | 128 | 0.080 | 0.080 | 1.000 | 0.039 | 0.615 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | biohash_half | 2000 | 2000 | 64 | 0.075 | 0.075 | 1.000 | 0.026 | 0.894 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | biohash_half | 2000 | 2000 | 128 | 0.075 | 0.075 | 1.000 | 0.045 | 0.894 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | biohash_half | 3000 | 2998 | 64 | 0.073 | 0.073 | 0.999 | 0.026 | 1.173 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | biohash_half | 3000 | 2998 | 128 | 0.073 | 0.073 | 1.000 | 0.048 | 1.173 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | iom_256 | 1000 | 1000 | 64 | 0.086 | 0.086 | 1.000 | 0.025 | 0.668 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | iom_256 | 1000 | 1000 | 128 | 0.086 | 0.086 | 1.000 | 0.041 | 0.668 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | iom_256 | 2000 | 2000 | 64 | 0.083 | 0.083 | 0.999 | 0.028 | 0.971 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | iom_256 | 2000 | 2000 | 128 | 0.083 | 0.083 | 1.000 | 0.047 | 0.971 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | iom_256 | 3000 | 2998 | 64 | 0.081 | 0.081 | 0.998 | 0.029 | 1.274 |
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | iom_256 | 3000 | 2998 | 128 | 0.081 | 0.081 | 1.000 | 0.051 | 1.274 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | lrp_full | 10000 | 10000 | 64 | 0.130 | 0.130 | 0.947 | 0.041 | 8.780 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | lrp_full | 10000 | 10000 | 128 | 0.130 | 0.130 | 0.988 | 0.077 | 8.780 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | lrp_full | 50000 | 50000 | 64 | 0.124 | 0.118 | 0.821 | 0.062 | 40.153 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | lrp_full | 50000 | 50000 | 128 | 0.124 | 0.124 | 0.917 | 0.117 | 40.153 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | lrp_full | 100000 | 100000 | 64 | 0.122 | 0.112 | 0.742 | 0.113 | 79.362 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | lrp_full | 100000 | 100000 | 128 | 0.122 | 0.119 | 0.853 | 0.211 | 79.362 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | biohash_half | 10000 | 10000 | 64 | 0.072 | 0.071 | 0.995 | 0.029 | 3.135 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | biohash_half | 10000 | 10000 | 128 | 0.072 | 0.072 | 1.000 | 0.058 | 3.135 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | biohash_half | 50000 | 50000 | 64 | 0.063 | 0.062 | 0.983 | 0.030 | 14.348 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | biohash_half | 50000 | 50000 | 128 | 0.063 | 0.063 | 0.998 | 0.061 | 14.348 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | biohash_half | 100000 | 100000 | 64 | 0.060 | 0.058 | 0.970 | 0.034 | 28.357 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | biohash_half | 100000 | 100000 | 128 | 0.060 | 0.059 | 0.993 | 0.064 | 28.357 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | iom_256 | 10000 | 10000 | 64 | 0.084 | 0.083 | 0.995 | 0.031 | 3.404 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | iom_256 | 10000 | 10000 | 128 | 0.084 | 0.084 | 1.000 | 0.063 | 3.404 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | iom_256 | 50000 | 50000 | 64 | 0.078 | 0.077 | 0.978 | 0.035 | 15.577 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | iom_256 | 50000 | 50000 | 128 | 0.078 | 0.078 | 0.997 | 0.068 | 15.577 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | iom_256 | 100000 | 100000 | 64 | 0.072 | 0.069 | 0.966 | 0.038 | 30.786 |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | iom_256 | 100000 | 100000 | 128 | 0.072 | 0.072 | 0.992 | 0.074 | 30.786 |

### 7c. Scale distractor spectrum check: participation ratio of distractors vs primary dev population

| run | distractor data source | participation ratio, spectrum_primary_dev | participation ratio, spectrum_distractors | relative difference |d - p| / p | differs by more than 30% |
|---|---|---|---|---|---|
| scale_d1_base_secondary-20261114-213920 | D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real) | 269.8 | 198.5 | 26.4% | no |
| scale_d1_base_synthetic-20261114-205007 | D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended) | 269.8 | 239.2 | 11.4% | no |

### 8. Skips

none

### 9. Proposition 2 check: rec_cos_z vs cstar (exact anchors, proj decoder, lrp_full)

| encoder | pre | m | r | rec_cos_z (proj) | cstar | abs(rec_cos_z - cstar) | excess rec_cos_z - cstar (pre none) | rec_cos_z (ridge, shows the intercept gap) |
|---|---|---|---|---|---|---|---|---|
| d1_base | pca128_a1 | 128 | 20 | 0.453 | 0.391 | 0.062 |  | 0.406 |
| d1_base | pca128_a1 | 128 | 40 | 0.633 | 0.557 | 0.076 |  | 0.616 |
| d1_base | pca128_a1 | 128 | 80 | 0.845 | 0.790 | 0.055 |  | 0.839 |
| d1_base | fi128_p512 | 128 | 20 | 0.421 | 0.391 | 0.030 |  | 0.373 |
| d1_base | fi128_p512 | 128 | 40 | 0.593 | 0.557 | 0.036 |  | 0.575 |
| d1_base | fi128_p512 | 128 | 80 | 0.817 | 0.790 | 0.027 |  | 0.811 |
| d1_base | none | 768 | 20 | 0.253 | 0.159 |  | 0.094 | 0.194 |
| d1_base | none | 768 | 40 | 0.359 | 0.227 |  | 0.133 | 0.331 |
| d1_base | none | 768 | 80 | 0.490 | 0.322 |  | 0.168 | 0.479 |
| d1_large | pca128_a1 | 128 | 20 | 0.436 | 0.391 | 0.045 |  | 0.387 |
| d1_large | pca128_a1 | 128 | 40 | 0.613 | 0.557 | 0.056 |  | 0.595 |
| d1_large | pca128_a1 | 128 | 80 | 0.831 | 0.790 | 0.041 |  | 0.825 |
| d1_large | fi128_p512 | 128 | 20 | 0.418 | 0.391 | 0.027 |  | 0.370 |
| d1_large | fi128_p512 | 128 | 40 | 0.587 | 0.557 | 0.030 |  | 0.569 |
| d1_large | fi128_p512 | 128 | 80 | 0.813 | 0.790 | 0.023 |  | 0.807 |
| d1_large | none | 1024 | 20 | 0.200 | 0.138 |  | 0.062 | 0.140 |
| d1_large | none | 1024 | 40 | 0.286 | 0.196 |  | 0.090 | 0.253 |
| d1_large | none | 1024 | 80 | 0.400 | 0.279 |  | 0.121 | 0.386 |
| d2_fingervit | pca128_a1 | 128 | 20 | 0.469 | 0.391 | 0.078 |  | 0.428 |
| d2_fingervit | pca128_a1 | 128 | 40 | 0.650 | 0.557 | 0.093 |  | 0.635 |
| d2_fingervit | pca128_a1 | 128 | 80 | 0.868 | 0.790 | 0.079 |  | 0.863 |
| d2_fingervit | fi128_p512 | 128 | 20 | 0.428 | 0.391 | 0.037 |  | 0.379 |
| d2_fingervit | fi128_p512 | 128 | 40 | 0.602 | 0.557 | 0.046 |  | 0.583 |
| d2_fingervit | fi128_p512 | 128 | 80 | 0.830 | 0.790 | 0.040 |  | 0.824 |
| d2_fingervit | none | 1536 | 20 | 0.836 | 0.113 |  | 0.723 | 0.819 |
| d2_fingervit | none | 1536 | 40 | 0.913 | 0.160 |  | 0.753 | 0.908 |
| d2_fingervit | none | 1536 | 80 | 0.953 | 0.228 |  | 0.725 | 0.951 |
| d2_deepprint | pca128_a1 | 128 | 20 | 0.429 | 0.391 | 0.038 |  | 0.383 |
| d2_deepprint | pca128_a1 | 128 | 40 | 0.601 | 0.557 | 0.044 |  | 0.582 |
| d2_deepprint | pca128_a1 | 128 | 80 | 0.827 | 0.790 | 0.037 |  | 0.821 |
| d2_deepprint | fi128_p384 | 128 | 20 | 0.413 | 0.391 | 0.022 |  | 0.364 |
| d2_deepprint | fi128_p384 | 128 | 40 | 0.583 | 0.557 | 0.026 |  | 0.563 |
| d2_deepprint | fi128_p384 | 128 | 80 | 0.811 | 0.790 | 0.021 |  | 0.804 |
| d2_deepprint | none | 512 | 20 | 0.737 | 0.195 |  | 0.542 | 0.714 |
| d2_deepprint | none | 512 | 40 | 0.859 | 0.278 |  | 0.581 | 0.851 |
| d2_deepprint | none | 512 | 80 | 0.949 | 0.394 |  | 0.554 | 0.947 |
| d2_ctrlvit | pca128_a1 | 128 | 20 | 0.413 | 0.391 | 0.022 |  | 0.359 |
| d2_ctrlvit | pca128_a1 | 128 | 40 | 0.584 | 0.557 | 0.028 |  | 0.564 |
| d2_ctrlvit | pca128_a1 | 128 | 80 | 0.813 | 0.790 | 0.023 |  | 0.806 |
| d2_ctrlvit | fi128_p512 | 128 | 20 | 0.412 | 0.391 | 0.021 |  | 0.362 |
| d2_ctrlvit | fi128_p512 | 128 | 40 | 0.580 | 0.557 | 0.024 |  | 0.560 |
| d2_ctrlvit | fi128_p512 | 128 | 80 | 0.810 | 0.790 | 0.020 |  | 0.803 |
| d2_ctrlvit | none | 768 | 20 | 0.324 | 0.159 |  | 0.164 | 0.267 |
| d2_ctrlvit | none | 768 | 40 | 0.437 | 0.227 |  | 0.210 | 0.412 |
| d2_ctrlvit | none | 768 | 80 | 0.575 | 0.322 |  | 0.253 | 0.566 |

The ridge column is shown to expose the intercept gap of the ridge-GCV decoder relative to the projection decoder; the proj decoder is the one compared with cstar.

### 10. Proposition 1 check: zero-pair on fi, rbig and firbig configurations

| encoder | pre | m | best_abs_cos_z [95% CI] | uniform E[abs cos] (m) | ratio measured/uniform | best_cos2_z [95% CI] | 1/m | ratio measured/(1/m) | best zero-pair E1 relative | zero-effort impostor level (target FMR / e1_genuine) | method | p (argmax of mean E1) | AUX |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1_base | fi64_p512 | 64 | 0.104 [0.098, 0.109] | 0.100 | 1.035 | 0.017 [0.015, 0.019] | 0.0156 | 1.096 | 0.005 [0.000, 0.019] | 0.0089 | ica | 64 | shifted |
| d1_base | fi128_p512 | 128 | 0.073 [0.070, 0.075] | 0.071 | 1.026 | 0.009 [0.008, 0.011] | 0.0078 | 1.184 | 0.010 [0.000, 0.028] | 0.0089 | ica | 128 | shifted |
| d1_base | fi256_p512 | 256 | 0.078 [0.066, 0.090] | 0.050 | 1.557 | 0.010 [0.007, 0.013] | 0.0039 | 2.519 | 0 successes; CP95 upper <= 0.024 | 0.0089 | n/a | n/a | n/a |
| d1_base | rbig128_i2 | 128 | 0.100 [0.088, 0.113] | 0.071 | 1.417 | 0.021 [0.014, 0.028] | 0.0078 | 2.664 | 0.011 [0.000, 0.029] | 0.0089 | ica | 128 | same_dist |
| d1_base | firbig128_p512_i2 | 128 | 0.072 [0.070, 0.074] | 0.071 | 1.020 | 0.008 [0.008, 0.009] | 0.0078 | 1.088 | 0.004 [0.000, 0.017] | 0.0089 | ica | 128 | shifted |
| d1_large | fi64_p512 | 64 | 0.105 [0.099, 0.111] | 0.100 | 1.045 | 0.018 [0.015, 0.021] | 0.0156 | 1.166 | 0.010 [0.000, 0.026] | 0.0094 | ica | 64 | shifted |
| d1_large | fi128_p512 | 128 | 0.072 [0.069, 0.075] | 0.071 | 1.015 | 0.009 [0.007, 0.010] | 0.0078 | 1.133 | 0.010 [0.000, 0.026] | 0.0094 | ica | 128 | shifted |
| d1_large | fi256_p512 | 256 | 0.049 [0.046, 0.052] | 0.050 | 0.981 | 0.004 [0.004, 0.005] | 0.0039 | 1.095 | 0.002 [0.000, 0.006] | 0.0094 | eigen | 8 | shifted |
| d1_large | rbig128_i2 | 128 | 0.080 [0.074, 0.086] | 0.071 | 1.137 | 0.014 [0.010, 0.018] | 0.0078 | 1.835 | 0.011 [0.000, 0.025] | 0.0094 | ica | 128 | same_dist |
| d1_large | firbig128_p512_i2 | 128 | 0.071 [0.070, 0.072] | 0.071 | 1.007 | 0.009 [0.008, 0.010] | 0.0078 | 1.135 | 0.010 [0.000, 0.026] | 0.0094 | ica | 128 | shifted |
| d2_fingervit | fi64_p512 | 64 | 0.139 [0.123, 0.155] | 0.100 | 1.387 | 0.036 [0.025, 0.047] | 0.0156 | 2.331 | 0.018 [0.000, 0.047] | 0.0082 | ica | 64 | same_dist |
| d2_fingervit | fi128_p512 | 128 | 0.088 [0.079, 0.097] | 0.071 | 1.241 | 0.015 [0.011, 0.018] | 0.0078 | 1.909 | 0.001 [0.000, 0.005] | 0.0082 | ica | 128 | same_dist |
| d2_fingervit | fi256_p512 | 256 | 0.055 [0.052, 0.057] | 0.050 | 1.095 | 0.005 [0.004, 0.006] | 0.0039 | 1.255 | 0 successes; CP95 upper <= 0.044 | 0.0082 | n/a | n/a | n/a |
| d2_fingervit | rbig128_i2 | 128 | 0.081 [0.074, 0.087] | 0.071 | 1.140 | 0.011 [0.009, 0.013] | 0.0078 | 1.437 | 0 successes; CP95 upper <= 0.044 | 0.0082 | n/a | n/a | n/a |
| d2_fingervit | firbig128_p512_i2 | 128 | 0.083 [0.072, 0.093] | 0.071 | 1.168 | 0.012 [0.008, 0.016] | 0.0078 | 1.483 | 0 successes; CP95 upper <= 0.044 | 0.0082 | n/a | n/a | n/a |
| d2_deepprint | fi64_p384 | 64 | 0.105 [0.099, 0.110] | 0.100 | 1.044 | 0.017 [0.015, 0.019] | 0.0156 | 1.093 | 0.001 [0.000, 0.004] | 0.0074 | ica | 64 | same_dist |
| d2_deepprint | fi128_p384 | 128 | 0.074 [0.071, 0.076] | 0.071 | 1.043 | 0.009 [0.008, 0.009] | 0.0078 | 1.093 | 0.001 [0.000, 0.005] | 0.0074 | wp | 128 | same_dist |
| d2_deepprint | fi256_p384 | 256 | 0.065 [0.064, 0.067] | 0.050 | 1.306 | 0.007 [0.006, 0.007] | 0.0039 | 1.700 | 0.001 [0.000, 0.004] | 0.0074 | eigen | 32 | same_dist |
| d2_deepprint | rbig128_i2 | 128 | 0.084 [0.078, 0.091] | 0.071 | 1.195 | 0.012 [0.010, 0.015] | 0.0078 | 1.594 | 0.004 [0.000, 0.014] | 0.0074 | ica | 128 | same_dist |
| d2_deepprint | firbig128_p384_i2 | 128 | 0.084 [0.079, 0.090] | 0.071 | 1.196 | 0.011 [0.010, 0.013] | 0.0078 | 1.428 | 0.001 [0.000, 0.005] | 0.0074 | eigen | 32 | same_dist |
| d2_ctrlvit | fi64_p512 | 64 | 0.119 [0.112, 0.125] | 0.100 | 1.184 | 0.023 [0.020, 0.026] | 0.0156 | 1.489 | 0.008 [0.001, 0.015] | 0.0012 | ica | 64 | same_dist |
| d2_ctrlvit | fi128_p512 | 128 | 0.079 [0.074, 0.084] | 0.071 | 1.124 | 0.011 [0.009, 0.012] | 0.0078 | 1.351 | 0.003 [0.000, 0.006] | 0.0012 | ica | 128 | same_dist |
| d2_ctrlvit | fi256_p512 | 256 | 0.068 [0.064, 0.072] | 0.050 | 1.361 | 0.007 [0.006, 0.008] | 0.0039 | 1.877 | 0 successes; CP95 upper <= 0.006 | 0.0012 | n/a | n/a | n/a |
| d2_ctrlvit | rbig128_i2 | 128 | 0.080 [0.077, 0.083] | 0.071 | 1.132 | 0.010 [0.009, 0.011] | 0.0078 | 1.327 | 0.002 [0.000, 0.006] | 0.0012 | ica | 128 | same_dist |
| d2_ctrlvit | firbig128_p512_i2 | 128 | 0.078 [0.073, 0.083] | 0.071 | 1.106 | 0.010 [0.008, 0.012] | 0.0078 | 1.315 | 0.003 [0.000, 0.007] | 0.0012 | ica | 128 | same_dist |

best_abs_cos_z and best_cos2_z come from the argmax-E1 row (selected with ground truth), so ratios above 1 include a selection effect. The chance level for abs cos is E[abs cos] of a uniform direction on the m-sphere and for cos^2 it is 1/m. Intervals on every metric other than E1/E2 relative success are Student-t intervals over the 5 seed means (4 df), mean +/- t(0.975, 4) * sd / sqrt(5): between-split variability over 5 overlapping splits; not an identity-level CI. Lower bounds of nonnegative quantities are truncated at 0; signed cosines are not truncated.

### 11. Anisotropy (pre none, proj decoder, exact anchors, lrp_full)

| encoder | m | r | rec_cos2_z (proj) [95% CI] | r/m | top-r explained-variance fraction | rec_cos2_z / (r/m) | rec_cos2_z / top-r variance fraction |
|---|---|---|---|---|---|---|---|
| d1_base | 768 | 5 | 0.020 [0.017, 0.023] | 0.007 | 0.051 | 3.105 | 0.394 |
| d1_base | 768 | 10 | 0.039 [0.036, 0.042] | 0.013 | 0.093 | 3.012 | 0.421 |
| d1_base | 768 | 20 | 0.075 [0.070, 0.079] | 0.026 | 0.162 | 2.864 | 0.460 |
| d1_base | 768 | 40 | 0.141 [0.133, 0.148] | 0.052 | 0.268 | 2.699 | 0.524 |
| d1_base | 768 | 80 | 0.251 [0.246, 0.255] | 0.104 | 0.423 | 2.406 | 0.592 |
| d1_base | 768 | 160 | 0.424 [0.420, 0.427] | 0.208 | 0.621 | 2.033 | 0.682 |
| d1_large | 1024 | 5 | 0.013 [0.011, 0.016] | 0.005 | 0.031 | 2.747 | 0.435 |
| d1_large | 1024 | 10 | 0.027 [0.022, 0.032] | 0.010 | 0.058 | 2.761 | 0.464 |
| d1_large | 1024 | 20 | 0.050 [0.044, 0.056] | 0.020 | 0.107 | 2.559 | 0.467 |
| d1_large | 1024 | 40 | 0.094 [0.088, 0.100] | 0.039 | 0.190 | 2.399 | 0.493 |
| d1_large | 1024 | 80 | 0.172 [0.166, 0.178] | 0.078 | 0.321 | 2.202 | 0.537 |
| d1_large | 1024 | 160 | 0.307 [0.303, 0.311] | 0.156 | 0.505 | 1.964 | 0.608 |
| d2_fingervit | 1536 | 5 | 0.356 [0.341, 0.371] | 0.003 | 0.520 | 109.266 | 0.684 |
| d2_fingervit | 1536 | 10 | 0.530 [0.505, 0.554] | 0.007 | 0.704 | 81.382 | 0.752 |
| d2_fingervit | 1536 | 20 | 0.708 [0.684, 0.733] | 0.013 | 0.847 | 54.398 | 0.837 |
| d2_fingervit | 1536 | 40 | 0.838 [0.826, 0.849] | 0.026 | 0.921 | 32.166 | 0.909 |
| d2_fingervit | 1536 | 80 | 0.909 [0.904, 0.914] | 0.052 | 0.957 | 17.460 | 0.950 |
| d2_fingervit | 1536 | 160 | 0.949 [0.946, 0.952] | 0.104 | 0.976 | 9.108 | 0.972 |
| d2_deepprint | 512 | 5 | 0.218 [0.187, 0.249] | 0.010 | 0.407 | 22.323 | 0.535 |
| d2_deepprint | 512 | 10 | 0.362 [0.344, 0.380] | 0.020 | 0.566 | 18.557 | 0.641 |
| d2_deepprint | 512 | 20 | 0.550 [0.534, 0.566] | 0.039 | 0.724 | 14.083 | 0.759 |
| d2_deepprint | 512 | 40 | 0.740 [0.735, 0.746] | 0.078 | 0.866 | 9.473 | 0.855 |
| d2_deepprint | 512 | 80 | 0.901 [0.898, 0.903] | 0.156 | 0.962 | 5.763 | 0.936 |
| d2_deepprint | 512 | 160 | 0.981 [0.980, 0.982] | 0.312 | 0.995 | 3.139 | 0.986 |
| d2_ctrlvit | 768 | 5 | 0.030 [0.027, 0.032] | 0.007 | 0.098 | 4.537 | 0.302 |
| d2_ctrlvit | 768 | 10 | 0.058 [0.055, 0.060] | 0.013 | 0.165 | 4.442 | 0.351 |
| d2_ctrlvit | 768 | 20 | 0.110 [0.106, 0.114] | 0.026 | 0.269 | 4.231 | 0.409 |
| d2_ctrlvit | 768 | 40 | 0.196 [0.191, 0.200] | 0.052 | 0.410 | 3.758 | 0.477 |
| d2_ctrlvit | 768 | 80 | 0.334 [0.329, 0.339] | 0.104 | 0.571 | 3.207 | 0.585 |
| d2_ctrlvit | 768 | 160 | 0.519 [0.515, 0.524] | 0.208 | 0.729 | 2.493 | 0.712 |

rec_cos2_z against r/m (the isotropic value) and against the top-r explained-variance fraction of the centered, normalized embeddings (`spectra.json`, preconditioned.none.cumvar_first256).

### 12. FMR 1% operating point (attack threshold at FMR 1%; separate runs, summary_fmr1)

| encoder | pre | r50 E1 (lrp_full, exact, best decoder) [per-seed range] | r50 E2 (lrp_full, exact, best decoder) [per-seed range] | zero-pair best E1 (common methods eigen, wp) | key-known T2 E1, lrp_full | key-known T2 E2, lrp_full | key-known T2 E1, biohash_half | key-known T2 E2, biohash_half | zero-effort impostor level (target FMR / e1_genuine) |
|---|---|---|---|---|---|---|---|---|---|
| d1_base_fmr1 | none | 37.2 (proj) [35.5-41.2] | 37.2 (proj) [35.5-41.2] | 0.236 [0.172, 0.299] | 1.000 [1.000, 1.000] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.900 [0.868, 0.931] (learned) | 0.889 [0.846, 0.932] (learned) | 0.0357 |
| d1_base_fmr1 | pca128_a1 | 25.7 (proj) [24.4-27.1] | 17.4 (proj) [16.2-19.0] | 0.178 [0.130, 0.226] | 0.917 [0.883, 0.951] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.684 [0.640, 0.728] (learned) | 0.514 [0.454, 0.574] (learned) | 0.0357 |
| d1_base_fmr1 | fi128_p512 | 80.0 (ridge) [69.2-99.9] | 19.2 (proj) [17.2-20.7] | 0.059 [0.039, 0.079] | 0.670 [0.615, 0.725] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.284 [0.236, 0.332] (learned) | 0.545 [0.457, 0.634] (learned) | 0.0357 |
| d1_large_fmr1 | none | 40.5 (proj) [37.9-48.3] | 40.5 (proj) [37.9-48.3] | 0.265 [0.212, 0.317] | 1.000 [1.000, 1.000] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.846 [0.809, 0.883] (learned) | 0.821 [0.768, 0.874] (learned) | 0.0332 |
| d1_large_fmr1 | pca128_a1 | 26.4 (proj) [21.9-29.0] | 14.4 (proj) [13.5-14.8] | 0.216 [0.161, 0.272] | 0.766 [0.727, 0.805] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.615 [0.569, 0.660] (learned) | 0.482 [0.416, 0.548] (learned) | 0.0332 |
| d1_large_fmr1 | fi128_p512 | 45.8 (ridge) [37.4-52.5] | 20.4 (proj) [19.4-22.1] | 0.090 [0.059, 0.121] | 0.773 [0.714, 0.832] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.463 [0.417, 0.508] (learned) | 0.521 [0.452, 0.589] (learned) | 0.0332 |
| d2_fingervit_fmr1 | none | 23.2 (proj) [19.3-28.6] | 23.2 (proj) [19.3-28.6] | 0.248 [0.132, 0.364] | 1.001 [0.999, 1.004] (learned) | 1.001 [0.999, 1.004] (learned) | 0.979 [0.952, 1.006] (learned) | 0.977 [0.942, 1.013] (learned) | 0.0427 |
| d2_fingervit_fmr1 | pca128_a1 | 78.9 (proj) [64.8-90.9] | 30.3 (proj) [28.7-32.1] | 0.008 [0.000, 0.019] | 1.003 [0.989, 1.018] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.089 [0.044, 0.133] (analytic) | 0.401 [0.341, 0.462] (learned) | 0.0427 |
| d2_fingervit_fmr1 | fi128_p512 | 104.2 (proj) [89.7-122.4] | 25.7 (proj) [23.9-27.8] | 0.002 [0.000, 0.004] | 0.825 [0.712, 0.938] (analytic) | 1.000 [1.000, 1.001] (analytic) | 0.096 [0.016, 0.176] (learned) | 0.422 [0.376, 0.467] (learned) | 0.0427 |
| d2_deepprint_fmr1 | none | 13.0 (proj) [11.0-14.2] | 13.0 (proj) [11.0-14.2] | 0.065 [0.032, 0.098] | 1.000 [1.000, 1.000] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.985 [0.946, 1.023] (learned) | 0.998 [0.947, 1.050] (learned) | 0.0447 |
| d2_deepprint_fmr1 | pca128_a1 | 49.0 (ridge) [46.4-53.3] | 22.9 (proj) [19.2-26.3] | 0.016 [0.000, 0.032] | 1.003 [0.995, 1.011] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.308 [0.242, 0.375] (learned) | 0.400 [0.318, 0.482] (learned) | 0.0447 |
| d2_deepprint_fmr1 | fi128_p384 | 76.5 (proj) [63.9-88.9] | 25.2 (proj) [23.1-27.4] | 0.025 [0.006, 0.045] | 0.869 [0.795, 0.942] (analytic) | 1.001 [0.999, 1.002] (analytic) | 0.192 [0.138, 0.247] (learned) | 0.381 [0.292, 0.469] (learned) | 0.0447 |
| d2_ctrlvit_fmr1 | none | 25.3 (proj) [21.9-27.8] | 25.3 (proj) [21.9-27.8] | 0.137 [0.073, 0.201] | 1.000 [1.000, 1.000] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.985 [0.976, 0.995] (learned) | 0.965 [0.950, 0.981] (learned) | 0.0104 |
| d2_ctrlvit_fmr1 | pca128_a1 | 26.2 (proj) [24.1-28.2] | 16.8 (proj) [15.3-18.4] | 0.014 [0.005, 0.024] | 1.005 [0.999, 1.011] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.730 [0.697, 0.763] (learned) | 0.409 [0.373, 0.445] (learned) | 0.0104 |
| d2_ctrlvit_fmr1 | fi128_p512 | 34.7 (proj) [30.3-39.5] | 17.8 (proj) [16.4-19.4] | 0.018 [0.009, 0.027] | 0.979 [0.967, 0.990] (analytic) | 1.000 [1.000, 1.000] (analytic) | 0.585 [0.542, 0.629] (learned) | 0.391 [0.353, 0.429] (learned) | 0.0104 |

Encoders carry the suffix `_fmr1`. The E1 relative zero-effort impostor level (target FMR / e1_genuine) at FMR 1% (d1_base_fmr1: 0.0357, d1_large_fmr1: 0.0332, d2_fingervit_fmr1: 0.0427, d2_deepprint_fmr1: 0.0447, d2_ctrlvit_fmr1: 0.0104). Observed values can fall below it, because a reconstructed proxy is not a random impostor capture. Intervals on every metric other than E1/E2 relative success are Student-t intervals over the 5 seed means (4 df), mean +/- t(0.975, 4) * sd / sqrt(5): between-split variability over 5 overlapping splits; not an identity-level CI. Lower bounds of nonnegative quantities are truncated at 0; signed cosines are not truncated. E1/E2 relative success carries the wider of two 95% intervals, elementwise: the identity-level interval (the runner's per-row paired-bootstrap standard error, resampling targets, see `src/ijiss_btp/eval/stats.py:63-72`, averaged over keys and seeds; mean +/- 1.96 * SE_bar, which treats the overlapping splits as fully correlated) and the seed-level Student-t interval over the 5 seed means (4 df). The reported interval is their union: lower = min of the two lower bounds, upper = max of the two upper bounds; lower bounds are truncated at 0.

### 12b. FMR 1% operating point: trials, genuine acceptance and zero-effort impostor level

| encoder | target FMR (attack threshold) | E1 trials per zero-pair/T2 row | E1 trials per anchor row (min-max) | e1_genuine: raw-space genuine acceptance at the DEV threshold, mean over zero-pair rows | zero-effort impostor level (target FMR / e1_genuine) | measured FMR (%): target-split impostor pairs at the DEV threshold (fmr_actual, lrp_full, pre none, mean over seeds) | measured zero-effort level (measured FMR / e1_genuine) |
|---|---|---|---|---|---|---|---|
| d1_base_fmr1 | 0.01 | 1200 | 560-560 | 0.280 (sd 0.006, n = 7350 rows) | 0.0357 | 1.479 | 0.0528 |
| d1_large_fmr1 | 0.01 | 1200 | 560-560 | 0.301 (sd 0.004, n = 7350 rows) | 0.0332 | 1.442 | 0.0479 |
| d2_fingervit_fmr1 | 0.01 | 600 | 280-280 | 0.234 (sd 0.015, n = 3675 rows) | 0.0427 | 0.725 | 0.0310 |
| d2_deepprint_fmr1 | 0.01 | 600 | 280-280 | 0.224 (sd 0.016, n = 3675 rows) | 0.0447 | 0.990 | 0.0443 |
| d2_ctrlvit_fmr1 | 0.01 | 600 | 280-280 | 0.964 (sd 0.003, n = 3675 rows) | 0.0104 | 0.882 | 0.0091 |

### 12c. FMR 1% operating point: sign-resolution reference (eigen_oracle, pre none, lrp_full)

| encoder | max E1 relative of eigen_oracle over AUX and p (pre none) | at p, AUX | absolute E1 acceptance at that entry (mean e1_proxy) | max rec_cos_z (pre-key) of eigen_oracle | at p, AUX | oracle gap (eigen_oracle - eigen E1) at the key-free eigen selection | selected p, AUX |
|---|---|---|---|---|---|---|---|
| d1_base_fmr1 | 0.333 | 8, same_dist | 0.093 | 0.090 | 32, same_dist | 0.097 | 8, same_dist |
| d1_large_fmr1 | 0.333 | 8, same_dist | 0.100 | 0.056 | 128, same_dist | 0.068 | 8, same_dist |
| d2_fingervit_fmr1 | 0.012 | 8, same_dist | 0.003 | 0.526 | 8, same_dist | -0.026 | 8, same_dist |
| d2_deepprint_fmr1 | 0.160 | 8, same_dist | 0.036 | 0.327 | 64, same_dist | 0.095 | 8, same_dist |
| d2_ctrlvit_fmr1 | 0.211 | 16, same_dist | 0.203 | 0.183 | 16, same_dist | 0.063 | 8, same_dist |

`eigen_oracle` is eigen alignment of the stolen templates with AUX (leading p principal axes matched by rank, variances rescaled) whose per-component signs come from the key: alignment followed by exact sign resolution, which an attacker could at best obtain from anchors. It is a diagnostic, never a key-free attack. Values are means over keys and seeds; the maximum over AUX type and p (grid 8 to 128) is selected with ground truth and has no interval. A maximum with zero successes in every row gives the largest relative one-sided 95% Clopper-Pearson bound. The joint alignment-plus-anchor attack, with anchor-voted signs, is in sections 13 and 13b and in the paper's Table S5.

### 12d. FMR 1% operating point: sign-resolution reference after whitening (eigen_oracle, pca128, pca256, fi64, fi128, fi256, lrp_full)

| encoder | pre | max E1 relative of eigen_oracle over AUX and p | max E1 absolute (mean e1_proxy) | max E2 relative | max E2 absolute (mean e2_proxy) | max rec_cos_z (pre-key) |
|---|---|---|---|---|---|---|
| d1_base_fmr1 | pca128 | 0.244 | 0.068 | 0.270 | 0.073 | 0.123 |
| d1_base_fmr1 | pca256 | 0.210 | 0.059 | 0.301 | 0.100 | 0.088 |
| d1_base_fmr1 | fi64 | 0.050 | 0.014 | 0.220 | 0.060 | 0.191 |
| d1_base_fmr1 | fi128 | 0.057 | 0.016 | 0.240 | 0.079 | 0.124 |
| d1_base_fmr1 | fi256 | 0.071 | 0.020 | 0.297 | 0.113 | 0.081 |
| d1_large_fmr1 | pca128 | 0.257 | 0.077 | 0.296 | 0.066 | 0.110 |
| d1_large_fmr1 | pca256 | 0.240 | 0.072 | 0.316 | 0.094 | 0.080 |
| d1_large_fmr1 | fi64 | 0.098 | 0.030 | 0.203 | 0.047 | 0.159 |
| d1_large_fmr1 | fi128 | 0.108 | 0.032 | 0.230 | 0.065 | 0.117 |
| d1_large_fmr1 | fi256 | 0.140 | 0.042 | 0.274 | 0.094 | 0.077 |
| d2_fingervit_fmr1 | pca128 | 0.001 | <0.001 | 0.029 | 0.015 | 0.085 |
| d2_fingervit_fmr1 | pca256 | 0 successes; CP95 upper <= 0.023 | 0.000 | 0.041 | 0.027 | 0.061 |
| d2_fingervit_fmr1 | fi64 | 0.019 | 0.005 | 0.027 | 0.022 | 0.152 |
| d2_fingervit_fmr1 | fi128 | 0.002 | 0.001 | 0.035 | 0.029 | 0.100 |
| d2_fingervit_fmr1 | fi256 | 0.001 | <0.001 | 0.070 | 0.059 | 0.069 |
| d2_deepprint_fmr1 | pca128 | 0.023 | 0.005 | 0.063 | 0.016 | 0.091 |
| d2_deepprint_fmr1 | pca256 | 0.015 | 0.003 | 0.107 | 0.028 | 0.065 |
| d2_deepprint_fmr1 | fi64 | 0.040 | 0.009 | 0.078 | 0.021 | 0.142 |
| d2_deepprint_fmr1 | fi128 | 0.025 | 0.005 | 0.102 | 0.028 | 0.089 |
| d2_deepprint_fmr1 | fi256 | 0.019 | 0.004 | 0.135 | 0.037 | 0.060 |
| d2_ctrlvit_fmr1 | pca128 | 0.025 | 0.024 | 0.053 | 0.052 | 0.105 |
| d2_ctrlvit_fmr1 | pca256 | 0.016 | 0.016 | 0.085 | 0.083 | 0.070 |
| d2_ctrlvit_fmr1 | fi64 | 0.041 | 0.039 | 0.034 | 0.031 | 0.148 |
| d2_ctrlvit_fmr1 | fi128 | 0.022 | 0.021 | 0.052 | 0.050 | 0.102 |
| d2_ctrlvit_fmr1 | fi256 | 0.013 | 0.013 | 0.103 | 0.101 | 0.066 |

`eigen_oracle` is eigen alignment of the stolen templates with AUX (leading p principal axes matched by rank, variances rescaled) whose per-component signs come from the key: alignment followed by exact sign resolution, which an attacker could at best obtain from anchors. It is a diagnostic, never a key-free attack. Values are means over keys and seeds; the maximum over AUX type and p (grid 8 to 128) is selected with ground truth and has no interval. A maximum with zero successes in every row gives the largest relative one-sided 95% Clopper-Pearson bound. The joint alignment-plus-anchor attack, with anchor-voted signs, is in sections 13 and 13b and in the paper's Table S5. After whitening, E1 maps the proxy back with the public back-map and E2 re-protects it with the renewed key (the round trip of the protocol); each maximum (relative, absolute, cosine) is taken on its own and can come from a different entry.

### 13. Joint alignment-plus-anchor attack (T1, lrp_full, exact anchors; FMR 0.1%)

| encoder | pre | m | E2 r50 proj | E2 r50 best attack [per-seed range] | change (best - proj) | E2 r50 J1 (align_anchor) | E2 r50 J2 (joint) | E1 r50 proj | E1 r50 best of proj/J1/J2 | best E2 at r = 1: relative / absolute (cosine, method) | at r = 2 | at r = 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1_base | none | 768 | 211.3 | 211.3 [197.4-228.9] | 0.0 | >= 640 | 215.8 | 211.3 | 211.3 | 0 successes; CP95 upper <= 0.056 | 0.005 / 0.001 (cos 0.067, proj) | 0.010 / 0.001 (cos 0.118, proj) |
| d1_base | pca128 | 128 | 67.2 | 67.2 [63.4-71.6] | 0.0 | >= 640 | 69.5 | >= 640 | >= 640 | 0 successes; CP95 upper <= 0.067 | 0.005 / <0.001 (cos 0.124, proj) | 0.010 / 0.001 (cos 0.217, proj) |
| d1_base | pca256 | 256 | 87.7 | 87.7 [81.5-96.0] | 0.0 | >= 640 | 92.8 | 217.0 | 217.0 | 0.002 / <0.001 (cos 0.056, proj) | 0.010 / 0.001 (cos 0.088, proj) | 0.017 / 0.002 (cos 0.154, proj) |
| d1_base | fi64 | 64 | 27.5 | 27.5 [27.2-27.7] | 0.0 | >= 640 | 28.4 | >= 640 | >= 640 | 0.014 / 0.002 (cos 0.076, align_anchor) | 0.023 / 0.003 (cos 0.153, joint) | 0.037 / 0.005 (cos 0.262, joint) |
| d1_base | fi128 | 128 | 40.9 | 40.9 [39.1-43.7] | 0.0 | >= 640 | 44.9 | >= 640 | >= 640 | 0.014 / 0.002 (cos 0.059, joint) | 0.014 / 0.002 (cos 0.079, joint) | 0.024 / 0.004 (cos 0.167, joint) |
| d1_base | fi256 | 256 | 58.6 | 58.6 [57.8-59.6] | 0.0 | >= 640 | 63.5 | >= 640 | >= 640 | 0.013 / 0.002 (cos 0.033, joint) | 0.017 / 0.003 (cos 0.058, joint) | 0.022 / 0.004 (cos 0.116, joint) |
| d1_large | none | 1024 | 233.4 | 233.4 [224.2-243.6] | 0.0 | >= 640 | 236.9 | 233.4 | 233.4 | 0.002 / <0.001 (cos 0.032, proj) | 0.003 / <0.001 (cos 0.052, proj) | 0.016 / 0.002 (cos 0.092, proj) |
| d1_large | pca128 | 128 | 84.5 | 84.5 [78.3-94.5] | 0.0 | >= 640 | 86.1 | >= 640 | >= 640 | 0 successes; CP95 upper <= 0.087 | 0.001 / <0.001 (cos 0.033, align_anchor) | 0.010 / 0.001 (cos 0.208, proj) |
| d1_large | pca256 | 256 | 102.6 | 102.5 [98.2-108.4] | -0.1 | >= 640 | 103.3 | 249.2 | 249.2 | 0.005 / 0.001 (cos 0.054, proj) | 0.008 / 0.001 (cos 0.085, proj) | 0.020 / 0.002 (cos 0.150, proj) |
| d1_large | fi64 | 64 | 27.6 | 27.6 [26.9-28.6] | 0.0 | >= 640 | 29.5 | >= 640 | >= 640 | 0.014 / 0.001 (cos 0.051, align_anchor) | 0.013 / 0.001 (cos 0.063, align_anchor) | 0.024 / 0.003 (cos 0.209, joint) |
| d1_large | fi128 | 128 | 41.6 | 41.6 [38.9-44.5] | 0.0 | >= 640 | 45.4 | >= 640 | >= 640 | 0.005 / 0.001 (cos 0.076, proj) | 0.008 / 0.001 (cos 0.070, joint) | 0.014 / 0.002 (cos 0.146, joint) |
| d1_large | fi256 | 256 | 59.8 | 59.8 [53.7-62.6] | 0.0 | >= 640 | 62.0 | >= 640 | >= 640 | 0.010 / 0.001 (cos 0.029, joint) | 0.010 / 0.001 (cos 0.054, joint) | 0.020 / 0.003 (cos 0.149, proj) |
| d2_fingervit | none | 1536 | 53.0 | 53.0 [42.8-69.7] | 0.0 | >= 320 | 79.4 | 53.0 | 53.0 | 0.002 / <0.001 (cos 0.253, proj) | 0.003 / <0.001 (cos 0.370, proj) | 0.034 / 0.004 (cos 0.570, proj) |
| d2_fingervit | pca128 | 128 | 55.3 | 55.3 [53.5-56.4] | 0.0 | >= 320 | 58.8 | 106.9 | 106.9 | 0.005 / 0.001 (cos 0.045, joint) | 0.004 / 0.001 (cos 0.072, joint) | 0.010 / 0.002 (cos 0.226, proj) |
| d2_fingervit | pca256 | 256 | 79.4 | 79.4 [73.7-85.4] | 0.0 | >= 320 | 83.6 | 208.0 | 208.0 | 0.009 / 0.003 (cos 0.037, joint) | 0.008 / 0.003 (cos 0.053, joint) | 0.010 / 0.003 (cos 0.163, proj) |
| d2_fingervit | fi64 | 64 | 40.1 | 40.1 [37.4-43.6] | 0.0 | >= 320 | 44.1 | >= 320 | >= 320 | 0.002 / 0.001 (cos 0.059, joint) | 0.002 / 0.001 (cos 0.168, proj) | 0.008 / 0.003 (cos 0.293, proj) |
| d2_fingervit | fi128 | 128 | 58.8 | 58.8 [53.8-60.8] | 0.0 | >= 320 | 62.8 | 145.4 (> m: upper estimate) | 145.4 (> m: upper estimate) | 0.004 / 0.002 (cos 0.031, joint) | 0.005 / 0.002 (cos 0.066, joint) | 0.009 / 0.005 (cos 0.205, proj) |
| d2_fingervit | fi256 | 256 | 80.2 | 80.2 [71.6-86.7] | 0.0 | >= 320 | 85.0 | 270.2 (> m: upper estimate) | 270.2 (> m: upper estimate) | 0.007 / 0.004 (cos 0.027, joint) | 0.012 / 0.007 (cos 0.050, joint) | 0.017 / 0.009 (cos 0.101, joint) |
| d2_deepprint | none | 512 | 27.9 | 27.9 [27.2-29.7] | 0.0 | >= 320 | 32.6 | 27.9 | 27.9 | 0.019 / 0.003 (cos 0.191, align_anchor) | 0.029 / 0.004 (cos 0.303, joint) | 0.031 / 0.004 (cos 0.386, joint) |
| d2_deepprint | pca128 | 128 | 48.2 | 48.2 [46.3-50.3] | 0.0 | >= 320 | 51.6 | 86.5 | 86.5 | 0.007 / 0.001 (cos 0.023, align_anchor) | 0.006 / 0.001 (cos 0.120, proj) | 0.009 / 0.002 (cos 0.205, proj) |
| d2_deepprint | pca256 | 256 | 68.9 | 68.9 [66.7-71.5] | 0.0 | >= 320 | 72.6 | 172.9 | 172.9 | 0.008 / 0.001 (cos 0.056, proj) | 0.007 / 0.001 (cos 0.088, proj) | 0.013 / 0.002 (cos 0.103, joint) |
| d2_deepprint | fi64 | 64 | 30.4 | 30.4 [28.8-32.8] | 0.0 | >= 320 | 32.6 | >= 320 | >= 320 | 0.007 / 0.001 (cos 0.026, align_anchor) | 0.009 / 0.001 (cos 0.088, joint) | 0.017 / 0.003 (cos 0.202, joint) |
| d2_deepprint | fi128 | 128 | 44.3 | 44.3 [42.3-46.5] | 0.0 | >= 320 | 48.3 | >= 320 | >= 320 | 0.008 / 0.002 (cos 0.074, proj) | 0.009 / 0.002 (cos 0.115, proj) | 0.013 / 0.002 (cos 0.128, joint) |
| d2_deepprint | fi256 | 256 | 61.7 | 61.7 [60.1-65.3] | 0.0 | >= 320 | 67.0 | 231.7 | 231.7 | 0.012 / 0.002 (cos 0.006, align_anchor) | 0.010 / 0.002 (cos 0.009, align_anchor) | 0.013 / 0.003 (cos 0.146, proj) |
| d2_ctrlvit | none | 768 | 87.2 | 83.8 [72.0-91.0] | -3.4 | >= 320 | 83.8 | 87.2 | 83.8 | 0.019 / 0.015 (cos 0.099, joint) | 0.022 / 0.018 (cos 0.132, joint) | 0.030 / 0.025 (cos 0.200, joint) |
| d2_ctrlvit | pca128 | 128 | 44.8 | 44.8 [41.8-48.0] | 0.0 | >= 320 | 49.5 | 65.6 | 65.6 | 0.002 / 0.002 (cos 0.073, proj) | 0.003 / 0.003 (cos 0.116, proj) | 0.007 / 0.006 (cos 0.198, proj) |
| d2_ctrlvit | pca256 | 256 | 51.0 | 51.0 [46.5-53.5] | 0.0 | >= 320 | 56.3 | 118.8 | 118.8 | 0.004 / 0.003 (cos 0.028, joint) | 0.007 / 0.006 (cos 0.046, joint) | 0.013 / 0.012 (cos 0.145, proj) |
| d2_ctrlvit | fi64 | 64 | 37.2 | 37.2 [35.2-40.9] | 0.0 | >= 320 | 40.5 | 75.5 (> m: upper estimate) | 75.5 (> m: upper estimate) | 0.001 / <0.001 (cos 0.047, joint) | 0.002 / 0.001 (cos 0.162, proj) | 0.003 / 0.002 (cos 0.276, proj) |
| d2_ctrlvit | fi128 | 128 | 51.4 | 51.4 [48.1-55.1] | 0.0 | >= 320 | 55.0 | 100.1 | 100.1 | 0.001 / 0.001 (cos 0.073, proj) | 0.002 / 0.002 (cos 0.117, proj) | 0.005 / 0.004 (cos 0.198, proj) |
| d2_ctrlvit | fi256 | 256 | 51.2 | 51.2 [47.3-54.2] | 0.0 | >= 320 | 56.9 | 150.6 | 150.6 | 0.006 / 0.005 (cos 0.030, joint) | 0.006 / 0.005 (cos 0.045, joint) | 0.015 / 0.014 (cos 0.093, joint) |

Joint alignment-plus-anchor stage (results/joint/, real embeddings; same seeds, splits, keys, thresholds, colluders and victims as the anchor stage; parity of proj with anchor.csv is exact, results/joint/PARITY.md). proj is the anchor-only projection decoder; J1 (align_anchor) aligns the stolen templates with AUX and takes the axis signs from the anchors; J2 (joint) adds J1 outside the anchors' span to the projection decoder's exact component. The best evaluated attack is, per r, the largest seed-mean relative success (mean over keys within a seed, then over seeds) over proj, J1 and J2 and their AUX and p settings: an upper envelope selected with ground truth; the true-sign reference (eigen_oracle) is not a candidate, and ties go to proj. r50 is interpolated log-linearly on that curve; a value above m is an upper estimate (with m exact anchors the projection decoder recovers z). The per-seed range takes, per seed, the smallest r50 over the settings, so the seed-mean envelope's r50 can lie above that range. E1 compares proj, J1 and J2 only (the anchor stage's ridge and mlp are not candidates, so section 12 can show a smaller E1 r50). A zero count at r = 1, 2 or 5 gives the largest relative one-sided 95% Clopper-Pearson bound over the rows of that r (a descriptive binomial reference limit). Provenance of the ten joint runs: results/joint/*/metadata.json (seed, configuration, data source, environment, git commit) and results/joint/PARITY.md.

### 13b. Joint alignment-plus-anchor attack at the FMR 1% operating point

| encoder | pre | m | E2 r50 proj | E2 r50 best attack [per-seed range] | change (best - proj) | E2 r50 J1 (align_anchor) | E2 r50 J2 (joint) | E1 r50 proj | E1 r50 best of proj/J1/J2 | best E2 at r = 1: relative / absolute (cosine, method) | at r = 2 | at r = 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1_base_fmr1 | none | 768 | 37.2 | 27.3 [22.7-36.4] | -9.9 | >= 640 | 27.3 | 37.2 | 27.3 | 0.269 / 0.075 (cos 0.049, joint) | 0.281 / 0.078 (cos 0.067, joint) | 0.319 / 0.088 (cos 0.111, joint) |
| d1_base_fmr1 | pca128 | 128 | 17.4 | 16.0 [12.5-19.0] | -1.4 | >= 640 | 16.0 | 25.7 | 24.6 | 0.220 / 0.059 (cos 0.061, joint) | 0.225 / 0.061 (cos 0.089, joint) | 0.275 / 0.074 (cos 0.162, joint) |
| d1_base_fmr1 | pca256 | 256 | 21.2 | 16.8 [13.1-20.4] | -4.3 | >= 640 | 16.8 | 38.8 | 37.8 | 0.249 / 0.083 (cos 0.043, joint) | 0.261 / 0.086 (cos 0.064, joint) | 0.304 / 0.101 (cos 0.118, joint) |
| d1_base_fmr1 | fi64 | 64 | 15.9 | 14.7 [11.7-16.1] | -1.1 | >= 640 | 14.7 | >= 640 | >= 640 | 0.173 / 0.048 (cos 0.104, joint) | 0.202 / 0.055 (cos 0.153, joint) | 0.276 / 0.075 (cos 0.262, joint) |
| d1_base_fmr1 | fi128 | 128 | 19.2 | 19.0 [13.9-20.7] | -0.2 | >= 640 | 19.0 | 80.6 | 80.6 | 0.189 / 0.061 (cos 0.059, joint) | 0.204 / 0.066 (cos 0.079, joint) | 0.279 / 0.090 (cos 0.167, joint) |
| d1_base_fmr1 | fi256 | 256 | 22.3 | 20.7 [15.7-24.6] | -1.6 | >= 640 | 20.7 | 84.1 | 83.9 | 0.233 / 0.088 (cos 0.033, joint) | 0.263 / 0.099 (cos 0.058, joint) | 0.314 / 0.118 (cos 0.116, joint) |
| d1_large_fmr1 | none | 1024 | 40.5 | 28.8 [24.4-33.9] | -11.7 | >= 640 | 28.8 | 40.5 | 28.8 | 0.294 / 0.088 (cos 0.012, align_anchor) | 0.295 / 0.089 (cos 0.016, align_anchor) | 0.316 / 0.095 (cos 0.071, joint) |
| d1_large_fmr1 | pca128 | 128 | 14.4 | 12.9 [10.1-14.5] | -1.5 | >= 640 | 12.9 | 26.4 | 25.5 | 0.233 / 0.052 (cos 0.051, joint) | 0.260 / 0.058 (cos 0.078, joint) | 0.327 / 0.073 (cos 0.156, joint) |
| d1_large_fmr1 | pca256 | 256 | 19.7 | 14.3 [10.6-18.6] | -5.4 | >= 640 | 14.3 | 34.3 | 28.9 | 0.293 / 0.086 (cos 0.038, joint) | 0.322 / 0.095 (cos 0.063, joint) | 0.342 / 0.101 (cos 0.113, joint) |
| d1_large_fmr1 | fi64 | 64 | 14.9 | 14.9 [14.1-14.8] | 0.0 | >= 640 | 15.7 | 45.1 | 45.1 | 0.182 / 0.042 (cos 0.051, align_anchor) | 0.207 / 0.048 (cos 0.120, joint) | 0.236 / 0.055 (cos 0.209, joint) |
| d1_large_fmr1 | fi128 | 128 | 20.4 | 20.4 [15.5-22.1] | 0.0 | >= 640 | 20.5 | 48.1 | 48.0 | 0.189 / 0.054 (cos 0.051, joint) | 0.190 / 0.054 (cos 0.070, joint) | 0.241 / 0.069 (cos 0.146, joint) |
| d1_large_fmr1 | fi256 | 256 | 24.8 | 22.1 [16.2-25.0] | -2.7 | >= 640 | 22.1 | 53.2 | 50.1 | 0.245 / 0.085 (cos 0.029, joint) | 0.267 / 0.092 (cos 0.054, joint) | 0.286 / 0.099 (cos 0.105, joint) |
| d2_fingervit_fmr1 | none | 1536 | 23.2 | 23.2 [19.3-28.6] | 0.0 | >= 320 | 25.8 | 23.2 | 23.2 | 0.021 / 0.005 (cos 0.253, proj) | 0.037 / 0.009 (cos 0.501, joint) | 0.090 / 0.022 (cos 0.570, proj) |
| d2_fingervit_fmr1 | pca128 | 128 | 30.3 | 30.3 [28.7-32.1] | 0.0 | >= 320 | 33.9 | 78.9 | 78.9 | 0.023 / 0.012 (cos 0.045, joint) | 0.027 / 0.015 (cos 0.130, proj) | 0.071 / 0.039 (cos 0.226, proj) |
| d2_fingervit_fmr1 | pca256 | 256 | 41.2 | 41.2 [38.4-44.5] | 0.0 | >= 320 | 44.7 | 157.8 | 157.8 | 0.032 / 0.021 (cos 0.037, joint) | 0.047 / 0.030 (cos 0.053, joint) | 0.073 / 0.048 (cos 0.163, proj) |
| d2_fingervit_fmr1 | fi64 | 64 | 21.5 | 21.5 [20.7-22.1] | 0.0 | >= 320 | 24.9 | 56.0 | 56.0 | 0.016 / 0.013 (cos 0.059, joint) | 0.023 / 0.019 (cos 0.168, proj) | 0.071 / 0.059 (cos 0.293, proj) |
| d2_fingervit_fmr1 | fi128 | 128 | 25.7 | 25.7 [23.9-27.8] | 0.0 | >= 320 | 29.8 | 104.2 | 104.2 | 0.028 / 0.024 (cos 0.031, joint) | 0.039 / 0.033 (cos 0.066, joint) | 0.083 / 0.071 (cos 0.205, proj) |
| d2_fingervit_fmr1 | fi256 | 256 | 32.6 | 32.6 [27.5-37.9] | 0.0 | >= 320 | 37.4 | 199.2 | 199.2 | 0.051 / 0.042 (cos 0.027, joint) | 0.066 / 0.055 (cos 0.050, joint) | 0.100 / 0.084 (cos 0.150, proj) |
| d2_deepprint_fmr1 | none | 512 | 13.0 | 13.0 [11.0-14.2] | 0.0 | >= 320 | 14.8 | 13.0 | 13.0 | 0.122 / 0.027 (cos 0.214, joint) | 0.174 / 0.039 (cos 0.303, joint) | 0.207 / 0.046 (cos 0.386, joint) |
| d2_deepprint_fmr1 | pca128 | 128 | 22.9 | 22.9 [19.2-26.3] | 0.0 | >= 320 | 26.9 | 50.1 | 50.1 | 0.064 / 0.016 (cos 0.049, joint) | 0.061 / 0.016 (cos 0.069, joint) | 0.097 / 0.025 (cos 0.139, joint) |
| d2_deepprint_fmr1 | pca256 | 256 | 29.8 | 29.8 [26.3-33.3] | 0.0 | >= 320 | 33.0 | 100.8 | 100.8 | 0.081 / 0.021 (cos 0.024, joint) | 0.089 / 0.023 (cos 0.055, joint) | 0.117 / 0.031 (cos 0.103, joint) |
| d2_deepprint_fmr1 | fi64 | 64 | 19.8 | 19.8 [17.9-22.1] | 0.0 | >= 320 | 22.3 | 42.4 | 42.4 | 0.074 / 0.020 (cos 0.057, joint) | 0.076 / 0.021 (cos 0.088, joint) | 0.119 / 0.032 (cos 0.202, joint) |
| d2_deepprint_fmr1 | fi128 | 128 | 25.2 | 25.2 [23.1-27.4] | 0.0 | >= 320 | 27.7 | 76.5 | 76.5 | 0.093 / 0.026 (cos 0.033, joint) | 0.100 / 0.028 (cos 0.057, joint) | 0.119 / 0.033 (cos 0.128, joint) |
| d2_deepprint_fmr1 | fi256 | 256 | 29.8 | 29.8 [27.7-32.3] | 0.0 | >= 320 | 32.9 | 127.7 | 127.7 | 0.116 / 0.032 (cos 0.018, joint) | 0.126 / 0.034 (cos 0.043, joint) | 0.145 / 0.039 (cos 0.090, joint) |
| d2_ctrlvit_fmr1 | none | 768 | 25.3 | 21.2 [12.8-25.2] | -4.0 | >= 320 | 21.2 | 25.3 | 21.2 | 0.127 / 0.123 (cos 0.099, joint) | 0.157 / 0.152 (cos 0.132, joint) | 0.236 / 0.228 (cos 0.200, joint) |
| d2_ctrlvit_fmr1 | pca128 | 128 | 16.8 | 16.8 [15.3-18.4] | 0.0 | >= 320 | 22.0 | 26.2 | 26.2 | 0.040 / 0.039 (cos 0.045, joint) | 0.051 / 0.050 (cos 0.066, joint) | 0.110 / 0.107 (cos 0.198, proj) |
| d2_ctrlvit_fmr1 | pca256 | 256 | 17.9 | 17.9 [16.6-20.2] | 0.0 | >= 320 | 23.4 | 48.4 | 48.4 | 0.070 / 0.068 (cos 0.031, joint) | 0.086 / 0.084 (cos 0.046, joint) | 0.138 / 0.135 (cos 0.145, proj) |
| d2_ctrlvit_fmr1 | fi64 | 64 | 16.7 | 16.7 [16.1-18.7] | 0.0 | >= 320 | 20.5 | 21.9 | 21.9 | 0.018 / 0.017 (cos 0.047, joint) | 0.031 / 0.029 (cos 0.099, joint) | 0.082 / 0.077 (cos 0.276, proj) |
| d2_ctrlvit_fmr1 | fi128 | 128 | 17.8 | 17.8 [16.4-19.4] | 0.0 | >= 320 | 22.8 | 34.7 | 34.7 | 0.034 / 0.033 (cos 0.037, joint) | 0.050 / 0.048 (cos 0.065, joint) | 0.117 / 0.113 (cos 0.198, proj) |
| d2_ctrlvit_fmr1 | fi256 | 256 | 16.3 | 16.3 [14.9-17.4] | 0.0 | >= 320 | 22.2 | 58.2 | 58.2 | 0.084 / 0.082 (cos 0.030, joint) | 0.098 / 0.095 (cos 0.045, joint) | 0.162 / 0.158 (cos 0.145, proj) |

Joint alignment-plus-anchor stage (results/joint/, real embeddings; same seeds, splits, keys, thresholds, colluders and victims as the anchor stage; parity of proj with anchor.csv is exact, results/joint/PARITY.md). proj is the anchor-only projection decoder; J1 (align_anchor) aligns the stolen templates with AUX and takes the axis signs from the anchors; J2 (joint) adds J1 outside the anchors' span to the projection decoder's exact component. The best evaluated attack is, per r, the largest seed-mean relative success (mean over keys within a seed, then over seeds) over proj, J1 and J2 and their AUX and p settings: an upper envelope selected with ground truth; the true-sign reference (eigen_oracle) is not a candidate, and ties go to proj. r50 is interpolated log-linearly on that curve; a value above m is an upper estimate (with m exact anchors the projection decoder recovers z). The per-seed range takes, per seed, the smallest r50 over the settings, so the seed-mean envelope's r50 can lie above that range. E1 compares proj, J1 and J2 only (the anchor stage's ridge and mlp are not candidates, so section 12 can show a smaller E1 r50). A zero count at r = 1, 2 or 5 gives the largest relative one-sided 95% Clopper-Pearson bound over the rows of that r (a descriptive binomial reference limit). Provenance of the ten joint runs: results/joint/*/metadata.json (seed, configuration, data source, environment, git commit) and results/joint/PARITY.md.

## Theory checks

Proposition 2 (rec_cos_z vs cstar, exact anchors, proj decoder, lrp_full):
- d1_base: largest |rec_cos_z - cstar| (proj decoder) over r in {20, 40, 80} for pca128_a1 and fi128 = 0.076 (pre pca128_a1, r = 40; ridge rec_cos_z there = 0.616). Excess of rec_cos_z over cstar for pre none (m = 768): r = 20: 0.094, r = 40: 0.133, r = 80: 0.168.
- d1_large: largest |rec_cos_z - cstar| (proj decoder) over r in {20, 40, 80} for pca128_a1 and fi128 = 0.056 (pre pca128_a1, r = 40; ridge rec_cos_z there = 0.595). Excess of rec_cos_z over cstar for pre none (m = 1024): r = 20: 0.062, r = 40: 0.090, r = 80: 0.121.
- d2_fingervit: largest |rec_cos_z - cstar| (proj decoder) over r in {20, 40, 80} for pca128_a1 and fi128 = 0.093 (pre pca128_a1, r = 40; ridge rec_cos_z there = 0.635). Excess of rec_cos_z over cstar for pre none (m = 1536): r = 20: 0.723, r = 40: 0.753, r = 80: 0.725.
- d2_deepprint: largest |rec_cos_z - cstar| (proj decoder) over r in {20, 40, 80} for pca128_a1 and fi128 = 0.044 (pre pca128_a1, r = 40; ridge rec_cos_z there = 0.582). Excess of rec_cos_z over cstar for pre none (m = 512): r = 20: 0.542, r = 40: 0.581, r = 80: 0.554.
- d2_ctrlvit: largest |rec_cos_z - cstar| (proj decoder) over r in {20, 40, 80} for pca128_a1 and fi128 = 0.028 (pre pca128_a1, r = 40; ridge rec_cos_z there = 0.564). Excess of rec_cos_z over cstar for pre none (m = 768): r = 20: 0.164, r = 40: 0.210, r = 80: 0.253.

Proposition 1 (zero-pair, key-free; abs cos and cos^2 against the uniform-direction chance level, and best E1 relative against the zero-effort impostor level, on fi, rbig and firbig configurations):
- d1_base, fi64_p512: best_abs_cos_z = 0.104 vs uniform 0.100 (ratio 1.04); best_cos2_z = 0.0171 vs 1/m = 0.0156 (ratio 1.10); best E1 relative = 0.005 [0.000, 0.019] next to the zero-effort impostor level 0.0089 (selected: ica, p 64, AUX shifted).
- d1_base, fi128_p512: best_abs_cos_z = 0.073 vs uniform 0.071 (ratio 1.03); best_cos2_z = 0.0092 vs 1/m = 0.0078 (ratio 1.18); best E1 relative = 0.010 [0.000, 0.028] next to the zero-effort impostor level 0.0089 (selected: ica, p 128, AUX shifted).
- d1_base, fi256_p512: best_abs_cos_z = 0.078 vs uniform 0.050 (ratio 1.56); best_cos2_z = 0.0098 vs 1/m = 0.0039 (ratio 2.52); best E1 relative = 0 successes; CP95 upper <= 0.024 next to the zero-effort impostor level 0.0089 (selected: n/a).
- d1_base, rbig128_i2: best_abs_cos_z = 0.100 vs uniform 0.071 (ratio 1.42); best_cos2_z = 0.0208 vs 1/m = 0.0078 (ratio 2.66); best E1 relative = 0.011 [0.000, 0.029] next to the zero-effort impostor level 0.0089 (selected: ica, p 128, AUX same_dist).
- d1_base, firbig128_p512_i2: best_abs_cos_z = 0.072 vs uniform 0.071 (ratio 1.02); best_cos2_z = 0.0085 vs 1/m = 0.0078 (ratio 1.09); best E1 relative = 0.004 [0.000, 0.017] next to the zero-effort impostor level 0.0089 (selected: ica, p 128, AUX shifted).
- d1_large, fi64_p512: best_abs_cos_z = 0.105 vs uniform 0.100 (ratio 1.05); best_cos2_z = 0.0182 vs 1/m = 0.0156 (ratio 1.17); best E1 relative = 0.010 [0.000, 0.026] next to the zero-effort impostor level 0.0094 (selected: ica, p 64, AUX shifted).
- d1_large, fi128_p512: best_abs_cos_z = 0.072 vs uniform 0.071 (ratio 1.01); best_cos2_z = 0.0089 vs 1/m = 0.0078 (ratio 1.13); best E1 relative = 0.010 [0.000, 0.026] next to the zero-effort impostor level 0.0094 (selected: ica, p 128, AUX shifted).
- d1_large, fi256_p512: best_abs_cos_z = 0.049 vs uniform 0.050 (ratio 0.98); best_cos2_z = 0.0043 vs 1/m = 0.0039 (ratio 1.10); best E1 relative = 0.002 [0.000, 0.006] next to the zero-effort impostor level 0.0094 (selected: eigen, p 8, AUX shifted).
- d1_large, rbig128_i2: best_abs_cos_z = 0.080 vs uniform 0.071 (ratio 1.14); best_cos2_z = 0.0143 vs 1/m = 0.0078 (ratio 1.84); best E1 relative = 0.011 [0.000, 0.025] next to the zero-effort impostor level 0.0094 (selected: ica, p 128, AUX same_dist).
- d1_large, firbig128_p512_i2: best_abs_cos_z = 0.071 vs uniform 0.071 (ratio 1.01); best_cos2_z = 0.0089 vs 1/m = 0.0078 (ratio 1.13); best E1 relative = 0.010 [0.000, 0.026] next to the zero-effort impostor level 0.0094 (selected: ica, p 128, AUX shifted).
- d2_fingervit, fi64_p512: best_abs_cos_z = 0.139 vs uniform 0.100 (ratio 1.39); best_cos2_z = 0.0364 vs 1/m = 0.0156 (ratio 2.33); best E1 relative = 0.018 [0.000, 0.047] next to the zero-effort impostor level 0.0082 (selected: ica, p 64, AUX same_dist).
- d2_fingervit, fi128_p512: best_abs_cos_z = 0.088 vs uniform 0.071 (ratio 1.24); best_cos2_z = 0.0149 vs 1/m = 0.0078 (ratio 1.91); best E1 relative = 0.001 [0.000, 0.005] next to the zero-effort impostor level 0.0082 (selected: ica, p 128, AUX same_dist).
- d2_fingervit, fi256_p512: best_abs_cos_z = 0.055 vs uniform 0.050 (ratio 1.09); best_cos2_z = 0.0049 vs 1/m = 0.0039 (ratio 1.26); best E1 relative = 0 successes; CP95 upper <= 0.044 next to the zero-effort impostor level 0.0082 (selected: n/a).
- d2_fingervit, rbig128_i2: best_abs_cos_z = 0.081 vs uniform 0.071 (ratio 1.14); best_cos2_z = 0.0112 vs 1/m = 0.0078 (ratio 1.44); best E1 relative = 0 successes; CP95 upper <= 0.044 next to the zero-effort impostor level 0.0082 (selected: n/a).
- d2_fingervit, firbig128_p512_i2: best_abs_cos_z = 0.083 vs uniform 0.071 (ratio 1.17); best_cos2_z = 0.0116 vs 1/m = 0.0078 (ratio 1.48); best E1 relative = 0 successes; CP95 upper <= 0.044 next to the zero-effort impostor level 0.0082 (selected: n/a).
- d2_deepprint, fi64_p384: best_abs_cos_z = 0.105 vs uniform 0.100 (ratio 1.04); best_cos2_z = 0.0171 vs 1/m = 0.0156 (ratio 1.09); best E1 relative = 0.001 [0.000, 0.004] next to the zero-effort impostor level 0.0074 (selected: ica, p 64, AUX same_dist).
- d2_deepprint, fi128_p384: best_abs_cos_z = 0.074 vs uniform 0.071 (ratio 1.04); best_cos2_z = 0.0085 vs 1/m = 0.0078 (ratio 1.09); best E1 relative = 0.001 [0.000, 0.005] next to the zero-effort impostor level 0.0074 (selected: wp, p 128, AUX same_dist).
- d2_deepprint, fi256_p384: best_abs_cos_z = 0.065 vs uniform 0.050 (ratio 1.31); best_cos2_z = 0.0066 vs 1/m = 0.0039 (ratio 1.70); best E1 relative = 0.001 [0.000, 0.004] next to the zero-effort impostor level 0.0074 (selected: eigen, p 32, AUX same_dist).
- d2_deepprint, rbig128_i2: best_abs_cos_z = 0.084 vs uniform 0.071 (ratio 1.19); best_cos2_z = 0.0124 vs 1/m = 0.0078 (ratio 1.59); best E1 relative = 0.004 [0.000, 0.014] next to the zero-effort impostor level 0.0074 (selected: ica, p 128, AUX same_dist).
- d2_deepprint, firbig128_p384_i2: best_abs_cos_z = 0.084 vs uniform 0.071 (ratio 1.20); best_cos2_z = 0.0112 vs 1/m = 0.0078 (ratio 1.43); best E1 relative = 0.001 [0.000, 0.005] next to the zero-effort impostor level 0.0074 (selected: eigen, p 32, AUX same_dist).
- d2_ctrlvit, fi64_p512: best_abs_cos_z = 0.119 vs uniform 0.100 (ratio 1.18); best_cos2_z = 0.0233 vs 1/m = 0.0156 (ratio 1.49); best E1 relative = 0.008 [0.001, 0.015] next to the zero-effort impostor level 0.0012 (selected: ica, p 64, AUX same_dist).
- d2_ctrlvit, fi128_p512: best_abs_cos_z = 0.079 vs uniform 0.071 (ratio 1.12); best_cos2_z = 0.0106 vs 1/m = 0.0078 (ratio 1.35); best E1 relative = 0.003 [0.000, 0.006] next to the zero-effort impostor level 0.0012 (selected: ica, p 128, AUX same_dist).
- d2_ctrlvit, fi256_p512: best_abs_cos_z = 0.068 vs uniform 0.050 (ratio 1.36); best_cos2_z = 0.0073 vs 1/m = 0.0039 (ratio 1.88); best E1 relative = 0 successes; CP95 upper <= 0.006 next to the zero-effort impostor level 0.0012 (selected: n/a).
- d2_ctrlvit, rbig128_i2: best_abs_cos_z = 0.080 vs uniform 0.071 (ratio 1.13); best_cos2_z = 0.0104 vs 1/m = 0.0078 (ratio 1.33); best E1 relative = 0.002 [0.000, 0.006] next to the zero-effort impostor level 0.0012 (selected: ica, p 128, AUX same_dist).
- d2_ctrlvit, firbig128_p512_i2: best_abs_cos_z = 0.078 vs uniform 0.071 (ratio 1.11); best_cos2_z = 0.0103 vs 1/m = 0.0078 (ratio 1.32); best E1 relative = 0.003 [0.000, 0.007] next to the zero-effort impostor level 0.0012 (selected: ica, p 128, AUX same_dist).

Key compromise (in-system acceptance and E2 per scheme, none vs fi128): see table 6.

## Deviations from the spec, failed runs and reruns

- The code changed after the first runs (projection decoder, abs-cos and cos^2 columns, identity-level intervals, deduplicated scale distractors with full metadata, FMR 1% configs, `rec_cos_z` and the exact collusion optimum), and the corpora were corrected afterwards: D1 identity labels re-derived from each source's native naming by content matching, D2 rebuilt from the development folds of an internal compilation, finger-disjoint from D1, and every finger shared with D1 removed from the secondary corpus (shifted AUX, real scale-run distractors). All 12 runs listed in table 1 use the final corpora. The runs of d1_base, d1_large, d2_fingervit and d2_deepprint (and the scale runs) were rerun on the final code; the d2_ctrlvit runs were added afterwards on the same corpora, with the same seeds, splits and pre-conditioners as d2_fingervit. Earlier runs are superseded and are not included.
- scale_d1_base_secondary: requested distractor sizes above the rows available after the near-duplicate filter were capped (3000 -> 2998); see table 7a.
- Hashing-scheme collusion numbers hold only for the ridge/proj/mlp decoders evaluated. A stronger IoM decoder exists in the synthetic proof check (+0.08 cos), so hashing collusion resistance is not claimed.
- FMR 1% runs (`*_fmr1`) are separate runs with the attack threshold at FMR 1%; they are aggregated separately (`results/isotropy/summary_fmr1`) and are not mixed with the FMR 0.1% runs.
- Skips: see table 8.
- Warnings in stdout logs (POT Sinkhorn did not converge, Wasserstein-Procrustes): d1_base: 0, d1_base_fmr1: 0, d1_large: 5, d1_large_fmr1: 5, d2_ctrlvit: 0, d2_ctrlvit_fmr1: 0, d2_deepprint: 0, d2_deepprint_fmr1: 0, d2_fingervit: 0, d2_fingervit_fmr1: 0. No Traceback in any log.
- Failed runs: none among the 12 runs listed in table 1.
- `configs/scale_d1_base_synthetic.yaml`: provenance: features extracted with gazi_base_extended; the extraction tool's default model is a different one and was not used.

## Data sources

- d1_base, d1_large (and their `_fmr1` runs): D1: SIN 2026 image set, identity labels re-derived from each source's native naming by content matching (real); their shifted AUX: the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real).
- d2_fingervit, d2_deepprint (and their `_fmr1` runs): D2: development folds of an internal compilation with labels parsed from each source's native naming, finger-disjoint from D1 (real).
- d2_ctrlvit (and its `_fmr1` run): D2: development folds of an internal compilation with labels parsed from each source's native naming, finger-disjoint from D1 (real), ctrl-vit: a control ViT trained by us on the compilation images of the D1 fingers.
- scale_d1_base_secondary: D1 targets + distractors from the secondary corpus (FVC set B, SD301a, SD302b/d), fingers shared with D1 removed (real).
- scale_d1_base_synthetic: D1 targets + SYNTHETIC distractors (gazi_synthetic_fp images embedded with gazi_base_extended).
- Synthetic data enter only as HNSW distractors in the scale run scale_d1_base_synthetic (table 7); every other result is on real embeddings.
