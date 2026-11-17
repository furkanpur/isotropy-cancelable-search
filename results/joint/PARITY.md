# Parity of the joint runs with the committed isotropy runs

For each joint run, `scripts/experiments/check_joint_parity.py` compared the `proj` rows (every seed, key, joint
pre-conditioner and committed budget) with the committed `anchor.csv` rows of the same configuration (variant
exact, decoder proj, scheme lrp_full): counts exactly, the floating columns within 1e-9. Before that it checked
that the configuration (apart from the stage selection) and every data field of `metadata.json` are equal; the
environment line is informational. Command, from the repository root:
`.venv/bin/python scripts/experiments/check_joint_parity.py <joint run> <committed run>`.

| joint run | committed run | rows | checker | runtime (s) | threads |
|---|---|---|---|---|---|
| `joint_d1_base-20261117-103155` | `isotropy_d1_base-20261114-194618` | 720 | OK | 834 | 4 |
| `joint_d1_base_fmr1-20261117-103206` | `isotropy_d1_base_fmr1-20261114-194626` | 720 | OK | 829 | 4 |
| `joint_d1_large-20261117-103214` | `isotropy_d1_large-20261114-194603` | 720 | OK | 912 | 4 |
| `joint_d1_large_fmr1-20261117-103222` | `isotropy_d1_large_fmr1-20261114-194611` | 720 | OK | 903 | 4 |
| `joint_d2_ctrlvit-20261117-104900` | `isotropy_d2_ctrlvit-20261115-125450` | 630 | OK | 146 | 12 |
| `joint_d2_ctrlvit_fmr1-20261117-104907` | `isotropy_d2_ctrlvit_fmr1-20261115-125457` | 630 | OK | 145 | 12 |
| `joint_d2_deepprint-20261117-103230` | `isotropy_d2_deepprint-20261114-194634` | 630 | OK | 409 | 4 |
| `joint_d2_deepprint_fmr1-20261117-103238` | `isotropy_d2_deepprint_fmr1-20261114-194642` | 630 | OK | 415 | 4 |
| `joint_d2_fingervit-20261117-103245` | `isotropy_d2_fingervit-20261114-194546` | 630 | OK | 688 | 4 |
| `joint_d2_fingervit_fmr1-20261117-103253` | `isotropy_d2_fingervit_fmr1-20261114-194553` | 630 | OK | 692 | 4 |

## joint_d1_base-20261117-103155

- committed run: `results/isotropy/isotropy_d1_base-20261114-194618`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:31:56.803025+00:00
- checker exit status: 0

```
parity: OK (720 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```

## joint_d1_base_fmr1-20261117-103206

- committed run: `results/isotropy/isotropy_d1_base_fmr1-20261114-194626`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:32:08.197047+00:00
- checker exit status: 0

```
parity: OK (720 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```

## joint_d1_large-20261117-103214

- committed run: `results/isotropy/isotropy_d1_large-20261114-194603`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:32:16.149873+00:00
- checker exit status: 0

```
parity: OK (720 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```

## joint_d1_large_fmr1-20261117-103222

- committed run: `results/isotropy/isotropy_d1_large_fmr1-20261114-194611`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:32:24.012598+00:00
- checker exit status: 0

```
parity: OK (720 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```

## joint_d2_ctrlvit-20261117-104900

- committed run: `results/isotropy/isotropy_d2_ctrlvit-20261115-125450`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:49:01.149882+00:00
- checker exit status: 0

```
parity: OK (630 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```

## joint_d2_ctrlvit_fmr1-20261117-104907

- committed run: `results/isotropy/isotropy_d2_ctrlvit_fmr1-20261115-125457`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:49:09.180698+00:00
- checker exit status: 0

```
parity: OK (630 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```

## joint_d2_deepprint-20261117-103230

- committed run: `results/isotropy/isotropy_d2_deepprint-20261114-194634`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:32:31.495702+00:00
- checker exit status: 0

```
parity: OK (630 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```

## joint_d2_deepprint_fmr1-20261117-103238

- committed run: `results/isotropy/isotropy_d2_deepprint_fmr1-20261114-194642`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:32:40.638362+00:00
- checker exit status: 0

```
parity: OK (630 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```

## joint_d2_fingervit-20261117-103245

- committed run: `results/isotropy/isotropy_d2_fingervit-20261114-194546`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:32:48.313156+00:00
- checker exit status: 0

```
parity: OK (630 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```

## joint_d2_fingervit_fmr1-20261117-103253

- committed run: `results/isotropy/isotropy_d2_fingervit_fmr1-20261114-194553`
- joint run commit `41c45b5b9ad3fe65d1c3e91886bfdd4824f7f027`, started 2026-11-17T07:32:55.981225+00:00
- checker exit status: 0

```
parity: OK (630 rows; exact n_targets, e1_successes, e1_trials, e2_successes, e2_trials; max |diff| e1_rel 0, e2_rel 0, e1_genuine 0, e2_genuine 0, rec_cos_z 0; equal data_source, data_files, corpus_counts, split_counts, aux_sizes, shifted_aux_source, shifted_aux_removed, secondary_removed_shared_identity and config)
environment: same image freeze, Python, NumPy, threads and CPU
```
