# Data

## Why no images or embeddings are released

- Embeddings are biometric data. They can be inverted into usable proxies of the fingerprint they were computed from, so releasing them is not safe even without the images.
- The fingerprint images, the compiled corpora and all embeddings cannot be shared, for privacy reasons and under the terms of use of the source datasets. The fingerprint data comprise public and licensed benchmark datasets used under their terms of use; no new biometric data were collected for this study.

This repository therefore releases only aggregates:

- the per-run CSV/JSON outputs, `metadata.json` files, summaries and figures of the experiments (`results/isotropy/`, `results/scale/`);
- the JSON summaries of the data preparation (`results/data/`), from which `scripts/manuscript/data_numbers.py` writes the data numbers of the paper.

The experiment runners never write embeddings, proxies or per-sample scores. The data-preparation scripts in `scripts/data/` write per-image outputs (label maps, embeddings, content keys) to their output directory on the data host. Those outputs stay there; only their aggregate JSON summaries are released.

## Datasets

All results in `results/isotropy/` and the secondary-corpus scale run are computed on real fingerprint embeddings. Synthetic data are labelled as such everywhere. Every count below is copied from the committed aggregate named after it; paths under `results/data/` are given without that prefix.

### D1

- **Source.** D1 is the image set of SIN 2026 (<https://github.com/furkanpur/cancelable-vit-fingerprint>). Its images come from public and licensed benchmarks. By image: CASIA-FingerprintV5 4,941, NIST SD302 3,026, CASIA-Fingerprint-Subject-Ageing 1,565, NIST SD301 455, the U.are.U and CrossMatch vendor sample sets 90 and 66, and FVC2000, FVC2004 and FVC2002 36, 30 and 24 (`relabel_d1/summary.json`, `per_source`).
- **Labels.** For this study its identity labels were re-derived at finger level. Each image was matched by its content to the compilation from which D2 is drawn (below), whose labels come from each source's native naming.
- **Size.** Of the 10,359 images, 10,233 were placed on 3,694 fingers. 126 were dropped: 119 unmatched and 7 ambiguous (`relabel_d1/summary.json`).
- **Images per finger.** A finger has 1 to 18 images, and 699 fingers have a single image (`relabel_d1/summary.json`, `images_per_finger_hist`). One identity is one finger.
- **Resolution.** The aggregates record no image resolution for D1.
- **Label map.** The map itself is not released. The D1 runs record its SHA-256 (`metadata.json`, `data_files`).

### Secondary corpus

- **Source.** FVC2000, FVC2002 and FVC2004 set B, NIST SD301a and SD302b/d: 14,863 images (`relabel_secondary/summary.json`).
- **Labels.** Each image was located in the compilation through its dataset path. This placed 14,614 images on 2,330 fingers. 249 images were dropped (`relabel_secondary/composition.json`):
  - the 240 images of the three FVC DB4 sets, which the compilation does not hold;
  - 9 unreadable or ambiguous images.
- **Label check.** The labels were checked on the images that the secondary corpus shares with D1, found as near-identical embeddings (cosine at least 0.999). Under the base encoder, 2,009 of the 2,009 such images that both labellings place on a finger carry the same finger, and all 11 path entries are covered (`relabel_secondary/mapping_check.json`).
- **Fingers shared with D1.** 1,810 of the 2,330 fingers (11,387 images) also occur in D1 and are removed wherever the corpus is used. 520 fingers (3,227 images) remain: 421 from SD302, 39 from SD301a and 60 from FVC (`relabel_secondary/composition.json`).
- **Use.** The corpus has two uses, each after near-duplicates of D1 images (cosine at least 0.95) are removed:
  - the shifted auxiliary population of the D1 runs, 3,225 embeddings for the base encoder and 3,224 for the large one (`metadata.json` of the D1 runs, `aux_sizes`);
  - the real HNSW distractors of one scale run: of the 3,000 rows drawn for the largest requested size, 2 are removed, which leaves 2,998 (`metadata.json` of `results/scale/scale_d1_base_secondary-*`).

### D2

- **Source.** D2 consists of the development folds 0 to 4 of an internal compilation of public and licensed fingerprint benchmarks. Its labels are parsed from each source's native naming, so that one identity is one finger. The compilation is not redistributable.
- **Size.** 18,493 images of 3,140 fingers with 719 (source, subject label) pairs, which are not persons in every source, with 5 to 30 images per finger (`corpus_d2/extract_summary.json`, `composition`).
- **Sources.** Nine sources. By finger: CASIA-FingerprintV5 2,237, NIST SD302 413, the contact-based images of the PolyU Contactless 2D to Contact-based 2D Fingerprint Images Database 336, NIST SD301 39, U.are.U 35, CrossMatch 29, FVC2002 22, FVC2000 18 and CASIA-Fingerprint-Subject-Ageing 11 (same file, `per_source`). D2 alone holds 2,976 PolyU images of 336 fingers, more than the 1,800 contact-based images of 300 clients described in the database's reference paper (Lin and Kumar, IEEE Transactions on Image Processing, 2018), so the compilation holds a larger release of this database than the one described there.
- **Devices.** 34 devices as the compilation records them. The compilation records no device for four sources (CASIA-FingerprintV5, the PolyU database and the two vendor sets), and each of these counts as one device (same file, `devices_by_source`).
- **Resolution.** The 3,170 NIST SD301 and SD302 images are recorded at 500 dpi. The compilation records no resolution for the other 15,323 images (same file, `dpi_by_source`).
- **Exclusions** (same file, `selection_counts`):
  - LivDet (the compilation holds its 2009, 2011, 2013 and 2015 editions): 3,372 identity keys and 37,020 images. Its identity key in the compilation does not name the edition, so subject numbers collide across editions.
  - Every finger of D1: 3,694 fingers, of which 3,684 lie in the development folds.
  - 8 fingers (53 images) with a compilation image, selected for D2 or not, that equals a compilation image of a D1 finger. All 8 were found through images not selected for D2. A separate check against the 10,359 images of D1 itself found no further finger.
  - 5,974 identities with fewer than two images.
- **Embeddings.** `scripts/data/extract_d2.py` embeds D2 with finger-vit and with DeepPrint (flx). `corpus_d2/parity.json` holds its parity check against the earlier features. The third D2 encoder, the control encoder ctrl-vit, embeds D2 with `scripts/train/train_control.py embed` (see [ctrl-vit](#ctrl-vit-control-encoder) below).
- **Label check.** `corpus_d2/label_check.json` compares all 53,100 genuine pairs of D2 with impostor pairs. The genuine-versus-impostor AUC is 0.893 for finger-vit and 0.739 for DeepPrint, and on the 12,858 cross-device pairs 0.713 and 0.509.
- **Earlier label check.** `corpus_d2/label_check_fold0_with_livdet.json` is the label check of the first, superseded extraction: development fold 0 with LivDet, 16,431 images. It is the evidence for the LivDet exclusion.

### Training-data overlap and clean subset

The training data of the three original ViT-based encoders are not recorded with their weights. `results/data/overlap/` holds the worst-case overlap with the most plausible candidate training pools. The control encoder ctrl-vit (next section) was trained by us on recorded data for this reason.

- `d1_vs_ext_train.json`: D1 against candidate pool A (90,557 files), by exact image content. 8,260 of the 10,233 D1 images are exact copies of pool images, and 3,339 of the 3,694 D1 fingers have an impression in the pool when any image of the finger in the compilation is counted.
- `d2_vs_packaged_train.json`: D2 against candidate pool B (51,938 resized and re-encoded files), by finger-vit near-duplicates at a calibrated threshold. At that threshold, 14,332 of the 18,493 D2 images and 2,633 of the 3,140 D2 fingers have a near-duplicate in the pool.
- `deepprint_sources.json`: the training sets named in the flx documentation (SFinGe synthetic, MCYT330), none of which is a D2 source.
- `clean_subset_stats.json`: utility and spectrum statistics on the fingers absent from the pools against size-matched samples of the present ones. It is descriptive, and it runs no attack.

### ctrl-vit (control encoder)

ctrl-vit is a control ViT trained by us. Its training data are recorded and disjoint from D2's fingers and images by identity labels and exact content, so it is a check on training exposure that the three original ViT-based encoders lack. It is an in-domain control: it shares most of D2's sources, and 2,748 of D2's 3,140 fingers have a subject key with training fingers (`overlap_audit.json`, `subjects`); a subject key is a finger's identity without its finger code, and it does not separate persons in every source (the 8 keys without training fingers hold 392 fingers). Every count below is copied from `results/data/control/` (paths given without that prefix).

- **Initialisation.** DINOv2 ViT-B/14 (`facebook/dinov2-base`; Oquab et al., Transactions on Machine Learning Research, 2024), pretrained without labels on natural images (LVD-142M), with no documented fingerprint data. The embedding is the 768-dimensional CLS token, L2-normalised. `scripts/train/fetch_weights.py` downloads the pretrained weights in a container that mounts no data; their file hashes are recorded in `training_summary.json` (`weights_fetch`).
- **Training data.** The compilation images of the D1 fingers that lie in the compilation's development partition, selected by `scripts/train/build_training_set.py`: 27,342 images of 3,684 fingers from 8 sources (10 D1 fingers lie outside the partition), with 27 devices named in the compilation's device records (D2: 30 named devices). 3,315 fingers (24,588 images) were used for fitting and 369 fingers (2,754 images) for validation (`training_set.json`).
- **Recipe.** `scripts/train/train_control.py train`: an ArcFace head (Deng et al., CVPR 2019), AdamW with layer-wise learning-rate decay, one warm-up epoch and cosine decay over 30 epochs, batch 128, and mild affine and contrast augmentation of 224 x 224 inputs. The recipe and the selection rule were fixed before training; `training_summary.json` records them (`recipe`, with no overrides). There was one training run, on one GPU, and it is not bitwise reproducible (`reproducibility`).
- **Selection.** On the validation fingers only: epoch 27 has the lowest validation EER, 0.031032, against 0.290245 for the pretrained model (`training_summary.json`, `selected_epoch`, `selected_val_eer`, `pretrained_val_eer`). No accuracy target was set; the accuracy is what this run reached.
- **Freeze, then evaluate.** D2 was embedded (`scripts/train/train_control.py embed`, `embed_d2.json`) only after `training_summary.json`, which holds the SHA-256 hashes of the trained weights, had been committed. The trained weights are not released, because they derive from licensed biometric data.
- **Disjointness.** D2 excludes every D1 finger and every finger with a compilation image identical to one of a D1 finger, so the training set is disjoint from D2 by identity labels and exact content. The audit (`scripts/data/control_audit.py`, `overlap_audit.json`) confirms identity overlap 0 and content overlap 0 over all 18,493 D2 rows.
- **Near-duplicate scan.** Under ctrl-vit, no D2 image has a cosine to a training image at or above the 99.9th percentile of the D2 genuine cosines (0.938450): 0 of 18,493 images and 0 of 3,140 fingers, with one-sided 95% Clopper-Pearson upper bounds of 1.6e-4 per image and 9.5e-4 per finger. The largest cosine is 0.896 (`near_duplicates`).
- **Finger-level statistic.** For each D2 finger it compares the best mean cross-cosine to a training finger with the finger's own genuine mean. It flags 50 D2 fingers (CASIA-FingerprintV5 31, NIST SD302 18, NIST SD301 1) and 0 of the 336 PolyU fingers (one-sided 95% Clopper-Pearson upper bound 8.9e-3 per finger), the null source with no training source and no subject key with training fingers (`finger_level`). The best matches share the subject key for 36 of the 50 flagged fingers, a lower bound of the share among flagged fingers whose subject key has training fingers, against 1,133 of the 2,748 D2 fingers whose key has them; they share the finger position for 8 of 50, against 911 of 2,764; and the median own genuine mean of the flagged fingers is 0.325, against 0.561 for all D2 fingers (`characterisation`). They are mostly low-quality fingers matched to another finger with the same subject key. The statistic neither bounds nor detects same-finger cases, and the null source is not matched on own genuine mean, so finger-disjointness is stated by identity labels and the statistic is an additional check, not a proof.
- **Label-structure scan.** 54 D2 fingers pair with a training finger if numeric subject ids are assumed to correspond across CASIA-FingerprintV5 and CASIA-Fingerprint-Subject-Ageing and a finger-position map is assumed. Both assumptions are unverified. 0 of the 54 are flagged (one-sided 95% Clopper-Pearson upper bound 5.4e-2 per candidate), and for each of them the best embedding match is a training finger other than the paired one (`label_structure`); they are not evidence of duplicates.
- **Sensitivity.** Removing the flagged fingers, all fingers of their subjects, or the 54 fingers of the label-structure scan changes the genuine-versus-impostor AUC of ctrl-vit by at most 0.0003 and that of any D2 encoder by at most 0.0043; these are differences of AUCs at four decimals (`sensitivity`).
- **Subject overlap.** 8 of D2's 9 sources are training sources. 710 of the 718 D2 subject keys (the compilation's subject link; the 719 above count (source, subject label) pairs, which do not separate persons either) have other fingers in training, and 2,748 of the 3,140 D2 fingers have such a key (`training_set.json`, `disjointness`). For FVC the subject label is the database, so a shared FVC subject means the same database, not the same person. For PolyU, which has no training source, the absence of a training subject restates the absence of the training source.
- **Subject check.** `subject_check.json` (`scripts/data/control_subject_check.py`) compares, within CASIA-FingerprintV5, the 56 fingers whose subject has no training finger with 20 size-matched random samples of the 2,181 fingers whose subject has one. In the raw space the EER of ctrl-vit is 0.0470 on the subject-disjoint fingers against a mean of 0.0244 (range 0.0155 to 0.0352) on the samples, whereas finger-vit (0.1743 against 0.2174) and DeepPrint (0.3036 against 0.4034), for which the split has no training meaning, do better on the subject-disjoint fingers. Subject overlap thus appears to favour ctrl-vit on in-domain fingers. The composition controls do not cover a same-subject similarity specific to ctrl-vit, however: the 56 fingers come from few subjects, so a noticeable share of their impostor pairs join two fingers of one person, and on them the raw mean impostor cosine of ctrl-vit is 0.0385 against a mean of 0.0092 (range 0.0062 to 0.0125) on the samples, whereas that of finger-vit lies inside its range and that of DeepPrint below it. The check is a single draw and descriptive.
- **PolyU.** The only D2 source with no training source and no subject key with training fingers (336 fingers, 2,976 images). The genuine-versus-impostor AUC of the label checks and the raw-space EER are 0.992 and 0.043 for ctrl-vit, 0.930 and 0.155 for finger-vit and 0.893 and 0.185 for DeepPrint (`label_check_ctrlvit.json`, `corpus_d2/label_check.json`, `subject_check.json` `polyu_descriptive`); these are single descriptive values without intervals.
- **Caveat.** ctrl-vit's accuracy on D2 (genuine-versus-impostor AUC 0.995 over all genuine pairs in `label_check_ctrlvit.json`) comes from in-domain training. It is not a measure of out-of-domain generalisation.

### Synthetic sets

- **Scale-run distractors.** The distractors of the synthetic scale run (`results/scale/scale_d1_base_synthetic-*`) are synthetic: `gazi_synthetic_fp` images embedded with the D1 base encoder, as recorded in `results/isotropy/RESULTS.md`. The images were generated by FPGAN-Control (Shoshan et al., WACV 2024), a StyleGAN-based controllable fingerprint generator, after training it on real fingerprint images; the run uses one image per synthetic identity (`load_features_by_id` with `per_id=1`), and the near-duplicate filter against D1 removed none of them (`distractors_removed` in the run's `metadata.json`).
- **Simulations and smoke runs.** The simulations in `docs/theory/proofcheck/` and the `synthetic` loader kind used by `configs/isotropy_smoke.yaml` are synthetic as well.

## Known provenance gaps

- **Relabel maps.** The relabel summaries (`relabel_d1/summary.json`, `relabel_secondary/summary.json`) do not record the SHA-256 of the map they summarise. The hashes of both maps are in the `data_files` of the runs that use them and in the provenance of `relabel_secondary/composition.json` and `mapping_check.json`.
- **Earlier label check.** `corpus_d2/label_check_fold0_with_livdet.json` records the SHA-256 of its inputs, but it has no provenance block (script hash, code commit, container image).
- **Label check image.** `corpus_d2/label_check.json` records the script hash and the code commit, but not the container image id.
- **Compilation manifest.** The D2 runs record the SHA-256 of the D2 files they read (`manifest.parquet`, `emb_<model>.npy`), not that of the compilation's manifest. That link goes through `corpus_d2/extract_summary.json` (`outputs.files` and `compilation.manifest_sha256`).
- **Code commits.** `corpus_d2/extract_summary.json`, `label_check.json` and `parity.json` record a code commit that ends in `-dirty`: they were written before the commit that holds their script. The script SHA-256 they record (`code` in `extract_summary.json` and `parity.json`, `provenance.script_sha256` in `label_check.json`) equals the script committed in the authors' working repository. `scripts/data/label_check.py` is released unchanged; the released `scripts/data/extract_d2.py` carries the renaming described in [Names in this release](#names-in-this-release), so it no longer matches its recorded hash.

## Names in this release

This release does not name the compilation from which D2 is drawn, as the paper does not. It was exported from the authors' working repository with one fixed renaming, applied to every path and to every text file. Identifiers, options, configuration values, environment variables, mount points, record fields and prose that named the compilation now say `compilation` (for example `--compilation`, `/data/compilation` and `compilation_keys.parquet`). The D2 extraction is `corpus_d2`: `results/data/corpus_d2/`, the mount point `/data/corpus_d2`, the data-source label `sin:corpus_d2:<model>` of the D2 runs and the environment variable `CANVEC_D2_ROOT`; its script is `scripts/data/extract_d2.py`. Binary files (the figure PDFs) were copied unchanged. Before the release, the released generators were run on the released files and regenerated `results/isotropy/RESULTS.md` and the numbers, tables and figures of the paper byte for byte (the figures with `SOURCE_DATE_EPOCH=1700000000`, which fixes the PDF creation date).

The SHA-256 hashes recorded in the aggregates and in the run records refer to the files of the working repository. A released file that carries the renaming therefore no longer matches its recorded hash. This concerns the scripts that name the compilation (for example `scripts/data/extract_d2.py`, `scripts/data/training_overlap.py` and `scripts/train/build_training_set.py`), `docker/Dockerfile.extract`, the D2 configs (`configs/isotropy_d2_*.yaml`) and the aggregates that other aggregates record as inputs (for example `corpus_d2/extract_summary.json`).

## Expected input format for rerunning

The loaders are in `src/ijiss_btp/data/loaders.py`.

- **Output.** Every loader returns an embedding matrix `x` of shape `(n, d)` (cast to float64), one identity label per row in `ids` (strings in the first two kinds below), and a provenance string that is recorded in the run metadata.
- **Identities.** Rows of the same identity are different impressions of the same finger.
- **Normalisation.** The loaders do not normalise; the helper `center_normalize` subtracts the DEV mean and L2-normalises each row.

### Data roots

`docker/run.sh` mounts every data root read-only inside the container:

- `/data/primary`: D1, in the `kind: sin` layout;
- `/data/secondary`: the secondary corpus, in the same layout;
- `/data/corpus_d2`: D2, the output directory of `scripts/data/extract_d2.py` in the same layout (`manifest.parquet`, `emb_fingervit.npy`, `emb_deepprint.npy`); for ctrl-vit, the output directory of `scripts/train/train_control.py embed` (`manifest.parquet`, `emb_ctrlvit.npy`) is mounted here instead;
- `/data/relabel`: the relabel maps (`d1_relabel.csv`, `secondary_relabel.csv`);
- `/data/scale`: the synthetic distractors;
- `/data/bench`: `keyed_npz` files.

### `kind: sin`

Config fields: `kind: sin`, `root`, `model`, and optionally `relabel` and `relabel_key`. In `configs/`, `data.root` is `/data/primary` (D1) or `/data/corpus_d2` (D2), and the secondary-corpus specs use `/data/secondary`.

Reads two files from `root`:

- `emb_<model>.npy`: a `.npy` array of shape `(n, d)`.
- `manifest.parquet`: a table of length `n` with an `identity` column (and a `path` column, which `relabel` needs); row `i` of the manifest describes row `i` of the array. Identities are converted to strings.

Optional fields:

- **`relabel`.** A CSV file with the columns `file,identity`, for example `/data/relabel/d1_relabel.csv`.
  - The identity of each manifest row is replaced by the map's, and rows without an entry are dropped.
  - The provenance string records the map's file name, the key mode and the dropped count, and the run metadata record the map's SHA-256.
  - A duplicate `file` entry or an empty identity raises.
- **`relabel_key`.** `basename` (the default when a map is given) matches `file` against the basename of the manifest `path`. `path` matches the full `path` string, for corpora whose basenames repeat across folders, such as the secondary corpus. A `relabel_key` without `relabel` raises.
- **`exclude_primary_identities: true`.** Accepted in a secondary-corpus spec (`shifted_aux` or `distractors`).
  - Every row whose identity occurs in the loaded primary corpus is dropped, before the near-duplicate filter.
  - Both specs must carry a relabel map, so that the two corpora share one identity namespace; otherwise the loader raises instead of comparing two namespaces.
- **`near_dup_cos`.** Accepted in `shifted_aux` and in `distractors`. Rows whose uncentred cosine to any row of the primary corpus is at least this value are removed.

Example (`configs/isotropy_d1_base.yaml`):

```yaml
data:
  kind: sin
  root: /data/primary
  model: base
  relabel: /data/relabel/d1_relabel.csv
  shifted_aux:
    kind: sin
    root: /data/secondary
    model: base
    relabel: /data/relabel/secondary_relabel.csv
    relabel_key: path
    exclude_primary_identities: true
    near_dup_cos: 0.95
```

The D2 configs need no map: `data: {kind: sin, root: /data/corpus_d2, model: fingervit}`.

### `kind: keyed_npz`

Config fields: `kind: keyed_npz`, `path`.

- **File.** One `.npz` file, loaded with `allow_pickle=False`.
- **Keys.** Each key is `<identity>_<impression>`, and each value is a one-dimensional vector of the same length `d`.
- **Loading.** The loader sorts the keys and stacks the vectors into an `(n, d)` matrix. It takes the identity as the part of the key before the last underscore (`key.rsplit("_", 1)[0]`).

Example: `data: {kind: keyed_npz, path: /data/bench/<model>_result/features.npz}`.

### `kind: features_by_id` (scale runs, distractors only)

Config fields: `kind: features_by_id`, `path`, `sizes`, `seed`, `near_dup_cos`.

- **File.** A `.npz` file with a key `features_by_id` that holds a pickled dictionary `{identity: [vector, ...]}`.
- **Loading.** The loader sorts the identities, draws `n` identities without replacement (seeded by `seed`, with `n` the largest requested size), and keeps the first vector of each.
- **Safety.** This loader uses `allow_pickle=True`, which can execute code embedded in the file. Use it only on files you produced yourself, inside the read-only, network-less container; never on downloaded or third-party files.

### `kind: synthetic` and `kind: synthetic_gaussian`

For tests and smoke runs only. Neither reads any file.

- **`synthetic`** takes `n_ids`, `per_id`, `dim`, `decay` and `seed`, and generates Gaussian identity clusters with a power-law spectrum (`decay: 0` is isotropic).
- **`synthetic_gaussian`** (scale distractors only) takes `dim` and `seed`, and generates standard normal rows.

### Where each kind is accepted

- **Isotropy runs** (`ijiss_btp.experiments.isotropy`): `data` and `data.shifted_aux` accept `sin`, `keyed_npz` and `synthetic`.
- **Scale runs** (`ijiss_btp.experiments.hnsw_scale`): `data` accepts the same kinds; `distractors` accepts `sin`, `features_by_id`, `synthetic` and `synthetic_gaussian`.
- **Targets** need at least `min_tgt_images` impressions per identity (3 in the released configs). Per `IdentitySplit` in `src/ijiss_btp/data/splits.py`, row 1 is the enrolled impression, row 2 the colluder/probe impression and row 3 a second probe.
