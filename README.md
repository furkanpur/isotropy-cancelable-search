# Embedding isotropy as a security parameter for shared-key cancelable fingerprint search — code and aggregate results

Manuscript under review. The citation will be added on publication.

This repository contains the library, the experiment configurations, the aggregate results and the generators behind the paper. It contains no embeddings and no images; see [DATA.md](DATA.md).

## Contents

- `src/ijiss_btp` is the library: pre-conditioners, schemes, attacks, evaluation and HNSW search.
- `configs/` holds the experiment configs.
- `docker/` holds the container and the runner. Experiments run inside the container with read-only data, no network and the invoking user.
- `results/isotropy/` holds the per-run CSV/JSON outputs of the real-embedding runs, the summaries and `RESULTS.md`.
- `results/joint/` holds the joint alignment-plus-anchor stage (real embeddings): one `joint.csv` and `metadata.json` per run and `PARITY.md`, the check that its anchor-only projection decoder reproduces `results/isotropy/`; `RESULTS.md` sections 13 and 13b summarise it.
- `results/scale/` holds the HNSW scale runs. One run uses real distractors from the secondary corpus; the other uses **synthetic** distractors.
- `results/data/` holds the JSON summaries of the data preparation: the D1 and secondary-corpus relabelling, the D2 extraction and its label checks, the secondary-corpus composition, the training-overlap and clean-subset checks, and, in `results/data/control/`, the training set, training record, D2 embedding record, overlap audit and subject check of the control encoder ctrl-vit. See [DATA.md](DATA.md).
- `scripts/train/` holds the code that builds the training set of ctrl-vit, fetches its pretrained weights, trains it and embeds with it.
- `docs/theory/proofcheck/` holds the **synthetic** simulations that check the theory.
- `scripts/` holds the generators.
- `tests/` holds the tests.

## Data source labels

Every result in `results/isotropy/`, `results/joint/` and the secondary-corpus scale run comes from real fingerprint embeddings. The synthetic-distractor scale run and the proofcheck simulations are synthetic.

## Not included

No embeddings and no images. See [DATA.md](DATA.md) for the reasons and for the input format that the experiment runners expect.

## Quick start

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Run every command from the repository root.

```bash
uv sync --frozen                       # create .venv from uv.lock

uv run pytest tests -q                 # library tests

# Regenerate results/isotropy/RESULTS.md from the released aggregate results
uv run python scripts/figures/results_tables.py --md results/isotropy/RESULTS.md

# Regenerate the figures (PDF) and their point tables into build/figures
uv run python scripts/figures/isotropy_figures.py \
    --summary results/isotropy/summary \
    --runs $(ls -d results/isotropy/isotropy_d*-* | grep -v _fmr1) \
    --out build/figures

# Regenerate the numbers and tables used by the paper (written to manuscript/generated/)
uv run python scripts/manuscript/make_numbers.py
uv run python scripts/manuscript/make_tables.py
```

`make_numbers.py` reads the run results and, through `scripts/manuscript/data_numbers.py`, the data aggregates in `results/data/` (the corpus relabelling, composition, training-overlap and clean-subset numbers), so every result number and every generated table of the paper is regenerated from this repository.

## Rerunning the experiments on your own embeddings

Write a config modelled on `configs/isotropy_d1_base.yaml` (isotropy study) or `configs/scale_d1_base_secondary.yaml` (HNSW scale run) that points to your data, as described in [DATA.md](DATA.md). Build the image once, then run the experiment through `docker/run.sh`:

```bash
docker build -t canvec/exp:core -f docker/Dockerfile docker/

# usage: docker/run.sh <host_out_dir> <command> [args...]
docker/run.sh ~/canvec-runs/my-run \
    python -m ijiss_btp.experiments.isotropy --config configs/isotropy_d1_base.yaml --out /out

# HNSW scale run
docker/run.sh ~/canvec-runs/my-scale-run \
    python -m ijiss_btp.experiments.hnsw_scale --config configs/scale_d1_base_secondary.yaml --out /out
```

`docker/run.sh` mounts the repository read-only at `/work`, the output directory at `/out`, and the data under `/data/primary`, `/data/secondary`, `/data/bench` and `/data/scale`, plus `/data/relabel` when `CANVEC_RELABEL_ROOT` is set (an optional directory of identity relabel maps, mounted read-only; a data spec may point at one with `relabel:`, and its SHA-256 is recorded in `metadata.json`) and `/data/corpus_d2` when `CANVEC_D2_ROOT` is set (the read-only output directory of the D2 embedding extraction, in the `kind: sin` layout). The roots on the host are set by `CANVEC_SIN_ROOT` (mounted as `/data/primary` from `results_qf/` and `/data/secondary` from `results/` inside it), `CANVEC_BENCH_ROOT` and `CANVEC_SCALE_ROOT`; edit the mounts to match your layout. Set `CANVEC_CPUS` and `CANVEC_MEM` to fit your machine (defaults 16 CPUs, 32 GB). Each run writes CSV/JSON files and a `metadata.json` with the seed, configuration, data-file hashes and environment. The `git_sha` recorded for the released runs refers to the authors' working repository, of which this repository is a curated snapshot. In this snapshot, the names that referred to the compilation from which D2 is drawn were replaced by neutral ones; see [DATA.md](DATA.md#names-in-this-release).

The D2 embeddings are extracted from fingerprint images by `scripts/data/extract_d2.py` in a separate GPU image, `docker/Dockerfile.extract` (CUDA base image, pinned packages, the flx package copied into the build context); the script's docstring lists the preprocessing, the parity check against the earlier features and the selection rules. It runs with the images and the weights mounted read-only, `--network none` and `--gpus all`, and writes the embeddings only to its output directory. The control encoder ctrl-vit is trained and applied by `scripts/train/train_control.py` (`train`, `embed`) in the same image and under the same conditions; its training set comes from `scripts/train/build_training_set.py`, and only `scripts/train/fetch_weights.py`, which downloads the pretrained DINOv2 weights, uses the network, in a container that mounts no data.

## Related artifact

The D1 images and the base-encoder measurements come from the SIN 2026 artifact: <https://github.com/furkanpur/cancelable-vit-fingerprint>. The finger-level identity labels used in this study were re-derived from each source's native naming (see [DATA.md](DATA.md)); the label maps are not public, only their aggregates in `results/data/` are.

## Encoders

Three ViT-based encoders are described in a separate manuscript under review. DeepPrint uses the public flx implementation with its released weights, without retraining. The fifth encoder, ctrl-vit, is a control ViT trained by us with `scripts/train/`: DINOv2 ViT-B/14 initialisation, fine-tuned with an ArcFace head on the compilation images of the D1 fingers, which are disjoint from D2's fingers and images by identity labels and exact content. It is an in-domain control, because it shares most of D2's sources and, for most D2 fingers, a subject key of the compilation (which does not separate persons in every source), so its accuracy on D2 is not a measure of out-of-domain generalisation. It comes from a single training run, and its trained weights are not released, because they derive from licensed biometric data. See [DATA.md](DATA.md) for its training data, audits and subject check.

## License

MIT; see [LICENSE](LICENSE).

## Authors

Sezai Furkan Pur, Bilgehan Arslan, Seref Sagiroglu. Department of Computer Engineering, Gazi University.
