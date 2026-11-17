#!/usr/bin/env bash
# Run a command inside the CanVec experiment container.
#
# Usage (from the repository root):
#   docker/run.sh <host_out_dir> <command> [args...]
# Example:
#   docker/run.sh ~/canvec-runs/spike-001 python -m ijiss_btp.experiments.isotropy --config configs/isotropy_d1_base.yaml --out /out
#
# Guarantees:
# - real embeddings are mounted read-only under /data
# - the code is mounted read-only under /work
# - only <host_out_dir> is writable (mounted at /out)
# - no network access, runs as the invoking host user, bounded CPU/memory
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: docker/run.sh <host_out_dir> <command> [args...]" >&2
  exit 2
fi

OUT_DIR="$1"; shift
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

CPUS="${CANVEC_CPUS:-16}"
MEM="${CANVEC_MEM:-32g}"
IMAGE="${CANVEC_IMAGE:-canvec/exp:core}"
SIN_ROOT="${CANVEC_SIN_ROOT:-$HOME/sin2026-embeddings}"
BENCH_ROOT="${CANVEC_BENCH_ROOT:-$HOME/Workspaces/fingerprint-model-benchmark-tool/result}"
GIT_SHA="${CANVEC_GIT_SHA:-$(cat "$PWD/GIT_SHA" 2>/dev/null || echo unknown)}"

# Optional read-only data sources.
EXTRA_MOUNTS=()
if [[ -d "$BENCH_ROOT" ]]; then
  EXTRA_MOUNTS+=(-v "$BENCH_ROOT:/data/bench:ro")
fi
SCALE_ROOT="${CANVEC_SCALE_ROOT:-$HOME/Workspaces/fingerprint-evaluate-tools/output/ann_scalability_benchmark}"
if [[ -d "$SCALE_ROOT" ]]; then
  EXTRA_MOUNTS+=(-v "$SCALE_ROOT:/data/scale:ro")
fi
RELABEL_ROOT="${CANVEC_RELABEL_ROOT:-}"
if [[ -n "$RELABEL_ROOT" && -d "$RELABEL_ROOT" ]]; then
  EXTRA_MOUNTS+=(-v "$RELABEL_ROOT:/data/relabel:ro")
fi
# SIN-layout output of scripts/data/extract_d2.py (manifest.parquet, emb_fingervit.npy, emb_deepprint.npy).
D2_ROOT="${CANVEC_D2_ROOT:-}"
if [[ -n "$D2_ROOT" && -d "$D2_ROOT" ]]; then
  EXTRA_MOUNTS+=(-v "$D2_ROOT:/data/corpus_d2:ro")
fi

exec docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  --cpus "$CPUS" --memory "$MEM" \
  -e OMP_NUM_THREADS="$CPUS" -e OPENBLAS_NUM_THREADS="$CPUS" -e MKL_NUM_THREADS="$CPUS" \
  -e HOME=/tmp \
  -e PYTHONPATH=/work/src \
  -e CANVEC_GIT_SHA="$GIT_SHA" \
  -v "$SIN_ROOT/results_qf:/data/primary:ro" \
  -v "$SIN_ROOT/results:/data/secondary:ro" \
  ${EXTRA_MOUNTS[@]+"${EXTRA_MOUNTS[@]}"} \
  -v "$PWD:/work:ro" \
  -v "$OUT_DIR:/out" \
  -w /work \
  "$IMAGE" "$@"
