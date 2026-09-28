#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" != "--execute" ]]; then
  echo "Usage: FASTOPD_DATASET_ROOT=... FASTOPD_BASE_CHECKPOINT=... FASTOPD_TRAIN_ANCHOR_DIR=... FASTOPD_QWEN_MODEL=... FASTOPD_QWEN_PYTHON=... $0 --execute" >&2
  exit 2
fi
for name in FASTOPD_DATASET_ROOT FASTOPD_BASE_CHECKPOINT FASTOPD_TRAIN_ANCHOR_DIR FASTOPD_QWEN_MODEL FASTOPD_QWEN_PYTHON; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is missing: $name" >&2
    exit 2
  fi
done

GPU="${FASTOPD_GPU:-0}"
OUTPUT_DIR="${FASTOPD_OUTPUT_DIR:-$ROOT/outputs/full_fashioniq}"
mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="$GPU" python -m blip2_maxsim.train \
  --phase train --execute-qwen \
  --dataset-root "$FASTOPD_DATASET_ROOT" \
  --base-checkpoint "$FASTOPD_BASE_CHECKPOINT" \
  --train-anchor-dir "$FASTOPD_TRAIN_ANCHOR_DIR" \
  --qwen-python "$FASTOPD_QWEN_PYTHON" --qwen-model-path "$FASTOPD_QWEN_MODEL" \
  --qwen-gpu "$GPU" --student-physical-gpu "$GPU" \
  --qwen-gpu-memory-utilization 0.78 --qwen-cpu-offload-gb 0 \
  --qwen-max-num-seqs 8 --qwen-max-model-len 3072 \
  --qwen-max-num-batched-tokens 4096 --qwen-inference-batch-size 8 \
  --qwen-cmi-inference-batch-size 24 --cmi-three-branch --ranking-spec-k 3 \
  --qwen-sleep-between-requests \
  --epochs 1 --categories dress,shirt,toptee --seed 1234 \
  --train-batch-size 64 --max-train-batches 0 \
  --coarse-topk 100 --live-shortlist-size 20 --teacher-list-size 10 \
  --teacher-min-gt-rank 2 --teacher-max-gt-rank 20 \
  --list-micro-batch-size 32 --candidate-encode-batch-size 64 \
  --gt-weight 0.8 --list-weight 0.2 --qformer-lr 5e-6 --head-lr 2e-5 \
  --workers 12 --resume --checkpoint-percent-interval 10 \
  --experiment full_fashioniq_spec_k3_cmi_branch \
  --output-dir "$OUTPUT_DIR" "${@:2}"
