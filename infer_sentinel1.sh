#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
INPUT="${INPUT:-$ROOT_DIR/inference_tif}"
OUTPUT="${OUTPUT:-$ROOT_DIR/inference_png/sentinel1_best}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-$ROOT_DIR/inference_model/model_sentinel1_best.pth}"
MODEL="${MODEL:-UNet}"
GPU_ID="${GPU_ID:-0}"
PATCH_SIZE="${PATCH_SIZE:-512}"
MODEL_INPUT_SIZE="${MODEL_INPUT_SIZE:-512}"

CMD=(
  python infer.py
  --cuda
  --gpu_id "$GPU_ID"
  --patch_size "$PATCH_SIZE"
  --model_input_size "$MODEL_INPUT_SIZE"
  --model "$MODEL"
  --input "$INPUT"
  --output "$OUTPUT"
  --pretrained "$PRETRAINED_MODEL"
)

if [[ -n "${STRIDE:-}" ]]; then
  CMD+=(--stride "$STRIDE")
fi

if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "$LIMIT")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry_run)
fi

if [[ "${FP16:-0}" == "1" ]]; then
  CMD+=(--fp16)
fi

if [[ "${PRESERVE_DIRS:-0}" == "1" ]]; then
  CMD+=(--preserve_dirs)
fi

"${CMD[@]}"
