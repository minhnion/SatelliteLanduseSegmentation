#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif [[ -x "/mnt/disk1/aiotlab/envs/landuse/bin/python" ]]; then
  PYTHON_BIN="/mnt/disk1/aiotlab/envs/landuse/bin/python"
else
  echo "Python executable not found. Activate the landuse environment or set PYTHON_BIN." >&2
  exit 1
fi

INPUT="${INPUT:-$ROOT_DIR/inference_tif}"
OUTPUT="${OUTPUT:-$ROOT_DIR/inference_png/sentinel1_best}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-$ROOT_DIR/inference_model/model_sentinel1_best.pth}"
MODEL="${MODEL:-UNet}"
GPU_ID="${GPU_ID:-0}"
PATCH_SIZE="${PATCH_SIZE:-140}"
MODEL_INPUT_SIZE="${MODEL_INPUT_SIZE:-512}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-4}"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/infer.py"
  --cuda
  --gpu_id "$GPU_ID"
  --patch_size "$PATCH_SIZE"
  --model_input_size "$MODEL_INPUT_SIZE"
  --patch_batch_size "$PATCH_BATCH_SIZE"
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
