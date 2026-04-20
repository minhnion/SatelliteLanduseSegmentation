#!/bin/bash
set -euo pipefail

ROOT_DIR="/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR}"
EXPORT_PATH="${EXPORT_PATH:-$ROOT_DIR/inference_model/model_sentinel1_vitunet_scratch.pth}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-0.0001}"
DEPTH="${DEPTH:-12}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sentinel1_vitunet_scratch}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
PRETRAINED="${PRETRAINED:-}"
EARLY_STOP="${EARLY_STOP:-1}"
PATIENCE="${PATIENCE:-5}"

CMD=(
  python main.py
  --cuda
  --gpu_id "$GPU_ID"
  --model ViTUnet
  --dataset north_vn
  --data_path "$DATA_PATH"
  --log_path "$LOG_PATH"
  --epoch "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --depth "$DEPTH"
  --heads "$HEADS"
  --dropout "$DROPOUT"
  --experiment_name "$EXPERIMENT_NAME"
  --export_inference_path "$EXPORT_PATH"
  --patience "$PATIENCE"
)

if [[ "$DISABLE_WANDB" == "1" ]]; then
  CMD+=(--disable_wandb)
fi

if [[ "$EARLY_STOP" == "1" ]]; then
  CMD+=(--early_stop)
fi

if [[ -n "$PRETRAINED" ]]; then
  CMD+=(--pretrained "$PRETRAINED")
fi

"${CMD[@]}"
