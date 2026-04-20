#!/bin/bash
set -euo pipefail

ROOT_DIR="/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation"
INPUT="${INPUT:-$ROOT_DIR/inference_tif/Resolution3x3}"
OUTPUT="${OUTPUT:-$ROOT_DIR/inference_png/Resolution3x3_sentinel1_vitunet_resume_aug}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-$ROOT_DIR/inference_model/model_sentinel1_vitunet_resume_aug.pth}"
MODEL="${MODEL:-UNet}"
GPU_ID="${GPU_ID:-0}"
PATCH_SIZE="${PATCH_SIZE:-128}"

python infer.py \
  --cuda \
  --gpu_id "$GPU_ID" \
  --patch_size "$PATCH_SIZE" \
  --model "$MODEL" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --pretrained "$PRETRAINED_MODEL"
