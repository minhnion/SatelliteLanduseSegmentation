#!/bin/bash
set -euo pipefail

ROOT_DIR="/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation"
SOURCE="${SOURCE:-s1}"
S1_PRETRAINED_DEFAULT="$ROOT_DIR/weights/datasetcfg:north_vn/data:dataset/channels:2/model:ViTUnet/run:sentinel1_vitunet_scratch/datetime:20260409_223308_epoch:100_bs:2_lr:0.0001/weight.pth"
S2_PRETRAINED_DEFAULT="$ROOT_DIR/inference_model/model.pth"

if [[ "$SOURCE" == "s2" ]]; then
  PRETRAINED_DEFAULT="$S2_PRETRAINED_DEFAULT"
  EXPORT_PATH_DEFAULT="$ROOT_DIR/inference_model/model_sentinel1_vitunet_from_s2_finetune.pth"
  EXPERIMENT_NAME_DEFAULT="sentinel1_vitunet_finetune_from_s2_ckpt"
else
  PRETRAINED_DEFAULT="$S1_PRETRAINED_DEFAULT"
  EXPORT_PATH_DEFAULT="$ROOT_DIR/inference_model/model_sentinel1_vitunet_resume_aug.pth"
  EXPERIMENT_NAME_DEFAULT="sentinel1_vitunet_resume_aug"
fi

PRETRAINED="${PRETRAINED:-$PRETRAINED_DEFAULT}"
EXPORT_PATH="${EXPORT_PATH:-$EXPORT_PATH_DEFAULT}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$EXPERIMENT_NAME_DEFAULT}"
EPOCHS="${EPOCHS:-15}"
LR="${LR:-0.00001}"
PATIENCE="${PATIENCE:-5}"

PRETRAINED="$PRETRAINED" \
EXPORT_PATH="$EXPORT_PATH" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
EPOCHS="$EPOCHS" \
LR="$LR" \
PATIENCE="$PATIENCE" \
EARLY_STOP=1 \
bash "$ROOT_DIR/train_sentinel1_vitunet.sh"
