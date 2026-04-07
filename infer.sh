#!/bin/bash
INPUT="/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/inference_tif/Resolution3x3"
OUTPUT="/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/inference_png/Resolution3x3"
PRETRAINED_MODEL="/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation/inference_model/model.pth"
MODEL="UNet"
GPU_ID="${GPU_ID:-0}"
PATCH_SIZE="${PATCH_SIZE:-128}"
python infer.py --cuda --gpu_id "$GPU_ID" --patch_size "$PATCH_SIZE" --model "$MODEL" --input "$INPUT" --output "$OUTPUT" --pretrained "$PRETRAINED_MODEL"
