#!/bin/bash
set -euo pipefail

ROOT_DIR="/mnt/disk1/aiotlab/anhnd2468/SatelliteLanduseSegmentation"
INPUT="${INPUT:-$ROOT_DIR/inference_png/Resolution3x3_sentinel1_vitunet_resume_aug}"
OUTPUT="${OUTPUT:-$ROOT_DIR/area_output/Resolution3x3_sentinel1_vitunet_resume_aug}"
PROVINCE_LABEL="${PROVINCE_LABEL:-$ROOT_DIR/mapbox/gadm_resolution_3_province_mapbox_label.json}"
NORMALIZE_RATIOS="${NORMALIZE_RATIOS:-0}"

CMD=(
  python calculate_area.py
  --input "$INPUT"
  --output "$OUTPUT"
  --province_label "$PROVINCE_LABEL"
)

if [[ "$NORMALIZE_RATIOS" == "1" ]]; then
  CMD+=(--normalize_intersection_ratios)
fi

"${CMD[@]}"
