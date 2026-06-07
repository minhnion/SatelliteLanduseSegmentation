#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-s1_vitunet_scratch}"
export PRETRAINED="${PRETRAINED:-none}"
exec "$ROOT_DIR/finetune_sentinel1_vitunet_from_s2_best.sh" "$@"
