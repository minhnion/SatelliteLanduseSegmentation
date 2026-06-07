#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${RESUME_FROM_RUN:-}" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <previous_run_dir>" >&2
    echo "Example: $0 finetune_runs/sentinel1_vitunet_from_s2/s1_vitunet_from_s2_quality_20260607_120000" >&2
    exit 1
  fi
  RESUME_FROM_RUN="$1"
fi

if [[ ! -d "$RESUME_FROM_RUN" ]]; then
  echo "Previous run directory not found: $RESUME_FROM_RUN" >&2
  exit 1
fi

export RESUME_FROM_RUN
export RUN_NAME="${RUN_NAME:-$(basename "$RESUME_FROM_RUN")_continue_from_best}"
exec "$ROOT_DIR/finetune_sentinel1_vitunet_from_s2_best.sh"
