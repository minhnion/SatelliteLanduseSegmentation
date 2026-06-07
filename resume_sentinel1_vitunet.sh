#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  exec "$ROOT_DIR/finetune_sentinel1_vitunet_from_s2_best.sh" "$@"
fi

if [[ -z "${RESUME_FROM_RUN:-}" && $# -gt 0 ]]; then
  export RESUME_FROM_RUN="$1"
fi

if [[ -n "${RESUME_FROM_RUN:-}" ]]; then
  exec "$ROOT_DIR/continue_sentinel1_vitunet_from_run_best.sh"
fi

echo "Set RESUME_CHECKPOINT=<run>/checkpoints/latest_training_state.pth to continue an interrupted run," >&2
echo "or pass a previous run dir / set RESUME_FROM_RUN to finetune from that run's best weights." >&2
exit 1
