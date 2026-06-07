#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "/home/azureuser/anaconda3/envs/landuse/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/azureuser/anaconda3/envs/landuse/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/finetune_runs/sentinel1_vitunet_from_s2}"
RUN_NAME="${RUN_NAME:-s1_vitunet_from_s2_quality}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/${RUN_NAME}_${TIMESTAMP}}"

GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-0.00005}"
DEPTH="${DEPTH:-12}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.2}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
MONITOR_METRIC="${MONITOR_METRIC:-val_miou}"
SAVE_EVERY="${SAVE_EVERY:-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PATIENCE="${PATIENCE:-30}"
EARLY_STOP="${EARLY_STOP:-1}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
TF32="${TF32:-0}"
SAVE_FULL_SNAPSHOTS="${SAVE_FULL_SNAPSHOTS:-0}"
DRY_RUN="${DRY_RUN:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PRETRAINED="${PRETRAINED:-}"
RESUME_FROM_RUN="${RESUME_FROM_RUN:-}"

if [[ "${PRETRAINED,,}" == "none" ]]; then
  PRETRAINED=""
fi

if [[ -n "$RESUME_CHECKPOINT" && -z "$PRETRAINED" ]]; then
  :
elif [[ -n "$RESUME_FROM_RUN" && -z "$PRETRAINED" ]]; then
  if [[ -f "$RESUME_FROM_RUN/checkpoints/best_model_cpu.pth" ]]; then
    PRETRAINED="$RESUME_FROM_RUN/checkpoints/best_model_cpu.pth"
  elif [[ -f "$RESUME_FROM_RUN/checkpoints/best_model.pth" ]]; then
    PRETRAINED="$RESUME_FROM_RUN/checkpoints/best_model.pth"
  else
    echo "Cannot find best_model_cpu.pth or best_model.pth in: $RESUME_FROM_RUN/checkpoints" >&2
    exit 1
  fi
elif [[ -z "$PRETRAINED" ]]; then
  PRETRAINED="$ROOT_DIR/inference_model/model.pth"
fi

if [[ -n "$PRETRAINED" && ! -f "$PRETRAINED" ]]; then
  echo "Pretrained checkpoint not found: $PRETRAINED" >&2
  exit 1
fi
if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Resume checkpoint not found: $RESUME_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/logs"

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/main.py"
  --cuda
  --gpu_id "$GPU_ID"
  --model ViTUnet
  --dataset north_vn
  --data_path "$DATA_PATH"
  --run_dir "$RUN_DIR"
  --experiment_name "$RUN_NAME"
  --epoch "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --depth "$DEPTH"
  --heads "$HEADS"
  --dropout "$DROPOUT"
  --optimizer "$OPTIMIZER"
  --weight_decay "$WEIGHT_DECAY"
  --monitor_metric "$MONITOR_METRIC"
  --save_every "$SAVE_EVERY"
  --num_workers "$NUM_WORKERS"
  --patience "$PATIENCE"
)

if [[ "$EARLY_STOP" == "1" ]]; then
  CMD+=(--early_stop)
fi
if [[ "$DISABLE_WANDB" == "1" ]]; then
  CMD+=(--disable_wandb)
fi
if [[ "$TF32" == "1" ]]; then
  CMD+=(--tf32)
fi
if [[ "$SAVE_FULL_SNAPSHOTS" == "1" ]]; then
  CMD+=(--save_full_snapshots)
fi
if [[ -n "$PRETRAINED" ]]; then
  CMD+=(--pretrained "$PRETRAINED")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi

echo "Run dir: $RUN_DIR"
echo "Data path: $DATA_PATH"
echo "Pretrained: ${PRETRAINED:-<none>}"
echo "Resume checkpoint: ${RESUME_CHECKPOINT:-<none>}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, command was not executed."
  exit 0
fi

"${CMD[@]}" | tee "$RUN_DIR/logs/train_${TIMESTAMP}.log"
