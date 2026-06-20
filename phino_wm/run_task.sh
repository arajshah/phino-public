#!/usr/bin/env bash
# Train the DINO-WM encoder experiments for ONE task, running 2 in parallel —
# one single-GPU training per card — keeping both H100s saturated.
#
# Why not DDP (both cards per run)? In this environment Lightning's DDP deadlocks
# in the DataLoader (num_workers>0 + persistent_workers, hangs after CUDA/NCCL
# init). Single-GPU training is proven to work here, so we run independent
# single-GPU jobs in parallel instead: same aggregate throughput, no hangs.
#
# Encoder order: phino_full, phino_norest, baseline (baseline last — no official
# pretrained DINO-WM predictor exists to drop in).
#
# Speed: batch_size / LR are bumped over stock prejepa.yaml (32 / 5e-4). Override:
#   BATCH=96 LR=8e-4 phino_wm/run_task.sh pusht
#   GPUS="0 1"  (space-separated list of cards to use as parallel slots)
#
# Usage:   phino_wm/run_task.sh <pusht|ogbcube|ogbscene> [extra hydra overrides...]
set -euo pipefail

TASK="${1:?usage: run_task.sh <pusht|ogbcube|ogbscene> [overrides...]}"; shift || true
EXTRA="$*"

REPO=/home/colligo/phino
SWM="$REPO/stable-worldmodel"
PREJEPA="$SWM/scripts/train/prejepa.py"
# _ddp_shim/sitecustomize.py is auto-imported at interpreter startup to inject the
# optional CPUOffloadCallback that prejepa.py hardcodes but stable_pretraining
# 0.1.7 lacks — so prejepa.py runs unmodified.
PYPATH="$REPO/phino_wm/_ddp_shim:$SWM"
LOGS="$REPO/phino_wm/logs"; mkdir -p "$LOGS"

BATCH="${BATCH:-128}"          # stock prejepa.yaml = 32
LR="${LR:-1e-3}"             # stock prejepa.yaml = 5e-4
read -r -a SLOTS <<< "${GPUS:-0 1}"   # GPU ids used as parallel slots

ENC=(phino_full phino_norest baseline)
SUF=(full norest baseline)

WANDB="wandb.enabled=true wandb.config.project=phino-dinowm wandb.config.entity=null +wandb.config.group=${TASK}"

LAUNCHED_PID=""
launch() {  # $1=gpu  $2=queue index
  local gpu="$1" i="$2"
  local cfg="phino_${TASK}_${SUF[$i]}" log="$LOGS/${TASK}_${ENC[$i]}.log"
  echo "=== [$(date '+%F %T')] START ${cfg} on GPU${gpu}  (batch ${BATCH}, lr ${LR})  -> ${log} ==="
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYPATH" \
    python "$PREJEPA" --config-name "$cfg" \
      batch_size="$BATCH" optimizer.lr="$LR" trainer.devices=1 trainer.strategy=auto \
      $WANDB $EXTRA > "$log" 2>&1 &
  LAUNCHED_PID=$!
}

declare -A SLOT_PID            # gpu -> running pid
qi=0; N=${#ENC[@]}
while :; do
  # reap finished slots
  for gpu in "${SLOTS[@]}"; do
    pid="${SLOT_PID[$gpu]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then echo "=== [$(date '+%F %T')] DONE  GPU${gpu} (pid $pid) ==="
      else echo "=== [$(date '+%F %T')] FAILED GPU${gpu} (pid $pid) — see log ==="; fi
      unset 'SLOT_PID[$gpu]'
    fi
  done
  # fill free slots from the queue
  for gpu in "${SLOTS[@]}"; do
    if [[ -z "${SLOT_PID[$gpu]:-}" && $qi -lt $N ]]; then
      launch "$gpu" "$qi"; SLOT_PID[$gpu]="$LAUNCHED_PID"; qi=$((qi+1))
    fi
  done
  # done when queue drained and every slot idle
  [[ $qi -ge $N && ${#SLOT_PID[@]} -eq 0 ]] && break
  sleep 5
done
echo "=== [$(date '+%F %T')] All ${TASK} encoders complete ==="
