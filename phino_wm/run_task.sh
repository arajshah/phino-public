#!/usr/bin/env bash
# Launch the 3 DINO-WM encoder runs for ONE task across the 2 H100s, in tmux.
# Each run is light (frozen ViT-B ~2GB), so we pack [GPU0, GPU1, GPU0].
#
# Usage:   phino_wm/run_task.sh <pusht|ogbcube|ogbscene> [extra hydra overrides...]
# Example: phino_wm/run_task.sh pusht trainer.max_epochs=3
#
# W&B: logs to project "phino-dinowm" under your default entity (run `wandb login`
#      first). Disable with:  phino_wm/run_task.sh pusht wandb.enabled=false
#
# Monitor:  tmux ls ; tmux attach -t dwm_pusht_phino_full
#           tail -f phino_wm/logs/pusht_phino_full.log
set -euo pipefail

TASK="${1:?usage: run_task.sh <pusht|ogbcube|ogbscene> [overrides...]}"; shift || true
EXTRA="$*"

REPO=/home/colligo/phino
SWM="$REPO/stable-worldmodel"
LOGS="$REPO/phino_wm/logs"; mkdir -p "$LOGS"

# encoder tag -> config-name suffix ; GPU assignment
ENCODERS=(baseline phino_full phino_norest)
SUFFIX=(baseline full norest)
GPUS=(0 1 0)

# Weights & Biases (entity=null -> your default entity). Override via EXTRA.
WANDB="wandb.enabled=true wandb.config.project=phino-dinowm wandb.config.entity=null +wandb.config.group=${TASK}"

for i in "${!ENCODERS[@]}"; do
  enc="${ENCODERS[$i]}"; cfg="phino_${TASK}_${SUFFIX[$i]}"; gpu="${GPUS[$i]}"
  sess="dwm_${TASK}_${enc}"; log="$LOGS/${TASK}_${enc}.log"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$SWM python $REPO/phino_wm/train_dinowm.py \
        --config-name $cfg trainer.devices=1 trainer.strategy=auto $WANDB $EXTRA \
        2>&1 | tee $log"
  echo "launched $sess on GPU$gpu  (config=$cfg)  log=$log"
done
echo "---"; tmux ls | grep "dwm_${TASK}_" || true
