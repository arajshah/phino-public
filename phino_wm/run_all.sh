#!/usr/bin/env bash
# One command to train every PHINO x DINO-WM experiment end-to-end:
#   for task in pusht, ogbcube, ogbscene:
#     download dataset (if missing) -> train [phino_full, phino_norest, baseline]
#
# Within a task, 2 experiments run in parallel (one single-GPU job per card) to
# keep both H100s busy. Baseline is trained last in each task (no official
# pretrained predictor exists to drop in). See run_task.sh for why not DDP.
#
# DISK: local disk is small (~50G free) and ogbscene alone is ~38G, so by
# default only the CURRENT task's dataset is kept on disk — datasets from other
# tasks are deleted just before each task downloads. Set KEEP_ALL_DATA=1 to keep
# every dataset (needs a big disk; lets you run benchmark.sh between tasks).
#
# Usage:   phino_wm/run_all.sh [extra hydra overrides...]
# Example: BATCH=96 LR=8e-4 phino_wm/run_all.sh trainer.max_epochs=3
#
# Runs detached in tmux session 'dwm_all'.
#   Monitor:  tmux attach -t dwm_all
#             tail -f phino_wm/logs/pusht_phino_full.log
set -euo pipefail

REPO=/home/colligo/phino
DATA_DIR="$HOME/.stable_worldmodel/datasets"
EXTRA="$*"

TASKS=(pusht ogbcube ogbscene)
# task -> HF dataset repo
declare -A HF_REPO=(
  [pusht]=galilai-group/lewm-pusht
  [ogbcube]=galilai-group/ogb_cube_single
  [ogbscene]=galilai-group/ogb_scene_single
)
# task -> local dir name (== "<org>--<name>")
declare -A LOCAL_DIR=(
  [pusht]=galilai-group--lewm-pusht
  [ogbcube]=galilai-group--ogb_cube_single
  [ogbscene]=galilai-group--ogb_scene_single
)
# task -> the .lance subdir that signals "already downloaded"
declare -A LANCE=(
  [pusht]=pusht_expert_train.lance
  [ogbcube]=ogb_cube_single.lance
  [ogbscene]=ogb_scene_single.lance
)

run_pipeline() {
  for task in "${TASKS[@]}"; do
    dir="$DATA_DIR/${LOCAL_DIR[$task]}"
    lance="$dir/${LANCE[$task]}"

    # free disk: drop other tasks' datasets (unless told to keep everything)
    if [[ "${KEEP_ALL_DATA:-0}" != "1" ]]; then
      for other in "${TASKS[@]}"; do
        [[ "$other" == "$task" ]] && continue
        od="$DATA_DIR/${LOCAL_DIR[$other]}"
        if [[ -d "$od" ]]; then
          echo "=== [$(date '+%F %T')] freeing disk: rm $od ==="
          rm -rf "$od"
        fi
      done
    fi

    # download if the lance dir isn't there yet
    if [[ ! -d "$lance" ]]; then
      echo "=== [$(date '+%F %T')] downloading ${HF_REPO[$task]} -> $dir ==="
      hf download "${HF_REPO[$task]}" --repo-type dataset --local-dir "$dir"
    else
      echo "=== [$(date '+%F %T')] dataset present: $lance ==="
    fi

    echo "=== [$(date '+%F %T')] TASK $task: training all encoders ==="
    bash "$REPO/phino_wm/run_task.sh" "$task" $EXTRA
  done
  echo "=== [$(date '+%F %T')] ALL TASKS COMPLETE ==="
}

# If called with RUN_INLINE=1, execute directly (used inside tmux). Otherwise
# spawn a detached tmux session so the pipeline survives disconnects.
if [[ "${RUN_INLINE:-0}" == "1" ]]; then
  run_pipeline
else
  tmux kill-session -t dwm_all 2>/dev/null || true
  tmux new-session -d -s dwm_all \
    "RUN_INLINE=1 KEEP_ALL_DATA='${KEEP_ALL_DATA:-0}' BATCH='${BATCH:-}' LR='${LR:-}' bash '$REPO/phino_wm/run_all.sh' $EXTRA"
  echo "Launched tmux session 'dwm_all':"
  echo "  pusht -> ogbcube -> ogbscene  (each: phino_full, phino_norest, baseline; 2 parallel single-GPU)"
  echo "Monitor:  tmux attach -t dwm_all   |   tail -f phino_wm/logs/pusht_phino_full.log"
fi
