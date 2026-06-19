#!/usr/bin/env bash
# Evaluate the 3 trained DINO-WM encoders for a task (CEM planning) and emit a
# success-rate table. Run AFTER training and BEFORE deleting the task dataset
# (eval samples start/goal states from it).
#
# Usage:   phino_wm/benchmark.sh <pusht|ogbcube|ogbscene> [epoch=10] [num_eval=50]
# Output:  phino_wm/results/<task>_benchmark.md  (+ per-run eval logs)
set -euo pipefail

TASK="${1:?usage: benchmark.sh <pusht|ogbcube|ogbscene> [epoch] [num_eval]}"
EPOCH="${2:-10}"; NUM="${3:-50}"

REPO=/home/colligo/phino; SWM="$REPO/stable-worldmodel"
RESULTS="$REPO/phino_wm/results"; mkdir -p "$RESULTS"
OUT="$RESULTS/${TASK}_benchmark.md"

# task -> (plan config name, eval dataset .lance path under datasets cache)
case "$TASK" in
  pusht)    PLAN=pusht; DS=galilai-group--lewm-pusht/pusht_expert_train.lance ;;
  ogbcube)  PLAN=cube;  DS=galilai-group--ogb_cube_single/ogb_cube_single.lance ;;
  ogbscene) PLAN=scene; DS=galilai-group--ogb_scene_single/ogb_scene_single.lance ;;
  *) echo "unknown task $TASK"; exit 1 ;;
esac

{ echo "# ${TASK} — DINO-WM success rate (epoch ${EPOCH}, ${NUM} eval episodes)";
  echo; echo "| encoder | success_rate (%) |"; echo "|---|---|"; } > "$OUT"

for enc in baseline phino_full phino_norest; do
  run="${TASK}_${enc}"; log="$RESULTS/eval_${run}.log"
  echo "== eval $run =="
  CUDA_VISIBLE_DEVICES=0 SDL_VIDEODRIVER=dummy PYTHONPATH="$SWM" MUJOCO_GL=egl \
    python "$REPO/phino_wm/eval_dinowm.py" --config-name "$PLAN" \
      policy="${run}/weights_epoch_${EPOCH}.pt" \
      eval.dataset_name="$DS" eval.num_eval="$NUM" \
      > "$log" 2>&1 || echo "  eval failed (see $log)"
  sr=$(grep -oE "'success_rate': [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$" || true)
  echo "| ${enc} | ${sr:-FAIL} |" >> "$OUT"
done

echo "---"; cat "$OUT"
