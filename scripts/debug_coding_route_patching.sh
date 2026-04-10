#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

resolve_frozen_root() {
  local candidate
  for candidate in \
    "${FROZEN_ROOT:-}" \
    "$REPO_ROOT/pressuretrace-frozen/coding_v1_qwen-qwen3-14b_off_v1_behavior_final" \
    "$REPO_ROOT/pressuretrace-frozen/coding_v1_qwen-qwen3-14b_off_neutralcue_tune_01" \
    "$REPO_ROOT/pressuretrace-frozen/coding_v1_qwen-qwen3-14b_off"
  do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if [[ -f "$candidate/data/manifests/coding_paper_slice_qwen-qwen3-14b_off.jsonl" ]] && \
       [[ -f "$candidate/results/coding_paper_slice_qwen-qwen3-14b_off.jsonl" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "${FROZEN_ROOT:-$REPO_ROOT/pressuretrace-frozen/coding_v1_qwen-qwen3-14b_off}"
}

FROZEN_ROOT="$(resolve_frozen_root)"
PRESSURE_TYPE="${PRESSURE_TYPE:-neutral_wrong_answer_cue}"
LAYER="${LAYER:--8}"
POSITION_WINDOW="${POSITION_WINDOW:-gen_1}"
PAIR_INDEX="${PAIR_INDEX:-0}"
PATCH_PAIRS_PATH="$FROZEN_ROOT/results/coding_patch_pairs_qwen-qwen3-14b_off.jsonl"

echo "Frozen root: $FROZEN_ROOT"
echo "Pressure type: $PRESSURE_TYPE"
echo "Layer: $LAYER"
echo "Position window: $POSITION_WINDOW"
echo "Patch pairs: $PATCH_PAIRS_PATH"

uv run pressuretrace coding-patch-pairs-v1 \
  --results-path "$FROZEN_ROOT/results/coding_paper_slice_qwen-qwen3-14b_off.jsonl" \
  --control-slice-path "$FROZEN_ROOT/data/splits/coding_control_robust_slice_qwen-qwen3-14b_off.jsonl" \
  --output-path "$PATCH_PAIRS_PATH" \
  --pressure-types "$PRESSURE_TYPE"

DEBUG_ARGS=(
  uv run pressuretrace coding-route-patching-debug-v1
  --frozen-root "$FROZEN_ROOT"
  --patch-pairs-path "$PATCH_PAIRS_PATH"
  --pressure-types "$PRESSURE_TYPE"
  --layer "$LAYER"
  --position-window "$POSITION_WINDOW"
  --pair-index "$PAIR_INDEX"
)

if [[ -n "${BASE_TASK_ID:-}" ]]; then
  DEBUG_ARGS+=(--base-task-id "$BASE_TASK_ID")
fi

"${DEBUG_ARGS[@]}"
