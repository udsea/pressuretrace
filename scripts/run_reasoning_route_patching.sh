#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

FROZEN_ROOT="${FROZEN_ROOT:-$REPO_ROOT/pressuretrace-frozen/reasoning_v2_qwen3_14b_off}"
MANIFEST_PATH="${MANIFEST_PATH:-$FROZEN_ROOT/data/manifests/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl}"
PAPER_RESULTS_PATH="${PAPER_RESULTS_PATH:-$FROZEN_ROOT/results/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl}"
PATCH_PAIRS_PATH="${PATCH_PAIRS_PATH:-$FROZEN_ROOT/results/reasoning_patch_pairs_qwen-qwen3-14b_off.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${RESULTS_PATH:-$FROZEN_ROOT/results/reasoning_route_patching_qwen-qwen3-14b_off.jsonl}}"
SUMMARY_TXT_PATH="${SUMMARY_TXT_PATH:-$FROZEN_ROOT/results/reasoning_route_patching_summary_qwen-qwen3-14b_off.txt}"
SUMMARY_CSV_PATH="${SUMMARY_CSV_PATH:-$FROZEN_ROOT/results/reasoning_route_patching_summary_qwen-qwen3-14b_off.csv}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
THINKING_MODE="${THINKING_MODE:-off}"
LAYERS="${LAYERS:--10,-8,-6}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/pressuretrace-uv-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pressuretrace-mpl-cache}"

mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR"

echo "Frozen root: $FROZEN_ROOT"
echo "Manifest: $MANIFEST_PATH"
echo "Paper results: $PAPER_RESULTS_PATH"
echo "Patch pairs: $PATCH_PAIRS_PATH"
echo "Model: $MODEL_NAME"
echo "Thinking mode: $THINKING_MODE"
echo "Layers: $LAYERS"
echo "Paper results: $PAPER_RESULTS_PATH"
echo "Output: $OUTPUT_PATH"
echo "Summary TXT: $SUMMARY_TXT_PATH"
echo "Summary CSV: $SUMMARY_CSV_PATH"

ARGS=(
  --frozen-root "$FROZEN_ROOT"
  --manifest-path "$MANIFEST_PATH"
  --results-path "$PAPER_RESULTS_PATH"
  --patch-pairs-path "$PATCH_PAIRS_PATH"
  --output-path "$OUTPUT_PATH"
  --summary-txt-path "$SUMMARY_TXT_PATH"
  --summary-csv-path "$SUMMARY_CSV_PATH"
  --model-name "$MODEL_NAME"
  --thinking-mode "$THINKING_MODE"
  "--layers=$LAYERS"
)

UV_CACHE_DIR="$UV_CACHE_DIR" MPLCONFIGDIR="$MPLCONFIGDIR" \
  uv run python -m pressuretrace.patching.run_reasoning_route_patching "${ARGS[@]}"
