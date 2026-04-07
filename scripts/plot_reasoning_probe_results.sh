#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

resolve_frozen_root() {
  local candidate
  for candidate in \
    "${FROZEN_ROOT:-}" \
    "/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off" \
    "$REPO_ROOT/pressuretrace-frozen/reasoning_v2_qwen3_14b_off"
  do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if [[ -f "$candidate/results/reasoning_probe_metrics_qwen-qwen3-14b_off.jsonl" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "${FROZEN_ROOT:-$REPO_ROOT/pressuretrace-frozen/reasoning_v2_qwen3_14b_off}"
}

FROZEN_ROOT="$(resolve_frozen_root)"
METRICS_PATH="${METRICS_PATH:-$FROZEN_ROOT/results/reasoning_probe_metrics_qwen-qwen3-14b_off.jsonl}"
DATASET_PATH="${DATASET_PATH:-$FROZEN_ROOT/results/reasoning_probe_dataset_qwen-qwen3-14b_off.jsonl}"
SUMMARY_PATH="${SUMMARY_PATH:-$FROZEN_ROOT/results/reasoning_probe_summary_qwen-qwen3-14b_off.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$FROZEN_ROOT/results}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pressuretrace-mpl-cache}"

mkdir -p "$MPLCONFIGDIR"

echo "Frozen root: $FROZEN_ROOT"
echo "Metrics: $METRICS_PATH"
echo "Dataset: $DATASET_PATH"
echo "Summary: $SUMMARY_PATH"
echo "Output dir: $OUTPUT_DIR"

uv run python -m pressuretrace.analysis.plot_reasoning_probe_results \
  --metrics-path "$METRICS_PATH" \
  --dataset-path "$DATASET_PATH" \
  --summary-path "$SUMMARY_PATH" \
  --output-dir "$OUTPUT_DIR"
