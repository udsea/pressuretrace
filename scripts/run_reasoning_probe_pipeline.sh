#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

resolve_frozen_root() {
  local candidate
  for candidate in \
    "${FROZEN_ROOT:-}" \
    "/Dev/tinkering/pressuretrace/pressuretrace-frozen" \
    "/Dev/tinkering/pressuretrace/pressuretrace-frozen/reasoning_v2_qwen3_14b_off" \
    "$REPO_ROOT/pressuretrace-frozen" \
    "$REPO_ROOT/pressuretrace-frozen/reasoning_v2_qwen3_14b_off"
  do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if [[ -f "$candidate/data/manifests/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl" ]] && \
       [[ -f "$candidate/results/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "${FROZEN_ROOT:-$REPO_ROOT/pressuretrace-frozen/reasoning_v2_qwen3_14b_off}"
}

FROZEN_ROOT="$(resolve_frozen_root)"
MANIFEST_PATH="$FROZEN_ROOT/data/manifests/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl"
RESULTS_PATH="$FROZEN_ROOT/results/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl"
HIDDEN_STATES_PATH="$FROZEN_ROOT/results/reasoning_probe_hidden_states_qwen-qwen3-14b_off.jsonl"
DATASET_PATH="$FROZEN_ROOT/results/reasoning_probe_dataset_qwen-qwen3-14b_off.jsonl"
METRICS_PATH="$FROZEN_ROOT/results/reasoning_probe_metrics_qwen-qwen3-14b_off.jsonl"
SUMMARY_PATH="$FROZEN_ROOT/results/reasoning_probe_summary_qwen-qwen3-14b_off.txt"

resolve_cache_root() {
  local user_name="${USER:-$(id -un)}"
  local candidate
  for candidate in \
    "/scratch/${user_name}/pressuretrace_cache" \
    "/local_scratch/${user_name}/pressuretrace_cache" \
    "/tmp/${user_name}/pressuretrace_cache"
  do
    mkdir -p "${candidate}" >/dev/null 2>&1 && {
      echo "${candidate}"
      return
    }
  done
  echo "${PWD}/.pressuretrace_cache"
}

CACHE_ROOT="${CACHE_ROOT:-$(resolve_cache_root)}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

mkdir -p "$FROZEN_ROOT/results"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"

echo "Frozen root: $FROZEN_ROOT"
echo "Manifest: $MANIFEST_PATH"
echo "Results: $RESULTS_PATH"
echo "Hidden states: $HIDDEN_STATES_PATH"
echo "Dataset: $DATASET_PATH"
echo "Metrics: $METRICS_PATH"
echo "Summary: $SUMMARY_PATH"
echo "HF cache: $HF_HOME"

export HF_HOME HF_HUB_CACHE HF_XET_CACHE TMPDIR HF_HUB_DISABLE_XET TOKENIZERS_PARALLELISM TRANSFORMERS_VERBOSITY
uv run python -m pressuretrace.probes.extract_hidden_states_reasoning \
  --manifest-path "$MANIFEST_PATH" \
  --results-path "$RESULTS_PATH" \
  --output-path "$HIDDEN_STATES_PATH"

uv run python -m pressuretrace.probes.build_reasoning_probe_dataset \
  --input-path "$HIDDEN_STATES_PATH" \
  --output-path "$DATASET_PATH"

uv run python -m pressuretrace.probes.train_reasoning_probes \
  --input-path "$DATASET_PATH" \
  --metrics-path "$METRICS_PATH" \
  --summary-path "$SUMMARY_PATH"
