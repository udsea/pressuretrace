#!/usr/bin/env bash
set -euo pipefail

# Build manifests for the factual (TriviaQA) and logical (BBH logical_deduction)
# task families, run the benchmarks, and print per-pressure-type route summaries.
#
# Environment overrides:
#   MODEL_NAME       HF model id                      (default Qwen/Qwen3-14B)
#   THINKING_MODE    off | on | default               (default off)
#   FACTUAL_LIMIT    base tasks to load for factual   (default 300)
#   LOGICAL_LIMIT    base tasks to load for logical   (default 200)
#   RUN_LIMIT        cap episodes per family runner   (default unset = all)
#   RUN_TAG          timestamp tag for output dir     (default: date)
#   SKIP_FACTUAL     set to 1 to skip factual family
#   SKIP_LOGICAL     set to 1 to skip logical family

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

slugify() {
  local value="$1"
  echo "${value}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
THINKING_MODE="${THINKING_MODE:-off}"
FACTUAL_LIMIT="${FACTUAL_LIMIT:-300}"
LOGICAL_LIMIT="${LOGICAL_LIMIT:-200}"
RUN_LIMIT="${RUN_LIMIT:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SKIP_FACTUAL="${SKIP_FACTUAL:-0}"
SKIP_LOGICAL="${SKIP_LOGICAL:-0}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export TRANSFORMERS_VERBOSITY TOKENIZERS_PARALLELISM

MODEL_SLUG="$(slugify "${MODEL_NAME}")"
RUN_DIR="results/factual_logical_${RUN_TAG}"
mkdir -p "${RUN_DIR}"

FACTUAL_MANIFEST="data/manifests/factual_paper_slice_v1_validation_${FACTUAL_LIMIT}.jsonl"
LOGICAL_MANIFEST="data/manifests/logical_paper_slice_v1_test_${LOGICAL_LIMIT}.jsonl"

FACTUAL_RESULTS="${RUN_DIR}/factual_results_${MODEL_SLUG}_${THINKING_MODE}.jsonl"
LOGICAL_RESULTS="${RUN_DIR}/logical_results_${MODEL_SLUG}_${THINKING_MODE}.jsonl"

printf 'Model: %s\n' "${MODEL_NAME}"
printf 'Thinking mode: %s\n' "${THINKING_MODE}"
printf 'Run dir: %s\n' "${RUN_DIR}"
if [[ -n "${RUN_LIMIT}" ]]; then
  printf 'Episode limit per family: %s\n' "${RUN_LIMIT}"
else
  printf 'Episode limit per family: all\n'
fi

run_factual() {
  if [[ ! -s "${FACTUAL_MANIFEST}" ]]; then
    printf '\n=== Building factual manifest (limit=%s) ===\n' "${FACTUAL_LIMIT}"
    uv run python -m pressuretrace.generation.factual.make_factual_tasks_v1
  else
    printf '\nFactual manifest exists: %s\n' "${FACTUAL_MANIFEST}"
  fi

  printf '\n=== Running factual benchmark ===\n'
  local limit_args=()
  if [[ -n "${RUN_LIMIT}" ]]; then
    limit_args+=(--limit "${RUN_LIMIT}")
  fi
  uv run python -m pressuretrace.behavior.run_factual_benchmark_v1 \
    --manifest-path "${FACTUAL_MANIFEST}" \
    --output-path "${FACTUAL_RESULTS}" \
    --model-name "${MODEL_NAME}" \
    --thinking-mode "${THINKING_MODE}" \
    "${limit_args[@]}"

  printf '\n=== Factual summary ===\n'
  uv run python -m pressuretrace.behavior.summarize_factual_behavior_v1 \
    --results-path "${FACTUAL_RESULTS}" | tee "${FACTUAL_RESULTS%.jsonl}.summary.txt"
}

run_logical() {
  if [[ ! -s "${LOGICAL_MANIFEST}" ]]; then
    printf '\n=== Building logical manifest (limit=%s) ===\n' "${LOGICAL_LIMIT}"
    uv run python -m pressuretrace.generation.logical.make_logical_tasks_v1
  else
    printf '\nLogical manifest exists: %s\n' "${LOGICAL_MANIFEST}"
  fi

  printf '\n=== Running logical benchmark ===\n'
  local limit_args=()
  if [[ -n "${RUN_LIMIT}" ]]; then
    limit_args+=(--limit "${RUN_LIMIT}")
  fi
  uv run python -m pressuretrace.behavior.run_logical_benchmark_v1 \
    --manifest-path "${LOGICAL_MANIFEST}" \
    --output-path "${LOGICAL_RESULTS}" \
    --model-name "${MODEL_NAME}" \
    --thinking-mode "${THINKING_MODE}" \
    "${limit_args[@]}"

  printf '\n=== Logical summary ===\n'
  uv run python -m pressuretrace.behavior.summarize_logical_behavior_v1 \
    --results-path "${LOGICAL_RESULTS}" | tee "${LOGICAL_RESULTS%.jsonl}.summary.txt"
}

if [[ "${SKIP_FACTUAL}" != "1" ]]; then
  run_factual
else
  printf '\nSkipping factual family (SKIP_FACTUAL=1)\n'
fi

if [[ "${SKIP_LOGICAL}" != "1" ]]; then
  run_logical
else
  printf '\nSkipping logical family (SKIP_LOGICAL=1)\n'
fi

printf '\nAll outputs in: %s\n' "${RUN_DIR}"
