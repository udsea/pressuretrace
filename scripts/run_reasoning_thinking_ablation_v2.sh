#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-25}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-results/reasoning_v2_thinking_ablation_${RUN_TAG}}"
MANIFEST_PATH="${MANIFEST_PATH:-data/manifests/reasoning_v2_ablation_${SPLIT}_${LIMIT}.jsonl}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODELS=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
)

THINKING_MODES=(
  "on"
  "off"
)

PRESSURE_TYPES=(
  "teacher_anchor"
  "urgency"
)

slugify() {
  local value="$1"
  echo "${value}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

if [[ "${SHOW_PROGRESS}" == "1" && -t 1 ]]; then
  PROGRESS_FLAG="--progress"
else
  PROGRESS_FLAG="--no-progress"
fi

mkdir -p "${RUN_DIR}"

SUMMARY_FILE="${RUN_DIR}/ablation_summary.txt"
INDEX_FILE="${RUN_DIR}/runs.tsv"

{
  echo "PressureTrace reasoning v2 thinking ablation"
  echo "run_tag=${RUN_TAG}"
  echo "split=${SPLIT}"
  echo "limit=${LIMIT}"
  echo "manifest_path=${MANIFEST_PATH}"
  echo
} > "${SUMMARY_FILE}"

printf 'model\tthinking_mode\tpressure_type\tresults_path\tsummary_path\n' > "${INDEX_FILE}"

for MODEL_NAME in "${MODELS[@]}"; do
  MODEL_SLUG="$(slugify "${MODEL_NAME}")"
  for THINKING_MODE in "${THINKING_MODES[@]}"; do
    for PRESSURE_TYPE in "${PRESSURE_TYPES[@]}"; do
      RESULTS_PATH="${RUN_DIR}/${MODEL_SLUG}_${THINKING_MODE}_${PRESSURE_TYPE}.jsonl"
      MODEL_SUMMARY_PATH="${RUN_DIR}/${MODEL_SLUG}_${THINKING_MODE}_${PRESSURE_TYPE}.summary.txt"

      printf '\n[%s] model=%s thinking=%s pressure=%s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" \
        "${MODEL_NAME}" \
        "${THINKING_MODE}" \
        "${PRESSURE_TYPE}"

      TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY}" \
      TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
      uv run pressuretrace reasoning-pilot-v2 \
        --split "${SPLIT}" \
        --limit "${LIMIT}" \
        --model-name "${MODEL_NAME}" \
        --pressure-type "${PRESSURE_TYPE}" \
        --manifest-path "${MANIFEST_PATH}" \
        --output-path "${RESULTS_PATH}" \
        --thinking-mode "${THINKING_MODE}" \
        --include-control \
        "${PROGRESS_FLAG}" \
        "$@"

      uv run pressuretrace summarize-v2 --input-path "${RESULTS_PATH}" | tee "${MODEL_SUMMARY_PATH}"

      {
        printf '\n=== model=%s thinking=%s pressure=%s ===\n' \
          "${MODEL_NAME}" \
          "${THINKING_MODE}" \
          "${PRESSURE_TYPE}"
        cat "${MODEL_SUMMARY_PATH}"
        printf '\n'
      } >> "${SUMMARY_FILE}"

      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${MODEL_NAME}" \
        "${THINKING_MODE}" \
        "${PRESSURE_TYPE}" \
        "${RESULTS_PATH}" \
        "${MODEL_SUMMARY_PATH}" >> "${INDEX_FILE}"
    done
  done
done

printf '\nThinking ablation complete.\n'
printf 'Run index: %s\n' "${INDEX_FILE}"
printf 'Combined summary: %s\n' "${SUMMARY_FILE}"
