#!/usr/bin/env bash
set -euo pipefail

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
PRESSURE_TYPE="${PRESSURE_TYPE:-all}"
LIMIT="${LIMIT:-24}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

MODEL_SLUG="$(slugify "${MODEL_NAME}")"
FROZEN_ROOT="${FROZEN_ROOT:-pressuretrace-frozen/coding_v1_${MODEL_SLUG}_${THINKING_MODE}}"

POOL_MANIFEST="${FROZEN_ROOT}/data/manifests/coding_all_valid_transforms.jsonl"
CONTROL_RESULTS="${FROZEN_ROOT}/results/coding_control_only_${MODEL_SLUG}_${THINKING_MODE}.jsonl"
ROBUST_SLICE="${FROZEN_ROOT}/data/splits/coding_control_robust_slice_${MODEL_SLUG}_${THINKING_MODE}.jsonl"
PAPER_MANIFEST="${FROZEN_ROOT}/data/manifests/coding_paper_slice_${MODEL_SLUG}_${THINKING_MODE}.jsonl"
PAPER_RESULTS="${FROZEN_ROOT}/results/coding_paper_slice_${MODEL_SLUG}_${THINKING_MODE}.jsonl"
SUMMARY_TXT="${FROZEN_ROOT}/results/coding_paper_slice_summary_${MODEL_SLUG}_${THINKING_MODE}.txt"
SUMMARY_CSV="${FROZEN_ROOT}/results/coding_paper_slice_summary_${MODEL_SLUG}_${THINKING_MODE}.csv"

mkdir -p "${FROZEN_ROOT}/data/manifests" "${FROZEN_ROOT}/data/splits" "${FROZEN_ROOT}/results"

if [[ "${SHOW_PROGRESS}" == "1" && -t 1 ]]; then
  PROGRESS_FLAG="--progress"
else
  PROGRESS_FLAG="--no-progress"
fi

printf 'Coding-family frozen root: %s\n' "${FROZEN_ROOT}"
printf 'Model: %s\n' "${MODEL_NAME}"
printf 'Thinking mode: %s\n' "${THINKING_MODE}"
printf 'Pressure type: %s\n' "${PRESSURE_TYPE}"
printf 'Base-task limit: %s\n' "${LIMIT}"
printf 'Batch size: %s\n' "${BATCH_SIZE}"

uv run pressuretrace coding-eval-debug-v1

uv run pressuretrace coding-debug-run-v1 \
  --model-name "${MODEL_NAME}" \
  --thinking-mode "${THINKING_MODE}" \
  --source fixtures \
  --require-shortcut-signal

uv run pressuretrace coding-build-pool-v1 \
  --limit "${LIMIT}" \
  --output-path "${POOL_MANIFEST}"

uv run pressuretrace coding-control-only-v1 \
  --manifest-path "${POOL_MANIFEST}" \
  --model-name "${MODEL_NAME}" \
  --thinking-mode "${THINKING_MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --output-path "${CONTROL_RESULTS}" \
  "${PROGRESS_FLAG}"

uv run pressuretrace coding-freeze-slice-v1 \
  --input-path "${CONTROL_RESULTS}" \
  --output-path "${ROBUST_SLICE}"

uv run pressuretrace coding-materialize-slice-v1 \
  --manifest-path "${POOL_MANIFEST}" \
  --slice-path "${ROBUST_SLICE}" \
  --output-path "${PAPER_MANIFEST}"

uv run pressuretrace coding-paper-slice-v1 \
  --manifest-path "${PAPER_MANIFEST}" \
  --model-name "${MODEL_NAME}" \
  --thinking-mode "${THINKING_MODE}" \
  --pressure-type "${PRESSURE_TYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --output-path "${PAPER_RESULTS}" \
  "${PROGRESS_FLAG}"

uv run pressuretrace coding-summarize-v1 \
  --input-path "${PAPER_RESULTS}" \
  --text-output-path "${SUMMARY_TXT}" \
  --csv-output-path "${SUMMARY_CSV}"

printf '\nCoding-family pipeline complete.\n'
printf 'Transform pool: %s\n' "${POOL_MANIFEST}"
printf 'Control-only results: %s\n' "${CONTROL_RESULTS}"
printf 'Control-robust slice: %s\n' "${ROBUST_SLICE}"
printf 'Paper-slice manifest: %s\n' "${PAPER_MANIFEST}"
printf 'Paper-slice results: %s\n' "${PAPER_RESULTS}"
printf 'Summary text: %s\n' "${SUMMARY_TXT}"
printf 'Summary CSV: %s\n' "${SUMMARY_CSV}"
