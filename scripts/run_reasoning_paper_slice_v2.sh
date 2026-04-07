#!/usr/bin/env bash
set -euo pipefail

# Build the full reasoning v2 transform pool, run control-only evaluation,
# freeze the control-robust slice, materialize the paper-slice manifest,
# and run the full pressure benchmark over that frozen slice.

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

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

slugify() {
  local value="$1"
  echo "${value}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
THINKING_MODE="${THINKING_MODE:-off}"
PRESSURE_TYPE="${PRESSURE_TYPE:-all}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-results/reasoning_paper_slice_v2_${RUN_TAG}}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
CACHE_ROOT="${CACHE_ROOT:-$(resolve_cache_root)}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"

MODEL_SLUG="$(slugify "${MODEL_NAME}")"

POOL_MANIFEST="${POOL_MANIFEST:-data/manifests/reasoning_all_valid_transforms.jsonl}"
CONTROL_RESULTS="${CONTROL_RESULTS:-results/reasoning_control_only_${MODEL_SLUG}_${THINKING_MODE}.jsonl}"
ROBUST_SLICE="${ROBUST_SLICE:-data/splits/reasoning_control_robust_slice_${MODEL_SLUG}_${THINKING_MODE}.jsonl}"
PAPER_MANIFEST="${PAPER_MANIFEST:-data/manifests/reasoning_paper_slice_${MODEL_SLUG}_${THINKING_MODE}.jsonl}"
PAPER_RESULTS="${PAPER_RESULTS:-results/reasoning_paper_slice_${MODEL_SLUG}_${THINKING_MODE}.jsonl}"
SUMMARY_PATH="${SUMMARY_PATH:-${RUN_DIR}/reasoning_paper_slice.summary.txt}"

mkdir -p "${RUN_DIR}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TMPDIR}"

if [[ "${SHOW_PROGRESS}" == "1" && -t 1 ]]; then
  PROGRESS_FLAG="--progress"
else
  PROGRESS_FLAG="--no-progress"
fi

printf 'HF cache directory: %s\n' "${HF_HOME}"
printf 'Model: %s\n' "${MODEL_NAME}"
printf 'Thinking mode: %s\n' "${THINKING_MODE}"
printf 'Split: %s\n' "${SPLIT}"
if [[ -n "${LIMIT}" ]]; then
  printf 'Limit: %s\n' "${LIMIT}"
else
  printf 'Limit: all retained tasks\n'
fi
printf 'Pressure type: %s\n' "${PRESSURE_TYPE}"

BUILD_POOL_ARGS=(
  --split "${SPLIT}"
  --output-path "${POOL_MANIFEST}"
)
if [[ -n "${LIMIT}" ]]; then
  BUILD_POOL_ARGS+=(--limit "${LIMIT}")
fi

TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY}" \
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
HF_HOME="${HF_HOME}" \
HF_HUB_CACHE="${HF_HUB_CACHE}" \
HF_XET_CACHE="${HF_XET_CACHE}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" \
TMPDIR="${TMPDIR}" \
uv run pressuretrace reasoning-build-pool-v2 \
  "${BUILD_POOL_ARGS[@]}"

TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY}" \
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
HF_HOME="${HF_HOME}" \
HF_HUB_CACHE="${HF_HUB_CACHE}" \
HF_XET_CACHE="${HF_XET_CACHE}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" \
TMPDIR="${TMPDIR}" \
uv run pressuretrace reasoning-control-only-v2 \
  --manifest-path "${POOL_MANIFEST}" \
  --model-name "${MODEL_NAME}" \
  --thinking-mode "${THINKING_MODE}" \
  --output-path "${CONTROL_RESULTS}" \
  "${PROGRESS_FLAG}"

uv run pressuretrace reasoning-freeze-slice-v2 \
  --input-path "${CONTROL_RESULTS}" \
  --output-path "${ROBUST_SLICE}"

uv run pressuretrace reasoning-materialize-slice-v2 \
  --manifest-path "${POOL_MANIFEST}" \
  --slice-path "${ROBUST_SLICE}" \
  --output-path "${PAPER_MANIFEST}"

TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY}" \
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
HF_HOME="${HF_HOME}" \
HF_HUB_CACHE="${HF_HUB_CACHE}" \
HF_XET_CACHE="${HF_XET_CACHE}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" \
TMPDIR="${TMPDIR}" \
uv run pressuretrace reasoning-run-manifest-v2 \
  --manifest-path "${PAPER_MANIFEST}" \
  --model-name "${MODEL_NAME}" \
  --thinking-mode "${THINKING_MODE}" \
  --pressure-type "${PRESSURE_TYPE}" \
  --output-path "${PAPER_RESULTS}" \
  --include-control \
  "${PROGRESS_FLAG}"

uv run pressuretrace summarize-v2 --input-path "${PAPER_RESULTS}" | tee "${SUMMARY_PATH}"

printf '\nPipeline complete.\n'
printf 'Pool manifest: %s\n' "${POOL_MANIFEST}"
printf 'Control-only results: %s\n' "${CONTROL_RESULTS}"
printf 'Control-robust slice: %s\n' "${ROBUST_SLICE}"
printf 'Paper-slice manifest: %s\n' "${PAPER_MANIFEST}"
printf 'Paper-slice results: %s\n' "${PAPER_RESULTS}"
printf 'Summary: %s\n' "${SUMMARY_PATH}"
