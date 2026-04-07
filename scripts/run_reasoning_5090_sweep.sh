#!/usr/bin/env bash
set -euo pipefail

# Single-GPU reasoning sweep intended for a desktop-class 32 GB card such as
# an RTX 5090. The default model set stays in the range that is realistic for
# plain transformers inference on one GPU.

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

SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-50}"
PRESSURE_TYPE="${PRESSURE_TYPE:-all}"
INCLUDE_CONTROL="${INCLUDE_CONTROL:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-results/reasoning_5090_sweep_${RUN_TAG}}"
MANIFEST_PATH="${MANIFEST_PATH:-data/manifests/reasoning_5090_sweep_${SPLIT}_${LIMIT}.jsonl}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INCLUDE_EXPERIMENTAL_32B="${INCLUDE_EXPERIMENTAL_32B:-0}"
CACHE_ROOT="${CACHE_ROOT:-$(resolve_cache_root)}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"

mkdir -p "${RUN_DIR}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TMPDIR}"

SAFE_MODELS=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-14B"
)

EXPERIMENTAL_MODELS=(
  "Qwen/QwQ-32B"
)

slugify() {
  local value="$1"
  echo "${value}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

if [[ "${INCLUDE_CONTROL}" == "1" ]]; then
  INCLUDE_CONTROL_FLAG="--include-control"
else
  INCLUDE_CONTROL_FLAG="--pressure-only"
fi

if [[ "${SHOW_PROGRESS}" == "1" && -t 1 ]]; then
  PROGRESS_FLAG="--progress"
else
  PROGRESS_FLAG="--no-progress"
fi

SUMMARY_FILE="${RUN_DIR}/sweep_summary.txt"
MODEL_INDEX_FILE="${RUN_DIR}/models.tsv"

{
  echo "PressureTrace reasoning single-5090 sweep"
  echo "run_tag=${RUN_TAG}"
  echo "split=${SPLIT}"
  echo "limit=${LIMIT}"
  echo "pressure_type=${PRESSURE_TYPE}"
  echo "manifest_path=${MANIFEST_PATH}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "include_experimental_32b=${INCLUDE_EXPERIMENTAL_32B}"
  echo "cache_root=${CACHE_ROOT}"
  echo "hf_home=${HF_HOME}"
  echo
} > "${SUMMARY_FILE}"

printf 'tier\tmodel\tresults_path\tsummary_path\n' > "${MODEL_INDEX_FILE}"

MODELS=("${SAFE_MODELS[@]}")
if [[ "${INCLUDE_EXPERIMENTAL_32B}" == "1" ]]; then
  MODELS+=("${EXPERIMENTAL_MODELS[@]}")
fi

printf 'Running on CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'Results directory: %s\n' "${RUN_DIR}"
printf 'HF cache directory: %s\n' "${HF_HOME}"

for MODEL_NAME in "${MODELS[@]}"; do
  MODEL_TIER="safe"
  if [[ "${MODEL_NAME}" == "${EXPERIMENTAL_MODELS[0]}" ]]; then
    MODEL_TIER="experimental"
  fi

  MODEL_SLUG="$(slugify "${MODEL_NAME}")"
  RESULTS_PATH="${RUN_DIR}/${MODEL_SLUG}.jsonl"
  MODEL_SUMMARY_PATH="${RUN_DIR}/${MODEL_SLUG}.summary.txt"

  printf '\n[%s] Running %s model: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${MODEL_TIER}" "${MODEL_NAME}"
  printf 'Results: %s\n' "${RESULTS_PATH}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY}" \
  TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  HF_HOME="${HF_HOME}" \
  HF_HUB_CACHE="${HF_HUB_CACHE}" \
  HF_XET_CACHE="${HF_XET_CACHE}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" \
  TMPDIR="${TMPDIR}" \
  uv run pressuretrace reasoning-pilot \
    --split "${SPLIT}" \
    --limit "${LIMIT}" \
    --pressure-type "${PRESSURE_TYPE}" \
    --model-name "${MODEL_NAME}" \
    --manifest-path "${MANIFEST_PATH}" \
    --output-path "${RESULTS_PATH}" \
    "${INCLUDE_CONTROL_FLAG}" \
    "${PROGRESS_FLAG}" \
    "$@"

  uv run pressuretrace summarize --input-path "${RESULTS_PATH}" | tee "${MODEL_SUMMARY_PATH}"

  {
    printf '\n=== %s (%s) ===\n' "${MODEL_NAME}" "${MODEL_TIER}"
    cat "${MODEL_SUMMARY_PATH}"
    printf '\n'
  } >> "${SUMMARY_FILE}"

  printf '%s\t%s\t%s\t%s\n' \
    "${MODEL_TIER}" \
    "${MODEL_NAME}" \
    "${RESULTS_PATH}" \
    "${MODEL_SUMMARY_PATH}" >> "${MODEL_INDEX_FILE}"
done

printf '\nSingle-GPU sweep complete.\n'
printf 'Model index: %s\n' "${MODEL_INDEX_FILE}"
printf 'Combined summary: %s\n' "${SUMMARY_FILE}"
