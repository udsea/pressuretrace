#!/usr/bin/env bash
set -euo pipefail

# Sequential reasoning sweep for 5 strong open-weight models.
# Each run uses the existing PressureTrace CLI and can shard a model across
# all visible CUDA devices via transformers + accelerate.

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
MODEL_GROUP="${MODEL_GROUP:-all}" # one of: all, high, mid
INCLUDE_CONTROL="${INCLUDE_CONTROL:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-results/reasoning_frontier_sweep_${RUN_TAG}}"
MANIFEST_PATH="${MANIFEST_PATH:-data/manifests/reasoning_frontier_sweep_${SPLIT}_${LIMIT}.jsonl}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
CACHE_ROOT="${CACHE_ROOT:-$(resolve_cache_root)}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"

mkdir -p "${RUN_DIR}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TMPDIR}"

HIGH_END_MODELS=(
  "Qwen/Qwen3-32B"
  "Qwen/QwQ-32B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

MID_END_MODELS=(
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-8B"
)

slugify() {
  local value="$1"
  echo "${value}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

build_model_list() {
  case "${MODEL_GROUP}" in
    all)
      printf '%s\n' "${HIGH_END_MODELS[@]}" "${MID_END_MODELS[@]}"
      ;;
    high)
      printf '%s\n' "${HIGH_END_MODELS[@]}"
      ;;
    mid)
      printf '%s\n' "${MID_END_MODELS[@]}"
      ;;
    *)
      echo "MODEL_GROUP must be one of: all, high, mid" >&2
      exit 1
      ;;
  esac
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
  echo "PressureTrace reasoning frontier sweep"
  echo "run_tag=${RUN_TAG}"
  echo "split=${SPLIT}"
  echo "limit=${LIMIT}"
  echo "pressure_type=${PRESSURE_TYPE}"
  echo "model_group=${MODEL_GROUP}"
  echo "manifest_path=${MANIFEST_PATH}"
  echo "cache_root=${CACHE_ROOT}"
  echo "hf_home=${HF_HOME}"
  echo
} > "${SUMMARY_FILE}"

printf 'category\tmodel\tresults_path\tsummary_path\n' > "${MODEL_INDEX_FILE}"

mapfile -t MODELS < <(build_model_list)

printf 'HF cache directory: %s\n' "${HF_HOME}"

for MODEL_NAME in "${MODELS[@]}"; do
  MODEL_CATEGORY="mid"
  for HIGH_END_MODEL in "${HIGH_END_MODELS[@]}"; do
    if [[ "${MODEL_NAME}" == "${HIGH_END_MODEL}" ]]; then
      MODEL_CATEGORY="high"
      break
    fi
  done

  MODEL_SLUG="$(slugify "${MODEL_NAME}")"
  RESULTS_PATH="${RUN_DIR}/${MODEL_SLUG}.jsonl"
  MODEL_SUMMARY_PATH="${RUN_DIR}/${MODEL_SLUG}.summary.txt"

  printf '\n[%s] Running %s model: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${MODEL_CATEGORY}" "${MODEL_NAME}"
  printf 'Results: %s\n' "${RESULTS_PATH}"

  TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY}" \
  TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
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
    printf '\n=== %s (%s) ===\n' "${MODEL_NAME}" "${MODEL_CATEGORY}"
    cat "${MODEL_SUMMARY_PATH}"
    printf '\n'
  } >> "${SUMMARY_FILE}"

  printf '%s\t%s\t%s\t%s\n' \
    "${MODEL_CATEGORY}" \
    "${MODEL_NAME}" \
    "${RESULTS_PATH}" \
    "${MODEL_SUMMARY_PATH}" >> "${MODEL_INDEX_FILE}"
done

printf '\nSweep complete.\n'
printf 'Model index: %s\n' "${MODEL_INDEX_FILE}"
printf 'Combined summary: %s\n' "${SUMMARY_FILE}"
