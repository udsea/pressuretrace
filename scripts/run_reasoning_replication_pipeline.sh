#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_reasoning_replication_pipeline.sh <model-name> [pipeline flags...]" >&2
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

MODEL_NAME="$1"
shift

CACHE_ROOT="${CACHE_ROOT:-$(resolve_cache_root)}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
PRESSURETRACE_TORCH_DTYPE="${PRESSURETRACE_TORCH_DTYPE:-fp16}"
REASONING_BATCH_SIZE="${REASONING_BATCH_SIZE:-4}"
PIPELINE_RESUME="${PIPELINE_RESUME:-1}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" "${TMPDIR}"

echo "Model: ${MODEL_NAME}"
echo "Repo root: ${REPO_ROOT}"
echo "HF cache: ${HF_HOME}"
echo "Torch dtype override: ${PRESSURETRACE_TORCH_DTYPE}"
echo "Reasoning batch size: ${REASONING_BATCH_SIZE}"
echo "Resume existing stages: ${PIPELINE_RESUME}"

export HF_HOME HF_HUB_CACHE HF_XET_CACHE TMPDIR HF_HUB_DISABLE_XET
export TOKENIZERS_PARALLELISM TRANSFORMERS_VERBOSITY PRESSURETRACE_TORCH_DTYPE

ARGS=(
  "${MODEL_NAME}"
  --batch-size "${REASONING_BATCH_SIZE}"
)
if [[ "${PIPELINE_RESUME}" == "1" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi
ARGS+=("$@")

uv run python -m pressuretrace.behavior.run_reasoning_model_pipeline "${ARGS[@]}"
