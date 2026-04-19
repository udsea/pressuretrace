#!/usr/bin/env bash
set -euo pipefail

# Cross-model probe transfer pipeline:
#   1. Extract paired final-prompt-token activations from two models on the
#      same control prompts.
#   2. Fit linear alignment between the two models' hidden spaces.
#   3. Transfer the source model's probe direction into target space and
#      evaluate AUC against the target model's native probe.
#
# Environment overrides:
#   MODEL_A                       source model id     (default Qwen/Qwen3-14B)
#   MODEL_B                       target model id     (default google/gemma-3-27b-it)
#   MANIFEST_PATH                 manifest with control rows
#   PRESSURE_TYPE                 which rows to pair  (default control)
#   LIMIT                         number of prompts   (default 229)
#   LAYER                         decoder layer index (default -4)
#   THINKING_MODE                 off|on|default      (default off)
#   SOURCE_HIDDEN_STATES_PATH     source frozen probe hidden states
#   TARGET_HIDDEN_STATES_PATH     target frozen probe hidden states
#   N_PCA_COMPONENTS              PCA dim before align (default 50)
#   RIDGE_ALPHA                   comma-separated sweep (default 100,1000,10000,100000)
#   RUN_TAG                       run tag             (default: date)

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

slugify() {
  local value="$1"
  echo "${value}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

MODEL_A="${MODEL_A:-Qwen/Qwen3-14B}"
MODEL_B="${MODEL_B:-google/gemma-3-27b-it}"
MANIFEST_PATH="${MANIFEST_PATH:-pressuretrace-frozen/reasoning_v2_qwen3_14b_off/data/manifests/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl}"
PRESSURE_TYPE="${PRESSURE_TYPE:-control}"
LIMIT="${LIMIT:-229}"
LAYER="${LAYER:--4}"
THINKING_MODE="${THINKING_MODE:-off}"
N_PCA_COMPONENTS="${N_PCA_COMPONENTS:-50}"
RIDGE_ALPHA="${RIDGE_ALPHA:-100,1000,10000,100000}"
SOURCE_HIDDEN_STATES_PATH="${SOURCE_HIDDEN_STATES_PATH:-pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/reasoning_probe_hidden_states_qwen-qwen3-14b_off.jsonl}"
TARGET_HIDDEN_STATES_PATH="${TARGET_HIDDEN_STATES_PATH:-pressuretrace-frozen/reasoning_v2_google-gemma-3-27b-it_off/results/reasoning_probe_hidden_states_google-gemma-3-27b-it_off.jsonl}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export TRANSFORMERS_VERBOSITY TOKENIZERS_PARALLELISM

SLUG_A="$(slugify "${MODEL_A}")"
SLUG_B="$(slugify "${MODEL_B}")"
RUN_DIR="results/cross_model_${SLUG_A}__${SLUG_B}_${RUN_TAG}"
mkdir -p "${RUN_DIR}"

PAIRED_PATH="${RUN_DIR}/paired_activations_layer${LAYER}.jsonl"
TRANSFER_SUMMARY="${RUN_DIR}/probe_transfer_summary_layer${LAYER}.json"

printf 'Source model: %s\n' "${MODEL_A}"
printf 'Target model: %s\n' "${MODEL_B}"
printf 'Layer: %s\n' "${LAYER}"
printf 'Pressure type: %s\n' "${PRESSURE_TYPE}"
printf 'Limit: %s\n' "${LIMIT}"
printf 'Run dir: %s\n' "${RUN_DIR}"

printf '\n=== Step 1: Extract paired activations ===\n'
uv run python -m pressuretrace.analysis.extract_paired_activations \
  --manifest-path "${MANIFEST_PATH}" \
  --pressure-type "${PRESSURE_TYPE}" \
  --limit "${LIMIT}" \
  --layer "${LAYER}" \
  --model-a "${MODEL_A}" \
  --model-b "${MODEL_B}" \
  --thinking-mode "${THINKING_MODE}" \
  --output-path "${PAIRED_PATH}"

printf '\n=== Step 2: Fit alignment + evaluate probe transfer ===\n'
uv run python -m pressuretrace.analysis.cross_model_probe_transfer \
  --paired-path "${PAIRED_PATH}" \
  --source-hidden-states-path "${SOURCE_HIDDEN_STATES_PATH}" \
  --target-hidden-states-path "${TARGET_HIDDEN_STATES_PATH}" \
  --layer "${LAYER}" \
  --n-pca-components "${N_PCA_COMPONENTS}" \
  --ridge-alpha "${RIDGE_ALPHA}" \
  --output-path "${TRANSFER_SUMMARY}"

printf '\nCross-model probe transfer summary: %s\n' "${TRANSFER_SUMMARY}"
