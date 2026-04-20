#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Adversarial + generation-patching layer sweep.
#
# Runs the full adversarial chain at each layer in LAYERS:
#   attack search -> detector robustness -> control pipeline
# plus the reasoning generation patching at the same layers.
# Every stage writes layer-tagged outputs so nothing is overwritten.
#
# Environment overrides:
#   MODEL_NAME      HF model id         (default Qwen/Qwen3-14B)
#   THINKING_MODE   off|on|default      (default off)
#   LAYERS          space-separated list of layer indices
#                   (default "-8 -6 -4 -2")
#   THRESHOLD       risk-score threshold for intervention (default 50.0)
#   SKIP_ATTACK     set to 1 to skip stage 1 at every layer
#   SKIP_DETECTOR   set to 1 to skip stage 2 at every layer
#   SKIP_CONTROL    set to 1 to skip stage 3 at every layer
#   SKIP_GEN_PATCH  set to 1 to skip reasoning generation patching
#   HIDDEN_STATES_PATH   override probe hidden states file (model-specific)
#   MANIFEST_PATH        override reasoning manifest (model-specific)
#
# Paths default to the per-model frozen artefacts under pressuretrace-frozen/.
#
# Examples:
#   MODEL_NAME=Qwen/Qwen3-14B bash scripts/layer_sweeper.sh
#   MODEL_NAME=google/gemma-3-27b-it bash scripts/layer_sweeper.sh
#   LAYERS="-4" bash scripts/layer_sweeper.sh       # single layer
#   SKIP_GEN_PATCH=1 bash scripts/layer_sweeper.sh  # adversarial chain only

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
THINKING_MODE="${THINKING_MODE:-off}"
LAYERS_STR="${LAYERS:--8 -6 -4 -2}"
read -r -a LAYERS <<< "${LAYERS_STR}"
THRESHOLD="${THRESHOLD:-50.0}"
SKIP_ATTACK="${SKIP_ATTACK:-0}"
SKIP_DETECTOR="${SKIP_DETECTOR:-0}"
SKIP_CONTROL="${SKIP_CONTROL:-0}"
SKIP_GEN_PATCH="${SKIP_GEN_PATCH:-0}"

SLUG="$(slugify "${MODEL_NAME}")"
TAG="${SLUG}_${THINKING_MODE}"

case "${SLUG}" in
  qwen-qwen3-14b)
    FROZEN_DIR="pressuretrace-frozen/reasoning_v2_qwen3_14b_off"
    ;;
  *)
    FROZEN_DIR="pressuretrace-frozen/reasoning_v2_${SLUG}_${THINKING_MODE}"
    ;;
esac
DEFAULT_HIDDEN_STATES_PATH="${FROZEN_DIR}/results/reasoning_probe_hidden_states_${SLUG}_${THINKING_MODE}.jsonl"
DEFAULT_MANIFEST_PATH="${FROZEN_DIR}/data/manifests/reasoning_paper_slice_${SLUG}_${THINKING_MODE}.jsonl"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-${DEFAULT_HIDDEN_STATES_PATH}}"
MANIFEST_PATH="${MANIFEST_PATH:-${DEFAULT_MANIFEST_PATH}}"

for required_file in "${HIDDEN_STATES_PATH}" "${MANIFEST_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    echo "Override with HIDDEN_STATES_PATH / MANIFEST_PATH env vars if the path differs." >&2
    exit 1
  fi
done

mkdir -p results

printf 'Model:         %s\n' "${MODEL_NAME}"
printf 'Thinking:      %s\n' "${THINKING_MODE}"
printf 'Layers:        %s\n' "${LAYERS[*]}"
printf 'Threshold:     %s\n' "${THRESHOLD}"
printf 'Hidden states: %s\n' "${HIDDEN_STATES_PATH}"
printf 'Manifest:      %s\n' "${MANIFEST_PATH}"

attack_path_for()    { echo "results/adversarial_attack_search_layer${1}_${TAG}.jsonl"; }
detector_path_for()  { echo "results/adversarial_detector_robustness_layer${1}_${TAG}.jsonl"; }
control_path_for()   { echo "results/adversarial_control_pipeline_layer${1}_${TAG}.jsonl"; }
gen_patch_path_for() { echo "results/reasoning_generation_patching_layer${1}_${TAG}.jsonl"; }
gen_patch_sum_for()  { echo "results/reasoning_generation_patching_layer${1}_summary_${TAG}.json"; }

for L in "${LAYERS[@]}"; do
  ATTACK_PATH="$(attack_path_for "${L}")"
  DETECTOR_PATH="$(detector_path_for "${L}")"
  CONTROL_PATH="$(control_path_for "${L}")"

  printf '\n########## LAYER %s ##########\n' "${L}"

  if [[ "${SKIP_ATTACK}" != "1" ]]; then
    printf '\n=== Stage 1: attack search (layer=%s) ===\n' "${L}"
    uv run python -m pressuretrace.adversarial.run_attack_search \
      --layer "${L}" \
      --model-name "${MODEL_NAME}" --thinking-mode "${THINKING_MODE}" \
      --manifest-path "${MANIFEST_PATH}" \
      --hidden-states-path "${HIDDEN_STATES_PATH}" \
      --output-path "${ATTACK_PATH}"
  else
    printf '\n=== Stage 1: skipped (expecting %s) ===\n' "${ATTACK_PATH}"
  fi

  if [[ "${SKIP_DETECTOR}" != "1" ]]; then
    printf '\n=== Stage 2: detector robustness (layer=%s) ===\n' "${L}"
    uv run python -m pressuretrace.adversarial.run_detector_robustness \
      --layer "${L}" \
      --attack-results-path "${ATTACK_PATH}" \
      --hidden-states-path "${HIDDEN_STATES_PATH}" \
      --model-name "${MODEL_NAME}" --thinking-mode "${THINKING_MODE}" \
      --output-path "${DETECTOR_PATH}"
  else
    printf '\n=== Stage 2: skipped (expecting %s) ===\n' "${DETECTOR_PATH}"
  fi

  if [[ "${SKIP_CONTROL}" != "1" ]]; then
    printf '\n=== Stage 3: control pipeline (layer=%s) ===\n' "${L}"
    uv run python -m pressuretrace.adversarial.run_control_pipeline \
      --layer "${L}" --threshold "${THRESHOLD}" \
      --adversarial-results-path "${DETECTOR_PATH}" \
      --static-hidden-states-path "${HIDDEN_STATES_PATH}" \
      --control-prompts-path "${MANIFEST_PATH}" \
      --model-name "${MODEL_NAME}" --thinking-mode "${THINKING_MODE}" \
      --output-path "${CONTROL_PATH}"
  else
    printf '\n=== Stage 3: skipped (would write %s) ===\n' "${CONTROL_PATH}"
  fi
done

if [[ "${SKIP_GEN_PATCH}" != "1" ]]; then
  for L in "${LAYERS[@]}"; do
    printf '\n=== Reasoning generation patching layer=%s ===\n' "${L}"
    uv run python -m pressuretrace.patching.run_reasoning_generation_patching \
      --layer "${L}" \
      --model-name "${MODEL_NAME}" --thinking-mode "${THINKING_MODE}" \
      --output-path "$(gen_patch_path_for "${L}")" \
      --summary-path "$(gen_patch_sum_for "${L}")"
  done
fi

printf '\nSummaries:\n'
ls -1 results/adversarial_control_pipeline_layer*_"${TAG}"_summary.json 2>/dev/null || true
ls -1 results/reasoning_generation_patching_layer*_summary_"${TAG}".json 2>/dev/null || true
