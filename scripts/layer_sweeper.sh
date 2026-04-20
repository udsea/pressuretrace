#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Adversarial + generation-patching layer sweep.
#
# Attack search and detector robustness run once per model at REF_LAYER
# (adversarial prompts are layer-agnostic; only the detection/intervention
# stage is layer-specific). The control pipeline and reasoning generation
# patching then sweep the LAYERS array.
#
# Environment overrides:
#   MODEL_NAME      HF model id         (default Qwen/Qwen3-14B)
#   THINKING_MODE   off|on|default      (default off)
#   LAYERS          space-separated list of layer indices
#                   (default "-8 -6 -4 -2")
#   REF_LAYER       layer used for one-shot upstream stages (default -4)
#   THRESHOLD       risk-score threshold for intervention (default 50.0)
#   SKIP_ATTACK     set to 1 to skip stage 1 (use existing attack results)
#   SKIP_DETECTOR   set to 1 to skip stage 2 (use existing detector results)
#   SKIP_GEN_PATCH  set to 1 to skip reasoning generation patching
#
# Examples:
#   MODEL_NAME=Qwen/Qwen3-14B bash scripts/layer_sweeper.sh
#   MODEL_NAME=google/gemma-3-27b-it bash scripts/layer_sweeper.sh

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
REF_LAYER="${REF_LAYER:--4}"
THRESHOLD="${THRESHOLD:-50.0}"
SKIP_ATTACK="${SKIP_ATTACK:-0}"
SKIP_DETECTOR="${SKIP_DETECTOR:-0}"
SKIP_GEN_PATCH="${SKIP_GEN_PATCH:-0}"

SLUG="$(slugify "${MODEL_NAME}")"
ATTACK_PATH="results/adversarial_attack_search_layer${REF_LAYER}_${SLUG}_${THINKING_MODE}.jsonl"
DETECTOR_PATH="results/adversarial_detector_robustness_layer${REF_LAYER}_${SLUG}_${THINKING_MODE}.jsonl"

mkdir -p results

printf 'Model:        %s\n' "${MODEL_NAME}"
printf 'Thinking:     %s\n' "${THINKING_MODE}"
printf 'Layers:       %s\n' "${LAYERS[*]}"
printf 'Ref layer:    %s\n' "${REF_LAYER}"
printf 'Threshold:    %s\n' "${THRESHOLD}"
printf 'Attack out:   %s\n' "${ATTACK_PATH}"
printf 'Detector out: %s\n' "${DETECTOR_PATH}"

if [[ "${SKIP_ATTACK}" != "1" ]]; then
  printf '\n=== Stage 1: attack search (layer=%s) ===\n' "${REF_LAYER}"
  uv run python -m pressuretrace.adversarial.run_attack_search \
    --layer "${REF_LAYER}" \
    --model-name "${MODEL_NAME}" --thinking-mode "${THINKING_MODE}" \
    --output-path "${ATTACK_PATH}"
else
  printf '\n=== Stage 1: skipped (using %s) ===\n' "${ATTACK_PATH}"
fi

if [[ "${SKIP_DETECTOR}" != "1" ]]; then
  printf '\n=== Stage 2: detector robustness (layer=%s) ===\n' "${REF_LAYER}"
  uv run python -m pressuretrace.adversarial.run_detector_robustness \
    --layer "${REF_LAYER}" \
    --attack-results-path "${ATTACK_PATH}" \
    --model-name "${MODEL_NAME}" --thinking-mode "${THINKING_MODE}" \
    --output-path "${DETECTOR_PATH}"
else
  printf '\n=== Stage 2: skipped (using %s) ===\n' "${DETECTOR_PATH}"
fi

for L in "${LAYERS[@]}"; do
  printf '\n=== Stage 3: control pipeline layer=%s ===\n' "${L}"
  uv run python -m pressuretrace.adversarial.run_control_pipeline \
    --layer "${L}" --threshold "${THRESHOLD}" \
    --adversarial-results-path "${DETECTOR_PATH}" \
    --model-name "${MODEL_NAME}" --thinking-mode "${THINKING_MODE}" \
    --output-path "results/adversarial_control_pipeline_layer${L}_${SLUG}_${THINKING_MODE}.jsonl"
done

if [[ "${SKIP_GEN_PATCH}" != "1" ]]; then
  for L in "${LAYERS[@]}"; do
    printf '\n=== Reasoning generation patching layer=%s ===\n' "${L}"
    uv run python -m pressuretrace.patching.run_reasoning_generation_patching \
      --layer "${L}" \
      --model-name "${MODEL_NAME}" --thinking-mode "${THINKING_MODE}" \
      --output-path "results/reasoning_generation_patching_layer${L}_${SLUG}_${THINKING_MODE}.jsonl" \
      --summary-path "results/reasoning_generation_patching_layer${L}_summary_${SLUG}_${THINKING_MODE}.json"
  done
fi

printf '\nSummaries:\n'
ls -1 results/adversarial_control_pipeline_layer*_"${SLUG}"_"${THINKING_MODE}"_summary.json 2>/dev/null || true
ls -1 results/reasoning_generation_patching_layer*_summary_"${SLUG}"_"${THINKING_MODE}".json 2>/dev/null || true
