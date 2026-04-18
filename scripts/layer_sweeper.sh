#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LAYERS=(-8 -6 -4 -2)
THRESHOLD=50.0

# Task 2: generation patching per layer
for L in "${LAYERS[@]}"; do
  echo "=== Task 2 layer=${L} ==="
  uv run python -m pressuretrace.patching.run_reasoning_generation_patching \
    --layer "${L}" \
    --output-path "pressuretrace-frozen/reasoning_v2_qwen3_14b_seq_off/results/reasoning_generation_patching_layer${L}_qwen-qwen3-14b_off.jsonl" \
    --summary-path "pressuretrace-frozen/reasoning_v2_qwen3_14b_seq_off/results/reasoning_generation_patching_layer${L}_summary_qwen-qwen3-14b_off.json"
done

# Task 6: control pipeline per layer (uses attack results from layer -4 attack search)
for L in "${LAYERS[@]}"; do
  echo "=== Task 6 layer=${L} ==="
  uv run python -m pressuretrace.adversarial.run_control_pipeline \
    --layer "${L}" --threshold "${THRESHOLD}" \
    --output-path "results/adversarial_control_pipeline_layer${L}_qwen-qwen3-14b_off.jsonl"
done

echo "Done. Summaries:"
ls -1 pressuretrace-frozen/reasoning_v2_qwen3_14b_seq_off/results/reasoning_generation_patching_layer*_summary_*.json
ls -1 results/adversarial_control_pipeline_layer*_summary.json
