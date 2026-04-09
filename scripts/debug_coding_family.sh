#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
THINKING_MODE="${THINKING_MODE:-off}"
DEBUG_SOURCE="${DEBUG_SOURCE:-fixtures}"
BATCH_SIZE="${BATCH_SIZE:-1}"

printf 'Coding-family evaluator self-test\n'
uv run pressuretrace coding-eval-debug-v1 --require-pass

printf '\nCoding-family debug run\n'
uv run pressuretrace coding-debug-run-v1 \
  --model-name "${MODEL_NAME}" \
  --thinking-mode "${THINKING_MODE}" \
  --source "${DEBUG_SOURCE}" \
  --batch-size "${BATCH_SIZE}" \
  --require-shortcut-signal
