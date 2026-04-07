#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  echo "Install it with: bash scripts/install_uv.sh" >&2
  exit 1
fi

uv sync --dev

echo
echo "PressureTrace bootstrap complete."
echo "Next steps:"
echo "  1. Inspect repo paths: uv run pressuretrace print-paths"
echo "  2. Run a dry reasoning pilot: bash scripts/run_reasoning_pilot.sh"
echo "  3. Review the working spec: paper/pressuretrace_v1_spec.md"
