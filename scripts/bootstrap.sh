#!/usr/bin/env bash
set -euo pipefail

uv sync --dev

echo
echo "PressureTrace bootstrap complete."
echo "Next steps:"
echo "  1. Inspect repo paths: uv run pressuretrace print-paths"
echo "  2. Run a dry reasoning pilot: bash scripts/run_reasoning_pilot.sh"
echo "  3. Review the working spec: paper/pressuretrace_v1_spec.md"
