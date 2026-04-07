#!/usr/bin/env bash
set -euo pipefail

uv run pressuretrace coding-pilot \
  --split test \
  --limit-per-dataset 5 \
  --pressure-profile medium \
  --dry-run
