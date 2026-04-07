#!/usr/bin/env bash
set -euo pipefail

uv run pressuretrace reasoning-pilot \
  --split test \
  --limit 20 \
  --pressure-type authority_conflict
