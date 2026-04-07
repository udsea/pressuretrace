#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  echo "uv is already installed: $(command -v uv)"
  uv --version
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to install uv." >&2
  exit 1
fi

INSTALLER_URL="${INSTALLER_URL:-https://astral.sh/uv/install.sh}"
UV_BIN_DIR="${UV_BIN_DIR:-$HOME/.local/bin}"
TMP_INSTALLER="$(mktemp)"
trap 'rm -f "$TMP_INSTALLER"' EXIT

echo "Installing uv from $INSTALLER_URL"
curl -LsSf "$INSTALLER_URL" -o "$TMP_INSTALLER"
sh "$TMP_INSTALLER"

if ! command -v uv >/dev/null 2>&1 && [[ -x "$UV_BIN_DIR/uv" ]]; then
  export PATH="$UV_BIN_DIR:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was installed but is not on PATH for this shell." >&2
  echo "Try: export PATH=\"$UV_BIN_DIR:\$PATH\"" >&2
  exit 1
fi

echo "uv installed successfully: $(command -v uv)"
uv --version
