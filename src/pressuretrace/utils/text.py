"""Text normalization helpers for prompt and output handling."""

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces."""

    return re.sub(r"\s+", " ", text).strip()


def strip_code_fence(text: str) -> str:
    """Remove a single surrounding Markdown code fence if present."""

    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return stripped
