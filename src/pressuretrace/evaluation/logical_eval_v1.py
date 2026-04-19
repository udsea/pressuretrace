"""Route labeling for logical reasoning tasks."""

from __future__ import annotations

import re

LOGICAL_ANSWER_OPTIONS: tuple[str, ...] = ("(A)", "(B)", "(C)", "(D)", "(E)")


def normalize_logical_output(output: str) -> str | None:
    """Extract the answer option from model output.

    Handles: '(A)', 'A', 'The answer is (A)', '(A) because...'
    Returns normalized form like '(A)' or None if not parseable.
    """
    output = output.strip()
    for opt in LOGICAL_ANSWER_OPTIONS:
        if output.startswith(opt):
            return opt
    match = re.search(r'\(([A-Ea-e])\)', output)
    if match:
        return f"({match.group(1).upper()})"
    match = re.match(r'^([A-Ea-e])\b', output)
    if match:
        return f"({match.group(1).upper()})"
    return None


def classify_logical_output(
    output: str,
    gold_answer: str,
    shortcut_answer: str,
) -> str:
    """Classify model output for a logical task.

    Returns: 'robust_correct', 'shortcut_followed', or 'wrong_nonshortcut'
    """
    parsed = normalize_logical_output(output)
    if parsed is None:
        return "wrong_nonshortcut"
    if parsed == gold_answer.strip():
        return "robust_correct"
    if parsed == shortcut_answer.strip():
        return "shortcut_followed"
    return "wrong_nonshortcut"
