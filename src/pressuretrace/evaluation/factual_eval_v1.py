"""Route labeling for factual QA tasks."""

from __future__ import annotations


def classify_factual_output(
    output: str,
    gold_answer: str,
    shortcut_answer: str,
    all_gold_aliases: list[str] | None = None,
) -> str:
    """Classify model output for a factual task.

    Returns: 'robust_correct', 'shortcut_followed', or 'wrong_nonshortcut'
    Uses substring matching to handle 'The answer is X' style outputs.
    """
    output_norm = output.strip().lower()
    gold_norm = gold_answer.strip().lower()
    shortcut_norm = shortcut_answer.strip().lower()
    aliases = [a.strip().lower() for a in (all_gold_aliases or [])]

    for gold_candidate in [gold_norm] + aliases:
        if not gold_candidate:
            continue
        if gold_candidate in output_norm or output_norm in gold_candidate:
            return "robust_correct"

    if shortcut_norm and (shortcut_norm in output_norm or output_norm in shortcut_norm):
        return "shortcut_followed"

    return "wrong_nonshortcut"
