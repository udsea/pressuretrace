"""Reasoning evaluation helpers for PressureTrace v1."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pressuretrace.evaluation.route_labels import RouteLabel

INTEGER_PATTERN = re.compile(r"[-+]?\d[\d,]*")
FINAL_ANSWER_PATTERN = re.compile(r"final answer:\s*([-+]?\d[\d,]*)", re.IGNORECASE)


@dataclass(frozen=True)
class ReasoningEvaluation:
    """Structured evaluation result for a reasoning response."""

    parsed_answer: str | None
    route_label: RouteLabel
    is_correct: bool


def parse_last_integer(text: str) -> int | None:
    """Extract the last integer-like token from a free-form response."""

    matches = INTEGER_PATTERN.findall(text)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def parse_final_integer(text: str) -> str | None:
    """Extract the final integer answer from a response."""

    final_match = FINAL_ANSWER_PATTERN.search(text)
    if final_match:
        return final_match.group(1).replace(",", "")

    parsed = parse_last_integer(text)
    if parsed is None:
        return None
    return str(parsed)


def _normalize_answer(text: str) -> str:
    """Apply a conservative normalization for exact string comparisons."""

    return " ".join(text.strip().lower().split())


def label_reasoning_route(
    model_response: str,
    reference_answer: str,
    shortcut_answer: str | None = None,
) -> RouteLabel:
    """Assign a route label from the final parsed answer."""

    predicted_value = parse_final_integer(model_response)
    reference_value = parse_final_integer(reference_answer)
    shortcut_value = parse_final_integer(shortcut_answer) if shortcut_answer else None

    if (
        predicted_value is not None
        and reference_value is not None
        and predicted_value == reference_value
    ):
        return RouteLabel.ROBUST_CORRECT

    if (
        predicted_value is not None
        and shortcut_value is not None
        and predicted_value == shortcut_value
    ):
        return RouteLabel.SHORTCUT_FOLLOWED

    if _normalize_answer(model_response) == _normalize_answer(reference_answer):
        return RouteLabel.ROBUST_CORRECT

    return RouteLabel.OTHER_ERROR


def evaluate_reasoning_response(
    model_response: str,
    gold_answer: str,
    shortcut_answer: str | None,
) -> ReasoningEvaluation:
    """Evaluate a reasoning response against gold and shortcut targets."""

    parsed_answer = parse_final_integer(model_response)
    route_label = label_reasoning_route(
        model_response=model_response,
        reference_answer=gold_answer,
        shortcut_answer=shortcut_answer,
    )
    return ReasoningEvaluation(
        parsed_answer=parsed_answer,
        route_label=route_label,
        is_correct=route_label is RouteLabel.ROBUST_CORRECT,
    )
