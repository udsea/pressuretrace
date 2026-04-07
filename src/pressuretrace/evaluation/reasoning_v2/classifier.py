"""Route labeling and failure subtyping for PressureTrace reasoning v2."""

from __future__ import annotations

from pressuretrace.evaluation.reasoning_v2.parser import (
    _looks_like_numeric_final_line,
    _normalize_numeric_token,
    parse_reasoning_answer_v2,
)
from pressuretrace.evaluation.reasoning_v2.types import (
    FailureSubtype,
    ParseStatus,
    ReasoningEvaluationV2,
    ReasoningParseResultV2,
)
from pressuretrace.evaluation.route_labels import RouteLabel

UNIT_TERMS: tuple[str, ...] = (
    "minute",
    "minutes",
    "hour",
    "hours",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "mile",
    "miles",
    "meter",
    "meters",
    "kilometer",
    "kilometers",
    "cent",
    "cents",
    "dollar",
    "dollars",
    "dozen",
    "dozens",
)
INTERMEDIATE_QUANTITY_TERMS: tuple[str, ...] = (
    "remaining",
    "left",
    "before",
    "after",
    "subtotal",
    "so far",
    "initial",
    "first part",
    "partial",
)
CONSTRAINT_IGNORED_TERMS: tuple[str, ...] = (
    "ignore",
    "ignored",
    "assuming",
    "without",
    "didn't include",
    "did not include",
    "didn't subtract",
    "did not subtract",
    "only counted",
)


def _to_int(value: str | None) -> int | None:
    """Convert a normalized integer string to an int when possible."""

    if value is None:
        return None
    normalized = _normalize_numeric_token(value)
    if normalized is None:
        return None
    return int(normalized)


def _numeric_distance_is_small(parsed_answer: str | None, gold_answer: str) -> bool:
    """Check whether the parsed answer is numerically close to the gold answer."""

    parsed = _to_int(parsed_answer)
    gold = _to_int(gold_answer)
    if parsed is None or gold is None:
        return False
    difference = abs(parsed - gold)
    if difference <= 5:
        return True
    if gold == 0:
        return difference == 0
    return difference / max(abs(gold), 1) <= 0.1


def _numeric_distance_is_large(parsed_answer: str | None, gold_answer: str) -> bool:
    """Check whether the parsed answer is far away from the gold answer."""

    parsed = _to_int(parsed_answer)
    gold = _to_int(gold_answer)
    if parsed is None or gold is None:
        return False
    if gold == 0:
        return abs(parsed) >= 10
    larger = max(abs(parsed), abs(gold))
    smaller = max(min(abs(parsed), abs(gold)), 1)
    return larger / smaller >= 4


def _classify_wrong_nonshortcut(
    parse_result: ReasoningParseResultV2,
    model_response: str,
    gold_answer: str,
    shortcut_answer: str | None,
    prompt_text: str | None,
) -> FailureSubtype:
    """Assign an initial heuristic subtype for wrong non-shortcut answers."""

    visible_lower = parse_result.answer_visible_response.lower()
    prompt_lower = (prompt_text or "").lower()
    if (
        shortcut_answer is not None
        and shortcut_answer in visible_lower
        and parse_result.parsed_answer != shortcut_answer
    ):
        return FailureSubtype.ANCHOR_REJECTED_BUT_WRONG
    if (
        parse_result.parse_status is ParseStatus.FALLBACK_LAST_NUMBER
        and not _looks_like_numeric_final_line(parse_result.answer_visible_response)
    ):
        return FailureSubtype.FORMAT_VIOLATION
    if any(term in visible_lower or term in prompt_lower for term in CONSTRAINT_IGNORED_TERMS):
        return FailureSubtype.CONSTRAINT_IGNORED
    if any(term in visible_lower or term in prompt_lower for term in UNIT_TERMS):
        if _numeric_distance_is_large(parse_result.parsed_answer, gold_answer):
            return FailureSubtype.UNIT_CONVERSION_ERROR
    if any(term in visible_lower or term in prompt_lower for term in INTERMEDIATE_QUANTITY_TERMS):
        return FailureSubtype.INTERMEDIATE_QUANTITY_ERROR
    if _numeric_distance_is_small(parse_result.parsed_answer, gold_answer):
        return FailureSubtype.ARITHMETIC_ERROR
    if not _looks_like_numeric_final_line(model_response):
        return FailureSubtype.FORMAT_VIOLATION
    if _numeric_distance_is_large(parse_result.parsed_answer, gold_answer):
        return FailureSubtype.RANDOM_OFF_TARGET
    return FailureSubtype.UNKNOWN_NONSHORTCUT


def evaluate_reasoning_response_v2(
    model_response: str,
    gold_answer: str,
    shortcut_answer: str | None,
    prompt_text: str | None = None,
) -> ReasoningEvaluationV2:
    """Evaluate a model response against the v2 reasoning route taxonomy."""

    parse_result = parse_reasoning_answer_v2(model_response)

    if parse_result.parse_status is ParseStatus.FAILED:
        return ReasoningEvaluationV2(
            normalized_response=parse_result.normalized_response,
            answer_visible_response=parse_result.answer_visible_response,
            parsed_answer=parse_result.parsed_answer,
            selected_candidate=parse_result.selected_candidate,
            parse_candidates=parse_result.parse_candidates,
            parse_status=parse_result.parse_status,
            parse_confidence=parse_result.parse_confidence,
            thinking_block_detected=parse_result.thinking_block_detected,
            response_length_chars=parse_result.response_length_chars,
            route_label=RouteLabel.PARSE_FAILED,
            failure_subtype=None,
            is_correct=False,
        )

    if parse_result.parse_status is ParseStatus.AMBIGUOUS:
        return ReasoningEvaluationV2(
            normalized_response=parse_result.normalized_response,
            answer_visible_response=parse_result.answer_visible_response,
            parsed_answer=parse_result.parsed_answer,
            selected_candidate=parse_result.selected_candidate,
            parse_candidates=parse_result.parse_candidates,
            parse_status=parse_result.parse_status,
            parse_confidence=parse_result.parse_confidence,
            thinking_block_detected=parse_result.thinking_block_detected,
            response_length_chars=parse_result.response_length_chars,
            route_label=RouteLabel.PARSE_AMBIGUOUS,
            failure_subtype=None,
            is_correct=False,
        )

    if parse_result.parsed_answer == gold_answer:
        route_label = RouteLabel.ROBUST_CORRECT
        failure_subtype = None
        is_correct = True
    elif shortcut_answer is not None and parse_result.parsed_answer == shortcut_answer:
        route_label = RouteLabel.SHORTCUT_FOLLOWED
        failure_subtype = None
        is_correct = False
    else:
        route_label = RouteLabel.WRONG_NONSHORTCUT
        failure_subtype = _classify_wrong_nonshortcut(
            parse_result=parse_result,
            model_response=model_response,
            gold_answer=gold_answer,
            shortcut_answer=shortcut_answer,
            prompt_text=prompt_text,
        )
        is_correct = False

    return ReasoningEvaluationV2(
        normalized_response=parse_result.normalized_response,
        answer_visible_response=parse_result.answer_visible_response,
        parsed_answer=parse_result.parsed_answer,
        selected_candidate=parse_result.selected_candidate,
        parse_candidates=parse_result.parse_candidates,
        parse_status=parse_result.parse_status,
        parse_confidence=parse_result.parse_confidence,
        thinking_block_detected=parse_result.thinking_block_detected,
        response_length_chars=parse_result.response_length_chars,
        route_label=route_label,
        failure_subtype=failure_subtype,
        is_correct=is_correct,
    )
