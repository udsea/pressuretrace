"""Shared types for PressureTrace reasoning v2 parsing and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pressuretrace.evaluation.route_labels import RouteLabel


class ParseStatus(StrEnum):
    """V2 parse outcome for a reasoning response."""

    EXPLICIT_CUE = "explicit_cue"
    STRUCTURED_WRAPPER = "structured_wrapper"
    FINAL_LINE = "final_line"
    FALLBACK_LAST_NUMBER = "fallback_last_number"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


class ParseConfidence(StrEnum):
    """Confidence level for a selected parse candidate."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class FailureSubtype(StrEnum):
    """Initial heuristic subtypes for non-shortcut wrong answers."""

    ARITHMETIC_ERROR = "arithmetic_error"
    INTERMEDIATE_QUANTITY_ERROR = "intermediate_quantity_error"
    UNIT_CONVERSION_ERROR = "unit_conversion_error"
    CONSTRAINT_IGNORED = "constraint_ignored"
    FORMAT_VIOLATION = "format_violation"
    ANCHOR_REJECTED_BUT_WRONG = "anchor_rejected_but_wrong"
    RANDOM_OFF_TARGET = "random_off_target"
    UNKNOWN_NONSHORTCUT = "unknown_nonshortcut"


@dataclass(frozen=True)
class ParseCandidate:
    """Single normalized answer candidate extracted from a response."""

    value: str
    start: int
    end: int
    source: ParseStatus
    priority: int


@dataclass(frozen=True)
class ReasoningParseResultV2:
    """Structured parse result used to derive reasoning route labels."""

    normalized_response: str
    answer_visible_response: str
    parsed_answer: str | None
    selected_candidate: str | None
    parse_candidates: list[str]
    parse_status: ParseStatus
    parse_confidence: ParseConfidence
    thinking_block_detected: bool
    response_length_chars: int


@dataclass(frozen=True)
class ReasoningEvaluationV2:
    """Structured v2 evaluation result for a reasoning response."""

    normalized_response: str
    answer_visible_response: str
    parsed_answer: str | None
    selected_candidate: str | None
    parse_candidates: list[str]
    parse_status: ParseStatus
    parse_confidence: ParseConfidence
    thinking_block_detected: bool
    response_length_chars: int
    route_label: RouteLabel
    failure_subtype: FailureSubtype | None
    is_correct: bool
