"""Aggregate row types for reasoning v2 summaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorAggregateV2:
    """Aggregate row for the primary reasoning v2 summary table."""

    thinking_mode: str
    pressure_type: str
    total: int
    robust_rate: float
    shortcut_followed_rate: float
    wrong_nonshortcut_rate: float
    parse_failed_rate: float
    parse_ambiguous_rate: float
    average_response_length_chars: float
    average_candidate_count: float


@dataclass(frozen=True)
class CountAggregateV2:
    """Generic count aggregate for failure subtypes and parse statuses."""

    thinking_mode: str
    pressure_type: str
    label: str
    count: int


@dataclass(frozen=True)
class PairedRouteShiftAggregateV2:
    """Paired control-versus-pressure route-shift counts."""

    thinking_mode: str
    pressure_type: str
    control_correct_to_pressure_shortcut: int
    control_wrong_nonshortcut_to_pressure_shortcut: int
    control_correct_to_pressure_wrong_nonshortcut: int
    control_shortcut_to_pressure_shortcut: int
