"""Aggregation helpers for reasoning v2 result JSONL files."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from pressuretrace.behavior.reasoning_summary_v2.types import (
    BehaviorAggregateV2,
    CountAggregateV2,
    PairedRouteShiftAggregateV2,
)
from pressuretrace.utils.io import read_jsonl
from pressuretrace.utils.math_utils import safe_divide


def _ordered_group_keys(keys: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Sort groups with control first and default thinking mode first."""

    return sorted(
        keys,
        key=lambda value: (value[1] != "default", value[1], value[0] != "control", value[0]),
    )


def summarize_behavior_results_v2(input_path: Path) -> list[BehaviorAggregateV2]:
    """Summarize reasoning v2 behavior by pressure type and thinking mode."""

    rows = read_jsonl(input_path)
    grouped: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    response_length_sums: defaultdict[tuple[str, str], int] = defaultdict(int)
    candidate_count_sums: defaultdict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        pressure_type = str(row.get("pressure_type", "unknown"))
        thinking_mode = str(row.get("thinking_mode", "default"))
        route_label = str(row.get("route_label", "unknown"))
        group_key = (pressure_type, thinking_mode)
        grouped[(pressure_type, thinking_mode, route_label)] += 1
        totals[group_key] += 1
        response_length_sums[group_key] += int(row.get("response_length_chars", 0))
        candidate_count_sums[group_key] += len(row.get("parse_candidates", []))

    aggregates: list[BehaviorAggregateV2] = []
    for pressure_type, thinking_mode in _ordered_group_keys(set(totals)):
        total = totals[(pressure_type, thinking_mode)]
        group_key = (pressure_type, thinking_mode)
        aggregates.append(
            BehaviorAggregateV2(
                thinking_mode=thinking_mode,
                pressure_type=pressure_type,
                total=total,
                robust_rate=safe_divide(
                    grouped[(pressure_type, thinking_mode, "robust_correct")],
                    total,
                ),
                shortcut_followed_rate=safe_divide(
                    grouped[(pressure_type, thinking_mode, "shortcut_followed")],
                    total,
                ),
                wrong_nonshortcut_rate=safe_divide(
                    grouped[(pressure_type, thinking_mode, "wrong_nonshortcut")],
                    total,
                ),
                parse_failed_rate=safe_divide(
                    grouped[(pressure_type, thinking_mode, "parse_failed")],
                    total,
                ),
                parse_ambiguous_rate=safe_divide(
                    grouped[(pressure_type, thinking_mode, "parse_ambiguous")],
                    total,
                ),
                average_response_length_chars=safe_divide(
                    response_length_sums[group_key],
                    total,
                ),
                average_candidate_count=safe_divide(candidate_count_sums[group_key], total),
            )
        )
    return aggregates


def summarize_failure_subtypes_v2(input_path: Path) -> list[CountAggregateV2]:
    """Count wrong-nonshortcut failure subtypes by pressure type and thinking mode."""

    rows = read_jsonl(input_path)
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        if str(row.get("route_label")) != "wrong_nonshortcut":
            continue
        subtype = row.get("failure_subtype")
        if subtype is None:
            continue
        pressure_type = str(row.get("pressure_type", "unknown"))
        thinking_mode = str(row.get("thinking_mode", "default"))
        counts[(pressure_type, thinking_mode, str(subtype))] += 1

    return [
        CountAggregateV2(
            thinking_mode=thinking_mode,
            pressure_type=pressure_type,
            label=label,
            count=count,
        )
        for pressure_type, thinking_mode, label in sorted(
            counts,
            key=lambda value: (
                value[1] != "default",
                value[1],
                value[0] != "control",
                value[0],
                value[2],
            ),
        )
        for count in [counts[(pressure_type, thinking_mode, label)]]
    ]


def summarize_parse_status_counts_v2(input_path: Path) -> list[CountAggregateV2]:
    """Count parse statuses by pressure type and thinking mode."""

    rows = read_jsonl(input_path)
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        pressure_type = str(row.get("pressure_type", "unknown"))
        thinking_mode = str(row.get("thinking_mode", "default"))
        parse_status = str(row.get("parse_status", "unknown"))
        counts[(pressure_type, thinking_mode, parse_status)] += 1

    return [
        CountAggregateV2(
            thinking_mode=thinking_mode,
            pressure_type=pressure_type,
            label=label,
            count=count,
        )
        for pressure_type, thinking_mode, label in sorted(
            counts,
            key=lambda value: (
                value[1] != "default",
                value[1],
                value[0] != "control",
                value[0],
                value[2],
            ),
        )
        for count in [counts[(pressure_type, thinking_mode, label)]]
    ]


def summarize_paired_route_shifts_v2(input_path: Path) -> list[PairedRouteShiftAggregateV2]:
    """Summarize paired control-versus-pressure route shifts by base task id."""

    rows = read_jsonl(input_path)
    control_routes: dict[tuple[str, str], str] = {}
    pressure_routes: defaultdict[tuple[str, str], dict[str, str]] = defaultdict(dict)

    for row in rows:
        metadata = row.get("metadata", {})
        base_task_id = metadata.get("base_task_id")
        if base_task_id is None:
            continue
        thinking_mode = str(row.get("thinking_mode", "default"))
        pressure_type = str(row.get("pressure_type", "unknown"))
        route_label = str(row.get("route_label", "unknown"))
        key = (thinking_mode, str(base_task_id))
        if pressure_type == "control":
            control_routes[key] = route_label
        else:
            pressure_routes[(thinking_mode, pressure_type)][str(base_task_id)] = route_label

    aggregates: list[PairedRouteShiftAggregateV2] = []
    for thinking_mode, pressure_type in sorted(
        pressure_routes,
        key=lambda value: (value[0] != "default", value[0], value[1]),
    ):
        route_map = pressure_routes[(thinking_mode, pressure_type)]
        control_correct_to_pressure_shortcut = 0
        control_wrong_to_pressure_shortcut = 0
        control_correct_to_pressure_wrong = 0
        control_shortcut_to_pressure_shortcut = 0
        for base_task_id, pressure_route in route_map.items():
            control_route = control_routes.get((thinking_mode, base_task_id))
            if control_route is None:
                continue
            if control_route == "robust_correct" and pressure_route == "shortcut_followed":
                control_correct_to_pressure_shortcut += 1
            if control_route == "wrong_nonshortcut" and pressure_route == "shortcut_followed":
                control_wrong_to_pressure_shortcut += 1
            if control_route == "robust_correct" and pressure_route == "wrong_nonshortcut":
                control_correct_to_pressure_wrong += 1
            if control_route == "shortcut_followed" and pressure_route == "shortcut_followed":
                control_shortcut_to_pressure_shortcut += 1

        aggregates.append(
            PairedRouteShiftAggregateV2(
                thinking_mode=thinking_mode,
                pressure_type=pressure_type,
                control_correct_to_pressure_shortcut=control_correct_to_pressure_shortcut,
                control_wrong_nonshortcut_to_pressure_shortcut=control_wrong_to_pressure_shortcut,
                control_correct_to_pressure_wrong_nonshortcut=control_correct_to_pressure_wrong,
                control_shortcut_to_pressure_shortcut=control_shortcut_to_pressure_shortcut,
            )
        )
    return aggregates
