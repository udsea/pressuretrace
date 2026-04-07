"""Terminal rendering for reasoning v2 summary tables."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from pressuretrace.behavior.reasoning_summary_v2.aggregates import (
    summarize_behavior_results_v2,
    summarize_failure_subtypes_v2,
    summarize_paired_route_shifts_v2,
    summarize_parse_status_counts_v2,
)


def print_behavior_summary_v2(input_path: Path) -> None:
    """Render reasoning v2 summary tables to the terminal."""

    console = Console()
    aggregates = summarize_behavior_results_v2(input_path)

    main_table = Table(title=f"Reasoning V2 Summary: {input_path.name}")
    main_table.add_column("Thinking")
    main_table.add_column("Pressure Type")
    main_table.add_column("N", justify="right")
    main_table.add_column("Robust Rate", justify="right")
    main_table.add_column("Shortcut Rate", justify="right")
    main_table.add_column("Wrong Nonshortcut", justify="right")
    main_table.add_column("Parse Failed", justify="right")
    main_table.add_column("Parse Ambiguous", justify="right")
    for aggregate in aggregates:
        main_table.add_row(
            aggregate.thinking_mode,
            aggregate.pressure_type,
            str(aggregate.total),
            f"{aggregate.robust_rate:.2%}",
            f"{aggregate.shortcut_followed_rate:.2%}",
            f"{aggregate.wrong_nonshortcut_rate:.2%}",
            f"{aggregate.parse_failed_rate:.2%}",
            f"{aggregate.parse_ambiguous_rate:.2%}",
        )
    console.print(main_table)

    metrics_table = Table(title="Supplemental Metrics")
    metrics_table.add_column("Thinking")
    metrics_table.add_column("Pressure Type")
    metrics_table.add_column("Avg Response Chars", justify="right")
    metrics_table.add_column("Avg Candidate Count", justify="right")
    for aggregate in aggregates:
        metrics_table.add_row(
            aggregate.thinking_mode,
            aggregate.pressure_type,
            f"{aggregate.average_response_length_chars:.1f}",
            f"{aggregate.average_candidate_count:.2f}",
        )
    console.print(metrics_table)

    subtype_rows = summarize_failure_subtypes_v2(input_path)
    if subtype_rows:
        subtype_table = Table(title="Failure Subtypes")
        subtype_table.add_column("Thinking")
        subtype_table.add_column("Pressure Type")
        subtype_table.add_column("Failure Subtype")
        subtype_table.add_column("Count", justify="right")
        for subtype_row in subtype_rows:
            subtype_table.add_row(
                subtype_row.thinking_mode,
                subtype_row.pressure_type,
                subtype_row.label,
                str(subtype_row.count),
            )
        console.print(subtype_table)

    parse_status_rows = summarize_parse_status_counts_v2(input_path)
    if parse_status_rows:
        parse_status_table = Table(title="Parse Status Counts")
        parse_status_table.add_column("Thinking")
        parse_status_table.add_column("Pressure Type")
        parse_status_table.add_column("Parse Status")
        parse_status_table.add_column("Count", justify="right")
        for parse_status_row in parse_status_rows:
            parse_status_table.add_row(
                parse_status_row.thinking_mode,
                parse_status_row.pressure_type,
                parse_status_row.label,
                str(parse_status_row.count),
            )
        console.print(parse_status_table)

    route_shift_rows = summarize_paired_route_shifts_v2(input_path)
    if route_shift_rows:
        route_shift_table = Table(title="Paired Route Shifts")
        route_shift_table.add_column("Thinking")
        route_shift_table.add_column("Pressure Type")
        route_shift_table.add_column("control_correct->pressure_shortcut", justify="right")
        route_shift_table.add_column(
            "control_wrong_nonshortcut->pressure_shortcut",
            justify="right",
        )
        route_shift_table.add_column(
            "control_correct->pressure_wrong_nonshortcut",
            justify="right",
        )
        route_shift_table.add_column("control_shortcut->pressure_shortcut", justify="right")
        for route_shift_row in route_shift_rows:
            route_shift_table.add_row(
                route_shift_row.thinking_mode,
                route_shift_row.pressure_type,
                str(route_shift_row.control_correct_to_pressure_shortcut),
                str(route_shift_row.control_wrong_nonshortcut_to_pressure_shortcut),
                str(route_shift_row.control_correct_to_pressure_wrong_nonshortcut),
                str(route_shift_row.control_shortcut_to_pressure_shortcut),
            )
        console.print(route_shift_table)
