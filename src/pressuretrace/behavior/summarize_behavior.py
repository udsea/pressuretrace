"""Aggregation helpers for reasoning pilot result JSONL files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from pressuretrace.utils.io import read_jsonl
from pressuretrace.utils.math_utils import safe_divide


@dataclass(frozen=True)
class BehaviorAggregate:
    """Aggregate row for a reasoning pilot summary."""

    pressure_type: str
    total: int
    robust_rate: float
    shortcut_followed_rate: float
    other_error_rate: float


def summarize_behavior_results(input_path: Path) -> list[BehaviorAggregate]:
    """Summarize reasoning behavior by pressure type."""

    rows = read_jsonl(input_path)
    grouped: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()

    for row in rows:
        pressure_type = str(row.get("pressure_type", row.get("pressure_level", "unknown")))
        route_label = str(row.get("route_label", "unknown"))
        grouped[(pressure_type, route_label)] += 1
        totals[pressure_type] += 1

    aggregates: list[BehaviorAggregate] = []
    ordered_pressure_types = sorted(totals, key=lambda value: (value != "control", value))
    for pressure_type in ordered_pressure_types:
        total = totals[pressure_type]
        aggregates.append(
            BehaviorAggregate(
                pressure_type=pressure_type,
                total=total,
                robust_rate=safe_divide(grouped[(pressure_type, "robust_correct")], total),
                shortcut_followed_rate=safe_divide(
                    grouped[(pressure_type, "shortcut_followed")],
                    total,
                ),
                other_error_rate=safe_divide(grouped[(pressure_type, "other_error")], total),
            )
        )
    return aggregates


def print_behavior_summary(input_path: Path) -> None:
    """Render a clean reasoning-summary table to the terminal."""

    console = Console()
    table = Table(title=f"Reasoning Summary: {input_path.name}")
    table.add_column("Pressure Type")
    table.add_column("N", justify="right")
    table.add_column("Robust Rate", justify="right")
    table.add_column("Shortcut Rate", justify="right")
    table.add_column("Other Error Rate", justify="right")

    for row in summarize_behavior_results(input_path):
        table.add_row(
            row.pressure_type,
            str(row.total),
            f"{row.robust_rate:.2%}",
            f"{row.shortcut_followed_rate:.2%}",
            f"{row.other_error_rate:.2%}",
        )

    console.print(table)
