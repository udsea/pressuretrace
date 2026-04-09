"""Summary helpers for coding-family behavior results."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pressuretrace.utils.io import ensure_directory, read_jsonl
from pressuretrace.utils.math_utils import safe_divide


@dataclass(frozen=True)
class CodingBehaviorAggregate:
    """Aggregate row for a coding-family behavior summary table."""

    pressure_type: str
    total: int
    robust_rate: float
    shortcut_rate: float
    wrong_nonshortcut_rate: float
    parse_failed_rate: float
    execution_failed_rate: float


@dataclass(frozen=True)
class CodingFailureSubtypeCount:
    """Count aggregate for coding-family failure subtypes."""

    pressure_type: str
    label: str
    count: int


def _ordered_pressure_types(pressure_types: set[str]) -> list[str]:
    """Sort pressure types with control first."""

    return sorted(pressure_types, key=lambda value: (value != "control", value))


def summarize_coding_behavior_results(input_path: Path) -> list[CodingBehaviorAggregate]:
    """Summarize coding-family behavior by pressure type."""

    rows = read_jsonl(input_path)
    totals: Counter[str] = Counter()
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        pressure_type = str(row.get("pressure_type", "unknown"))
        route_label = str(row.get("route_label", "unknown"))
        totals[pressure_type] += 1
        counts[(pressure_type, route_label)] += 1

    aggregates: list[CodingBehaviorAggregate] = []
    for pressure_type in _ordered_pressure_types(set(totals)):
        total = totals[pressure_type]
        aggregates.append(
            CodingBehaviorAggregate(
                pressure_type=pressure_type,
                total=total,
                robust_rate=safe_divide(counts[(pressure_type, "robust_success")], total),
                shortcut_rate=safe_divide(counts[(pressure_type, "shortcut_success")], total),
                wrong_nonshortcut_rate=safe_divide(
                    counts[(pressure_type, "wrong_nonshortcut")],
                    total,
                ),
                parse_failed_rate=safe_divide(counts[(pressure_type, "parse_failed")], total),
                execution_failed_rate=safe_divide(
                    counts[(pressure_type, "execution_failed")],
                    total,
                ),
            )
        )
    return aggregates


def summarize_coding_control_robust_slice(input_path: Path) -> list[CodingBehaviorAggregate]:
    """Summarize pressure outcomes on base tasks with robust control routes."""

    rows = read_jsonl(input_path)
    control_routes: dict[str, str] = {}
    totals: Counter[str] = Counter()
    counts: Counter[tuple[str, str]] = Counter()

    for row in rows:
        base_task_id = str(row.get("base_task_id", ""))
        if not base_task_id:
            continue
        if str(row.get("pressure_type")) == "control":
            control_routes[base_task_id] = str(row.get("route_label", "unknown"))

    for row in rows:
        base_task_id = str(row.get("base_task_id", ""))
        pressure_type = str(row.get("pressure_type", "unknown"))
        if not base_task_id or pressure_type == "control":
            continue
        if control_routes.get(base_task_id) != "robust_success":
            continue
        route_label = str(row.get("route_label", "unknown"))
        totals[pressure_type] += 1
        counts[(pressure_type, route_label)] += 1

    aggregates: list[CodingBehaviorAggregate] = []
    for pressure_type in _ordered_pressure_types(set(totals)):
        total = totals[pressure_type]
        aggregates.append(
            CodingBehaviorAggregate(
                pressure_type=pressure_type,
                total=total,
                robust_rate=safe_divide(counts[(pressure_type, "robust_success")], total),
                shortcut_rate=safe_divide(counts[(pressure_type, "shortcut_success")], total),
                wrong_nonshortcut_rate=safe_divide(
                    counts[(pressure_type, "wrong_nonshortcut")],
                    total,
                ),
                parse_failed_rate=safe_divide(counts[(pressure_type, "parse_failed")], total),
                execution_failed_rate=safe_divide(
                    counts[(pressure_type, "execution_failed")],
                    total,
                ),
            )
        )
    return aggregates


def summarize_coding_failure_subtypes(input_path: Path) -> list[CodingFailureSubtypeCount]:
    """Count coding-family failure subtypes by pressure type."""

    rows = read_jsonl(input_path)
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        subtype = row.get("failure_subtype")
        if subtype is None:
            continue
        pressure_type = str(row.get("pressure_type", "unknown"))
        counts[(pressure_type, str(subtype))] += 1

    return [
        CodingFailureSubtypeCount(pressure_type=pressure_type, label=label, count=count)
        for pressure_type, label in sorted(
            counts, key=lambda value: (value[0] != "control", value[0], value[1])
        )
        for count in [counts[(pressure_type, label)]]
    ]


def render_coding_behavior_summary_text(input_path: Path) -> str:
    """Render a plain-text coding-family summary report."""

    full_rows = summarize_coding_behavior_results(input_path)
    slice_rows = summarize_coding_control_robust_slice(input_path)
    subtype_rows = summarize_coding_failure_subtypes(input_path)

    lines = [f"Coding Family Summary: {input_path.name}", ""]
    lines.append("All rows:")
    for row in full_rows:
        lines.append(
            "  "
            f"{row.pressure_type}: n={row.total}, robust={row.robust_rate:.2%}, "
            f"shortcut={row.shortcut_rate:.2%}, wrong={row.wrong_nonshortcut_rate:.2%}, "
            f"parse_failed={row.parse_failed_rate:.2%}, "
            f"execution_failed={row.execution_failed_rate:.2%}"
        )

    lines.append("")
    lines.append("Control-robust slice:")
    for row in slice_rows:
        lines.append(
            "  "
            f"{row.pressure_type}: n={row.total}, pressure_robust={row.robust_rate:.2%}, "
            f"pressure_shortcut={row.shortcut_rate:.2%}, "
            f"pressure_wrong={row.wrong_nonshortcut_rate:.2%}, "
            f"pressure_parse_failed={row.parse_failed_rate:.2%}, "
            f"pressure_execution_failed={row.execution_failed_rate:.2%}"
        )

    if subtype_rows:
        lines.append("")
        lines.append("Failure subtypes:")
        for row in subtype_rows:
            lines.append(f"  {row.pressure_type}: {row.label}={row.count}")

    return "\n".join(lines) + "\n"


def export_coding_behavior_summary(
    *,
    input_path: Path,
    text_output_path: Path,
    csv_output_path: Path,
) -> tuple[Path, Path]:
    """Write coding-family summary exports to text and CSV."""

    text = render_coding_behavior_summary_text(input_path)
    ensure_directory(text_output_path.parent)
    text_output_path.write_text(text, encoding="utf-8")

    ensure_directory(csv_output_path.parent)
    fieldnames = [
        "table",
        "pressure_type",
        "label",
        "total",
        "count",
        "robust_rate",
        "shortcut_rate",
        "wrong_nonshortcut_rate",
        "parse_failed_rate",
        "execution_failed_rate",
    ]
    with csv_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summarize_coding_behavior_results(input_path):
            writer.writerow(
                {
                    "table": "all_rows",
                    "pressure_type": row.pressure_type,
                    "label": "",
                    "total": row.total,
                    "count": "",
                    "robust_rate": row.robust_rate,
                    "shortcut_rate": row.shortcut_rate,
                    "wrong_nonshortcut_rate": row.wrong_nonshortcut_rate,
                    "parse_failed_rate": row.parse_failed_rate,
                    "execution_failed_rate": row.execution_failed_rate,
                }
            )
        for row in summarize_coding_control_robust_slice(input_path):
            writer.writerow(
                {
                    "table": "control_robust_slice",
                    "pressure_type": row.pressure_type,
                    "label": "",
                    "total": row.total,
                    "count": "",
                    "robust_rate": row.robust_rate,
                    "shortcut_rate": row.shortcut_rate,
                    "wrong_nonshortcut_rate": row.wrong_nonshortcut_rate,
                    "parse_failed_rate": row.parse_failed_rate,
                    "execution_failed_rate": row.execution_failed_rate,
                }
            )
        for row in summarize_coding_failure_subtypes(input_path):
            writer.writerow(
                {
                    "table": "failure_subtype",
                    "pressure_type": row.pressure_type,
                    "label": row.label,
                    "total": "",
                    "count": row.count,
                    "robust_rate": "",
                    "shortcut_rate": "",
                    "wrong_nonshortcut_rate": "",
                    "parse_failed_rate": "",
                    "execution_failed_rate": "",
                }
            )
    return text_output_path, csv_output_path
