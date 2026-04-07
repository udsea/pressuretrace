"""Export a compact paper table for the frozen reasoning probe run."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pressuretrace.analysis.reasoning_probe_reports import (
    compute_prompt_baselines,
    format_metric,
    get_frozen_reasoning_probe_paths,
    load_probe_dataset_rows,
    load_probe_metrics_rows,
    rank_hidden_state_probe_rows,
)
from pressuretrace.probes.train_reasoning_probes import deduplicate_episode_rows
from pressuretrace.utils.io import ensure_directory


def _table_rows(
    *,
    metrics_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, dict[str, float]],
) -> list[dict[str, str | int | float]]:
    """Build the ordered paper-table rows."""

    rows: list[dict[str, str | int | float]] = []
    for row in rank_hidden_state_probe_rows(metrics_rows)[:6]:
        rows.append(
            {
                "method": "hidden_state_probe",
                "layer": int(row["layer"]),
                "representation": str(row["representation"]),
                "roc_auc": float(row["roc_auc"]),
                "accuracy": float(row["accuracy"]),
                "f1": float(row["f1"]),
            }
        )
    for baseline_name in ("prompt_length", "tfidf_prompt"):
        baseline = baseline_metrics[baseline_name]
        rows.append(
            {
                "method": baseline_name,
                "layer": "",
                "representation": "",
                "roc_auc": float(baseline["roc_auc"]),
                "accuracy": float(baseline["accuracy"]),
                "f1": float(baseline["f1"]),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, str | int | float]], output_path: Path) -> Path:
    """Write the paper table as CSV."""

    ensure_directory(output_path.parent)
    fieldnames = ["method", "layer", "representation", "roc_auc", "accuracy", "f1"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "layer": row["layer"],
                    "representation": row["representation"],
                    "roc_auc": format_metric(row["roc_auc"]),
                    "accuracy": format_metric(row["accuracy"]),
                    "f1": format_metric(row["f1"]),
                }
            )
    return output_path


def _write_markdown(rows: list[dict[str, str | int | float]], output_path: Path) -> Path:
    """Write the paper table as Markdown."""

    lines = [
        "| method | layer | representation | roc_auc | accuracy | f1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["layer"]),
                    str(row["representation"]),
                    format_metric(row["roc_auc"]),
                    format_metric(row["accuracy"]),
                    format_metric(row["f1"]),
                ]
            )
            + " |"
        )
    ensure_directory(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def export_reasoning_probe_table(
    *,
    metrics_path: Path,
    dataset_path: Path,
    csv_path: Path,
    markdown_path: Path,
) -> dict[str, Path]:
    """Export the frozen reasoning probe paper table."""

    metrics_rows = load_probe_metrics_rows(metrics_path)
    episode_rows = deduplicate_episode_rows(load_probe_dataset_rows(dataset_path))
    baseline_metrics = compute_prompt_baselines(episode_rows)
    rows = _table_rows(metrics_rows=metrics_rows, baseline_metrics=baseline_metrics)
    return {
        "csv": _write_csv(rows, csv_path),
        "markdown": _write_markdown(rows, markdown_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for paper-table export."""

    paths = get_frozen_reasoning_probe_paths()
    parser = argparse.ArgumentParser(
        description="Export the frozen reasoning probe paper table.",
    )
    parser.add_argument("--metrics-path", type=Path, default=paths.probe_metrics_path)
    parser.add_argument("--dataset-path", type=Path, default=paths.probe_dataset_path)
    parser.add_argument("--csv-path", type=Path, default=paths.table_csv_path)
    parser.add_argument("--markdown-path", type=Path, default=paths.table_md_path)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    """Run the paper-table export."""

    args = _build_arg_parser().parse_args(argv)
    outputs = export_reasoning_probe_table(
        metrics_path=args.metrics_path,
        dataset_path=args.dataset_path,
        csv_path=args.csv_path,
        markdown_path=args.markdown_path,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


if __name__ == "__main__":
    main()
