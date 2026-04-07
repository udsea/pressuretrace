"""Generate plots for the frozen reasoning probe results."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pressuretrace.analysis.reasoning_probe_reports import (
    get_frozen_reasoning_probe_paths,
    load_or_compute_baselines,
    load_probe_dataset_rows,
    load_probe_metrics_rows,
)
from pressuretrace.config import REASONING_PROBE_LAYERS, REASONING_PROBE_REPRESENTATIONS
from pressuretrace.probes.train_reasoning_probes import deduplicate_episode_rows
from pressuretrace.utils.io import ensure_directory


def _indexed_hidden_state_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[int, dict[str, float]]]:
    """Index hidden-state metrics by representation and layer."""

    indexed: dict[str, dict[int, dict[str, float]]] = {
        representation: {} for representation in REASONING_PROBE_REPRESENTATIONS
    }
    for row in rows:
        if str(row.get("kind")) != "hidden_state_probe":
            continue
        indexed[str(row["representation"])][int(row["layer"])] = {
            "roc_auc": float(row["roc_auc"]),
            "accuracy": float(row["accuracy"]),
        }
    return indexed


def _plot_metric_by_layer(
    *,
    output_path: Path,
    indexed_metrics: dict[str, dict[int, dict[str, float]]],
    metric_name: str,
    y_label: str,
) -> Path:
    """Plot one metric across layers for each representation."""

    ensure_directory(output_path.parent)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for representation in REASONING_PROBE_REPRESENTATIONS:
        xs = list(REASONING_PROBE_LAYERS)
        ys = [
            indexed_metrics.get(representation, {}).get(layer, {}).get(metric_name, float("nan"))
            for layer in xs
        ]
        axis.plot(xs, ys, marker="o", linewidth=2, label=representation)
    axis.set_xlabel("Layer")
    axis.set_ylabel(y_label)
    axis.set_xticks(list(REASONING_PROBE_LAYERS))
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _plot_baseline_comparison(
    *,
    output_path: Path,
    best_hidden_state_auc: float,
    baseline_metrics: dict[str, dict[str, float]],
) -> Path:
    """Plot a simple ROC-AUC comparison against the prompt baselines."""

    ensure_directory(output_path.parent)
    labels = ["best_hidden_state_probe", "prompt_length", "tfidf_prompt"]
    values = [
        best_hidden_state_auc,
        baseline_metrics["prompt_length"]["roc_auc"],
        baseline_metrics["tfidf_prompt"]["roc_auc"],
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    bars = axis.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axis.set_ylabel("ROC AUC")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_reasoning_probe_plots(
    *,
    metrics_path: Path,
    dataset_path: Path,
    summary_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate the three frozen reasoning probe plots."""

    metrics_rows = load_probe_metrics_rows(metrics_path)
    indexed_metrics = _indexed_hidden_state_metrics(metrics_rows)
    episode_rows = deduplicate_episode_rows(load_probe_dataset_rows(dataset_path))
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else None
    baseline_metrics = load_or_compute_baselines(
        episode_rows=episode_rows,
        summary_text=summary_text,
    )
    best_hidden_state_auc = max(float(row["roc_auc"]) for row in metrics_rows)

    return [
        _plot_metric_by_layer(
            output_path=output_dir / "reasoning_probe_auc_by_layer_qwen-qwen3-14b_off.png",
            indexed_metrics=indexed_metrics,
            metric_name="roc_auc",
            y_label="ROC AUC",
        ),
        _plot_metric_by_layer(
            output_path=output_dir / "reasoning_probe_accuracy_by_layer_qwen-qwen3-14b_off.png",
            indexed_metrics=indexed_metrics,
            metric_name="accuracy",
            y_label="Accuracy",
        ),
        _plot_baseline_comparison(
            output_path=(
                output_dir / "reasoning_probe_baseline_comparison_qwen-qwen3-14b_off.png"
            ),
            best_hidden_state_auc=best_hidden_state_auc,
            baseline_metrics=baseline_metrics,
        ),
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for plot generation."""

    paths = get_frozen_reasoning_probe_paths()
    parser = argparse.ArgumentParser(
        description="Generate plots for the frozen reasoning probe results.",
    )
    parser.add_argument("--metrics-path", type=Path, default=paths.probe_metrics_path)
    parser.add_argument("--dataset-path", type=Path, default=paths.probe_dataset_path)
    parser.add_argument("--summary-path", type=Path, default=paths.probe_summary_path)
    parser.add_argument("--output-dir", type=Path, default=paths.frozen_root / "results")
    return parser


def main(argv: Sequence[str] | None = None) -> list[Path]:
    """Run the frozen reasoning probe plotting workflow."""

    args = _build_arg_parser().parse_args(argv)
    outputs = generate_reasoning_probe_plots(
        metrics_path=args.metrics_path,
        dataset_path=args.dataset_path,
        summary_path=args.summary_path,
        output_dir=args.output_dir,
    )
    for output_path in outputs:
        print(output_path)
    return outputs


if __name__ == "__main__":
    main()
