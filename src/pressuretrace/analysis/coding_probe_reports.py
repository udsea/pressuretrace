"""Reporting and export helpers for the frozen coding probe bundle."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pressuretrace.config import (
    CODING_PROBE_LAYERS,
    CODING_PROBE_PRESSURE_TYPES,
    CODING_PROBE_RANDOM_SEED,
    CODING_PROBE_REPRESENTATIONS,
    CODING_PROBE_TEST_SIZE,
    CODING_V1_MODEL_NAME,
    CODING_V1_THINKING_MODE,
    resolve_coding_frozen_root,
)
from pressuretrace.probes.train_coding_probes import (
    deduplicate_episode_rows,
    fit_prompt_length_baseline,
    fit_tfidf_baseline,
    split_base_tasks,
    split_rows_by_base_task,
)
from pressuretrace.utils.io import ensure_directory, read_jsonl


@dataclass(frozen=True)
class FrozenCodingProbePaths:
    """Canonical file locations for the frozen coding probe run."""

    frozen_root: Path
    manifest_path: Path
    control_slice_path: Path
    paper_results_path: Path
    hidden_states_path: Path
    probe_dataset_path: Path
    probe_metrics_path: Path
    probe_summary_path: Path
    artifact_index_path: Path
    metrics_csv_path: Path


@dataclass(frozen=True)
class ProbeSplitSummary:
    """Deterministic split metadata for the frozen coding probe dataset."""

    train_base_task_count: int
    test_base_task_count: int
    stratified: bool


def get_frozen_coding_probe_paths(root: Path | None = None) -> FrozenCodingProbePaths:
    """Resolve the frozen coding probe paths under the configured root."""

    frozen_root = root or resolve_coding_frozen_root()
    results_root = frozen_root / "results"
    return FrozenCodingProbePaths(
        frozen_root=frozen_root,
        manifest_path=(
            frozen_root / "data" / "manifests" / "coding_paper_slice_qwen-qwen3-14b_off.jsonl"
        ),
        control_slice_path=(
            frozen_root / "data" / "splits" / "coding_control_robust_slice_qwen-qwen3-14b_off.jsonl"
        ),
        paper_results_path=results_root / "coding_paper_slice_qwen-qwen3-14b_off.jsonl",
        hidden_states_path=results_root / "coding_probe_hidden_states_qwen-qwen3-14b_off.jsonl",
        probe_dataset_path=results_root / "coding_probe_dataset_qwen-qwen3-14b_off.jsonl",
        probe_metrics_path=results_root / "coding_probe_metrics_qwen-qwen3-14b_off.jsonl",
        probe_summary_path=results_root / "coding_probe_summary_qwen-qwen3-14b_off.txt",
        artifact_index_path=frozen_root / "ARTIFACTS.md",
        metrics_csv_path=results_root / "coding_probe_metrics_qwen-qwen3-14b_off.csv",
    )


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into plain dictionaries."""

    return [dict(row) for row in read_jsonl(path)]


def load_probe_metrics_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the frozen coding hidden-state probe metrics."""

    return load_jsonl_rows(path or get_frozen_coding_probe_paths().probe_metrics_path)


def load_probe_dataset_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the frozen coding probe dataset."""

    return load_jsonl_rows(path or get_frozen_coding_probe_paths().probe_dataset_path)


def load_hidden_state_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the frozen coding hidden-state rows."""

    return load_jsonl_rows(path or get_frozen_coding_probe_paths().hidden_states_path)


def count_by_pressure_type(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count retained episodes by pressure type."""

    counts = {pressure_type: 0 for pressure_type in CODING_PROBE_PRESSURE_TYPES}
    for row in rows:
        pressure_type = str(row["pressure_type"])
        counts[pressure_type] = counts.get(pressure_type, 0) + 1
    return counts


def count_by_binary_label(rows: list[dict[str, Any]]) -> dict[int, int]:
    """Count retained episodes by binary label."""

    label_counts: dict[int, int] = {0: 0, 1: 0}
    for row in rows:
        label_counts[int(row["binary_label"])] += 1
    return label_counts


def count_by_archetype(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count retained episodes by coding archetype."""

    counts: Counter[str] = Counter(str(row.get("archetype", "")) for row in rows)
    return dict(sorted(counts.items()))


def hidden_state_probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only hidden-state probe metrics."""

    return [row for row in rows if str(row.get("kind")) == "hidden_state_probe"]


def rank_hidden_state_probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort hidden-state probe rows by ROC AUC, then accuracy."""

    return sorted(
        hidden_state_probe_rows(rows),
        key=lambda row: (
            float("-inf") if math.isnan(float(row["roc_auc"])) else float(row["roc_auc"]),
            float(row["accuracy"]),
        ),
        reverse=True,
    )


def best_hidden_state_probe_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the best hidden-state probe row."""

    ranked_rows = rank_hidden_state_probe_rows(rows)
    if not ranked_rows:
        raise ValueError("No hidden-state probe rows were found.")
    return ranked_rows[0]


def probe_split_summary(
    episode_rows: list[dict[str, Any]],
    *,
    seed: int = CODING_PROBE_RANDOM_SEED,
    test_size: float = CODING_PROBE_TEST_SIZE,
) -> ProbeSplitSummary:
    """Recreate the deterministic base-task split used by probe training."""

    train_ids, test_ids, stratified = split_base_tasks(
        episode_rows,
        seed=seed,
        test_size=test_size,
    )
    return ProbeSplitSummary(
        train_base_task_count=len(train_ids),
        test_base_task_count=len(test_ids),
        stratified=stratified,
    )


def _binary_metrics(
    *,
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
) -> dict[str, float]:
    """Compute binary metrics for prompt-only baselines."""

    test_size = len(y_true)
    positives = sum(y_true)
    true_positives = sum(
        int(pred == 1 and true == 1) for pred, true in zip(y_pred, y_true, strict=True)
    )
    predicted_positives = sum(int(pred == 1) for pred in y_pred)
    accuracy = (
        sum(int(pred == true) for pred, true in zip(y_pred, y_true, strict=True)) / test_size
        if test_size
        else 0.0
    )
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    try:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")
    return {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "f1": f1,
    }


def compute_prompt_baselines(
    episode_rows: list[dict[str, Any]],
    *,
    seed: int = CODING_PROBE_RANDOM_SEED,
    test_size: float = CODING_PROBE_TEST_SIZE,
) -> dict[str, dict[str, float]]:
    """Recompute the two prompt-only baselines from the frozen dataset."""

    train_ids, test_ids, _ = split_base_tasks(
        episode_rows,
        seed=seed,
        test_size=test_size,
    )
    train_rows, test_rows = split_rows_by_base_task(episode_rows, train_ids, test_ids)
    y_true = [int(row["binary_label"]) for row in test_rows]
    prompt_predictions, prompt_probabilities = fit_prompt_length_baseline(
        train_rows,
        test_rows,
        seed,
    )
    tfidf_predictions, tfidf_probabilities = fit_tfidf_baseline(train_rows, test_rows, seed)
    return {
        "prompt_length": _binary_metrics(
            y_true=y_true,
            y_pred=prompt_predictions.tolist(),
            y_prob=prompt_probabilities.tolist(),
        ),
        "tfidf_prompt": _binary_metrics(
            y_true=y_true,
            y_pred=tfidf_predictions.tolist(),
            y_prob=tfidf_probabilities.tolist(),
        ),
    }


def format_metric(value: Any) -> str:
    """Format a metric value for human-readable exports."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "nan"
    return f"{numeric:.4f}"


def build_summary_text(paths: FrozenCodingProbePaths | None = None) -> str:
    """Render a polished summary from the frozen coding probe artifacts."""

    resolved = paths or get_frozen_coding_probe_paths()
    metrics_rows = load_probe_metrics_rows(resolved.probe_metrics_path)
    hidden_state_rows = load_hidden_state_rows(resolved.hidden_states_path)
    episode_rows = deduplicate_episode_rows(load_probe_dataset_rows(resolved.probe_dataset_path))
    pressure_counts = count_by_pressure_type(episode_rows)
    label_counts = count_by_binary_label(episode_rows)
    archetype_counts = count_by_archetype(episode_rows)
    split_summary = probe_split_summary(episode_rows)
    baselines = compute_prompt_baselines(episode_rows=episode_rows)
    ranked_rows = rank_hidden_state_probe_rows(metrics_rows)
    best_row = ranked_rows[0]

    lines = [
        "PressureTrace coding probe summary",
        "",
        "Filtering logic:",
        "  - pressure_type in {neutral_wrong_answer_cue, teacher_anchor}",
        "  - route_label in {robust_success, shortcut_success}",
        "  - binary_label: robust_success=0, shortcut_success=1",
        "",
        f"Frozen root: {resolved.frozen_root}",
        f"Source manifest: {resolved.manifest_path}",
        f"Source results: {resolved.paper_results_path}",
        f"Hidden-state file: {resolved.hidden_states_path}",
        f"Probe dataset: {resolved.probe_dataset_path}",
        f"Probe metrics JSONL: {resolved.probe_metrics_path}",
        "",
        f"Model: {CODING_V1_MODEL_NAME}",
        f"Thinking mode: {CODING_V1_THINKING_MODE}",
        f"Hidden-state rows: {len(hidden_state_rows)}",
        f"Retained episodes: {len(episode_rows)}",
        f"Unique base tasks: {len({str(row['base_task_id']) for row in episode_rows})}",
        (
            "Per-pressure counts: "
            + ", ".join(
                f"{pressure_type}={pressure_counts[pressure_type]}"
                for pressure_type in CODING_PROBE_PRESSURE_TYPES
            )
        ),
        (
            "Per-archetype counts: "
            + ", ".join(f"{name}={archetype_counts[name]}" for name in sorted(archetype_counts))
        ),
        f"Label balance: 0={label_counts[0]}, 1={label_counts[1]}",
        f"Train base tasks: {split_summary.train_base_task_count}",
        f"Test base tasks: {split_summary.test_base_task_count}",
        f"Split stratified by base-task label: {split_summary.stratified}",
        "Layers used: " + ", ".join(str(layer) for layer in CODING_PROBE_LAYERS),
        "Representations used: " + ", ".join(CODING_PROBE_REPRESENTATIONS),
        "",
        "Ranked hidden-state probe results:",
    ]
    for index, row in enumerate(ranked_rows[:6], start=1):
        lines.append(
            f"  {index}. layer={row['layer']}, representation={row['representation']}, "
            f"roc_auc={format_metric(row['roc_auc'])}, "
            f"accuracy={format_metric(row['accuracy'])}, "
            f"f1={format_metric(row['f1'])}"
        )
    lines.extend(
        [
            "",
            "Baseline comparison:",
            "  prompt_length: "
            f"roc_auc={format_metric(baselines['prompt_length']['roc_auc'])}, "
            f"accuracy={format_metric(baselines['prompt_length']['accuracy'])}",
            "  tfidf_prompt: "
            f"roc_auc={format_metric(baselines['tfidf_prompt']['roc_auc'])}, "
            f"accuracy={format_metric(baselines['tfidf_prompt']['accuracy'])}",
            "",
            "Best hidden-state probe result:",
            f"  layer={best_row['layer']}, representation={best_row['representation']}, "
            f"roc_auc={format_metric(best_row['roc_auc'])}, "
            f"accuracy={format_metric(best_row['accuracy'])}, "
            f"f1={format_metric(best_row['f1'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def export_metrics_csv(*, metrics_path: Path, output_path: Path) -> Path:
    """Convert the frozen metrics JSONL into a flat CSV."""

    rows = load_probe_metrics_rows(metrics_path)
    fieldnames = [
        "kind",
        "feature_set",
        "layer",
        "representation",
        "accuracy",
        "roc_auc",
        "inverse_auc",
        "precision",
        "recall",
        "f1",
        "train_size",
        "test_size",
        "train_pos",
        "train_neg",
        "test_pos",
        "test_neg",
        "mean_prob_y1",
        "mean_prob_y0",
    ]
    ensure_directory(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output_path


def export_artifact_index(*, paths: FrozenCodingProbePaths) -> Path:
    """Write a compact artifact index under the frozen root."""

    episode_rows = deduplicate_episode_rows(load_probe_dataset_rows(paths.probe_dataset_path))
    best_row = best_hidden_state_probe_row(load_probe_metrics_rows(paths.probe_metrics_path))
    baselines = compute_prompt_baselines(episode_rows=episode_rows)

    lines = [
        "# PressureTrace Frozen Coding Probe Artifacts",
        "",
        f"- Frozen root: `{paths.frozen_root}`",
        f"- Source manifest: `{paths.manifest_path}`",
        f"- Source results: `{paths.paper_results_path}`",
        f"- Hidden-state file: `{paths.hidden_states_path}`",
        f"- Probe dataset: `{paths.probe_dataset_path}`",
        f"- Probe metrics JSONL: `{paths.probe_metrics_path}`",
        f"- Probe summary TXT: `{paths.probe_summary_path}`",
        f"- Metrics CSV: `{paths.metrics_csv_path}`",
        "",
        "## Configuration",
        "",
        f"- Model name: `{CODING_V1_MODEL_NAME}`",
        f"- Thinking mode: `{CODING_V1_THINKING_MODE}`",
        (
            "- Pressure conditions used in probing: "
            f"`{', '.join(CODING_PROBE_PRESSURE_TYPES)}`"
        ),
        "- Label definition: `1 = shortcut_success`, `0 = robust_success`",
        "- Split rule: `base_task_id` train/test split, stratified when possible",
        f"- Layers used: `{', '.join(str(layer) for layer in CODING_PROBE_LAYERS)}`",
        f"- Representations used: `{', '.join(CODING_PROBE_REPRESENTATIONS)}`",
        "",
        "## Result Snapshot",
        "",
        (
            f"- Best hidden-state probe: layer `{best_row['layer']}`, "
            f"representation `{best_row['representation']}`, "
            f"ROC AUC `{format_metric(best_row['roc_auc'])}`"
        ),
        (
            "- Prompt-length baseline ROC AUC: "
            f"`{format_metric(baselines['prompt_length']['roc_auc'])}`"
        ),
        (
            "- TF-IDF baseline ROC AUC: "
            f"`{format_metric(baselines['tfidf_prompt']['roc_auc'])}`"
        ),
    ]
    ensure_directory(paths.artifact_index_path.parent)
    paths.artifact_index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths.artifact_index_path


def export_probe_reports(
    *,
    frozen_root: Path | None = None,
    write_artifact_index: bool = True,
    write_csv: bool = True,
    write_summary: bool = True,
) -> dict[str, Path]:
    """Export derived frozen coding probe reports."""

    paths = get_frozen_coding_probe_paths(frozen_root)
    outputs: dict[str, Path] = {}
    if write_artifact_index:
        outputs["artifact_index"] = export_artifact_index(paths=paths)
    if write_csv:
        outputs["metrics_csv"] = export_metrics_csv(
            metrics_path=paths.probe_metrics_path,
            output_path=paths.metrics_csv_path,
        )
    if write_summary:
        ensure_directory(paths.probe_summary_path.parent)
        paths.probe_summary_path.write_text(build_summary_text(paths), encoding="utf-8")
        outputs["summary"] = paths.probe_summary_path
    return outputs
