"""Train first-pass binary probes on frozen reasoning hidden states."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pressuretrace.config import (
    REASONING_PROBE_RANDOM_SEED,
    REASONING_PROBE_TEST_SIZE,
    reasoning_probe_dataset_path,
    reasoning_probe_metrics_path,
    reasoning_probe_summary_path,
)
from pressuretrace.utils.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class ProbeTrainingConfig:
    """Configuration for the reasoning probe trainer."""

    input_path: Path
    metrics_path: Path
    summary_path: Path
    seed: int = REASONING_PROBE_RANDOM_SEED
    test_size: float = REASONING_PROBE_TEST_SIZE


def default_probe_training_config() -> ProbeTrainingConfig:
    """Build the default training config for the frozen reasoning probe dataset."""

    return ProbeTrainingConfig(
        input_path=reasoning_probe_dataset_path(),
        metrics_path=reasoning_probe_metrics_path(),
        summary_path=reasoning_probe_summary_path(),
    )


def deduplicate_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse layer/representation duplicates into one row per task_id."""

    unique_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique_rows.setdefault(str(row["task_id"]), row)
    return list(unique_rows.values())


def _group_rows_by_base_task_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Collect dataset rows by latent base task id."""

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["base_task_id"])].append(row)
    return grouped_rows


def _group_label(rows: list[dict[str, Any]]) -> int:
    """Derive a single approximate label for one latent base task."""

    mean_label = sum(int(row["binary_label"]) for row in rows) / len(rows)
    return int(mean_label >= 0.5)


def split_base_tasks(
    rows: list[dict[str, Any]],
    seed: int,
    test_size: float,
) -> tuple[set[str], set[str], bool]:
    """Split latent base tasks into train/test partitions without leakage."""

    grouped_rows = _group_rows_by_base_task_id(rows)
    base_task_ids = sorted(grouped_rows)
    group_labels = [_group_label(grouped_rows[base_task_id]) for base_task_id in base_task_ids]
    label_counts = Counter(group_labels)
    can_stratify = len(label_counts) > 1 and min(label_counts.values()) >= 2

    split_kwargs: dict[str, Any] = {
        "test_size": test_size,
        "random_state": seed,
        "shuffle": True,
    }
    if can_stratify:
        split_kwargs["stratify"] = group_labels

    train_ids, test_ids = train_test_split(base_task_ids, **split_kwargs)
    return set(train_ids), set(test_ids), can_stratify


def split_rows_by_base_task(
    rows: list[dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition rows according to their base task split."""

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for row in rows:
        base_task_id = str(row["base_task_id"])
        if base_task_id in train_ids:
            train_rows.append(row)
        elif base_task_id in test_ids:
            test_rows.append(row)
    return train_rows, test_rows


def _build_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Build dense hidden-state features and labels from probe dataset rows."""

    features = np.asarray([row["hidden_state"] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["binary_label"]) for row in rows], dtype=np.int64)
    return features, labels


def _compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute standard binary classification metrics with graceful ROC-AUC fallback."""

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    metrics["inverse_auc"] = (
        1.0 - metrics["roc_auc"] if not math.isnan(metrics["roc_auc"]) else float("nan")
    )
    return metrics


def _fit_logistic_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a standard scaler + logistic regression probe and return predictions."""

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    probe.fit(train_features, train_labels)
    predicted_labels = probe.predict(test_features)
    predicted_probabilities = probe.predict_proba(test_features)[:, 1]
    return predicted_labels, predicted_probabilities


def _prompt_length_features(
    rows: list[dict[str, Any]],
    pressure_types: list[str],
) -> np.ndarray:
    """Construct a lightweight prompt baseline feature matrix."""

    pressure_type_index = {
        pressure_type: index for index, pressure_type in enumerate(pressure_types)
    }
    feature_rows: list[list[float]] = []
    for row in rows:
        prompt = str(row["prompt"])
        row_features = [
            float(len(prompt)),
            float(sum(character.isdigit() for character in prompt)),
            float(int(row.get("shortcut_answer_digit_count", 0))),
        ]
        row_features.extend([0.0] * len(pressure_types))
        row_features[3 + pressure_type_index[str(row["pressure_type"])]] = 1.0
        feature_rows.append(row_features)
    return np.asarray(feature_rows, dtype=np.float32)


def fit_prompt_length_baseline(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the prompt-length baseline."""

    pressure_types = sorted({str(row["pressure_type"]) for row in train_rows + test_rows})
    train_features = _prompt_length_features(train_rows, pressure_types)
    test_features = _prompt_length_features(test_rows, pressure_types)
    train_labels = np.asarray([int(row["binary_label"]) for row in train_rows], dtype=np.int64)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    probe.fit(train_features, train_labels)
    predicted_labels = probe.predict(test_features)
    predicted_probabilities = probe.predict_proba(test_features)[:, 1]
    return predicted_labels, predicted_probabilities


def fit_tfidf_baseline(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the prompt-only TF-IDF baseline."""

    probe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    train_text = [str(row["prompt"]) for row in train_rows]
    test_text = [str(row["prompt"]) for row in test_rows]
    train_labels = np.asarray([int(row["binary_label"]) for row in train_rows], dtype=np.int64)
    probe.fit(train_text, train_labels)
    predicted_labels = probe.predict(test_text)
    predicted_probabilities = probe.predict_proba(test_text)[:, 1]
    return predicted_labels, predicted_probabilities


def _mean_probability(probabilities: np.ndarray, labels: np.ndarray, target_label: int) -> float:
    """Average predicted probability for one label bucket."""

    mask = labels == target_label
    if not np.any(mask):
        return float("nan")
    return float(np.mean(probabilities[mask]))


def _metrics_row(
    *,
    kind: str,
    feature_set: str,
    layer: int | None,
    representation: str | None,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Assemble the standard metrics payload for one probe or baseline."""

    metrics = _compute_binary_metrics(y_true, y_pred, y_prob)
    train_pos = int(sum(int(row["binary_label"]) for row in train_rows))
    test_pos = int(np.sum(y_true))
    return {
        "kind": kind,
        "feature_set": feature_set,
        "layer": layer,
        "representation": representation,
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "inverse_auc": metrics["inverse_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "train_pos": train_pos,
        "train_neg": len(train_rows) - train_pos,
        "test_pos": test_pos,
        "test_neg": len(y_true) - test_pos,
        "mean_prob_y1": _mean_probability(y_prob, y_true, 1),
        "mean_prob_y0": _mean_probability(y_prob, y_true, 0),
    }


def _format_float(value: float) -> str:
    """Format a float for a compact human-readable summary."""

    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _rank_probe_rows(metrics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort hidden-state probe rows by ROC AUC, then accuracy."""

    return sorted(
        (row for row in metrics_rows if row["kind"] == "hidden_state_probe"),
        key=lambda row: (
            float("-inf") if math.isnan(row["roc_auc"]) else row["roc_auc"],
            row["accuracy"],
        ),
        reverse=True,
    )


def _write_summary(
    *,
    summary_path: Path,
    hidden_state_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    prompt_length_metrics: dict[str, Any],
    tfidf_metrics: dict[str, Any],
    train_base_task_count: int,
    test_base_task_count: int,
    stratified: bool,
) -> None:
    """Write a compact text summary for the probe sweep."""

    best_probe = _rank_probe_rows(probe_rows)[0]
    top_probe_rows = _rank_probe_rows(probe_rows)[:5]
    label_balance = Counter(int(row["binary_label"]) for row in episode_rows)

    lines = [
        "PressureTrace reasoning probe summary",
        "",
        "Dataset filtering logic:",
        "  - family == reasoning_conflict",
        "  - pressure_type in {neutral_wrong_answer_cue, teacher_anchor}",
        "  - route_label in {robust_correct, shortcut_followed}",
        "  - binary_label: robust_correct=0, shortcut_followed=1",
        "",
        f"Hidden-state rows: {len(hidden_state_rows)}",
        f"Retained episodes: {len(episode_rows)}",
        f"Unique base tasks: {len({str(row['base_task_id']) for row in episode_rows})}",
        f"Train base tasks: {train_base_task_count}",
        f"Test base tasks: {test_base_task_count}",
        f"Split stratified by base-task label: {stratified}",
        f"Label balance: 0={label_balance.get(0, 0)}, 1={label_balance.get(1, 0)}",
        "",
        "Best hidden-state probe result:",
        (
            f"  layer={best_probe['layer']}, representation={best_probe['representation']}, "
            f"roc_auc={_format_float(best_probe['roc_auc'])}, "
            f"accuracy={_format_float(best_probe['accuracy'])}, "
            f"f1={_format_float(best_probe['f1'])}"
        ),
        "",
        "Baseline results:",
        (
            f"  prompt_length: "
            f"roc_auc={_format_float(prompt_length_metrics['roc_auc'])}, "
            f"accuracy={_format_float(prompt_length_metrics['accuracy'])}"
        ),
        (
            f"  tfidf_prompt: "
            f"roc_auc={_format_float(tfidf_metrics['roc_auc'])}, "
            f"accuracy={_format_float(tfidf_metrics['accuracy'])}"
        ),
    ]
    lines.extend(["", "Top 5 hidden-state settings:"])
    for index, row in enumerate(top_probe_rows, start=1):
        lines.append(
            f"  {index}. layer={row['layer']}, representation={row['representation']}, "
            f"roc_auc={_format_float(row['roc_auc'])}, "
            f"accuracy={_format_float(row['accuracy'])}, "
            f"f1={_format_float(row['f1'])}"
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_ranked_summary(
    probe_rows: list[dict[str, Any]],
    prompt_length_metrics: dict[str, Any],
    tfidf_metrics: dict[str, Any],
) -> None:
    """Print a concise ranked table of probe and baseline outcomes."""

    print("layer\trepresentation\troc_auc\taccuracy\tf1")
    for row in _rank_probe_rows(probe_rows):
        print(
            "\t".join(
                [
                    str(row["layer"]),
                    str(row["representation"]),
                    _format_float(float(row["roc_auc"])),
                    _format_float(float(row["accuracy"])),
                    _format_float(float(row["f1"])),
                ]
            )
        )
    print(
        "prompt_length\t\t"
        f"{_format_float(float(prompt_length_metrics['roc_auc']))}\t"
        f"{_format_float(float(prompt_length_metrics['accuracy']))}\t"
        f"{_format_float(float(prompt_length_metrics['f1']))}"
    )
    print(
        "tfidf_prompt\t\t"
        f"{_format_float(float(tfidf_metrics['roc_auc']))}\t"
        f"{_format_float(float(tfidf_metrics['accuracy']))}\t"
        f"{_format_float(float(tfidf_metrics['f1']))}"
    )


def train_reasoning_probes(config: ProbeTrainingConfig) -> Path:
    """Train probes and baselines on the frozen reasoning probe dataset."""

    rows = [dict(row) for row in read_jsonl(config.input_path)]
    if not rows:
        raise ValueError("Probe dataset is empty.")

    episode_rows = deduplicate_episode_rows(rows)
    train_ids, test_ids, stratified = split_base_tasks(episode_rows, config.seed, config.test_size)
    train_rows_all, test_rows_all = split_rows_by_base_task(episode_rows, train_ids, test_ids)
    train_base_task_count = len(train_ids)
    test_base_task_count = len(test_ids)

    probe_rows: list[dict[str, Any]] = []
    for layer, representation in sorted(
        {(int(row["layer"]), str(row["representation"])) for row in rows}
    ):
        subset_rows = [
            row
            for row in rows
            if int(row["layer"]) == layer and str(row["representation"]) == representation
        ]
        train_rows, test_rows = split_rows_by_base_task(subset_rows, train_ids, test_ids)
        train_features, train_labels = _build_matrix(train_rows)
        test_features, test_labels = _build_matrix(test_rows)
        y_pred, y_prob = _fit_logistic_probe(
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            seed=config.seed,
        )
        probe_rows.append(
            _metrics_row(
                kind="hidden_state_probe",
                feature_set="hidden_state",
                layer=layer,
                representation=representation,
                train_rows=train_rows,
                test_rows=test_rows,
                y_true=test_labels,
                y_pred=y_pred,
                y_prob=y_prob,
            )
        )

    prompt_y_true = np.asarray([int(row["binary_label"]) for row in test_rows_all], dtype=np.int64)
    prompt_length_pred, prompt_length_prob = fit_prompt_length_baseline(
        train_rows_all,
        test_rows_all,
        config.seed,
    )
    prompt_length_metrics = _metrics_row(
        kind="baseline",
        feature_set="prompt_length",
        layer=None,
        representation=None,
        train_rows=train_rows_all,
        test_rows=test_rows_all,
        y_true=prompt_y_true,
        y_pred=prompt_length_pred,
        y_prob=prompt_length_prob,
    )

    tfidf_pred, tfidf_prob = fit_tfidf_baseline(train_rows_all, test_rows_all, config.seed)
    tfidf_metrics = _metrics_row(
        kind="baseline",
        feature_set="tfidf_prompt",
        layer=None,
        representation=None,
        train_rows=train_rows_all,
        test_rows=test_rows_all,
        y_true=prompt_y_true,
        y_pred=tfidf_pred,
        y_prob=tfidf_prob,
    )

    write_jsonl(config.metrics_path, probe_rows)
    _write_summary(
        summary_path=config.summary_path,
        hidden_state_rows=rows,
        episode_rows=episode_rows,
        probe_rows=probe_rows,
        prompt_length_metrics=prompt_length_metrics,
        tfidf_metrics=tfidf_metrics,
        train_base_task_count=train_base_task_count,
        test_base_task_count=test_base_task_count,
        stratified=stratified,
    )
    _print_ranked_summary(probe_rows, prompt_length_metrics, tfidf_metrics)
    return config.metrics_path


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the probe trainer."""

    defaults = default_probe_training_config()
    parser = argparse.ArgumentParser(
        description="Train linear probes on frozen PressureTrace reasoning hidden states.",
    )
    parser.add_argument("--input-path", type=Path, default=defaults.input_path)
    parser.add_argument("--metrics-path", type=Path, default=defaults.metrics_path)
    parser.add_argument("--summary-path", type=Path, default=defaults.summary_path)
    parser.add_argument("--seed", type=int, default=REASONING_PROBE_RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=REASONING_PROBE_TEST_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the reasoning probe trainer."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = ProbeTrainingConfig(
        input_path=args.input_path,
        metrics_path=args.metrics_path,
        summary_path=args.summary_path,
        seed=args.seed,
        test_size=args.test_size,
    )
    return train_reasoning_probes(config)


if __name__ == "__main__":
    main()
