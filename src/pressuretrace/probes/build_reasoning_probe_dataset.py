"""Normalize extracted reasoning hidden states into a compact probe dataset."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pressuretrace.config import (
    reasoning_probe_dataset_path,
    reasoning_probe_hidden_states_path,
)
from pressuretrace.utils.io import append_jsonl, prepare_results_file, read_jsonl


def _coerce_hidden_state(hidden_state: Any) -> list[float]:
    """Validate and normalize a hidden-state vector."""

    if not isinstance(hidden_state, list) or not hidden_state:
        raise ValueError("hidden_state must be a non-empty list.")
    vector = [float(value) for value in hidden_state]
    if not vector:
        raise ValueError("hidden_state must be a non-empty numeric vector.")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("hidden_state must contain only finite numeric values.")
    return vector


def build_reasoning_probe_dataset(
    input_path: Path,
    output_path: Path,
) -> Path:
    """Compact extracted hidden-state rows into a training-friendly JSONL file."""

    rows = [dict(row) for row in read_jsonl(input_path)]
    output_file = prepare_results_file(output_path)

    layer_counts: Counter[int] = Counter()
    representation_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    written_rows = 0

    for row in rows:
        binary_label = int(row["binary_label"])
        if binary_label not in {0, 1}:
            raise ValueError(f"binary_label must be 0 or 1, observed {binary_label}.")

        prompt = str(row.get("prompt", ""))
        layer = int(row["layer"])
        representation = str(row["representation"])
        compact_row = {
            "task_id": str(row["task_id"]),
            "base_task_id": str(row["base_task_id"]),
            "pressure_type": str(row["pressure_type"]),
            "route_label": str(row["route_label"]),
            "binary_label": binary_label,
            "layer": layer,
            "representation": representation,
            "hidden_state": _coerce_hidden_state(row["hidden_state"]),
            "prompt": prompt,
        }
        append_jsonl(output_file, compact_row)
        layer_counts[layer] += 1
        representation_counts[representation] += 1
        label_counts[binary_label] += 1
        written_rows += 1

    print(f"Total rows: {written_rows}")
    print(
        "Per-layer counts: "
        + ", ".join(f"{layer}={layer_counts[layer]}" for layer in sorted(layer_counts))
    )
    print(
        "Per-representation counts: "
        + ", ".join(
            f"{representation}={representation_counts[representation]}"
            for representation in sorted(representation_counts)
        )
    )
    print(f"Per-label counts: 0={label_counts.get(0, 0)}, 1={label_counts.get(1, 0)}")

    return output_file


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the dataset builder."""

    parser = argparse.ArgumentParser(
        description="Normalize extracted reasoning hidden states into a compact probe dataset.",
    )
    parser.add_argument("--input-path", type=Path, default=reasoning_probe_hidden_states_path())
    parser.add_argument("--output-path", type=Path, default=reasoning_probe_dataset_path())
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the reasoning probe dataset builder."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return build_reasoning_probe_dataset(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
