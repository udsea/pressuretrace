"""Freeze model-specific control-robust factual-family slices.

Mirrors build_control_robust_slice for the factual family: reads the output
of run_factual_benchmark_v1 and writes a JSONL of base tasks where the
target model solved the control prompt correctly. Pressure conditions can
then be evaluated on this filtered slice for interpretable effects.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pressuretrace.behavior.reasoning_runtime import slugify_model_name
from pressuretrace.paths import splits_dir
from pressuretrace.utils.io import read_jsonl, write_jsonl


def _require_uniform_string(rows: list[dict[str, Any]], field_name: str) -> str:
    values = {str(row.get(field_name, "")) for row in rows}
    if len(values) != 1:
        observed = ", ".join(sorted(values))
        raise ValueError(f"Expected one value for '{field_name}', observed: {observed}.")
    return next(iter(values))


def _default_slice_path(model_name: str, thinking_mode: str) -> Path:
    model_slug = slugify_model_name(model_name)
    return (
        splits_dir() / f"factual_control_robust_slice_{model_slug}_{thinking_mode}.jsonl"
    )


def build_factual_control_robust_slice(
    control_results_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Write a slice listing base tasks where the model was robust on control."""

    rows = read_jsonl(control_results_path)
    if not rows:
        raise ValueError("Factual results file is empty.")

    control_rows = [row for row in rows if str(row.get("pressure_type")) == "control"]
    if not control_rows:
        raise ValueError("No control rows found in the provided results file.")

    model_name = _require_uniform_string(control_rows, "model_name")
    thinking_mode = _require_uniform_string(control_rows, "thinking_mode")

    retained_by_base_task: dict[str, dict[str, Any]] = {}
    for row in control_rows:
        metadata = row.get("metadata", {})
        base_task_id = metadata.get("base_task_id") or row.get("base_task_id")
        if base_task_id is None:
            raise ValueError("Control result row is missing base_task_id.")
        base_task_id_str = str(base_task_id)
        if base_task_id_str in retained_by_base_task:
            raise ValueError(f"Duplicate control row for base task '{base_task_id_str}'.")
        if str(row.get("route_label")) != "robust_correct":
            continue
        retained_by_base_task[base_task_id_str] = {
            "base_task_id": base_task_id_str,
            "control_task_id": str(row.get("task_id", "")),
            "source_dataset": str(row.get("source_dataset", "")),
            "split": str(metadata.get("split", "")),
            "prompt_family": str(metadata.get("prompt_family", "")),
            "transformation_version": str(metadata.get("transformation_version", "")),
            "model_name": model_name,
            "thinking_mode": thinking_mode,
            "control_route_label": "robust_correct",
        }

    destination = output_path or _default_slice_path(
        model_name=model_name,
        thinking_mode=thinking_mode,
    )
    retained_rows = [retained_by_base_task[key] for key in sorted(retained_by_base_task)]
    written = write_jsonl(destination, retained_rows)
    print(
        f"Retained {len(retained_rows)} / {len(control_rows)} control-robust base tasks "
        f"-> {written}"
    )
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Freeze a control-robust slice from factual benchmark results.",
    )
    parser.add_argument("--control-results-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args(argv)
    build_factual_control_robust_slice(
        control_results_path=args.control_results_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
