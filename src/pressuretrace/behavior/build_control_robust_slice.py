"""Freeze model-specific control-robust reasoning slices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pressuretrace.behavior.reasoning_runtime import slugify_model_name
from pressuretrace.paths import splits_dir
from pressuretrace.utils.io import read_jsonl, write_jsonl


def _require_uniform_string(rows: list[dict[str, Any]], field_name: str) -> str:
    """Require a single stable string value for a field across all rows."""

    values = {str(row.get(field_name, "")) for row in rows}
    if len(values) != 1:
        observed = ", ".join(sorted(values))
        raise ValueError(f"Expected one value for '{field_name}', observed: {observed}.")
    return next(iter(values))


def _default_control_robust_slice_path(
    model_name: str,
    thinking_mode: str,
) -> Path:
    """Build the default split path for a frozen control-robust slice."""

    model_slug = slugify_model_name(model_name)
    return (
        splits_dir() / f"reasoning_control_robust_slice_{model_slug}_{thinking_mode}.jsonl"
    )


def build_control_robust_slice(
    control_results_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Freeze the latent tasks whose control rows are robust on the target model."""

    rows = read_jsonl(control_results_path)
    if not rows:
        raise ValueError("Control-only results file is empty.")

    control_rows = [row for row in rows if str(row.get("pressure_type")) == "control"]
    if not control_rows:
        raise ValueError("No control rows found in the provided results file.")

    model_name = _require_uniform_string(control_rows, "model_name")
    thinking_mode = _require_uniform_string(control_rows, "thinking_mode")

    retained_by_base_task: dict[str, dict[str, Any]] = {}
    for row in control_rows:
        metadata = row.get("metadata", {})
        base_task_id = metadata.get("base_task_id")
        if base_task_id is None:
            raise ValueError("Control result row is missing metadata.base_task_id.")
        base_task_id_str = str(base_task_id)
        if base_task_id_str in retained_by_base_task:
            raise ValueError(f"Duplicate control row for base task '{base_task_id_str}'.")
        if str(row.get("route_label")) != "robust_correct":
            continue
        retained_by_base_task[base_task_id_str] = {
            "base_task_id": base_task_id_str,
            "control_task_id": str(row.get("task_id", "")),
            "source_dataset": str(row.get("source_dataset", "")),
            "source_id": str(row.get("source_id", "")),
            "split": str(metadata.get("split", "")),
            "prompt_family": str(metadata.get("prompt_family", "")),
            "transformation_version": str(metadata.get("transformation_version", "")),
            "model_name": model_name,
            "thinking_mode": thinking_mode,
            "control_route_label": "robust_correct",
        }

    destination = output_path or _default_control_robust_slice_path(
        model_name=model_name,
        thinking_mode=thinking_mode,
    )
    retained_rows = [retained_by_base_task[key] for key in sorted(retained_by_base_task)]
    return write_jsonl(destination, retained_rows)
