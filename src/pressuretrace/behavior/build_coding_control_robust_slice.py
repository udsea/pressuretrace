"""Freeze model-specific control-robust coding-family slices."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pressuretrace.paths import splits_dir
from pressuretrace.utils.io import read_jsonl, write_jsonl


def _slugify_model_name(model_name: str) -> str:
    """Convert a model identifier into a filename-safe slug."""

    return re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()


def _require_uniform_string(rows: list[dict[str, Any]], field_name: str) -> str:
    """Require a single stable string value for a field across all rows."""

    values = {str(row.get(field_name, "")) for row in rows}
    if len(values) != 1:
        observed = ", ".join(sorted(values))
        raise ValueError(f"Expected one value for '{field_name}', observed: {observed}.")
    return next(iter(values))


def _default_slice_path(
    *,
    model_name: str,
    thinking_mode: str,
) -> Path:
    """Build the default split path for a frozen coding-family control-robust slice."""

    model_slug = _slugify_model_name(model_name)
    return splits_dir() / f"coding_control_robust_slice_{model_slug}_{thinking_mode}.jsonl"


def build_coding_control_robust_slice(
    *,
    control_results_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Freeze the base tasks whose control rows are robust on the target model."""

    rows = read_jsonl(control_results_path)
    if not rows:
        raise ValueError("Control-only coding results file is empty.")

    control_rows = [row for row in rows if str(row.get("pressure_type")) == "control"]
    if not control_rows:
        raise ValueError("No control rows found in the provided coding results file.")

    model_name = _require_uniform_string(control_rows, "model_name")
    thinking_mode = _require_uniform_string(control_rows, "thinking_mode")

    retained_by_base_task: dict[str, dict[str, Any]] = {}
    for row in control_rows:
        base_task_id = str(row.get("base_task_id", "")).strip()
        if not base_task_id:
            raise ValueError("Control result row is missing base_task_id.")
        if base_task_id in retained_by_base_task:
            raise ValueError(f"Duplicate control row for base task '{base_task_id}'.")
        if str(row.get("route_label")) != "robust_success":
            continue
        retained_by_base_task[base_task_id] = {
            "base_task_id": base_task_id,
            "control_task_id": str(row.get("task_id", "")),
            "source_family": str(row.get("source_family", "")),
            "source_task_name": str(row.get("source_task_name", "")),
            "archetype": str(row.get("archetype", "")),
            "model_name": model_name,
            "thinking_mode": thinking_mode,
            "control_route_label": "robust_success",
        }

    destination = output_path or _default_slice_path(
        model_name=model_name,
        thinking_mode=thinking_mode,
    )
    retained_rows = [retained_by_base_task[key] for key in sorted(retained_by_base_task)]
    return write_jsonl(destination, retained_rows)
