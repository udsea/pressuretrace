"""Materialize frozen coding-family manifests from retained base-task ids."""

from __future__ import annotations

from pathlib import Path

from pressuretrace.paths import manifests_dir
from pressuretrace.utils.io import read_jsonl, write_jsonl


def _default_paper_slice_manifest_path(slice_path: Path) -> Path:
    """Build the default manifest path for a frozen coding-family paper slice."""

    stem = slice_path.stem
    if stem.startswith("coding_control_robust_slice_"):
        filename = stem.replace(
            "coding_control_robust_slice_",
            "coding_paper_slice_",
            1,
        )
    else:
        filename = f"{stem}_paper_slice"
    return manifests_dir() / f"{filename}.jsonl"


def materialize_coding_paper_slice(
    *,
    manifest_path: Path,
    slice_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Filter a broad coding manifest down to the frozen retained base-task slice."""

    manifest_rows = read_jsonl(manifest_path)
    slice_rows = read_jsonl(slice_path)
    retained_base_task_ids = {
        str(row.get("base_task_id", "")) for row in slice_rows if row.get("base_task_id")
    }
    if not retained_base_task_ids and slice_rows:
        raise ValueError("Slice rows are present but none contain base_task_id.")

    filtered_rows = [
        row for row in manifest_rows if str(row.get("base_task_id", "")) in retained_base_task_ids
    ]
    destination = output_path or _default_paper_slice_manifest_path(slice_path)
    return write_jsonl(destination, filtered_rows)
