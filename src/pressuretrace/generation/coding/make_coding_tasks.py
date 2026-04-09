"""Manifest builders for the separate coding-family benchmark."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from pressuretrace.generation.coding.load_coding_base_tasks import (
    CODING_V1_ARCHETYPES,
    load_coding_base_tasks,
)
from pressuretrace.generation.coding.transform_coding_tasks import build_coding_episode_family
from pressuretrace.paths import manifests_dir
from pressuretrace.utils.io import read_jsonl, write_jsonl


def load_coding_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a previously written coding-family manifest."""

    return [dict(row) for row in read_jsonl(path)]


def build_coding_all_valid_transforms(
    *,
    limit: int | None = None,
    archetypes: tuple[str, ...] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Build the full transformed coding-family pool."""

    selected_archetypes = archetypes or CODING_V1_ARCHETYPES
    base_tasks = load_coding_base_tasks(limit=limit, archetypes=selected_archetypes)
    manifest_rows: list[dict[str, Any]] = []
    for base_task in base_tasks:
        manifest_rows.extend(build_coding_episode_family(base_task))

    if not manifest_rows:
        raise ValueError("No coding-family transforms were generated.")

    destination = output_path or manifests_dir() / "coding_all_valid_transforms.jsonl"
    archetype_counts = Counter(
        row["archetype"] for row in manifest_rows if row["pressure_type"] == "control"
    )

    print(f"Loaded coding base tasks: {len(base_tasks)}")
    for archetype in sorted(archetype_counts):
        print(f"  {archetype}: {archetype_counts[archetype]}")
    print(f"Transformed coding rows: {len(manifest_rows)}")
    return write_jsonl(destination, manifest_rows)


def build_coding_manifest(
    *,
    limit: int | None = None,
    archetypes: tuple[str, ...] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Backward-compatible wrapper for building the coding-family transform pool."""

    return build_coding_all_valid_transforms(
        limit=limit,
        archetypes=archetypes,
        output_path=output_path,
    )
