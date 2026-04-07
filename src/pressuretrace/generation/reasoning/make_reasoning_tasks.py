"""Build paired reasoning benchmark episodes from GSM8K."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pressuretrace.generation.reasoning.gsm8k_loader import load_gsm8k_tasks
from pressuretrace.generation.reasoning.reasoning_transform import (
    PRESSURE_TYPES,
    derive_shortcut_answer,
    make_control_variant,
    make_pressure_variant,
)
from pressuretrace.paths import manifests_dir
from pressuretrace.types import ReasoningTask
from pressuretrace.utils.io import read_jsonl, write_jsonl


def _build_episode_group(task: ReasoningTask) -> list[dict[str, Any]] | None:
    """Create control and pressure episodes for a base task."""

    shortcut_answer = derive_shortcut_answer(task)
    if shortcut_answer is None:
        return None

    episodes = [make_control_variant(task, shortcut_answer)]
    for pressure_type in PRESSURE_TYPES:
        episodes.append(
            make_pressure_variant(
                task=task,
                pressure_profile=pressure_type,
                shortcut_answer=shortcut_answer,
            )
        )
    return episodes


def load_reasoning_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a previously written reasoning manifest."""

    return [dict(row) for row in read_jsonl(path)]


def build_reasoning_manifest(
    split: str = "test",
    limit: int | None = None,
    output_path: Path | None = None,
) -> Path:
    """Build and write paired reasoning benchmark episodes."""

    base_tasks = load_gsm8k_tasks(split=split, limit=None)
    destination = output_path or manifests_dir() / "reasoning_v1.jsonl"

    manifest_rows: list[dict[str, Any]] = []
    total_base_tasks = len(base_tasks)
    tasks_with_shortcut_candidates = 0
    retained_base_tasks = 0
    for task_index, task in enumerate(base_tasks, start=1):
        task.task_id = f"gsm8k_reasoning_{task_index:06d}_base"
        episode_group = _build_episode_group(task)
        if episode_group is None:
            continue

        tasks_with_shortcut_candidates += 1
        if limit is not None and retained_base_tasks >= limit:
            continue

        manifest_rows.extend(episode_group)
        retained_base_tasks += 1

    if not manifest_rows:
        raise ValueError("No GSM8K examples produced valid shortcut-derived reasoning episodes.")

    retention_rate = tasks_with_shortcut_candidates / total_base_tasks
    print(f"Loaded base tasks: {total_base_tasks}")
    print(f"Tasks with shortcut candidates: {tasks_with_shortcut_candidates}")
    print(f"Retained tasks written to manifest: {retained_base_tasks}")
    print(f"Retention rate: {retention_rate:.3f}")

    return write_jsonl(destination, manifest_rows)
