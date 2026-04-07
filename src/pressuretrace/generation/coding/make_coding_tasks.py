"""Manifest builders for coding tasks."""

from __future__ import annotations

from pathlib import Path

from pressuretrace.generation.coding.humaneval_loader import load_humaneval_tasks
from pressuretrace.generation.coding.mbpp_loader import load_mbpp_tasks
from pressuretrace.paths import manifests_dir
from pressuretrace.types import CodingTask
from pressuretrace.utils.io import write_jsonl


def build_coding_manifest(
    split: str = "test",
    limit_per_dataset: int | None = None,
    include_humaneval: bool = True,
    include_mbpp: bool = True,
    output_path: Path | None = None,
) -> Path:
    """Build and write a combined coding-task manifest."""

    tasks: list[CodingTask] = []
    if include_humaneval:
        tasks.extend(load_humaneval_tasks(split=split, limit=limit_per_dataset))
    if include_mbpp:
        tasks.extend(load_mbpp_tasks(split=split, limit=limit_per_dataset))

    destination = output_path or manifests_dir() / f"coding_{split}.jsonl"
    return write_jsonl(destination, tasks)
