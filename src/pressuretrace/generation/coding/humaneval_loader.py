"""HumanEval-style loader utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.ipc as ipc
from datasets import load_dataset

from pressuretrace.config import BASE_DATASETS
from pressuretrace.types import CodingTask


def _cached_humaneval_arrow_path(split: str) -> Path | None:
    """Return the cached HumanEval Arrow path when available."""

    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / "openai___openai_humaneval"
        / "openai_humaneval"
        / "0.0.0"
    )
    if not cache_root.exists():
        return None
    matches = sorted(cache_root.glob(f"*/openai_humaneval-{split}.arrow"))
    if not matches:
        return None
    return matches[-1]


def _load_cached_humaneval_rows(split: str) -> list[dict[str, Any]] | None:
    """Load cached HumanEval rows directly from Arrow when possible."""

    arrow_path = _cached_humaneval_arrow_path(split)
    if arrow_path is None:
        return None
    with ipc.open_stream(arrow_path.open("rb")) as reader:
        return [dict(row) for row in reader.read_all().to_pylist()]


def load_humaneval_tasks(split: str = "test", limit: int | None = None) -> list[CodingTask]:
    """Load HumanEval-style coding tasks from Hugging Face datasets."""

    dataset_ref = BASE_DATASETS.humaneval
    cached_rows = _load_cached_humaneval_rows(split)
    if cached_rows is None:
        dataset = load_dataset(dataset_ref.path, split=split)
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        rows = [dict(example) for example in dataset]
    else:
        rows = cached_rows[: limit or None]

    tasks: list[CodingTask] = []
    for index, example in enumerate(rows):
        tasks.append(
            CodingTask(
                task_id=example.get("task_id", f"humaneval-{split}-{index}"),
                source_dataset=dataset_ref.path,
                source_split=split,
                prompt=example["prompt"],
                entry_point=example.get("entry_point"),
                canonical_solution=example.get("canonical_solution"),
                test_code=example.get("test"),
                metadata={
                    "hf_dataset_path": dataset_ref.path,
                    "original_index": index,
                    "prompt_examples_source": "docstring_examples",
                },
            )
        )
    return tasks
