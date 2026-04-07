"""HumanEval-style loader utilities."""

from __future__ import annotations

from datasets import load_dataset

from pressuretrace.config import BASE_DATASETS
from pressuretrace.types import CodingTask


def load_humaneval_tasks(split: str = "test", limit: int | None = None) -> list[CodingTask]:
    """Load HumanEval-style coding tasks from Hugging Face datasets."""

    dataset_ref = BASE_DATASETS.humaneval
    dataset = load_dataset(dataset_ref.path, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    tasks: list[CodingTask] = []
    for index, example in enumerate(dataset):
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
                },
            )
        )
    return tasks
