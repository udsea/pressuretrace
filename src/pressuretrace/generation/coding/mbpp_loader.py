"""MBPP-style loader utilities."""

from __future__ import annotations

from datasets import load_dataset

from pressuretrace.config import BASE_DATASETS
from pressuretrace.types import CodingTask


def load_mbpp_tasks(split: str = "test", limit: int | None = None) -> list[CodingTask]:
    """Load MBPP-style coding tasks from Hugging Face datasets."""

    dataset_ref = BASE_DATASETS.mbpp
    dataset = load_dataset(dataset_ref.path, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    tasks: list[CodingTask] = []
    for index, example in enumerate(dataset):
        test_field = example.get("test_list") or example.get("test")
        if isinstance(test_field, list):
            test_code = "\n".join(test_field)
        else:
            test_code = test_field

        tasks.append(
            CodingTask(
                task_id=str(example.get("task_id", f"mbpp-{split}-{index}")),
                source_dataset=dataset_ref.path,
                source_split=split,
                prompt=example.get("text", example.get("prompt", "")),
                canonical_solution=example.get("code"),
                test_code=test_code,
                metadata={
                    "hf_dataset_path": dataset_ref.path,
                    "original_index": index,
                },
            )
        )
    return tasks
