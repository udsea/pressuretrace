"""GSM8K loading and normalization utilities."""

from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset

from pressuretrace.config import BASE_DATASETS
from pressuretrace.types import ReasoningTask

FINAL_ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*)")


def extract_final_answer(answer_text: str) -> str | None:
    """Extract the final scalar answer from GSM8K solution text."""

    match = FINAL_ANSWER_PATTERN.search(answer_text)
    if not match:
        return None
    return match.group(1).replace(",", "")


def load_gsm8k_split(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load a raw GSM8K split with stable source identifiers."""

    dataset_ref = BASE_DATASETS.gsm8k
    dataset = load_dataset(dataset_ref.path, dataset_ref.config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "source_dataset": "gsm8k",
                "source_id": f"{split}_{index}",
                "split": split,
                "question": str(example["question"]),
                "answer": str(example["answer"]),
                "original_index": index,
            }
        )
    return rows


def load_gsm8k_tasks(split: str = "test", limit: int | None = None) -> list[ReasoningTask]:
    """Load GSM8K tasks with parsed gold answers."""

    tasks: list[ReasoningTask] = []
    for row in load_gsm8k_split(split=split, limit=limit):
        gold_answer = extract_final_answer(row["answer"])
        if gold_answer is None:
            message = f"Could not parse final GSM8K answer for source_id={row['source_id']}."
            raise ValueError(message)

        tasks.append(
            ReasoningTask(
                task_id=f"gsm8k_{row['source_id']}_base",
                source_dataset=row["source_dataset"],
                source_id=row["source_id"],
                prompt=row["question"],
                reference_solution=row["answer"],
                reference_answer=gold_answer,
                metadata={
                    "split": row["split"],
                    "original_index": row["original_index"],
                },
            )
        )
    return tasks
