"""Load TriviaQA tasks as PressureTrace base tasks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from pressuretrace.types import ReasoningTask


def load_triviaqa_tasks(
    split: str = "validation",
    limit: int = 300,
    seed: int = 42,
) -> list[ReasoningTask]:
    """Load TriviaQA tasks and convert to PressureTrace ReasoningTask format.

    Requires: pip install datasets
    Uses rc.nocontext subset (question only, no passage).
    """
    from datasets import load_dataset

    dataset = load_dataset("trivia_qa", "rc.nocontext", split=split)
    dataset = dataset.shuffle(seed=seed)

    tasks = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        aliases = item["answer"].get("normalized_aliases", [])
        gold_answer = aliases[0] if aliases else item["answer"]["value"]
        task = ReasoningTask(
            task_id=f"triviaqa_factual_v1_{i:06d}_base",
            family="reasoning",
            source_dataset="triviaqa",
            source_id=str(item.get("question_id", i)),
            prompt=item["question"],
            reference_solution=gold_answer,
            reference_answer=gold_answer,
            metadata={
                "split": split,
                "original_index": i,
                "all_aliases": aliases[:5],
            },
        )
        tasks.append(task)
    return tasks


def derive_factual_shortcut_answer(
    task: ReasoningTask,
    all_tasks: list[ReasoningTask],
    rng: random.Random,
) -> str | None:
    """Sample a wrong answer from a different task in the pool.

    This produces wrong answers that are the right type (names, years, places)
    but wrong for this specific question.
    """
    gold = task.reference_answer.strip().lower()
    gold_aliases = [
        a.strip().lower()
        for a in task.metadata.get("all_aliases", [])
    ]

    candidates = [
        t.reference_answer.strip()
        for t in all_tasks
        if t.task_id != task.task_id
        and t.reference_answer.strip().lower() != gold
        and t.reference_answer.strip().lower() not in gold_aliases
        and 1 <= len(t.reference_answer.strip()) <= 60
    ]
    if not candidates:
        return None
    return rng.choice(candidates)
