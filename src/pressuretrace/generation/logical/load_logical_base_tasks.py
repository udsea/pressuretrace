"""Load BIG-Bench Hard logical deduction tasks as PressureTrace base tasks."""

from __future__ import annotations

import random

from pressuretrace.types import ReasoningTask

BBH_THREE_OBJECTS_OPTIONS: tuple[str, ...] = ("(A)", "(B)", "(C)")


def load_bbh_logical_tasks(
    limit: int = 200,
) -> list[ReasoningTask]:
    """Load BBH logical_deduction_three_objects tasks.

    Requires: pip install datasets
    Each task is a logical deduction problem with a multiple-choice answer.
    """
    from datasets import load_dataset

    dataset = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")

    tasks = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        gold_answer = item["target"].strip()
        if gold_answer not in BBH_THREE_OBJECTS_OPTIONS:
            continue
        task = ReasoningTask(
            task_id=f"bbh_logical_v1_{i:06d}_base",
            family="reasoning",
            source_dataset="bbh_logical_deduction",
            source_id=str(i),
            prompt=item["input"],
            reference_solution=gold_answer,
            reference_answer=gold_answer,
            metadata={
                "split": "test",
                "original_index": i,
                "answer_options": list(BBH_THREE_OBJECTS_OPTIONS),
            },
        )
        tasks.append(task)
    return tasks


def derive_logical_shortcut_answer(
    task: ReasoningTask,
    rng: random.Random | None = None,
) -> str | None:
    """Sample one of the two wrong options from the actual answer space.

    BBH logical_deduction_three_objects only uses (A)/(B)/(C). Picking from
    (D)/(E) produces an impossible shortcut the model correctly ignores,
    making the pressure look ineffective even when the model is susceptible.
    """
    r = rng or random.Random(42)

    gold = task.reference_answer.strip()
    valid_options = [opt for opt in BBH_THREE_OBJECTS_OPTIONS if opt != gold]
    if not valid_options:
        return None
    return r.choice(valid_options)
