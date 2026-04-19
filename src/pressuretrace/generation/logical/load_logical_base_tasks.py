"""Load BIG-Bench Hard logical deduction tasks as PressureTrace base tasks."""

from __future__ import annotations

import random

from pressuretrace.types import ReasoningTask

LOGICAL_ANSWER_OPTIONS: tuple[str, ...] = ("(A)", "(B)", "(C)", "(D)", "(E)")


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
        if gold_answer not in LOGICAL_ANSWER_OPTIONS:
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
                "answer_options": list(LOGICAL_ANSWER_OPTIONS),
            },
        )
        tasks.append(task)
    return tasks


def derive_logical_shortcut_answer(
    task: ReasoningTask,
    rng: random.Random | None = None,
) -> str | None:
    """Pick a wrong multiple-choice option as the shortcut.

    Excludes both the gold answer AND the adjacent option so the shortcut is
    not the most common distractor, which otherwise inflates the control
    shortcut rate (models default to the adjacent option when uncertain).
    """
    r = rng or random.Random(42)

    gold = task.reference_answer.strip()
    options = list(LOGICAL_ANSWER_OPTIONS)
    if gold not in options:
        return None

    gold_idx = options.index(gold)
    adjacent_idx = (gold_idx + 1) % len(options)
    excluded = {gold_idx, adjacent_idx}
    remaining = [opt for i, opt in enumerate(options) if i not in excluded]

    if remaining:
        return r.choice(remaining)
    return options[adjacent_idx]
