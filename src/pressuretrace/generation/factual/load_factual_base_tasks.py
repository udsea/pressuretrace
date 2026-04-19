"""Load TriviaQA tasks as PressureTrace base tasks."""

from __future__ import annotations

import random
import re
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


def detect_answer_type(answer: str) -> str:
    """Rough type detection for shortcut generation.

    Returns one of: 'year', 'number', 'proper_noun', 'phrase'.
    """

    stripped = answer.strip()
    if re.match(r"^\d{4}$", stripped):
        return "year"
    if re.match(r"^\d+$", stripped):
        return "number"
    if len(stripped.split()) == 1 and stripped and stripped[0].isupper():
        return "proper_noun"
    return "phrase"


def derive_factual_shortcut_answer(
    task: ReasoningTask,
    all_tasks: list[ReasoningTask],
    rng: random.Random,
) -> str | None:
    """Sample a wrong answer of the same type as the gold answer.

    Same-type sampling produces plausible distractors (year -> different year,
    proper_noun -> different proper noun) rather than obviously-wrong-shaped
    strings. Falls back to a year perturbation if the gold is a year and no
    other year appears in the pool. Returns None if no reasonable shortcut
    candidate is available.
    """

    gold = task.reference_answer.strip()
    gold_type = detect_answer_type(gold)
    gold_norm = gold.lower()
    aliases = [
        a.strip().lower()
        for a in task.metadata.get("all_aliases", [])
    ]

    same_type_candidates = [
        t.reference_answer.strip()
        for t in all_tasks
        if t.task_id != task.task_id
        and detect_answer_type(t.reference_answer.strip()) == gold_type
        and t.reference_answer.strip().lower() != gold_norm
        and t.reference_answer.strip().lower() not in aliases
        and 1 <= len(t.reference_answer.strip()) <= 50
    ]

    if same_type_candidates:
        return rng.choice(same_type_candidates)

    if gold_type == "year":
        try:
            year = int(gold)
        except ValueError:
            return None
        offset = rng.choice([-3, -2, -1, 1, 2, 3])
        return str(year + offset)

    return None
