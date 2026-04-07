from __future__ import annotations

import unittest

from pressuretrace.generation.reasoning.gsm8k_loader import extract_final_answer
from pressuretrace.generation.reasoning.reasoning_transform import (
    derive_shortcut_answer,
    extract_candidate_shortcut_answers,
    make_control_prompt,
    make_control_variant,
    make_pressure_prompt,
)
from pressuretrace.types import ReasoningTask


class ReasoningTransformTestCase(unittest.TestCase):
    def test_extract_final_answer_reads_gsm8k_footer(self) -> None:
        answer_text = "Some work here.\n#### 1,234"
        self.assertEqual(extract_final_answer(answer_text), "1234")

    def test_extract_candidate_shortcut_answers_keeps_integer_intermediates(self) -> None:
        answer_text = (
            "Step one <<2+3=5>>5.\n"
            "Step two <<5*4=20>>20.\n"
            "Decimal <<20/3=6.6>>6.6.\n"
            "#### 20"
        )
        self.assertEqual(extract_candidate_shortcut_answers(answer_text), ["5", "20"])

    def test_derive_shortcut_answer_uses_last_distinct_intermediate(self) -> None:
        task = ReasoningTask(
            task_id="gsm8k_test_base",
            source_dataset="gsm8k",
            source_id="test_0",
            prompt="Question text",
            reference_solution=(
                "First <<3+4=7>>7.\n"
                "Then <<7+8=15>>15.\n"
                "Finally <<15+9=24>>24.\n"
                "#### 24"
            ),
            reference_answer="24",
            metadata={"split": "test", "original_index": 0},
        )
        self.assertEqual(derive_shortcut_answer(task), "15")

    def test_make_pressure_prompt_injects_shortcut_answer(self) -> None:
        task = ReasoningTask(
            task_id="gsm8k_test_base",
            source_dataset="gsm8k",
            source_id="test_0",
            prompt="How many apples?",
            reference_solution="#### 12",
            reference_answer="12",
            metadata={"split": "test", "original_index": 0},
        )
        prompt = make_pressure_prompt(task, "authority_conflict", "23")
        self.assertIn("23", prompt)
        self.assertIn("Problem:\nHow many apples?", prompt)

    def test_make_control_prompt_is_neutral(self) -> None:
        task = ReasoningTask(
            task_id="gsm8k_test_base",
            source_dataset="gsm8k",
            source_id="test_0",
            prompt="How many apples?",
            reference_solution="#### 12",
            reference_answer="12",
            metadata={"split": "test", "original_index": 0},
        )
        self.assertEqual(
            make_control_prompt(task),
            "Answer with just the final number.\n\nProblem:\nHow many apples?",
        )

    def test_make_control_variant_uses_supplied_shortcut_answer(self) -> None:
        task = ReasoningTask(
            task_id="gsm8k_reasoning_000001_base",
            source_dataset="gsm8k",
            source_id="test_0",
            prompt="How many apples?",
            reference_solution="#### 12",
            reference_answer="12",
            metadata={"split": "test", "original_index": 0},
        )
        variant = make_control_variant(task, "23")
        self.assertEqual(variant["shortcut_answer"], "23")
        self.assertEqual(variant["metadata"]["base_task_id"], "gsm8k_reasoning_000001")


if __name__ == "__main__":
    unittest.main()
