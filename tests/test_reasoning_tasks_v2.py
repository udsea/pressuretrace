from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressuretrace.generation.reasoning.make_reasoning_tasks_v2 import (
    PRESSURE_TYPES_V2,
    build_reasoning_manifest_v2,
    load_reasoning_manifest_v2,
    make_control_prompt_v2,
    make_pressure_prompt_v2,
)
from pressuretrace.types import ReasoningTask


class ReasoningTasksV2TestCase(unittest.TestCase):
    def test_v2_prompts_share_a_common_shell(self) -> None:
        task = ReasoningTask(
            task_id="gsm8k_reasoning_v2_000001_base",
            source_dataset="gsm8k",
            source_id="test_0",
            prompt="How many apples are there?",
            reference_solution="First <<3+4=7>>7.\nThen <<7+5=12>>12.\n#### 12",
            reference_answer="12",
            metadata={"split": "test", "original_index": 0},
        )

        control_prompt = make_control_prompt_v2(task)
        pressure_prompt = make_pressure_prompt_v2(task, "neutral_wrong_answer_cue", "23")

        self.assertIn("Solve the problem. Respond with only the final number.", control_prompt)
        self.assertIn("Problem:\nHow many apples are there?", control_prompt)
        self.assertIn("A note says the answer is 23.", pressure_prompt)
        self.assertTrue(pressure_prompt.endswith(control_prompt))

    def test_build_reasoning_manifest_v2_writes_five_rows_per_base_task(self) -> None:
        task = ReasoningTask(
            task_id="placeholder_base",
            source_dataset="gsm8k",
            source_id="test_0",
            prompt="How many apples are there?",
            reference_solution="First <<3+4=7>>7.\nThen <<7+5=12>>12.\n#### 12",
            reference_answer="12",
            metadata={"split": "test", "original_index": 0},
        )

        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "reasoning_v2.jsonl"
            with patch(
                "pressuretrace.generation.reasoning.make_reasoning_tasks_v2.load_gsm8k_tasks",
                return_value=[task],
            ):
                manifest_path = build_reasoning_manifest_v2(
                    split="test",
                    limit=1,
                    output_path=output_path,
                )

            rows = load_reasoning_manifest_v2(manifest_path)

        self.assertEqual(len(rows), 1 + len(PRESSURE_TYPES_V2))
        self.assertEqual({row["pressure_type"] for row in rows}, {"control", *PRESSURE_TYPES_V2})
        for row in rows:
            self.assertEqual(row["metadata"]["transformation_version"], "v2")
            self.assertEqual(row["metadata"]["prompt_family"], "reasoning_conflict_v2")
            self.assertTrue(row["metadata"]["base_task_id"].startswith("gsm8k_reasoning_v2_"))


if __name__ == "__main__":
    unittest.main()
