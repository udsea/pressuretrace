from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.behavior.build_control_robust_slice import build_control_robust_slice
from pressuretrace.utils.io import read_jsonl, write_jsonl


class BuildControlRobustSliceTestCase(unittest.TestCase):
    def test_build_control_robust_slice_keeps_only_robust_control_tasks(self) -> None:
        rows = [
            {
                "task_id": "gsm8k_reasoning_v2_000001_control",
                "pressure_type": "control",
                "route_label": "robust_correct",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "source_dataset": "gsm8k",
                "source_id": "test_0",
                "metadata": {
                    "base_task_id": "gsm8k_reasoning_v2_000001",
                    "split": "test",
                    "prompt_family": "reasoning_conflict_v2",
                    "transformation_version": "v2",
                },
            },
            {
                "task_id": "gsm8k_reasoning_v2_000002_control",
                "pressure_type": "control",
                "route_label": "shortcut_followed",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "source_dataset": "gsm8k",
                "source_id": "test_1",
                "metadata": {
                    "base_task_id": "gsm8k_reasoning_v2_000002",
                    "split": "test",
                    "prompt_family": "reasoning_conflict_v2",
                    "transformation_version": "v2",
                },
            },
            {
                "task_id": "gsm8k_reasoning_v2_000001_teacher_anchor",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_followed",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "source_dataset": "gsm8k",
                "source_id": "test_0",
                "metadata": {
                    "base_task_id": "gsm8k_reasoning_v2_000001",
                    "split": "test",
                    "prompt_family": "reasoning_conflict_v2",
                    "transformation_version": "v2",
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "control_results.jsonl"
            output_path = Path(tempdir) / "control_robust_slice.jsonl"
            write_jsonl(input_path, rows)

            slice_path = build_control_robust_slice(
                control_results_path=input_path,
                output_path=output_path,
            )
            retained_rows = read_jsonl(slice_path)

        self.assertEqual(len(retained_rows), 1)
        self.assertEqual(retained_rows[0]["base_task_id"], "gsm8k_reasoning_v2_000001")
        self.assertEqual(retained_rows[0]["model_name"], "Qwen/Qwen3-14B")
        self.assertEqual(retained_rows[0]["thinking_mode"], "off")
        self.assertEqual(retained_rows[0]["control_route_label"], "robust_correct")


if __name__ == "__main__":
    unittest.main()
