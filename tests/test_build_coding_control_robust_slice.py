from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.behavior.build_coding_control_robust_slice import (
    build_coding_control_robust_slice,
)
from pressuretrace.utils.io import read_jsonl, write_jsonl


class BuildCodingControlRobustSliceTestCase(unittest.TestCase):
    def test_slice_keeps_only_robust_control_base_tasks(self) -> None:
        rows = [
            {
                "task_id": "coding_v1_visible_test_hardcoding_001_control",
                "base_task_id": "coding_v1_visible_test_hardcoding_001",
                "pressure_type": "control",
                "route_label": "robust_success",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/mirror_words",
                "archetype": "visible_test_hardcoding",
            },
            {
                "task_id": "coding_v1_visible_test_hardcoding_002_control",
                "base_task_id": "coding_v1_visible_test_hardcoding_002",
                "pressure_type": "control",
                "route_label": "shortcut_success",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/rotate_left",
                "archetype": "visible_test_hardcoding",
            },
            {
                "task_id": "coding_v1_visible_test_hardcoding_001_teacher_anchor",
                "base_task_id": "coding_v1_visible_test_hardcoding_001",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_success",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/mirror_words",
                "archetype": "visible_test_hardcoding",
            },
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "control_results.jsonl"
            output_path = Path(tempdir) / "coding_control_robust_slice.jsonl"
            write_jsonl(input_path, rows)

            slice_path = build_coding_control_robust_slice(
                control_results_path=input_path,
                output_path=output_path,
            )
            retained_rows = read_jsonl(slice_path)

        self.assertEqual(len(retained_rows), 1)
        self.assertEqual(retained_rows[0]["base_task_id"], "coding_v1_visible_test_hardcoding_001")
        self.assertEqual(retained_rows[0]["control_route_label"], "robust_success")

    def test_slice_raises_when_no_control_row_is_robust(self) -> None:
        rows = [
            {
                "task_id": "coding_v1_visible_test_hardcoding_001_control",
                "base_task_id": "coding_v1_visible_test_hardcoding_001",
                "pressure_type": "control",
                "route_label": "wrong_nonshortcut",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
            }
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "control_results.jsonl"
            write_jsonl(input_path, rows)

            with self.assertRaisesRegex(ValueError, "No control rows achieved 'robust_success'"):
                build_coding_control_robust_slice(control_results_path=input_path)


if __name__ == "__main__":
    unittest.main()
