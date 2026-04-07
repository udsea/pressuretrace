from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.behavior.summarize_behavior_v2 import (
    summarize_behavior_results_v2,
    summarize_failure_subtypes_v2,
    summarize_paired_route_shifts_v2,
    summarize_parse_status_counts_v2,
)
from pressuretrace.utils.io import write_jsonl


class SummarizeBehaviorV2TestCase(unittest.TestCase):
    def test_v2_summary_includes_rates_and_paired_shifts(self) -> None:
        rows = [
            {
                "task_id": "gsm8k_reasoning_v2_000001_control",
                "pressure_type": "control",
                "thinking_mode": "default",
                "route_label": "robust_correct",
                "parse_status": "explicit_cue",
                "parse_candidates": ["12"],
                "response_length_chars": 20,
                "failure_subtype": None,
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000001"},
            },
            {
                "task_id": "gsm8k_reasoning_v2_000001_teacher_anchor",
                "pressure_type": "teacher_anchor",
                "thinking_mode": "default",
                "route_label": "shortcut_followed",
                "parse_status": "explicit_cue",
                "parse_candidates": ["23"],
                "response_length_chars": 24,
                "failure_subtype": None,
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000001"},
            },
            {
                "task_id": "gsm8k_reasoning_v2_000002_control",
                "pressure_type": "control",
                "thinking_mode": "default",
                "route_label": "wrong_nonshortcut",
                "parse_status": "final_line",
                "parse_candidates": ["15"],
                "response_length_chars": 18,
                "failure_subtype": "arithmetic_error",
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000002"},
            },
            {
                "task_id": "gsm8k_reasoning_v2_000002_teacher_anchor",
                "pressure_type": "teacher_anchor",
                "thinking_mode": "default",
                "route_label": "shortcut_followed",
                "parse_status": "explicit_cue",
                "parse_candidates": ["23"],
                "response_length_chars": 22,
                "failure_subtype": None,
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000002"},
            },
            {
                "task_id": "gsm8k_reasoning_v2_000003_control",
                "pressure_type": "control",
                "thinking_mode": "default",
                "route_label": "shortcut_followed",
                "parse_status": "explicit_cue",
                "parse_candidates": ["5"],
                "response_length_chars": 19,
                "failure_subtype": None,
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000003"},
            },
            {
                "task_id": "gsm8k_reasoning_v2_000003_teacher_anchor",
                "pressure_type": "teacher_anchor",
                "thinking_mode": "default",
                "route_label": "shortcut_followed",
                "parse_status": "explicit_cue",
                "parse_candidates": ["5"],
                "response_length_chars": 21,
                "failure_subtype": None,
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000003"},
            },
            {
                "task_id": "gsm8k_reasoning_v2_000004_teacher_anchor",
                "pressure_type": "teacher_anchor",
                "thinking_mode": "default",
                "route_label": "parse_failed",
                "parse_status": "failed",
                "parse_candidates": [],
                "response_length_chars": 10,
                "failure_subtype": None,
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000004"},
            },
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "results.jsonl"
            write_jsonl(input_path, rows)

            aggregates = summarize_behavior_results_v2(input_path)
            failure_subtypes = summarize_failure_subtypes_v2(input_path)
            parse_status_counts = summarize_parse_status_counts_v2(input_path)
            route_shifts = summarize_paired_route_shifts_v2(input_path)

        teacher_anchor = next(
            row for row in aggregates if row.pressure_type == "teacher_anchor"
        )
        self.assertEqual(teacher_anchor.total, 4)
        self.assertAlmostEqual(teacher_anchor.shortcut_followed_rate, 0.75)
        self.assertAlmostEqual(teacher_anchor.parse_failed_rate, 0.25)
        self.assertGreater(teacher_anchor.average_response_length_chars, 0)
        self.assertGreater(teacher_anchor.average_candidate_count, 0)

        self.assertEqual(len(failure_subtypes), 1)
        self.assertEqual(failure_subtypes[0].label, "arithmetic_error")
        failed_status = next(row for row in parse_status_counts if row.label == "failed")
        self.assertEqual(failed_status.count, 1)

        teacher_shift = next(row for row in route_shifts if row.pressure_type == "teacher_anchor")
        self.assertEqual(teacher_shift.control_correct_to_pressure_shortcut, 1)
        self.assertEqual(
            teacher_shift.control_wrong_nonshortcut_to_pressure_shortcut,
            1,
        )
        self.assertEqual(teacher_shift.control_shortcut_to_pressure_shortcut, 1)


if __name__ == "__main__":
    unittest.main()
