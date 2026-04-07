from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.patching.build_reasoning_patch_pairs import (
    build_reasoning_patch_pairs,
    load_reasoning_patch_pairs,
)
from pressuretrace.utils.io import read_jsonl, write_jsonl


class BuildReasoningPatchPairsTestCase(unittest.TestCase):
    def test_build_reasoning_patch_pairs_keeps_only_matched_shortcuts(self) -> None:
        control_rows = [
            {
                "base_task_id": "base_1",
                "control_task_id": "base_1_control",
                "source_dataset": "gsm8k",
                "source_id": "src_1",
                "split": "test",
                "prompt_family": "reasoning_conflict_v2",
                "transformation_version": "v2",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "robust_correct",
            },
            {
                "base_task_id": "base_2",
                "control_task_id": "base_2_control",
                "source_dataset": "gsm8k",
                "source_id": "src_2",
                "split": "test",
                "prompt_family": "reasoning_conflict_v2",
                "transformation_version": "v2",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "shortcut_followed",
            },
        ]
        result_rows = [
            {
                "task_id": "base_1_teacher_anchor",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_followed",
                "gold_answer": "3",
                "shortcut_answer": "1",
                "source_dataset": "gsm8k",
                "source_id": "src_1",
                "metadata": {"base_task_id": "base_1"},
            },
            {
                "task_id": "base_2_teacher_anchor",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_followed",
                "gold_answer": "4",
                "shortcut_answer": "2",
                "source_dataset": "gsm8k",
                "source_id": "src_2",
                "metadata": {"base_task_id": "base_2"},
            },
            {
                "task_id": "base_1_urgency",
                "pressure_type": "urgency",
                "route_label": "shortcut_followed",
                "gold_answer": "3",
                "shortcut_answer": "1",
                "source_dataset": "gsm8k",
                "source_id": "src_1",
                "metadata": {"base_task_id": "base_1"},
            },
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            control_path = Path(tempdir) / "control.jsonl"
            results_path = Path(tempdir) / "results.jsonl"
            output_path = Path(tempdir) / "pairs.jsonl"
            write_jsonl(control_path, control_rows)
            write_jsonl(results_path, result_rows)

            built_path = build_reasoning_patch_pairs(
                results_path=results_path,
                control_slice_path=control_path,
                output_path=output_path,
            )
            pairs = read_jsonl(built_path)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["base_task_id"], "base_1")
        self.assertEqual(pairs[0]["pressure_type"], "teacher_anchor")
        self.assertEqual(pairs[0]["control_task_id"], "base_1_control")
        self.assertEqual(pairs[0]["pressure_task_id"], "base_1_teacher_anchor")
        self.assertEqual(pairs[0]["gold_answer"], "3")
        self.assertEqual(pairs[0]["shortcut_answer"], "1")

    def test_load_reasoning_patch_pairs_reconstructs_prompts(self) -> None:
        patch_rows = [
            {
                "base_task_id": "base_1",
                "pressure_type": "teacher_anchor",
                "control_task_id": "base_1_control",
                "pressure_task_id": "base_1_teacher_anchor",
                "gold_answer": "3",
                "shortcut_answer": "1",
                "metadata": {"pairing_strategy": "control_robust_vs_pressure_shortcut"},
            }
        ]
        manifest_rows = [
            {"task_id": "base_1_control", "prompt": "Solve the problem."},
            {"task_id": "base_1_teacher_anchor", "prompt": "A teacher's note says 1."},
        ]
        result_rows = [
            {"task_id": "base_1_control", "prompt": "Solve the problem."},
            {"task_id": "base_1_teacher_anchor", "prompt": "A teacher's note says 1."},
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            patch_path = Path(tempdir) / "pairs.jsonl"
            manifest_path = Path(tempdir) / "manifest.jsonl"
            results_path = Path(tempdir) / "results.jsonl"
            write_jsonl(patch_path, patch_rows)
            write_jsonl(manifest_path, manifest_rows)
            write_jsonl(results_path, result_rows)

            pairs = load_reasoning_patch_pairs(
                patch_pairs_path=patch_path,
                manifest_path=manifest_path,
                results_path=results_path,
            )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].control_prompt, "Solve the problem.")
        self.assertEqual(pairs[0].pressure_prompt, "A teacher's note says 1.")
        self.assertEqual(pairs[0].pressure_type, "teacher_anchor")


if __name__ == "__main__":
    unittest.main()
