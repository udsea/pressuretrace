from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.patching.build_coding_patch_pairs import (
    build_coding_patch_pairs,
    load_coding_patch_pairs,
)
from pressuretrace.utils.io import read_jsonl, write_jsonl


class BuildCodingPatchPairsTestCase(unittest.TestCase):
    def test_build_coding_patch_pairs_keeps_only_matched_shortcuts(self) -> None:
        control_rows = [
            {
                "base_task_id": "base_1",
                "control_task_id": "base_1_control",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/prefix_sums",
                "archetype": "weak_checker_exploit",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "robust_success",
            },
            {
                "base_task_id": "base_2",
                "control_task_id": "base_2_control",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/compress_runs",
                "archetype": "visible_test_hardcoding",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "shortcut_success",
            },
        ]
        result_rows = [
            {
                "task_id": "base_1_control",
                "base_task_id": "base_1",
                "pressure_type": "control",
                "route_label": "robust_success",
                "entry_point": "prefix_sums",
                "extracted_code": "def prefix_sums(nums):\n    return nums\n",
            },
            {
                "task_id": "base_1_teacher_anchor",
                "base_task_id": "base_1",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_success",
                "entry_point": "prefix_sums",
                "extracted_code": "def prefix_sums(nums):\n    return [0] * len(nums)\n",
            },
            {
                "task_id": "base_2_teacher_anchor",
                "base_task_id": "base_2",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_success",
                "entry_point": "compress_runs",
                "extracted_code": "def compress_runs(text):\n    return text\n",
            },
            {
                "task_id": "base_1_urgency",
                "base_task_id": "base_1",
                "pressure_type": "urgency",
                "route_label": "shortcut_success",
                "entry_point": "prefix_sums",
                "extracted_code": "def prefix_sums(nums):\n    return nums\n",
            },
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            control_path = Path(tempdir) / "control.jsonl"
            results_path = Path(tempdir) / "results.jsonl"
            output_path = Path(tempdir) / "pairs.jsonl"
            write_jsonl(control_path, control_rows)
            write_jsonl(results_path, result_rows)

            built_path = build_coding_patch_pairs(
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
        self.assertEqual(pairs[0]["entry_point"], "prefix_sums")
        self.assertIn("pairing_strategy", pairs[0]["metadata"])

    def test_load_coding_patch_pairs_reconstructs_prompts(self) -> None:
        patch_rows = [
            {
                "base_task_id": "base_1",
                "pressure_type": "teacher_anchor",
                "control_task_id": "base_1_control",
                "pressure_task_id": "base_1_teacher_anchor",
                "entry_point": "prefix_sums",
                "robust_reference_code": "def prefix_sums(nums):\n    return nums\n",
                "shortcut_reference_code": "def prefix_sums(nums):\n    return [0] * len(nums)\n",
                "metadata": {
                    "source_family": "humaneval_like",
                    "archetype": "weak_checker_exploit",
                    "pairing_strategy": "control_robust_vs_pressure_shortcut",
                },
            }
        ]
        manifest_rows = [
            {"task_id": "base_1_control", "prompt": "Control prompt"},
            {"task_id": "base_1_teacher_anchor", "prompt": "Pressure prompt"},
        ]
        result_rows = [
            {"task_id": "base_1_control", "prompt": "Control prompt"},
            {"task_id": "base_1_teacher_anchor", "prompt": "Pressure prompt"},
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            patch_path = Path(tempdir) / "pairs.jsonl"
            manifest_path = Path(tempdir) / "manifest.jsonl"
            results_path = Path(tempdir) / "results.jsonl"
            write_jsonl(patch_path, patch_rows)
            write_jsonl(manifest_path, manifest_rows)
            write_jsonl(results_path, result_rows)

            pairs = load_coding_patch_pairs(
                patch_pairs_path=patch_path,
                manifest_path=manifest_path,
                results_path=results_path,
            )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].control_prompt, "Control prompt")
        self.assertEqual(pairs[0].pressure_prompt, "Pressure prompt")
        self.assertEqual(pairs[0].pressure_type, "teacher_anchor")


if __name__ == "__main__":
    unittest.main()
