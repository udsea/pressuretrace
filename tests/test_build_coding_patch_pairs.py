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
    def test_build_coding_patch_pairs_keeps_only_neutral_visible_hardcoding_pairs(self) -> None:
        control_rows = [
            {
                "base_task_id": "base_1",
                "control_task_id": "base_1_control",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/prefix_sums",
                "archetype": "visible_test_hardcoding",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "robust_success",
                "prompt": "Control prompt 1",
            },
            {
                "base_task_id": "base_2",
                "control_task_id": "base_2_control",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/compress_runs",
                "archetype": "weak_checker_exploit",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "robust_success",
                "prompt": "Control prompt 2",
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
                "passed_visible_tests": True,
                "passed_hidden_tests": True,
                "prompt": "Control prompt 1",
            },
            {
                "task_id": "base_1_neutral_wrong_answer_cue",
                "base_task_id": "base_1",
                "pressure_type": "neutral_wrong_answer_cue",
                "route_label": "shortcut_success",
                "entry_point": "prefix_sums",
                "extracted_code": "def prefix_sums(nums):\n    return [0] * len(nums)\n",
                "passed_visible_tests": True,
                "passed_hidden_tests": False,
                "prompt": "Pressure prompt 1",
            },
            {
                "task_id": "base_1_teacher_anchor",
                "base_task_id": "base_1",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_success",
                "entry_point": "prefix_sums",
                "extracted_code": "def prefix_sums(nums):\n    return nums\n",
                "passed_visible_tests": True,
                "passed_hidden_tests": False,
                "prompt": "Pressure prompt 2",
            },
            {
                "task_id": "base_2_neutral_wrong_answer_cue",
                "base_task_id": "base_2",
                "pressure_type": "neutral_wrong_answer_cue",
                "route_label": "shortcut_success",
                "entry_point": "compress_runs",
                "extracted_code": "def compress_runs(text):\n    return text\n",
                "passed_visible_tests": True,
                "passed_hidden_tests": False,
                "prompt": "Pressure prompt 3",
            },
            {
                "task_id": "base_1_urgency",
                "base_task_id": "base_1",
                "pressure_type": "urgency",
                "route_label": "shortcut_success",
                "entry_point": "prefix_sums",
                "extracted_code": "def prefix_sums(nums):\n    return nums\n",
                "passed_visible_tests": True,
                "passed_hidden_tests": False,
                "prompt": "Pressure prompt 4",
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
        self.assertEqual(pairs[0]["archetype"], "visible_test_hardcoding")
        self.assertEqual(pairs[0]["pressure_type"], "neutral_wrong_answer_cue")
        self.assertEqual(pairs[0]["control_task_id"], "base_1_control")
        self.assertEqual(pairs[0]["pressure_task_id"], "base_1_neutral_wrong_answer_cue")
        self.assertEqual(pairs[0]["entry_point"], "prefix_sums")
        self.assertEqual(pairs[0]["prompt_control"], "Control prompt 1")
        self.assertEqual(pairs[0]["prompt_pressure"], "Pressure prompt 1")
        self.assertEqual(pairs[0]["route_control"], "robust_success")
        self.assertEqual(pairs[0]["route_pressure"], "shortcut_success")
        self.assertTrue(pairs[0]["metadata"]["control_visible_pass"])
        self.assertTrue(pairs[0]["metadata"]["pressure_visible_pass"])
        self.assertIn("pairing_strategy", pairs[0]["metadata"])

    def test_build_coding_patch_pairs_allows_teacher_anchor_when_explicitly_requested(
        self,
    ) -> None:
        control_rows = [
            {
                "base_task_id": "base_1",
                "control_task_id": "base_1_control",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/prefix_sums",
                "archetype": "visible_test_hardcoding",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "robust_success",
                "prompt": "Control prompt 1",
            }
        ]
        result_rows = [
            {
                "task_id": "base_1_control",
                "base_task_id": "base_1",
                "pressure_type": "control",
                "route_label": "robust_success",
                "entry_point": "prefix_sums",
                "extracted_code": "def prefix_sums(nums):\n    return nums\n",
                "passed_visible_tests": True,
                "passed_hidden_tests": True,
                "prompt": "Control prompt 1",
            },
            {
                "task_id": "base_1_teacher_anchor",
                "base_task_id": "base_1",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_success",
                "entry_point": "prefix_sums",
                "extracted_code": "def prefix_sums(nums):\n    return [0] * len(nums)\n",
                "passed_visible_tests": True,
                "passed_hidden_tests": False,
                "prompt": "Pressure prompt 2",
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
                pressure_types=("teacher_anchor",),
            )
            pairs = read_jsonl(built_path)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["pressure_type"], "teacher_anchor")

    def test_build_coding_patch_pairs_raises_with_zero_pair_diagnostics(self) -> None:
        control_rows = [
            {
                "base_task_id": "base_1",
                "control_task_id": "base_1_control",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/compress_runs",
                "archetype": "visible_test_hardcoding",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "robust_success",
            }
        ]
        result_rows = [
            {
                "task_id": "base_1_control",
                "base_task_id": "base_1",
                "pressure_type": "control",
                "route_label": "robust_success",
                "entry_point": "compress_runs",
                "extracted_code": "def compress_runs(text):\n    return 'a2'\n",
                "passed_visible_tests": True,
                "passed_hidden_tests": True,
            },
            {
                "task_id": "base_1_neutral_wrong_answer_cue",
                "base_task_id": "base_1",
                "pressure_type": "neutral_wrong_answer_cue",
                "route_label": "robust_success",
                "entry_point": "compress_runs",
                "extracted_code": "def compress_runs(text):\n    return 'a2'\n",
                "passed_visible_tests": True,
                "passed_hidden_tests": True,
            },
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            control_path = Path(tempdir) / "control.jsonl"
            results_path = Path(tempdir) / "results.jsonl"
            output_path = Path(tempdir) / "pairs.jsonl"
            write_jsonl(control_path, control_rows)
            write_jsonl(results_path, result_rows)

            with self.assertRaisesRegex(
                ValueError,
                "No coding patch pairs matched the first-pass filter",
            ):
                build_coding_patch_pairs(
                    results_path=results_path,
                    control_slice_path=control_path,
                    output_path=output_path,
                )

    def test_load_coding_patch_pairs_reconstructs_prompts(self) -> None:
        patch_rows = [
            {
                "base_task_id": "base_1",
                "archetype": "visible_test_hardcoding",
                "pressure_type": "teacher_anchor",
                "control_task_id": "base_1_control",
                "pressure_task_id": "base_1_teacher_anchor",
                "entry_point": "prefix_sums",
                "prompt_control": "Control prompt",
                "prompt_pressure": "Pressure prompt",
                "route_control": "robust_success",
                "route_pressure": "shortcut_success",
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
                pressure_types=("teacher_anchor",),
            )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].control_prompt, "Control prompt")
        self.assertEqual(pairs[0].pressure_prompt, "Pressure prompt")
        self.assertEqual(pairs[0].pressure_type, "teacher_anchor")


if __name__ == "__main__":
    unittest.main()
