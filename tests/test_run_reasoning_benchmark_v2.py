"""Tests for reasoning v2 benchmark runner helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressuretrace.behavior.run_reasoning_benchmark_v2 import (
    _generation_profile_for_model_v2,
    run_reasoning_pilot_v2,
)
from pressuretrace.utils.io import read_jsonl, write_jsonl


class RunReasoningBenchmarkV2TestCase(unittest.TestCase):
    def test_qwen3_profile_supports_thinking_ablation(self) -> None:
        profile = _generation_profile_for_model_v2("Qwen/Qwen3-8B", "off")

        self.assertEqual(profile.backend, "manual_qwen3")
        self.assertFalse(profile.enable_thinking)
        self.assertTrue(profile.do_sample)

    def test_non_qwen3_models_reject_non_default_thinking_modes(self) -> None:
        with self.assertRaises(ValueError):
            _generation_profile_for_model_v2("Qwen/Qwen2.5-14B-Instruct", "on")

    def test_runner_writes_thinking_mode_into_dry_run_rows(self) -> None:
        manifest_rows = [
            {
                "task_id": "gsm8k_reasoning_v2_000001_control",
                "source_dataset": "gsm8k",
                "source_id": "test_0",
                "family": "reasoning_conflict",
                "template_type": "direct_question_control_v2",
                "pressure_type": "control",
                "base_question": "How many apples?",
                "prompt": (
                    "Solve the problem. Respond with only the final number.\n\nProblem:\n"
                    "How many apples?"
                ),
                "gold_answer": "12",
                "shortcut_answer": "7",
                "robust_path_available": True,
                "metadata": {
                    "base_task_id": "gsm8k_reasoning_v2_000001",
                    "transformation_version": "v2",
                    "prompt_family": "reasoning_conflict_v2",
                    "split": "test",
                    "original_index": 0,
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "manifest.jsonl"
            results_path = Path(tempdir) / "results.jsonl"
            write_jsonl(manifest_path, manifest_rows)

            with patch(
                "pressuretrace.behavior.run_reasoning_benchmark_v2.build_reasoning_manifest_v2",
                return_value=manifest_path,
            ), patch(
                "pressuretrace.behavior.run_reasoning_benchmark_v2.load_reasoning_manifest_v2",
                return_value=manifest_rows,
            ):
                artifacts = run_reasoning_pilot_v2(
                    split="test",
                    limit=1,
                    model_name="Qwen/Qwen3-8B",
                    pressure_type="all",
                    output_path=results_path,
                    manifest_path=manifest_path,
                    dry_run=True,
                    include_control=True,
                    thinking_mode="off",
                )

            rows = read_jsonl(artifacts.results_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["thinking_mode"], "off")
        self.assertEqual(rows[0]["route_label"], "pending_inference")


if __name__ == "__main__":
    unittest.main()
