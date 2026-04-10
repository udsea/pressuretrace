from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pressuretrace.evaluation.coding_eval import CodingEvaluationRecord
from pressuretrace.patching.build_coding_patch_pairs import CodingPatchPair
from pressuretrace.patching.run_coding_route_patching import (
    build_route_patching_config,
    run_coding_route_patching,
    select_eligible_patch_pairs,
)
from pressuretrace.utils.io import read_jsonl


class RunCodingRoutePatchingTestCase(unittest.TestCase):
    def _make_pair(self, base_task_id: str) -> CodingPatchPair:
        return CodingPatchPair(
            base_task_id=base_task_id,
            pressure_type="neutral_wrong_answer_cue",
            control_task_id=f"{base_task_id}_control",
            pressure_task_id=f"{base_task_id}_neutral_wrong_answer_cue",
            control_prompt="control prompt",
            pressure_prompt="pressure prompt",
            entry_point="solve",
            archetype="visible_test_hardcoding",
            source_family="humaneval_like",
            robust_reference_code="def solve(x):\n    return x\n",
            shortcut_reference_code=(
                "def solve(x):\n"
                "    if x == 1:\n"
                "        return 1\n"
                "    return x\n"
            ),
            metadata={"base_task_id": base_task_id, "pairing_strategy": "test"},
        )

    def test_select_eligible_patch_pairs_keeps_rows_with_task_context(self) -> None:
        pairs = [self._make_pair("base_1"), self._make_pair("base_2")]
        task_rows = {
            "base_1_control": {"task_id": "base_1_control"},
            "base_1_neutral_wrong_answer_cue": {"task_id": "base_1_neutral_wrong_answer_cue"},
        }
        result_rows = {
            "base_1_control": {"route_label": "robust_success"},
            "base_1_neutral_wrong_answer_cue": {"route_label": "shortcut_success"},
        }

        retained_pairs, skipped_missing = select_eligible_patch_pairs(
            pairs,
            task_rows_by_task_id=task_rows,
            results_by_task_id=result_rows,
        )

        self.assertEqual(len(retained_pairs), 1)
        self.assertEqual(skipped_missing, 1)
        self.assertEqual(retained_pairs[0].route_control, "robust_success")
        self.assertEqual(retained_pairs[0].route_pressure, "shortcut_success")

    def test_run_coding_route_patching_writes_outputs(self) -> None:
        pair_rows = [self._make_pair("base_1"), self._make_pair("base_2")]
        task_rows = {
            "base_1_control": {
                "task_id": "base_1_control",
                "entry_point": "solve",
                "visible_tests": [],
                "hidden_test_contract": [],
            },
            "base_1_neutral_wrong_answer_cue": {
                "task_id": "base_1_neutral_wrong_answer_cue",
                "entry_point": "solve",
                "visible_tests": [],
                "hidden_test_contract": [],
            },
            "base_2_control": {
                "task_id": "base_2_control",
                "entry_point": "solve",
                "visible_tests": [],
                "hidden_test_contract": [],
            },
            "base_2_neutral_wrong_answer_cue": {
                "task_id": "base_2_neutral_wrong_answer_cue",
                "entry_point": "solve",
                "visible_tests": [],
                "hidden_test_contract": [],
            },
        }
        result_rows = {
            "base_1_control": {"route_label": "robust_success"},
            "base_1_neutral_wrong_answer_cue": {"route_label": "shortcut_success"},
            "base_2_control": {"route_label": "robust_success"},
            "base_2_neutral_wrong_answer_cue": {"route_label": "shortcut_success"},
        }
        fake_bundle = SimpleNamespace()

        def fake_build_model_inputs(_bundle: object, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(prompt=prompt)

        def fake_capture_greedy_generation_trace(
            _bundle: object,
            prompt_inputs: SimpleNamespace,
            *,
            layer: int,
            max_new_tokens: int,
        ) -> SimpleNamespace:
            del max_new_tokens
            return SimpleNamespace(
                generated_token_ids=(11, 12, 13),
                generated_text=f"trace_{prompt_inputs.prompt}_{layer}",
                step_traces=tuple(range(5)),
            )

        def fake_generate_with_generation_window_patch(
            _bundle: object,
            prompt_inputs: SimpleNamespace,
            *,
            layer: int,
            patch_window: str,
            donor_trace: SimpleNamespace,
            max_new_tokens: int,
        ) -> SimpleNamespace:
            del donor_trace, max_new_tokens
            if prompt_inputs.prompt == "pressure prompt":
                text = f"rescued_{layer}_{patch_window}"
            else:
                text = f"induced_{layer}_{patch_window}"
            return SimpleNamespace(
                generated_token_ids=(21, 22, 23),
                generated_text=text,
                step_diagnostics=(
                    SimpleNamespace(
                        baseline_top1_token_id=1,
                        patched_top1_token_id=2,
                    ),
                ),
            )

        def fake_evaluate_coding_response(
            task_row: dict[str, object],
            completion: str,
        ) -> CodingEvaluationRecord:
            del task_row
            if completion.startswith("rescued_"):
                return CodingEvaluationRecord(
                    route_label="robust_success",
                    failure_subtype=None,
                    parse_status="ok",
                    passed_visible_tests=True,
                    passed_hidden_tests=True,
                    extracted_code="def solve(x):\n    return x\n",
                )
            return CodingEvaluationRecord(
                route_label="shortcut_success",
                failure_subtype="hardcoding_suspected",
                parse_status="ok",
                passed_visible_tests=True,
                passed_hidden_tests=False,
                extracted_code="def solve(x):\n    if x == 1:\n        return 1\n",
            )

        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            config = build_route_patching_config(
                frozen_root=temp_root,
                manifest_path=temp_root / "data" / "manifests" / "coding_paper_slice.jsonl",
                patch_pairs_path=temp_root / "results" / "coding_patch_pairs.jsonl",
                results_path=temp_root / "results" / "paper_results.jsonl",
                output_path=temp_root / "results" / "coding_route_patching.jsonl",
                summary_txt_path=temp_root / "results" / "coding_route_patching.txt",
                summary_csv_path=temp_root / "results" / "coding_route_patching.csv",
                rescue_success_plot_path=temp_root / "results" / "rescue.png",
                induction_success_plot_path=temp_root / "results" / "induction.png",
                position_window_comparison_plot_path=temp_root / "results" / "windows.png",
            )

            with (
                patch(
                    "pressuretrace.patching.run_coding_route_patching.load_coding_patch_pairs",
                    return_value=pair_rows,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching._load_task_rows_by_task_id",
                    return_value=(task_rows, result_rows),
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.load_model_and_tokenizer",
                    return_value=fake_bundle,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.build_model_inputs",
                    side_effect=fake_build_model_inputs,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.capture_greedy_generation_trace",
                    side_effect=fake_capture_greedy_generation_trace,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.greedy_generate_with_generation_window_patch",
                    side_effect=fake_generate_with_generation_window_patch,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.evaluate_coding_response",
                    side_effect=fake_evaluate_coding_response,
                ),
            ):
                artifacts = run_coding_route_patching(config)

            rows = read_jsonl(artifacts.output_path)
            self.assertTrue(artifacts.output_path.exists())
            self.assertTrue(artifacts.summary_txt_path.exists())
            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.rescue_success_plot_path.exists())
            self.assertTrue(artifacts.induction_success_plot_path.exists())
            self.assertTrue(artifacts.position_window_comparison_plot_path.exists())

        self.assertEqual(artifacts.total_pairs_loaded, 2)
        self.assertEqual(artifacts.retained_pairs, 2)
        self.assertEqual(artifacts.skipped_tokenization, 0)
        self.assertEqual(artifacts.skipped_missing_task_rows, 0)
        self.assertEqual(artifacts.rows_written, 36)
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["direction"] for row in rows}, {"rescue", "induction"})
        self.assertEqual({row["layer"] for row in rows}, {-10, -8, -6})
        self.assertEqual({row["position_window"] for row in rows}, {"gen_1", "gen_1_3", "gen_1_5"})
        self.assertTrue(all("rescue_success" in row for row in rows))
        self.assertTrue(all("induction_success" in row for row in rows))

    def test_run_coding_route_patching_raises_when_no_pairs_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            config = build_route_patching_config(
                frozen_root=temp_root,
                manifest_path=temp_root / "data" / "manifests" / "coding_paper_slice.jsonl",
                patch_pairs_path=temp_root / "results" / "coding_patch_pairs.jsonl",
                results_path=temp_root / "results" / "paper_results.jsonl",
                output_path=temp_root / "results" / "coding_route_patching.jsonl",
                summary_txt_path=temp_root / "results" / "coding_route_patching.txt",
                summary_csv_path=temp_root / "results" / "coding_route_patching.csv",
                rescue_success_plot_path=temp_root / "results" / "rescue.png",
                induction_success_plot_path=temp_root / "results" / "induction.png",
                position_window_comparison_plot_path=temp_root / "results" / "windows.png",
            )

            with (
                patch(
                    "pressuretrace.patching.run_coding_route_patching.load_coding_patch_pairs",
                    return_value=[],
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching._load_task_rows_by_task_id",
                    return_value=({}, {}),
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "No eligible coding patch pairs were available for route patching",
                ):
                    run_coding_route_patching(config)


if __name__ == "__main__":
    unittest.main()
