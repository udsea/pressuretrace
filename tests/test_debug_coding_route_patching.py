from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pressuretrace.patching.build_coding_patch_pairs import CodingPatchPair
from pressuretrace.patching.coding_patching_core import (
    GenerationStepPatchDebug,
    TokenCandidate,
)
from pressuretrace.patching.debug_coding_route_patching import (
    build_debug_config,
    run_coding_route_patching_debug,
)
from pressuretrace.utils.io import read_jsonl


class DebugCodingRoutePatchingTestCase(unittest.TestCase):
    def _make_pair(self) -> CodingPatchPair:
        return CodingPatchPair(
            base_task_id="base_1",
            pressure_type="neutral_wrong_answer_cue",
            control_task_id="base_1_control",
            pressure_task_id="base_1_neutral_wrong_answer_cue",
            control_prompt="control prompt",
            pressure_prompt="pressure prompt",
            entry_point="solve",
            archetype="visible_test_hardcoding",
            source_family="humaneval_like",
            robust_reference_code="def solve(x):\n    return x\n",
            shortcut_reference_code="def solve(x):\n    return 1\n",
            metadata={"pairing_strategy": "test"},
        )

    def _make_debug(self, *, generation_step: int, changed: bool) -> GenerationStepPatchDebug:
        return GenerationStepPatchDebug(
            generation_step=generation_step,
            prefix_token_ids=(),
            baseline_top1_token_id=11,
            baseline_top1_token_str="def",
            patched_top1_token_id=12 if changed else 11,
            patched_top1_token_str="if" if changed else "def",
            top1_changed=changed,
            hidden_delta_l2=1.5,
            hidden_delta_max_abs=0.4,
            donor_target_hidden_delta_l2=2.5,
            donor_target_hidden_delta_max_abs=0.8,
            logit_delta_l2=0.6,
            logit_delta_max_abs=0.2,
            baseline_top_tokens=(
                TokenCandidate(1, 11, "def", 4.0, 0.9),
                TokenCandidate(2, 12, "if", 3.0, 0.1),
            ),
            patched_top_tokens=(
                TokenCandidate(1, 12 if changed else 11, "if" if changed else "def", 4.1, 0.9),
                TokenCandidate(2, 11 if changed else 12, "def" if changed else "if", 3.1, 0.1),
            ),
        )

    def test_run_coding_route_patching_debug_writes_outputs(self) -> None:
        pair = self._make_pair()
        fake_bundle = SimpleNamespace()
        fake_trace = SimpleNamespace(
            generated_token_ids=(1,),
            generated_text="def",
            step_traces=(0,),
        )
        results_index = {
            "base_1_control": {"task_id": "base_1_control", "route_label": "robust_success"},
            "base_1_neutral_wrong_answer_cue": {
                "task_id": "base_1_neutral_wrong_answer_cue",
                "route_label": "shortcut_success",
            },
        }

        with tempfile.TemporaryDirectory() as tempdir:
            temp_root = Path(tempdir)
            config = build_debug_config(
                frozen_root=temp_root,
                manifest_path=temp_root / "data" / "manifests" / "coding_paper_slice.jsonl",
                results_path=temp_root / "results" / "coding_results.jsonl",
                patch_pairs_path=temp_root / "results" / "coding_patch_pairs.jsonl",
                output_path=temp_root / "results" / "coding_route_debug.jsonl",
                summary_txt_path=temp_root / "results" / "coding_route_debug.txt",
                layer=-8,
                position_window="gen_1",
            )

            with (
                patch(
                    "pressuretrace.patching.debug_coding_route_patching.load_coding_patch_pairs",
                    return_value=[pair],
                ),
                patch(
                    "pressuretrace.patching.debug_coding_route_patching._load_task_rows_by_task_id",
                    return_value=({}, results_index),
                ),
                patch(
                    "pressuretrace.patching.debug_coding_route_patching.load_model_and_tokenizer",
                    return_value=fake_bundle,
                ),
                patch(
                    "pressuretrace.patching.debug_coding_route_patching.build_model_inputs",
                    side_effect=[SimpleNamespace(), SimpleNamespace()],
                ),
                patch(
                    "pressuretrace.patching.debug_coding_route_patching.capture_greedy_generation_trace",
                    side_effect=[fake_trace, fake_trace],
                ),
                patch(
                    "pressuretrace.patching.debug_coding_route_patching.debug_generation_step_patch",
                    side_effect=[
                        self._make_debug(generation_step=0, changed=True),
                        self._make_debug(generation_step=0, changed=False),
                    ],
                ),
            ):
                artifacts = run_coding_route_patching_debug(config)

            rows = read_jsonl(artifacts.output_path)
            self.assertEqual(artifacts.base_task_id, "base_1")
            self.assertEqual(artifacts.pressure_type, "neutral_wrong_answer_cue")
            self.assertEqual(artifacts.layer, -8)
            self.assertEqual(artifacts.position_window, "gen_1")
            self.assertEqual(artifacts.rows_written, 2)
            self.assertTrue(artifacts.output_path.exists())
            self.assertTrue(artifacts.summary_txt_path.exists())
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["direction"] for row in rows}, {"rescue", "induction"})
            self.assertEqual(rows[0]["generation_step"], 0)
            self.assertIn("baseline_top_tokens", rows[0])
            self.assertIn("patched_top_tokens", rows[0])


if __name__ == "__main__":
    unittest.main()
