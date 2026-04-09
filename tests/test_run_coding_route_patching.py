from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pressuretrace.patching.build_coding_patch_pairs import CodingPatchPair
from pressuretrace.patching.coding_patching_metrics import ContinuationScore
from pressuretrace.patching.run_coding_route_patching import (
    build_route_patching_config,
    run_coding_route_patching,
    select_eligible_patch_pairs,
)
from pressuretrace.utils.io import read_jsonl


class FakeTokenizer:
    """Tiny tokenizer stub for runner-only tests."""

    def __init__(self, token_map: dict[str, list[int]]) -> None:
        self._token_map = token_map

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self._token_map.get(text, []))

    def convert_ids_to_tokens(self, token_ids: int | list[int]) -> str | list[str]:
        if isinstance(token_ids, list):
            return [f"tok-{token_id}" for token_id in token_ids]
        return f"tok-{token_ids}"

    def decode(self, token_ids: list[int], **_: object) -> str:
        return " ".join(f"tok-{token_id}" for token_id in token_ids)


class RunCodingRoutePatchingTestCase(unittest.TestCase):
    def _make_pair(
        self,
        base_task_id: str,
        pressure_type: str,
        robust_reference_code: str,
        shortcut_reference_code: str,
    ) -> CodingPatchPair:
        return CodingPatchPair(
            base_task_id=base_task_id,
            pressure_type=pressure_type,
            control_task_id=f"{base_task_id}_control",
            pressure_task_id=f"{base_task_id}_{pressure_type}",
            control_prompt="control prompt",
            pressure_prompt="pressure prompt",
            entry_point="solve",
            archetype="spec_omission",
            source_family="humaneval_like",
            robust_reference_code=robust_reference_code,
            shortcut_reference_code=shortcut_reference_code,
            metadata={"base_task_id": base_task_id, "pairing_strategy": "test"},
        )

    def test_select_eligible_patch_pairs_accepts_tokenizable_sequences(self) -> None:
        pairs = [
            self._make_pair("base_1", "neutral_wrong_answer_cue", "robust1", "shortcut1"),
            self._make_pair("base_2", "neutral_wrong_answer_cue", "robust2", "shortcut2"),
        ]
        tokenizer = FakeTokenizer(
            {
                "robust1": [1, 2],
                "shortcut1": [3],
                "robust2": [4],
                "shortcut2": [5, 6],
            }
        )
        bundle = SimpleNamespace(tokenizer=tokenizer)

        eligible_pairs, skipped_tokenization = select_eligible_patch_pairs(pairs, bundle)

        self.assertEqual(len(eligible_pairs), 2)
        self.assertEqual(skipped_tokenization, 0)
        self.assertEqual(eligible_pairs[0].continuations.robust_token_ids, (1, 2))
        self.assertEqual(eligible_pairs[1].continuations.shortcut_token_ids, (5, 6))

    def test_run_coding_route_patching_writes_outputs(self) -> None:
        pair_rows = [
            self._make_pair("base_1", "neutral_wrong_answer_cue", "robust1", "shortcut1"),
            self._make_pair("base_2", "neutral_wrong_answer_cue", "robust2", "shortcut2"),
        ]
        tokenizer = FakeTokenizer(
            {
                "robust1": [1, 2],
                "shortcut1": [3],
                "robust2": [4],
                "shortcut2": [5, 6],
            }
        )
        fake_bundle = SimpleNamespace(model=SimpleNamespace(), tokenizer=tokenizer)

        def fake_get_prompt_hidden_states(
            _bundle: object,
            prompt: str,
        ) -> tuple[str, object]:
            logits = (
                torch.tensor([[[2.0, 1.0, 0.5]]], dtype=torch.float32)
                if prompt == "control prompt"
                else torch.tensor([[[0.5, 2.0, 1.0]]], dtype=torch.float32)
            )
            outputs = SimpleNamespace(
                logits=logits,
                hidden_states=[
                    torch.zeros((1, 1, 3), dtype=torch.float32) for _ in range(13)
                ],
            )
            return prompt, outputs

        def fake_get_next_token_logits_from_outputs(outputs: object) -> torch.Tensor:
            return outputs.logits[0, -1, :]

        def fake_final_token_activation_for_layer(
            outputs: object,
            prompt_inputs: str,
            *,
            model: object,
            layer: int,
        ) -> torch.Tensor:
            del outputs, prompt_inputs, model
            return torch.tensor([float(layer), 0.0, 0.0], dtype=torch.float32)

        def fake_patch_final_token_activation(
            _bundle: object,
            prompt_inputs: str,
            *,
            layer: int,
            donor_final_token_activation: torch.Tensor,
        ) -> torch.Tensor:
            del donor_final_token_activation
            if prompt_inputs == "pressure prompt":
                return torch.tensor([2.5 + (layer / 100.0), 0.25, 0.1], dtype=torch.float32)
            return torch.tensor([0.35, 2.2 + (layer / 100.0), 0.2], dtype=torch.float32)

        def fake_score_continuation_sequence(
            _bundle: object,
            prompt_inputs: str,
            *,
            continuation_token_ids: tuple[int, ...] | list[int],
            layer: int | None = None,
            donor_final_token_activation: torch.Tensor | None = None,
        ) -> ContinuationScore:
            del donor_final_token_activation
            token_ids = tuple(int(token_id) for token_id in continuation_token_ids)
            token_strs = tuple(f"tok-{token_id}" for token_id in token_ids)
            base = -0.2 * len(token_ids)
            if prompt_inputs == "pressure prompt":
                robust_bonus = 0.4 if layer is not None else 0.0
                shortcut_bonus = 0.6 if layer is None else -0.2
            else:
                robust_bonus = 0.6 if layer is None else -0.2
                shortcut_bonus = 0.1 if layer is None else 0.5
            bonus = robust_bonus if token_ids[0] in {1, 4} else shortcut_bonus
            mean = base + bonus
            return ContinuationScore(
                token_ids=token_ids,
                token_strs=token_strs,
                logprob_sum=mean * len(token_ids),
                logprob_mean=mean,
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
            )

            with (
                patch(
                    "pressuretrace.patching.run_coding_route_patching.load_coding_patch_pairs",
                    return_value=pair_rows,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.load_model_and_tokenizer",
                    return_value=fake_bundle,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.get_prompt_hidden_states",
                    side_effect=fake_get_prompt_hidden_states,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.get_next_token_logits_from_outputs",
                    side_effect=fake_get_next_token_logits_from_outputs,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.final_token_activation_for_layer",
                    side_effect=fake_final_token_activation_for_layer,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.patch_final_token_activation",
                    side_effect=fake_patch_final_token_activation,
                ),
                patch(
                    "pressuretrace.patching.run_coding_route_patching.score_continuation_sequence",
                    side_effect=fake_score_continuation_sequence,
                ),
            ):
                artifacts = run_coding_route_patching(config)

            rows = read_jsonl(artifacts.output_path)
            self.assertTrue(artifacts.output_path.exists())
            self.assertTrue(artifacts.summary_txt_path.exists())
            self.assertTrue(artifacts.summary_csv_path.exists())
            self.assertTrue(artifacts.rescue_delta_robust_prob_plot_path.exists())
            self.assertTrue(artifacts.rescue_delta_margin_plot_path.exists())
            self.assertTrue(artifacts.induction_delta_shortcut_prob_plot_path.exists())

        self.assertEqual(artifacts.total_pairs_loaded, 2)
        self.assertEqual(artifacts.retained_pairs, 2)
        self.assertEqual(artifacts.skipped_tokenization, 0)
        self.assertEqual(artifacts.rows_written, 12)
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["direction"] for row in rows}, {"rescue", "induction"})
        self.assertEqual({row["layer"] for row in rows}, {-10, -8, -6})
        self.assertEqual({row["pressure_type"] for row in rows}, {"neutral_wrong_answer_cue"})
        self.assertTrue(all("delta_robust_prob" in row for row in rows))
        self.assertTrue(all("continuation_scoring_mode" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
