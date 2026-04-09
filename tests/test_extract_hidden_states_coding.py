"""Tests for coding hidden-state extraction helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pressuretrace.behavior.run_coding_paper_slice import CODING_SYSTEM_PROMPT
from pressuretrace.config import CODING_V1_MODEL_NAME
from pressuretrace.probes.extract_hidden_states_coding import (
    CodingProbeExtractionConfig,
    extract_coding_hidden_states,
    project_hidden_state_vector,
    select_coding_probe_rows,
)
from pressuretrace.utils.io import read_jsonl, write_jsonl


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.pad_token_id = 0
        self.eos_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return f"system::{messages[0]['content']}||user::{messages[1]['content']}"

    def __call__(self, texts: list[str], return_tensors: str) -> dict[str, torch.Tensor]:
        del return_tensors
        if len(texts) != 1:
            raise AssertionError("Expected a single prompt.")
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        }


class _FakeModel:
    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def eval(self) -> _FakeModel:
        return self

    def parameters(self):
        yield self._parameter

    def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        hidden_states = tuple(
            torch.full((1, 4, 2), float(index), dtype=torch.float32) for index in range(13)
        )
        return SimpleNamespace(hidden_states=hidden_states)


class ExtractCodingHiddenStatesTestCase(unittest.TestCase):
    def test_select_coding_probe_rows_filters_and_joins(self) -> None:
        manifest_rows = [
            {
                "task_id": "task_1",
                "base_task_id": "base_1",
                "archetype": "spec_omission",
                "prompt": "Prompt 1",
                "metadata": {"base_task_id": "base_1"},
            },
            {
                "task_id": "task_2",
                "base_task_id": "base_2",
                "archetype": "weak_checker_exploit",
                "prompt": "Prompt 2",
                "metadata": {"base_task_id": "base_2"},
            },
        ]
        result_rows = [
            {
                "task_id": "task_1",
                "pressure_type": "neutral_wrong_answer_cue",
                "route_label": "shortcut_success",
                "model_name": CODING_V1_MODEL_NAME,
                "thinking_mode": "off",
                "prompt": "Prompt 1",
            },
            {
                "task_id": "task_2",
                "pressure_type": "teacher_anchor",
                "route_label": "robust_success",
                "model_name": CODING_V1_MODEL_NAME,
                "thinking_mode": "off",
                "prompt": "Prompt 2",
            },
            {
                "task_id": "task_3",
                "pressure_type": "teacher_anchor",
                "route_label": "wrong_nonshortcut",
                "model_name": CODING_V1_MODEL_NAME,
                "thinking_mode": "off",
                "prompt": "Prompt 3",
            },
        ]

        selected_rows = select_coding_probe_rows(
            manifest_rows,
            result_rows,
            model_name=CODING_V1_MODEL_NAME,
            thinking_mode="off",
        )

        self.assertEqual(len(selected_rows), 2)
        self.assertEqual(
            {row["route_label"] for row in selected_rows},
            {"shortcut_success", "robust_success"},
        )
        self.assertEqual(
            {row["base_task_id"] for row in selected_rows},
            {"base_1", "base_2"},
        )

    def test_project_hidden_state_vector_supports_last_token_and_mean_pool(self) -> None:
        hidden_state = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        attention_mask = torch.tensor([[1, 1, 1]])

        last_token = project_hidden_state_vector(hidden_state, "last_token", attention_mask)
        mean_pool = project_hidden_state_vector(hidden_state, "mean_pool", attention_mask)

        self.assertEqual(last_token.tolist(), [5.0, 6.0])
        self.assertEqual(mean_pool.tolist(), [3.0, 4.0])

    def test_extractor_writes_eligible_rows_only(self) -> None:
        manifest_rows = [
            {
                "task_id": "task_1",
                "base_task_id": "base_1",
                "source_family": "humaneval_like",
                "source_task_name": "humaneval_like/example",
                "archetype": "spec_omission",
                "pressure_type": "neutral_wrong_answer_cue",
                "prompt": "Prompt 1",
                "metadata": {"base_task_id": "base_1"},
            },
            {
                "task_id": "task_2",
                "base_task_id": "base_2",
                "source_family": "mbpp_like",
                "source_task_name": "mbpp_like/example",
                "archetype": "weak_checker_exploit",
                "pressure_type": "teacher_anchor",
                "prompt": "Prompt 2",
                "metadata": {"base_task_id": "base_2"},
            },
        ]
        result_rows = [
            {
                **manifest_rows[0],
                "model_name": CODING_V1_MODEL_NAME,
                "thinking_mode": "off",
                "route_label": "shortcut_success",
            },
            {
                **manifest_rows[1],
                "model_name": CODING_V1_MODEL_NAME,
                "thinking_mode": "off",
                "route_label": "robust_success",
            },
        ]

        fake_tokenizer = _FakeTokenizer()
        fake_model = _FakeModel()
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "manifest.jsonl"
            results_path = Path(tempdir) / "results.jsonl"
            output_path = Path(tempdir) / "hidden.jsonl"
            write_jsonl(manifest_path, manifest_rows)
            write_jsonl(results_path, result_rows)

            with (
                patch(
                    "pressuretrace.probes.extract_hidden_states_coding.AutoTokenizer.from_pretrained",
                    return_value=fake_tokenizer,
                ),
                patch(
                    "pressuretrace.probes.extract_hidden_states_coding.AutoModelForCausalLM.from_pretrained",
                    return_value=fake_model,
                ),
            ):
                extracted_path = extract_coding_hidden_states(
                    CodingProbeExtractionConfig(
                        manifest_path=manifest_path,
                        results_path=results_path,
                        output_path=output_path,
                        model_name=CODING_V1_MODEL_NAME,
                        thinking_mode="off",
                        layers=(-1,),
                        representations=("last_token", "mean_pool"),
                        progress_every=0,
                    )
                )
            rows = read_jsonl(extracted_path)

        self.assertEqual(len(rows), 4)
        self.assertEqual({row["binary_label"] for row in rows}, {0, 1})
        self.assertEqual(fake_tokenizer.calls[0]["messages"][0]["content"], CODING_SYSTEM_PROMPT)
        self.assertEqual(rows[0]["archetype"], "spec_omission")


if __name__ == "__main__":
    unittest.main()
