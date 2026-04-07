from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pressuretrace.behavior.reasoning_runtime import ReasoningGenerationProfile
from pressuretrace.patching.reasoning_patching_core import (
    PromptInputs,
    ReasoningPatchingBundle,
    build_model_inputs,
    ensure_single_token_answer,
    extract_final_token_activation,
    get_next_token_logits,
    load_model_and_tokenizer,
    patch_final_token_activation,
    resolve_decoder_layer_module,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None
        self.eos_token_id = 2
        self.apply_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        self.apply_calls.append(
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
        text = texts[0]
        length = 4 if "Question" in text else 3
        start = 1 if "1+1" in text else 5 if "2+2" in text else 1
        return {
            "input_ids": torch.arange(start, start + length, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        mapping = {"42": [42], " 7": [7], "multi": [1, 2]}
        return mapping.get(text, [99])

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return f"tok{token_ids[0]}"

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"tok{token_id}"


class _DummyBlock(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(offset), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return (hidden_states + self.offset, None)


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_DummyBlock(1.0), _DummyBlock(10.0), _DummyBlock(100.0)]
        )
        self.embed_tokens = nn.Embedding(16, 4)
        self.lm_head = nn.Linear(4, 16, bias=False)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(16, 4))
            self.lm_head.weight.copy_(torch.eye(16, 4))
        self._parameter = nn.Parameter(torch.zeros(1))

    def eval(self) -> _DummyModel:
        return self

    def parameters(self):  # type: ignore[override]
        yield self._parameter

    def forward(  # type: ignore[no-untyped-def]
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        return_dict=False,
    ):
        del attention_mask, use_cache, return_dict
        hidden_states = self.embed_tokens(input_ids)
        all_hidden_states = [hidden_states]
        for block in self.model.layers:
            block_output = block(hidden_states)
            hidden_states = block_output[0]
            all_hidden_states.append(hidden_states)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


def _bundle() -> ReasoningPatchingBundle:
    tokenizer = _FakeTokenizer()
    model = _DummyModel()
    profile = ReasoningGenerationProfile(
        backend="manual_qwen3",
        do_sample=False,
        max_new_tokens=96,
        enable_thinking=False,
    )
    return ReasoningPatchingBundle(
        model=model,
        tokenizer=tokenizer,
        profile=profile,
        input_device=torch.device("cpu"),
    )


class ReasoningPatchingCoreTestCase(unittest.TestCase):
    def test_build_model_inputs_applies_chat_template(self) -> None:
        bundle = _bundle()
        prompt_inputs = build_model_inputs(bundle, "Question: 1+1?")

        self.assertIsInstance(prompt_inputs, PromptInputs)
        self.assertEqual(prompt_inputs.final_token_index, 3)
        self.assertIn("system::", prompt_inputs.rendered_prompt)
        self.assertEqual(len(bundle.tokenizer.apply_calls), 1)
        self.assertIs(bundle.tokenizer.apply_calls[0]["enable_thinking"], False)

    def test_ensure_single_token_answer_rejects_multi_token_answers(self) -> None:
        tokenizer = _FakeTokenizer()

        answer = ensure_single_token_answer(tokenizer, "42")

        self.assertEqual(answer.token_id, 42)
        self.assertEqual(answer.token_str, "tok42")
        self.assertEqual(answer.matched_text, "42")
        with self.assertRaises(ValueError):
            ensure_single_token_answer(tokenizer, "multi")

    def test_get_next_token_logits_and_activation_patch(self) -> None:
        bundle = _bundle()
        prompt_inputs = build_model_inputs(bundle, "Question: 1+1?")
        donor_inputs = build_model_inputs(bundle, "Question: 2+2?")

        baseline_logits = get_next_token_logits(bundle, prompt_inputs)
        donor_activation = extract_final_token_activation(bundle, donor_inputs, layer=-1)
        patched_logits = patch_final_token_activation(
            bundle,
            prompt_inputs,
            layer=-1,
            donor_final_token_activation=donor_activation,
        )

        self.assertEqual(baseline_logits.shape, patched_logits.shape)
        self.assertFalse(torch.allclose(baseline_logits, patched_logits))

    def test_resolve_decoder_layer_module_supports_negative_indexing(self) -> None:
        model = _DummyModel()

        resolved = resolve_decoder_layer_module(model, -1)

        self.assertIsInstance(resolved, _DummyBlock)

    def test_load_model_and_tokenizer_sets_pad_token_and_evals_model(self) -> None:
        fake_tokenizer = _FakeTokenizer()
        fake_model = _DummyModel()

        with patch(
            "pressuretrace.patching.reasoning_patching_core.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ), patch(
            "pressuretrace.patching.reasoning_patching_core.AutoModelForCausalLM.from_pretrained",
            return_value=fake_model,
        ), patch(
            "pressuretrace.patching.reasoning_patching_core.manual_model_load_kwargs",
            return_value={"torch_dtype": torch.float32},
        ), patch(
            "pressuretrace.patching.reasoning_patching_core.model_input_device",
            return_value=torch.device("cpu"),
        ):
            bundle = load_model_and_tokenizer("Qwen/Qwen3-14B")

        self.assertEqual(bundle.input_device, torch.device("cpu"))
        self.assertEqual(fake_tokenizer.pad_token_id, 2)
        self.assertIs(bundle.model, fake_model)


if __name__ == "__main__":
    unittest.main()
