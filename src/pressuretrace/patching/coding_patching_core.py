"""Core prompt-only activation patching helpers for coding route patching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pressuretrace.behavior.reasoning_runtime import (
    build_reasoning_messages,
    is_qwen3_model,
    manual_model_load_kwargs,
    model_input_device,
    validate_thinking_mode,
)
from pressuretrace.behavior.run_coding_paper_slice import CODING_SYSTEM_PROMPT
from pressuretrace.patching.coding_patching_metrics import ContinuationScore


@dataclass(frozen=True)
class CodingPatchingBundle:
    """Loaded model/tokenizer bundle for prompt-level coding patching."""

    model_name: str
    model: Any
    tokenizer: Any
    input_device: torch.device
    enable_thinking: bool
    system_prompt: str = CODING_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptInputs:
    """Rendered and tokenized prompt inputs for a single coding episode."""

    prompt: str
    rendered_prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    final_token_index: int


@dataclass(frozen=True)
class ContinuationInputs:
    """Prompt inputs extended with a teacher-forced continuation."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    final_prompt_token_index: int


def load_model_and_tokenizer(
    model_name: str,
    *,
    thinking_mode: str = "off",
    trust_remote_code: bool = True,
) -> CodingPatchingBundle:
    """Load the coding model and tokenizer bundle used for route patching."""

    validated_thinking_mode = validate_thinking_mode(thinking_mode)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        **manual_model_load_kwargs(),
    )
    model.eval()
    return CodingPatchingBundle(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        input_device=model_input_device(model),
        enable_thinking=is_qwen3_model(model_name) and validated_thinking_mode in {"default", "on"},
    )


def token_id_to_string(tokenizer: Any, token_id: int) -> str:
    """Decode a token id into a stable human-readable string."""

    if hasattr(tokenizer, "decode"):
        try:
            return str(
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ).strip()
            )
        except Exception:  # pragma: no cover - tokenizer-specific fallback
            pass
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        converted = tokenizer.convert_ids_to_tokens(token_id)
        return str(converted)
    return str(token_id)


def _render_prompt_text(bundle: CodingPatchingBundle, prompt: str) -> str:
    """Render the chat-formatted prompt text used in the frozen coding run."""

    messages = build_reasoning_messages(prompt=prompt, system_prompt=bundle.system_prompt)
    if not hasattr(bundle.tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support chat templates required for route patching.")

    if is_qwen3_model(bundle.model_name):
        return str(
            bundle.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bundle.enable_thinking,
            )
        )
    return str(
        bundle.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def _final_token_position(attention_mask: torch.Tensor) -> int:
    """Return the final valid prompt token index for a batch of size one."""

    if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
        raise ValueError("Coding patching expects a single prompt at a time.")
    valid_tokens = int(attention_mask[0].sum().item())
    if valid_tokens <= 0:
        raise ValueError("Prompt attention mask contains no valid tokens.")
    return valid_tokens - 1


def build_model_inputs(bundle: CodingPatchingBundle, prompt: str) -> PromptInputs:
    """Tokenize a prompt exactly as the frozen coding run did."""

    rendered_prompt = _render_prompt_text(bundle, prompt)
    model_inputs = bundle.tokenizer([rendered_prompt], return_tensors="pt")
    if "input_ids" not in model_inputs:
        raise ValueError("Tokenizer did not return input_ids.")

    input_ids = model_inputs["input_ids"].to(bundle.input_device)
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(bundle.input_device)

    return PromptInputs(
        prompt=prompt,
        rendered_prompt=rendered_prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        final_token_index=_final_token_position(attention_mask),
    )


def _decoder_layers(model: Any) -> Any:
    """Return the decoder-layer sequence for a causal LM."""

    candidate_paths = (
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("decoder", "layers"),
        ("layers",),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidate_paths:
        current = model
        for attribute in path:
            if not hasattr(current, attribute):
                break
            current = getattr(current, attribute)
        else:
            return current
    raise ValueError("Could not find decoder layers for the supplied model.")


def resolve_layer_index(model: Any, layer: int) -> int:
    """Resolve a possibly negative layer index against the model depth."""

    layers = _decoder_layers(model)
    layer_count = len(layers)
    resolved = layer if layer >= 0 else layer_count + layer
    if resolved < 0 or resolved >= layer_count:
        raise IndexError(f"Layer index {layer} is out of range for {layer_count} layers.")
    return resolved


def resolve_decoder_layer_module(model: Any, layer: int) -> Any:
    """Return the decoder block module for the requested layer."""

    return _decoder_layers(model)[resolve_layer_index(model, layer)]


def _forward_inputs(
    bundle: CodingPatchingBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    output_hidden_states: bool,
) -> Any:
    """Run a forward pass through the model for arbitrary prepared inputs."""

    with torch.no_grad():
        return bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
            return_dict=True,
        )


def get_prompt_hidden_states(
    bundle: CodingPatchingBundle,
    prompt: str,
) -> tuple[PromptInputs, Any]:
    """Tokenize a prompt and run a hidden-state forward pass."""

    prompt_inputs = build_model_inputs(bundle, prompt)
    outputs = _forward_inputs(
        bundle,
        prompt_inputs.input_ids,
        prompt_inputs.attention_mask,
        output_hidden_states=True,
    )
    return prompt_inputs, outputs


def get_next_token_logits_from_outputs(outputs: Any) -> torch.Tensor:
    """Return next-token logits from an existing model output object."""

    return outputs.logits[0, -1, :].detach().to(dtype=torch.float32).cpu()


def final_token_activation_for_layer(
    outputs: Any,
    prompt_inputs: PromptInputs,
    *,
    model: Any,
    layer: int,
) -> torch.Tensor:
    """Extract the final prompt-token activation from cached hidden states."""

    if outputs.hidden_states is None:
        raise ValueError("Model forward pass did not return hidden states.")
    resolved_layer = resolve_layer_index(model, layer)
    hidden_state = outputs.hidden_states[resolved_layer + 1]
    return hidden_state[0, prompt_inputs.final_token_index, :].detach()


def build_continuation_inputs(
    prompt_inputs: PromptInputs,
    continuation_token_ids: Sequence[int],
) -> ContinuationInputs:
    """Append continuation tokens to a prompt while preserving prompt-final index."""

    if not continuation_token_ids:
        raise ValueError("Continuation must contain at least one token.")
    continuation = torch.tensor(
        [list(int(token_id) for token_id in continuation_token_ids)],
        dtype=prompt_inputs.input_ids.dtype,
        device=prompt_inputs.input_ids.device,
    )
    continuation_mask = torch.ones_like(continuation, dtype=prompt_inputs.attention_mask.dtype)
    return ContinuationInputs(
        input_ids=torch.cat([prompt_inputs.input_ids, continuation], dim=-1),
        attention_mask=torch.cat([prompt_inputs.attention_mask, continuation_mask], dim=-1),
        final_prompt_token_index=prompt_inputs.final_token_index,
    )


def _patch_forward_outputs(
    bundle: CodingPatchingBundle,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    final_token_index: int,
    layer: int,
    donor_final_token_activation: torch.Tensor,
) -> Any:
    """Run a forward pass while replacing one final-prompt-token activation."""

    resolved_layer = resolve_layer_index(bundle.model, layer)
    target_module = resolve_decoder_layer_module(bundle.model, resolved_layer)
    donor_activation = donor_final_token_activation.detach().reshape(-1)

    def _hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        if torch.is_tensor(output):
            patched_hidden_state = output.clone()
            patched_hidden_state[:, final_token_index, :] = donor_activation.to(
                device=patched_hidden_state.device,
                dtype=patched_hidden_state.dtype,
            )
            return patched_hidden_state

        if isinstance(output, tuple):
            hidden_state = output[0]
            if not torch.is_tensor(hidden_state):
                raise TypeError("Expected the decoder block output to contain hidden states.")
            patched_hidden_state = hidden_state.clone()
            patched_hidden_state[:, final_token_index, :] = donor_activation.to(
                device=patched_hidden_state.device,
                dtype=patched_hidden_state.dtype,
            )
            return (patched_hidden_state, *output[1:])

        raise TypeError(
            "Unsupported decoder block output type for activation patching: "
            f"{type(output).__name__}."
        )

    handle = target_module.register_forward_hook(_hook)
    try:
        outputs = _forward_inputs(
            bundle,
            input_ids,
            attention_mask,
            output_hidden_states=False,
        )
    finally:
        handle.remove()
    return outputs


def patch_final_token_activation(
    bundle: CodingPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    layer: int,
    donor_final_token_activation: torch.Tensor,
) -> torch.Tensor:
    """Patch the final prompt-token activation at one layer and return next-token logits."""

    outputs = _patch_forward_outputs(
        bundle,
        input_ids=prompt_inputs.input_ids,
        attention_mask=prompt_inputs.attention_mask,
        final_token_index=prompt_inputs.final_token_index,
        layer=layer,
        donor_final_token_activation=donor_final_token_activation,
    )
    return get_next_token_logits_from_outputs(outputs)


def continuation_token_ids(bundle: CodingPatchingBundle, continuation_text: str) -> tuple[int, ...]:
    """Tokenize a continuation string without adding special tokens."""

    token_ids = tuple(
        int(token_id)
        for token_id in bundle.tokenizer.encode(continuation_text, add_special_tokens=False)
    )
    if not token_ids:
        raise ValueError("Continuation text tokenized to zero tokens.")
    return token_ids


def score_continuation_sequence(
    bundle: CodingPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    continuation_token_ids: Sequence[int],
    layer: int | None = None,
    donor_final_token_activation: torch.Tensor | None = None,
) -> ContinuationScore:
    """Score a continuation sequence with teacher forcing, optionally under a patch."""

    continuation_inputs = build_continuation_inputs(prompt_inputs, continuation_token_ids)
    if layer is None:
        outputs = _forward_inputs(
            bundle,
            continuation_inputs.input_ids,
            continuation_inputs.attention_mask,
            output_hidden_states=False,
        )
    else:
        if donor_final_token_activation is None:
            raise ValueError("A donor activation is required when scoring a patched sequence.")
        outputs = _patch_forward_outputs(
            bundle,
            input_ids=continuation_inputs.input_ids,
            attention_mask=continuation_inputs.attention_mask,
            final_token_index=continuation_inputs.final_prompt_token_index,
            layer=layer,
            donor_final_token_activation=donor_final_token_activation,
        )

    prompt_length = int(prompt_inputs.input_ids.shape[-1])
    answer_length = len(continuation_token_ids)
    prediction_positions = torch.arange(
        prompt_length - 1,
        prompt_length - 1 + answer_length,
        device=outputs.logits.device,
    )
    target_ids = torch.tensor(
        list(int(token_id) for token_id in continuation_token_ids),
        dtype=torch.long,
        device=outputs.logits.device,
    )
    token_logits = outputs.logits[0, prediction_positions, :]
    token_logprobs = F.log_softmax(token_logits, dim=-1)
    gathered_logprobs = token_logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_strs = tuple(
        token_id_to_string(bundle.tokenizer, int(token_id))
        for token_id in target_ids
    )
    return ContinuationScore(
        token_ids=tuple(int(token_id) for token_id in continuation_token_ids),
        token_strs=token_strs,
        logprob_sum=float(gathered_logprobs.sum().item()),
        logprob_mean=float(gathered_logprobs.mean().item()),
    )
