"""Core prompt-only activation patching helpers for reasoning route patching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pressuretrace.behavior.reasoning_runtime import (
    STRICT_INTEGER_SYSTEM_PROMPT,
    ReasoningGenerationProfile,
    build_reasoning_messages,
    generation_profile_for_reasoning_v2,
    is_qwen3_model,
    manual_model_load_kwargs,
    model_input_device,
)
from pressuretrace.patching.reasoning_patching_metrics import SequenceScore


@dataclass(frozen=True)
class ReasoningPatchingBundle:
    """Loaded model/tokenizer bundle for prompt-level reasoning patching."""

    model_name: str
    model: Any
    tokenizer: Any
    profile: ReasoningGenerationProfile
    input_device: torch.device
    system_prompt: str = STRICT_INTEGER_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptInputs:
    """Rendered and tokenized prompt inputs for a single reasoning episode."""

    prompt: str
    rendered_prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    final_token_index: int

    def to_model_inputs(self) -> dict[str, torch.Tensor]:
        """Return model-call kwargs for the stored prompt tensors."""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }


@dataclass(frozen=True)
class SingleTokenAnswer:
    """A prompt answer that tokenizes to exactly one token."""

    matched_text: str
    token_id: int
    token_str: str


@dataclass(frozen=True)
class ContinuationInputs:
    """Prompt inputs extended with a teacher-forced answer continuation."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    final_prompt_token_index: int

    def to_model_inputs(self) -> dict[str, torch.Tensor]:
        """Return model-call kwargs for the stored continuation tensors."""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }


def load_model_and_tokenizer(
    model_name: str,
    *,
    thinking_mode: str = "off",
    trust_remote_code: bool = True,
) -> ReasoningPatchingBundle:
    """Load the reasoning model and tokenizer bundle used for route patching."""

    profile = generation_profile_for_reasoning_v2(model_name, thinking_mode)

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
    return ReasoningPatchingBundle(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        profile=profile,
        input_device=model_input_device(model),
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


def ensure_single_token_answer(tokenizer: Any, answer: str) -> SingleTokenAnswer:
    """Validate that an answer tokenizes to exactly one token."""

    token_ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Answer '{answer}' tokenizes to {len(token_ids)} tokens, not 1.")
    token_id = int(token_ids[0])
    return SingleTokenAnswer(
        matched_text=answer,
        token_id=token_id,
        token_str=token_id_to_string(tokenizer, token_id),
    )


def _render_prompt_text(bundle: ReasoningPatchingBundle, prompt: str) -> str:
    """Render the chat-formatted prompt text used in the frozen reasoning run."""

    messages = build_reasoning_messages(prompt=prompt, system_prompt=bundle.system_prompt)
    if not hasattr(bundle.tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support chat templates required for route patching.")

    if is_qwen3_model(bundle.model_name):
        return str(
            bundle.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bundle.profile.enable_thinking,
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
        raise ValueError("Reasoning patching expects a single prompt at a time.")
    valid_tokens = int(attention_mask[0].sum().item())
    if valid_tokens <= 0:
        raise ValueError("Prompt attention mask contains no valid tokens.")
    return valid_tokens - 1


def build_model_inputs(bundle: ReasoningPatchingBundle, prompt: str) -> PromptInputs:
    """Tokenize a prompt exactly as the frozen reasoning run did."""

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


def _forward_prompt(
    bundle: ReasoningPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    output_hidden_states: bool,
) -> Any:
    """Run a prompt forward pass through the model."""

    return _forward_inputs(
        bundle,
        prompt_inputs.input_ids,
        prompt_inputs.attention_mask,
        output_hidden_states=output_hidden_states,
    )


def _forward_inputs(
    bundle: ReasoningPatchingBundle,
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


def _next_token_logits_from_outputs(outputs: Any) -> torch.Tensor:
    """Extract next-token logits as a CPU float32 tensor."""

    return outputs.logits[0, -1, :].detach().to(dtype=torch.float32).cpu()


def get_next_token_logits(
    bundle: ReasoningPatchingBundle,
    prompt_inputs: PromptInputs,
) -> torch.Tensor:
    """Return the baseline next-token logits for a prompt."""

    outputs = _forward_prompt(bundle, prompt_inputs, output_hidden_states=False)
    return _next_token_logits_from_outputs(outputs)


def get_next_token_logits_from_outputs(outputs: Any) -> torch.Tensor:
    """Return next-token logits from an existing model output object."""

    return _next_token_logits_from_outputs(outputs)


def build_continuation_inputs(
    prompt_inputs: PromptInputs,
    answer_token_ids: Sequence[int],
) -> ContinuationInputs:
    """Append answer tokens to a prompt while preserving the prompt-final token index."""

    if not answer_token_ids:
        raise ValueError("Answer continuation must contain at least one token.")
    continuation = torch.tensor(
        [list(int(token_id) for token_id in answer_token_ids)],
        dtype=prompt_inputs.input_ids.dtype,
        device=prompt_inputs.input_ids.device,
    )
    continuation_mask = torch.ones_like(continuation, dtype=prompt_inputs.attention_mask.dtype)
    return ContinuationInputs(
        input_ids=torch.cat([prompt_inputs.input_ids, continuation], dim=-1),
        attention_mask=torch.cat([prompt_inputs.attention_mask, continuation_mask], dim=-1),
        final_prompt_token_index=prompt_inputs.final_token_index,
    )


def get_prompt_hidden_states(
    bundle: ReasoningPatchingBundle,
    prompt: str,
) -> tuple[PromptInputs, Any]:
    """Tokenize a prompt and run a hidden-state forward pass."""

    prompt_inputs = build_model_inputs(bundle, prompt)
    outputs = _forward_prompt(bundle, prompt_inputs, output_hidden_states=True)
    return prompt_inputs, outputs


def get_final_token_activation(
    bundle: ReasoningPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    layer: int,
) -> torch.Tensor:
    """Extract the final prompt-token activation at a chosen decoder layer."""

    outputs = _forward_prompt(bundle, prompt_inputs, output_hidden_states=True)
    if outputs.hidden_states is None:
        raise ValueError("Model forward pass did not return hidden states.")
    resolved_layer = resolve_layer_index(bundle.model, layer)
    hidden_state = outputs.hidden_states[resolved_layer + 1]
    return hidden_state[0, prompt_inputs.final_token_index, :].detach()


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


extract_final_token_activation = get_final_token_activation


def _patch_forward_outputs(
    bundle: ReasoningPatchingBundle,
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
    bundle: ReasoningPatchingBundle,
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
    return _next_token_logits_from_outputs(outputs)


def score_answer_token_sequence(
    bundle: ReasoningPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    answer_token_ids: Sequence[int],
    layer: int | None = None,
    donor_final_token_activation: torch.Tensor | None = None,
) -> SequenceScore:
    """Score an answer sequence with teacher forcing, optionally under a patch."""

    continuation_inputs = build_continuation_inputs(prompt_inputs, answer_token_ids)
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
    answer_length = len(answer_token_ids)
    prediction_positions = torch.arange(
        prompt_length - 1,
        prompt_length - 1 + answer_length,
        device=outputs.logits.device,
    )
    target_ids = torch.tensor(
        list(int(token_id) for token_id in answer_token_ids),
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
    return SequenceScore(
        token_ids=tuple(int(token_id) for token_id in answer_token_ids),
        token_strs=token_strs,
        logprob_sum=float(gathered_logprobs.sum().item()),
        logprob_mean=float(gathered_logprobs.mean().item()),
    )
