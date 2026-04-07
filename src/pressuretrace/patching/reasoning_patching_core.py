"""Core prompt-only activation patching helpers for reasoning route patching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pressuretrace.behavior.reasoning_runtime import (
    STRICT_INTEGER_SYSTEM_PROMPT,
    ReasoningGenerationProfile,
    build_reasoning_messages,
    generation_profile_for_reasoning_v2,
    manual_model_load_kwargs,
    model_input_device,
)


@dataclass(frozen=True)
class ReasoningPatchingBundle:
    """Loaded model/tokenizer bundle for prompt-level reasoning patching."""

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


def load_model_and_tokenizer(
    model_name: str,
    *,
    thinking_mode: str = "off",
    trust_remote_code: bool = True,
) -> ReasoningPatchingBundle:
    """Load the Qwen3 reasoning model and tokenizer for patching."""

    profile = generation_profile_for_reasoning_v2(model_name, thinking_mode)
    if profile.backend != "manual_qwen3":
        raise ValueError("Reasoning route patching currently supports only Qwen3 models.")

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
        raise ValueError("Tokenizer does not support chat templates required for Qwen3.")
    return str(
        bundle.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bundle.profile.enable_thinking,
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

    with torch.no_grad():
        return bundle.model(
            **prompt_inputs.to_model_inputs(),
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


def patch_final_token_activation(
    bundle: ReasoningPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    layer: int,
    donor_final_token_activation: torch.Tensor,
) -> torch.Tensor:
    """Patch the final prompt-token activation at one layer and return next-token logits."""

    resolved_layer = resolve_layer_index(bundle.model, layer)
    target_module = resolve_decoder_layer_module(bundle.model, resolved_layer)
    donor_activation = donor_final_token_activation.detach().reshape(-1)

    def _hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        if torch.is_tensor(output):
            patched_hidden_state = output.clone()
            patched_hidden_state[:, prompt_inputs.final_token_index, :] = donor_activation.to(
                device=patched_hidden_state.device,
                dtype=patched_hidden_state.dtype,
            )
            return patched_hidden_state

        if isinstance(output, tuple):
            hidden_state = output[0]
            if not torch.is_tensor(hidden_state):
                raise TypeError("Expected the decoder block output to contain hidden states.")
            patched_hidden_state = hidden_state.clone()
            patched_hidden_state[:, prompt_inputs.final_token_index, :] = donor_activation.to(
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
        outputs = _forward_prompt(bundle, prompt_inputs, output_hidden_states=False)
    finally:
        handle.remove()
    return _next_token_logits_from_outputs(outputs)
