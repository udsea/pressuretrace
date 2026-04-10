"""Core prompt and generation-time activation patching helpers for coding."""

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


@dataclass(frozen=True)
class GenerationStepTrace:
    """One greedy generation step with its aligned decoder activation."""

    step_index: int
    token_id: int
    token_str: str
    top1_token_id: int
    top1_token_str: str
    activation: torch.Tensor


@dataclass(frozen=True)
class GenerationTrace:
    """Greedy generation trace with one cached activation snapshot per step."""

    generated_token_ids: tuple[int, ...]
    generated_text: str
    step_traces: tuple[GenerationStepTrace, ...]


@dataclass(frozen=True)
class StepPatchDiagnostic:
    """Per-step token diagnostic for one patched generation step."""

    generation_step: int
    baseline_top1_token_id: int
    baseline_top1_token_str: str
    patched_top1_token_id: int
    patched_top1_token_str: str


@dataclass(frozen=True)
class TokenCandidate:
    """One token candidate from a next-token distribution."""

    rank: int
    token_id: int
    token_str: str
    logit: float
    probability: float


@dataclass(frozen=True)
class GenerationStepPatchDebug:
    """Detailed hidden/logit diagnostics for one patched generation step."""

    generation_step: int
    prefix_token_ids: tuple[int, ...]
    baseline_top1_token_id: int
    baseline_top1_token_str: str
    patched_top1_token_id: int
    patched_top1_token_str: str
    top1_changed: bool
    hidden_delta_l2: float
    hidden_delta_max_abs: float
    donor_target_hidden_delta_l2: float
    donor_target_hidden_delta_max_abs: float
    logit_delta_l2: float
    logit_delta_max_abs: float
    baseline_top_tokens: tuple[TokenCandidate, ...]
    patched_top_tokens: tuple[TokenCandidate, ...]


@dataclass(frozen=True)
class GenerationPatchResult:
    """One greedy generation run with optional early-step activation patches."""

    generated_token_ids: tuple[int, ...]
    generated_text: str
    step_diagnostics: tuple[StepPatchDiagnostic, ...]


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


def _activation_for_context_step(outputs: Any, *, model: Any, layer: int) -> torch.Tensor:
    """Return the final-context-token activation for one forward pass."""

    if outputs.hidden_states is None:
        raise ValueError("Model forward pass did not return hidden states.")
    resolved_layer = resolve_layer_index(model, layer)
    hidden_state = outputs.hidden_states[resolved_layer + 1]
    return hidden_state[0, -1, :].detach()


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


def build_generation_step_inputs(
    prompt_inputs: PromptInputs,
    prefix_token_ids: Sequence[int],
) -> ContinuationInputs:
    """Build the autoregressive context used to predict a chosen generation step."""

    if not prefix_token_ids:
        return ContinuationInputs(
            input_ids=prompt_inputs.input_ids,
            attention_mask=prompt_inputs.attention_mask,
            final_prompt_token_index=prompt_inputs.final_token_index,
        )
    return build_continuation_inputs(prompt_inputs, prefix_token_ids)


def resolve_generation_step_window(
    window: int | Sequence[int] | str,
) -> tuple[int, ...]:
    """Resolve a coding-generation patch window into zero-based step indices.

    Supported labels are intentionally simple and coding-native:
    - ``gen_1`` patches step 0 only
    - ``gen_1_3`` patches steps 0, 1, 2
    - ``gen_1_5`` patches steps 0, 1, 2, 3, 4
    """

    if isinstance(window, int):
        if window <= 0:
            raise ValueError("Generation window size must be positive.")
        return tuple(range(window))

    if isinstance(window, str):
        normalized = window.strip().lower().replace("-", "_")
        aliases = {
            "gen_1": (0,),
            "gen_1_3": (0, 1, 2),
            "gen_1_5": (0, 1, 2, 3, 4),
        }
        if normalized in aliases:
            return aliases[normalized]
        if normalized.startswith("gen_"):
            suffix = normalized.removeprefix("gen_")
            parts = [part for part in suffix.split("_") if part]
            if not parts:
                raise ValueError(f"Unsupported generation window label: {window!r}")
            try:
                upper_bound = int(parts[-1])
            except ValueError as exc:  # pragma: no cover - defensive parsing
                raise ValueError(f"Unsupported generation window label: {window!r}") from exc
            if upper_bound <= 0:
                raise ValueError(f"Generation window upper bound must be positive: {window!r}")
            return tuple(range(upper_bound))
        raise ValueError(f"Unsupported generation window label: {window!r}")

    step_indices = tuple(int(step) for step in window)
    if not step_indices:
        raise ValueError("Generation window must contain at least one step.")
    if any(step < 0 for step in step_indices):
        raise ValueError("Generation window step indices must be non-negative.")
    return step_indices


def generation_step_window_label(window: int | Sequence[int] | str) -> str:
    """Return a canonical label for a generation-step patch window."""

    step_indices = resolve_generation_step_window(window)
    if step_indices == (0,):
        return "gen_1"
    if step_indices == (0, 1, 2):
        return "gen_1_3"
    if step_indices == (0, 1, 2, 3, 4):
        return "gen_1_5"
    return "gen_" + "_".join(str(step + 1) for step in step_indices)


def generation_trace_activation_by_step(trace: GenerationTrace) -> dict[int, torch.Tensor]:
    """Return the aligned decoder activations from a generation trace."""

    return {step_trace.step_index: step_trace.activation for step_trace in trace.step_traces}


def generation_trace_activation_at_step(trace: GenerationTrace, step_index: int) -> torch.Tensor:
    """Return the cached activation for one generation step."""

    for step_trace in trace.step_traces:
        if step_trace.step_index == step_index:
            return step_trace.activation
    raise KeyError(f"Generation trace does not contain step {step_index}.")


def _append_token(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one generated token to the current autoregressive context."""

    token = torch.tensor(
        [[int(token_id)]],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    token_mask = torch.ones_like(token, dtype=attention_mask.dtype)
    return (
        torch.cat([input_ids, token], dim=-1),
        torch.cat([attention_mask, token_mask], dim=-1),
    )


def _eos_token_ids(bundle: CodingPatchingBundle) -> set[int]:
    """Return the tokenizer eos ids as a stable integer set."""

    raw_eos = getattr(bundle.tokenizer, "eos_token_id", None)
    if raw_eos is None:
        return set()
    if isinstance(raw_eos, Sequence) and not isinstance(raw_eos, (str, bytes)):
        return {int(token_id) for token_id in raw_eos}
    return {int(raw_eos)}


def decode_generated_tokens(
    bundle: CodingPatchingBundle,
    token_ids: Sequence[int],
) -> str:
    """Decode generated token ids into the coding completion text."""

    return str(
        bundle.tokenizer.decode(
            list(int(token_id) for token_id in token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
    )
    ).strip()


def capture_greedy_generation_trace(
    bundle: CodingPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    layer: int,
    max_new_tokens: int,
) -> GenerationTrace:
    """Greedily generate and cache donor activations aligned by generation step.

    Step index 0 corresponds to the prompt-to-generation boundary, i.e. the final
    prompt token used to predict the first generated code token.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    input_ids = prompt_inputs.input_ids
    attention_mask = prompt_inputs.attention_mask
    eos_token_ids = _eos_token_ids(bundle)
    generated_token_ids: list[int] = []
    step_traces: list[GenerationStepTrace] = []

    for generation_step in range(max_new_tokens):
        outputs = _forward_inputs(
            bundle,
            input_ids,
            attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits[0, -1, :]
        next_token_id = int(torch.argmax(logits).item())
        activation = _activation_for_context_step(outputs, model=bundle.model, layer=layer)
        generated_token_ids.append(next_token_id)
        step_traces.append(
            GenerationStepTrace(
                step_index=generation_step,
                token_id=next_token_id,
                token_str=token_id_to_string(bundle.tokenizer, next_token_id),
                top1_token_id=next_token_id,
                top1_token_str=token_id_to_string(bundle.tokenizer, next_token_id),
                activation=activation.cpu(),
            )
        )
        input_ids, attention_mask = _append_token(input_ids, attention_mask, next_token_id)
        if next_token_id in eos_token_ids:
            break

    return GenerationTrace(
        generated_token_ids=tuple(generated_token_ids),
        generated_text=decode_generated_tokens(bundle, generated_token_ids),
        step_traces=tuple(step_traces),
    )


def teacher_forced_last_token_activation(
    bundle: CodingPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    continuation_token_ids: Sequence[int],
    layer: int,
) -> torch.Tensor:
    """Return the hidden state of the final teacher-forced continuation token."""

    continuation_inputs = build_continuation_inputs(prompt_inputs, continuation_token_ids)
    outputs = _forward_inputs(
        bundle,
        continuation_inputs.input_ids,
        continuation_inputs.attention_mask,
        output_hidden_states=True,
    )
    if outputs.hidden_states is None:
        raise ValueError("Model forward pass did not return hidden states.")
    resolved_layer = resolve_layer_index(bundle.model, layer)
    hidden_state = outputs.hidden_states[resolved_layer + 1]
    final_index = int(continuation_inputs.attention_mask[0].sum().item()) - 1
    return hidden_state[0, final_index, :].detach()


def _patch_forward_outputs(
    bundle: CodingPatchingBundle,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    final_token_index: int,
    layer: int,
    donor_final_token_activation: torch.Tensor,
    output_hidden_states: bool,
) -> Any:
    """Run a forward pass while replacing one token activation at a chosen index."""

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
            output_hidden_states=output_hidden_states,
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
        output_hidden_states=False,
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
            output_hidden_states=False,
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


def greedy_generate_with_step_window_patch(
    bundle: CodingPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    layer: int,
    donor_activations_by_step: dict[int, torch.Tensor],
    max_new_tokens: int,
) -> GenerationPatchResult:
    """Greedily generate a completion, patching one or more aligned steps.

    Step index 0 corresponds to the prompt-to-generation boundary. Later indices
    correspond to the first, second, third, ... generated code tokens.
    """

    if not donor_activations_by_step:
        raise ValueError("At least one patch step is required.")
    if any(step < 0 for step in donor_activations_by_step):
        raise ValueError("Patch steps must be non-negative.")

    input_ids = prompt_inputs.input_ids
    attention_mask = prompt_inputs.attention_mask
    eos_token_ids = _eos_token_ids(bundle)
    generated_token_ids: list[int] = []
    step_diagnostics: list[StepPatchDiagnostic] = []

    for generation_step in range(max_new_tokens):
        outputs = _forward_inputs(
            bundle,
            input_ids,
            attention_mask,
            output_hidden_states=False,
        )
        baseline_logits = outputs.logits[0, -1, :]
        next_token_logits = baseline_logits

        if generation_step in donor_activations_by_step:
            patched_outputs = _patch_forward_outputs(
                bundle,
                input_ids=input_ids,
                attention_mask=attention_mask,
                final_token_index=int(attention_mask[0].sum().item()) - 1,
                layer=layer,
                donor_final_token_activation=donor_activations_by_step[generation_step],
                output_hidden_states=False,
            )
            next_token_logits = patched_outputs.logits[0, -1, :]
            baseline_top1_token_id = int(torch.argmax(baseline_logits).item())
            patched_top1_token_id = int(torch.argmax(next_token_logits).item())
            step_diagnostics.append(
                StepPatchDiagnostic(
                    generation_step=generation_step,
                    baseline_top1_token_id=baseline_top1_token_id,
                    baseline_top1_token_str=token_id_to_string(
                        bundle.tokenizer,
                        baseline_top1_token_id,
                    ),
                    patched_top1_token_id=patched_top1_token_id,
                    patched_top1_token_str=token_id_to_string(
                        bundle.tokenizer,
                        patched_top1_token_id,
                    ),
                )
            )

        next_token_id = int(torch.argmax(next_token_logits).item())
        generated_token_ids.append(next_token_id)
        input_ids, attention_mask = _append_token(input_ids, attention_mask, next_token_id)
        if next_token_id in eos_token_ids:
            break

    return GenerationPatchResult(
        generated_token_ids=tuple(generated_token_ids),
        generated_text=decode_generated_tokens(bundle, generated_token_ids),
        step_diagnostics=tuple(step_diagnostics),
    )


def _top_k_token_candidates(
    bundle: CodingPatchingBundle,
    logits: torch.Tensor,
    *,
    k: int,
) -> tuple[TokenCandidate, ...]:
    """Render a compact top-k view of a next-token distribution."""

    if k <= 0:
        raise ValueError("k must be positive.")
    probabilities = torch.softmax(logits.detach().to(dtype=torch.float32), dim=-1)
    top_probabilities, top_indices = torch.topk(probabilities, k=min(k, probabilities.shape[-1]))
    candidates: list[TokenCandidate] = []
    for rank, (probability, token_id_tensor) in enumerate(
        zip(top_probabilities.tolist(), top_indices.tolist(), strict=False),
        start=1,
    ):
        token_id = int(token_id_tensor)
        candidates.append(
            TokenCandidate(
                rank=rank,
                token_id=token_id,
                token_str=token_id_to_string(bundle.tokenizer, token_id),
                logit=float(logits[token_id].item()),
                probability=float(probability),
            )
        )
    return tuple(candidates)


def debug_generation_step_patch(
    bundle: CodingPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    target_trace: GenerationTrace,
    donor_trace: GenerationTrace,
    generation_step: int,
    layer: int,
    top_k: int = 10,
) -> GenerationStepPatchDebug:
    """Inspect one generation-step patch with hidden and logit delta diagnostics."""

    if generation_step < 0:
        raise ValueError("generation_step must be non-negative.")
    donor_activation = generation_trace_activation_at_step(donor_trace, generation_step)
    prefix_token_ids = target_trace.generated_token_ids[:generation_step]
    step_inputs = build_generation_step_inputs(prompt_inputs, prefix_token_ids)
    final_token_index = int(step_inputs.attention_mask[0].sum().item()) - 1

    baseline_outputs = _forward_inputs(
        bundle,
        step_inputs.input_ids,
        step_inputs.attention_mask,
        output_hidden_states=True,
    )
    patched_outputs = _patch_forward_outputs(
        bundle,
        input_ids=step_inputs.input_ids,
        attention_mask=step_inputs.attention_mask,
        final_token_index=final_token_index,
        layer=layer,
        donor_final_token_activation=donor_activation,
        output_hidden_states=True,
    )

    baseline_logits = baseline_outputs.logits[0, -1, :].detach().to(dtype=torch.float32).cpu()
    patched_logits = patched_outputs.logits[0, -1, :].detach().to(dtype=torch.float32).cpu()
    baseline_activation = _activation_for_context_step(
        baseline_outputs,
        model=bundle.model,
        layer=layer,
    ).detach().to(dtype=torch.float32).cpu()
    patched_activation = _activation_for_context_step(
        patched_outputs,
        model=bundle.model,
        layer=layer,
    ).detach().to(dtype=torch.float32).cpu()
    donor_activation_cpu = donor_activation.detach().to(dtype=torch.float32).cpu()

    hidden_delta = patched_activation - baseline_activation
    donor_target_delta = donor_activation_cpu - baseline_activation
    logit_delta = patched_logits - baseline_logits

    baseline_top1_token_id = int(torch.argmax(baseline_logits).item())
    patched_top1_token_id = int(torch.argmax(patched_logits).item())
    return GenerationStepPatchDebug(
        generation_step=generation_step,
        prefix_token_ids=tuple(int(token_id) for token_id in prefix_token_ids),
        baseline_top1_token_id=baseline_top1_token_id,
        baseline_top1_token_str=token_id_to_string(bundle.tokenizer, baseline_top1_token_id),
        patched_top1_token_id=patched_top1_token_id,
        patched_top1_token_str=token_id_to_string(bundle.tokenizer, patched_top1_token_id),
        top1_changed=baseline_top1_token_id != patched_top1_token_id,
        hidden_delta_l2=float(torch.linalg.vector_norm(hidden_delta).item()),
        hidden_delta_max_abs=float(torch.max(torch.abs(hidden_delta)).item()),
        donor_target_hidden_delta_l2=float(torch.linalg.vector_norm(donor_target_delta).item()),
        donor_target_hidden_delta_max_abs=float(torch.max(torch.abs(donor_target_delta)).item()),
        logit_delta_l2=float(torch.linalg.vector_norm(logit_delta).item()),
        logit_delta_max_abs=float(torch.max(torch.abs(logit_delta)).item()),
        baseline_top_tokens=_top_k_token_candidates(bundle, baseline_logits, k=top_k),
        patched_top_tokens=_top_k_token_candidates(bundle, patched_logits, k=top_k),
    )


def greedy_generate_with_generation_window_patch(
    bundle: CodingPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    layer: int,
    patch_window: int | Sequence[int] | str,
    donor_trace: GenerationTrace,
    max_new_tokens: int,
) -> GenerationPatchResult:
    """Greedily generate with a named or explicit early-generation patch window."""

    patch_steps = resolve_generation_step_window(patch_window)
    donor_activations_by_step = generation_trace_activation_by_step(donor_trace)
    missing_steps = [step for step in patch_steps if step not in donor_activations_by_step]
    if missing_steps:
        raise ValueError(
            "Donor trace does not contain all requested generation steps: "
            f"{missing_steps}."
        )
    return greedy_generate_with_step_window_patch(
        bundle,
        prompt_inputs,
        layer=layer,
        donor_activations_by_step={
            step: donor_activations_by_step[step]
            for step in patch_steps
        },
        max_new_tokens=max_new_tokens,
    )


def greedy_generate_with_step_patch(
    bundle: CodingPatchingBundle,
    prompt_inputs: PromptInputs,
    *,
    layer: int,
    patch_step: int,
    donor_token_activation: torch.Tensor,
    max_new_tokens: int,
) -> GenerationPatchResult:
    """Backward-compatible wrapper for single-step patching."""

    return greedy_generate_with_step_window_patch(
        bundle,
        prompt_inputs,
        layer=layer,
        donor_activations_by_step={patch_step: donor_token_activation},
        max_new_tokens=max_new_tokens,
    )
