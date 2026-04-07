"""Extract hidden states for the frozen PressureTrace reasoning paper slice."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pressuretrace.behavior.reasoning_runtime import (
    STRICT_INTEGER_SYSTEM_PROMPT,
    build_reasoning_messages,
    is_qwen3_model,
    manual_model_load_kwargs,
    model_input_device,
)
from pressuretrace.config import (
    REASONING_PROBE_FAMILY,
    REASONING_PROBE_LAYERS,
    REASONING_PROBE_PRESSURE_TYPES,
    REASONING_PROBE_REPRESENTATIONS,
    REASONING_PROBE_ROUTE_LABELS,
    REASONING_V2_MODEL_NAME,
    REASONING_V2_THINKING_MODE,
    reasoning_probe_hidden_states_path,
    reasoning_probe_manifest_path,
    reasoning_probe_results_path,
)
from pressuretrace.utils.io import append_jsonl, prepare_results_file, read_jsonl

ELIGIBLE_PRESSURE_TYPES = set(REASONING_PROBE_PRESSURE_TYPES)
ELIGIBLE_ROUTE_LABELS = set(REASONING_PROBE_ROUTE_LABELS)


@dataclass(frozen=True)
class ReasoningProbeExtractionConfig:
    """Configuration for reasoning hidden-state extraction."""

    manifest_path: Path
    results_path: Path
    output_path: Path
    model_name: str = REASONING_V2_MODEL_NAME
    thinking_mode: str = REASONING_V2_THINKING_MODE
    layers: tuple[int, ...] = REASONING_PROBE_LAYERS
    representations: tuple[str, ...] = REASONING_PROBE_REPRESENTATIONS


def default_reasoning_probe_extraction_config() -> ReasoningProbeExtractionConfig:
    """Return the default frozen-paper-slice extraction config."""

    return ReasoningProbeExtractionConfig(
        manifest_path=reasoning_probe_manifest_path(),
        results_path=reasoning_probe_results_path(),
        output_path=reasoning_probe_hidden_states_path(),
    )


def _validate_config(config: ReasoningProbeExtractionConfig) -> None:
    """Reject unsupported extraction settings early."""

    unsupported_layers = set(config.layers) - set(REASONING_PROBE_LAYERS)
    if unsupported_layers:
        available = ", ".join(str(layer) for layer in REASONING_PROBE_LAYERS)
        unsupported = ", ".join(str(layer) for layer in sorted(unsupported_layers))
        raise ValueError(
            f"Unsupported layer selection: {unsupported}. Available layers: {available}."
        )
    unsupported_representations = set(config.representations) - set(REASONING_PROBE_REPRESENTATIONS)
    if unsupported_representations:
        available = ", ".join(REASONING_PROBE_REPRESENTATIONS)
        unsupported = ", ".join(sorted(unsupported_representations))
        raise ValueError(
            f"Unsupported hidden-state representation(s): {unsupported}. Available: {available}."
        )


def _build_chat_prompt(
    tokenizer: Any,
    prompt: str,
    model_name: str,
    thinking_mode: str,
) -> str:
    """Reconstruct the exact chat prompt used during the frozen run."""

    messages = build_reasoning_messages(prompt, STRICT_INTEGER_SYSTEM_PROMPT)
    if is_qwen3_model(model_name):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_mode in {"default", "on"},
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    """Load the model/tokenizer pair used to reproduce prompt hidden states."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, **manual_model_load_kwargs())
    model.eval()
    return model, tokenizer


def _select_prompt(manifest_row: dict[str, Any], result_row: dict[str, Any]) -> str:
    """Choose the prompt text from the joined manifest/result row pair."""

    manifest_prompt = str(manifest_row.get("prompt", ""))
    result_prompt = str(result_row.get("prompt", ""))
    if manifest_prompt and result_prompt and manifest_prompt != result_prompt:
        raise ValueError(f"Prompt mismatch for task_id={manifest_row.get('task_id')}.")
    return manifest_prompt or result_prompt


def _row_is_eligible(
    result_row: dict[str, Any],
    *,
    model_name: str,
    thinking_mode: str,
) -> bool:
    """Return whether a frozen result row should be probed."""

    if str(result_row.get("family")) != REASONING_PROBE_FAMILY:
        return False
    if str(result_row.get("pressure_type")) not in ELIGIBLE_PRESSURE_TYPES:
        return False
    if str(result_row.get("route_label")) not in ELIGIBLE_ROUTE_LABELS:
        return False
    if str(result_row.get("model_name")) != model_name:
        return False
    if str(result_row.get("thinking_mode")) != thinking_mode:
        return False
    return True


def select_reasoning_probe_rows(
    manifest_rows: Sequence[dict[str, Any]],
    result_rows: Sequence[dict[str, Any]],
    *,
    model_name: str = REASONING_V2_MODEL_NAME,
    thinking_mode: str = REASONING_V2_THINKING_MODE,
) -> list[dict[str, Any]]:
    """Join manifest and results rows, keeping only probe-eligible pressure rows."""

    manifest_by_task_id = {str(row.get("task_id")): dict(row) for row in manifest_rows}
    selected_rows: list[dict[str, Any]] = []
    missing_manifest_task_ids: list[str] = []

    for result_row in result_rows:
        if not _row_is_eligible(
            result_row,
            model_name=model_name,
            thinking_mode=thinking_mode,
        ):
            continue
        task_id = str(result_row.get("task_id"))
        manifest_row = manifest_by_task_id.get(task_id)
        if manifest_row is None:
            missing_manifest_task_ids.append(task_id)
            continue

        prompt = _select_prompt(manifest_row, dict(result_row))
        merged_metadata = dict(manifest_row.get("metadata", {}))
        merged_metadata.update(dict(result_row.get("metadata", {})))
        merged_row = dict(manifest_row)
        merged_row.update(dict(result_row))
        merged_row["prompt"] = prompt
        merged_row["metadata"] = merged_metadata
        selected_rows.append(merged_row)

    if missing_manifest_task_ids:
        sample = ", ".join(sorted(missing_manifest_task_ids)[:5])
        raise ValueError(
            f"Selected result rows were missing manifest entries. Sample task_ids: {sample}."
        )

    return selected_rows


def _normalize_hidden_state_tensor(layer_hidden_state: torch.Tensor) -> torch.Tensor:
    """Reduce a per-layer hidden-state tensor to the single-example sequence view."""

    if layer_hidden_state.ndim == 3:
        return layer_hidden_state[0]
    if layer_hidden_state.ndim == 2:
        return layer_hidden_state
    raise ValueError(
        f"Expected 2D or 3D hidden-state tensor, got shape {tuple(layer_hidden_state.shape)}."
    )


def project_hidden_state_vector(
    layer_hidden_state: torch.Tensor,
    representation: str,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Project a prompt-layer tensor into a fixed vector representation."""

    if representation not in REASONING_PROBE_REPRESENTATIONS:
        available = ", ".join(REASONING_PROBE_REPRESENTATIONS)
        raise ValueError(f"Unknown representation '{representation}'. Available: {available}.")

    sequence_hidden_state = _normalize_hidden_state_tensor(layer_hidden_state)
    if representation == "last_token":
        if attention_mask is not None and attention_mask.ndim >= 2:
            valid_length = int(attention_mask[0].sum().item())
            if valid_length <= 0:
                raise ValueError("Attention mask contains no valid tokens.")
            return sequence_hidden_state[valid_length - 1]
        return sequence_hidden_state[-1]

    if attention_mask is not None and attention_mask.ndim >= 2:
        token_mask = attention_mask[0].to(dtype=torch.bool)
        valid_tokens = sequence_hidden_state[token_mask]
        if valid_tokens.numel() == 0:
            raise ValueError("Attention mask contains no valid tokens.")
        return valid_tokens.mean(dim=0)
    return sequence_hidden_state.mean(dim=0)


def _flatten_hidden_state(hidden_state: torch.Tensor) -> list[float]:
    """Convert a tensor representation into a JSON-friendly list of floats."""

    return hidden_state.detach().to(dtype=torch.float32).cpu().tolist()


def extract_reasoning_hidden_states(
    config: ReasoningProbeExtractionConfig,
) -> Path:
    """Extract hidden states for the frozen reasoning paper slice."""

    _validate_config(config)
    manifest_rows = [dict(row) for row in read_jsonl(config.manifest_path)]
    result_rows = [dict(row) for row in read_jsonl(config.results_path)]
    selected_rows = select_reasoning_probe_rows(
        manifest_rows,
        result_rows,
        model_name=config.model_name,
        thinking_mode=config.thinking_mode,
    )
    if not selected_rows:
        raise ValueError("No eligible reasoning rows were found for hidden-state extraction.")

    eligible_count = len(selected_rows)
    pressure_counts = Counter(str(row["pressure_type"]) for row in selected_rows)
    label_counts = Counter(
        1 if str(row["route_label"]) == "shortcut_followed" else 0 for row in selected_rows
    )

    output_path = prepare_results_file(config.output_path)
    model, tokenizer = _load_model_and_tokenizer(config.model_name)

    rows_written = 0
    with torch.no_grad():
        for row in selected_rows:
            prompt_text = _build_chat_prompt(
                tokenizer=tokenizer,
                prompt=str(row["prompt"]),
                model_name=config.model_name,
                thinking_mode=config.thinking_mode,
            )
            model_inputs = tokenizer([prompt_text], return_tensors="pt")
            input_device = model_input_device(model)
            model_inputs = {name: tensor.to(input_device) for name, tensor in model_inputs.items()}

            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                task_id = row.get("task_id")
                raise ValueError(f"Model did not return hidden states for task_id={task_id}.")

            attention_mask = model_inputs.get("attention_mask")
            for layer in config.layers:
                if layer < -len(hidden_states) or layer >= len(hidden_states):
                    hidden_state_count = len(hidden_states)
                    raise ValueError(
                        "Requested layer "
                        f"{layer} is out of range for hidden_states length {hidden_state_count}."
                    )
                layer_hidden_state = hidden_states[layer]
                for representation in config.representations:
                    projected = project_hidden_state_vector(
                        layer_hidden_state=layer_hidden_state,
                        representation=representation,
                        attention_mask=attention_mask,
                    )
                    output_row = {
                        "task_id": row["task_id"],
                        "base_task_id": row["metadata"]["base_task_id"],
                        "source_id": row["source_id"],
                        "family": row["family"],
                        "pressure_type": row["pressure_type"],
                        "route_label": row["route_label"],
                        "binary_label": 1 if str(row["route_label"]) == "shortcut_followed" else 0,
                        "layer": layer,
                        "representation": representation,
                        "model_name": row["model_name"],
                        "prompt": row["prompt"],
                        "gold_answer": row["gold_answer"],
                        "shortcut_answer": row["shortcut_answer"],
                        "hidden_state": _flatten_hidden_state(projected),
                        "metadata": {
                            **dict(row.get("metadata", {})),
                            "thinking_mode": row.get("thinking_mode", config.thinking_mode),
                            "prompt_token_count": int(model_inputs["input_ids"].shape[-1]),
                            "extractor": "reasoning_hidden_states",
                            "layer": layer,
                            "representation": representation,
                        },
                    }
                    append_jsonl(output_path, output_row)
                    rows_written += 1

    print(f"Eligible rows: {eligible_count}")
    print(
        "Per-pressure counts: "
        + ", ".join(
            f"{pressure_type}={pressure_counts.get(pressure_type, 0)}"
            for pressure_type in REASONING_PROBE_PRESSURE_TYPES
        )
    )
    print(
        "Per-label counts: "
        + ", ".join(f"{label}={label_counts.get(label, 0)}" for label in (0, 1))
    )
    print(f"Hidden-state rows written: {rows_written}")

    return output_path


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the reasoning hidden-state extractor."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Extract hidden states for the frozen PressureTrace reasoning paper slice.",
    )
    parser.add_argument("--manifest-path", type=Path, default=reasoning_probe_manifest_path())
    parser.add_argument("--results-path", type=Path, default=reasoning_probe_results_path())
    parser.add_argument("--output-path", type=Path, default=reasoning_probe_hidden_states_path())
    parser.add_argument("--model-name", type=str, default=REASONING_V2_MODEL_NAME)
    parser.add_argument("--thinking-mode", type=str, default=REASONING_V2_THINKING_MODE)
    args = parser.parse_args(argv)
    config = ReasoningProbeExtractionConfig(
        manifest_path=args.manifest_path,
        results_path=args.results_path,
        output_path=args.output_path,
        model_name=args.model_name,
        thinking_mode=args.thinking_mode,
    )
    return extract_reasoning_hidden_states(config)


if __name__ == "__main__":
    main()
