"""Extract hidden states for the frozen PressureTrace coding paper slice."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pressuretrace.behavior.reasoning_runtime import (
    build_reasoning_messages,
    is_qwen3_model,
    manual_model_load_kwargs,
    model_input_device,
)
from pressuretrace.behavior.run_coding_paper_slice import CODING_SYSTEM_PROMPT
from pressuretrace.config import (
    CODING_PROBE_LAYERS,
    CODING_PROBE_PRESSURE_TYPES,
    CODING_PROBE_REPRESENTATIONS,
    CODING_PROBE_ROUTE_LABELS,
    CODING_V1_MODEL_NAME,
    CODING_V1_THINKING_MODE,
    coding_probe_hidden_states_path,
    coding_probe_manifest_path,
    coding_probe_results_path,
)
from pressuretrace.utils.io import append_jsonl, prepare_results_file, read_jsonl

ELIGIBLE_PRESSURE_TYPES = set(CODING_PROBE_PRESSURE_TYPES)
ELIGIBLE_ROUTE_LABELS = set(CODING_PROBE_ROUTE_LABELS)


@dataclass(frozen=True)
class CodingProbeExtractionConfig:
    """Configuration for coding hidden-state extraction."""

    manifest_path: Path
    results_path: Path
    output_path: Path
    model_name: str = CODING_V1_MODEL_NAME
    thinking_mode: str = CODING_V1_THINKING_MODE
    layers: tuple[int, ...] = CODING_PROBE_LAYERS
    representations: tuple[str, ...] = CODING_PROBE_REPRESENTATIONS
    progress_every: int = 10


def default_coding_probe_extraction_config() -> CodingProbeExtractionConfig:
    """Return the default frozen-paper-slice extraction config."""

    return CodingProbeExtractionConfig(
        manifest_path=coding_probe_manifest_path(),
        results_path=coding_probe_results_path(),
        output_path=coding_probe_hidden_states_path(),
    )


def _validate_config(config: CodingProbeExtractionConfig) -> None:
    """Reject unsupported extraction settings early."""

    unsupported_layers = set(config.layers) - set(CODING_PROBE_LAYERS)
    if unsupported_layers:
        available = ", ".join(str(layer) for layer in CODING_PROBE_LAYERS)
        unsupported = ", ".join(str(layer) for layer in sorted(unsupported_layers))
        raise ValueError(
            f"Unsupported layer selection: {unsupported}. Available layers: {available}."
        )
    unsupported_representations = set(config.representations) - set(CODING_PROBE_REPRESENTATIONS)
    if unsupported_representations:
        available = ", ".join(CODING_PROBE_REPRESENTATIONS)
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
    """Reconstruct the exact coding chat prompt used during the frozen run."""

    messages = build_reasoning_messages(prompt, CODING_SYSTEM_PROMPT)
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **manual_model_load_kwargs(),
    )
    model.eval()
    return model, tokenizer


def _summarize_model_devices(model: Any) -> str:
    """Render a compact summary of where the model lives."""

    if hasattr(model, "hf_device_map"):
        device_map = model.hf_device_map
        counts = Counter(str(device) for device in device_map.values())
        return ", ".join(f"{device}={count}" for device, count in sorted(counts.items()))
    return str(model_input_device(model))


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

    if str(result_row.get("pressure_type")) not in ELIGIBLE_PRESSURE_TYPES:
        return False
    if str(result_row.get("route_label")) not in ELIGIBLE_ROUTE_LABELS:
        return False
    if str(result_row.get("model_name")) != model_name:
        return False
    if str(result_row.get("thinking_mode")) != thinking_mode:
        return False
    return True


def select_coding_probe_rows(
    manifest_rows: Sequence[dict[str, Any]],
    result_rows: Sequence[dict[str, Any]],
    *,
    model_name: str = CODING_V1_MODEL_NAME,
    thinking_mode: str = CODING_V1_THINKING_MODE,
) -> list[dict[str, Any]]:
    """Join manifest and result rows, keeping only probe-eligible coding episodes."""

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


def coding_probe_output_counts(
    manifest_rows: Sequence[dict[str, Any]],
    result_rows: Sequence[dict[str, Any]],
    *,
    model_name: str = CODING_V1_MODEL_NAME,
    thinking_mode: str = CODING_V1_THINKING_MODE,
    layers: Sequence[int] = CODING_PROBE_LAYERS,
    representations: Sequence[str] = CODING_PROBE_REPRESENTATIONS,
) -> tuple[int, int]:
    """Return expected eligible-episode and hidden-state row counts for coding probes."""

    selected_rows = select_coding_probe_rows(
        manifest_rows,
        result_rows,
        model_name=model_name,
        thinking_mode=thinking_mode,
    )
    episode_count = len(selected_rows)
    output_row_count = episode_count * len(tuple(layers)) * len(tuple(representations))
    return episode_count, output_row_count


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

    if representation not in CODING_PROBE_REPRESENTATIONS:
        available = ", ".join(CODING_PROBE_REPRESENTATIONS)
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


def extract_coding_hidden_states(config: CodingProbeExtractionConfig) -> Path:
    """Extract hidden states for the frozen coding paper slice."""

    _validate_config(config)
    manifest_rows = [dict(row) for row in read_jsonl(config.manifest_path)]
    result_rows = [dict(row) for row in read_jsonl(config.results_path)]
    selected_rows = select_coding_probe_rows(
        manifest_rows,
        result_rows,
        model_name=config.model_name,
        thinking_mode=config.thinking_mode,
    )
    if not selected_rows:
        pressure_counts = Counter(str(row.get("pressure_type", "")) for row in result_rows)
        route_counts = Counter(str(row.get("route_label", "")) for row in result_rows)
        raise ValueError(
            "No eligible coding rows were found for hidden-state extraction. "
            f"results_path={config.results_path}; "
            f"pressure_counts={dict(pressure_counts)}; "
            f"route_counts={dict(route_counts)}"
        )

    eligible_count = len(selected_rows)
    pressure_counts = Counter(str(row["pressure_type"]) for row in selected_rows)
    label_counts = Counter(
        1 if str(row["route_label"]) == "shortcut_success" else 0 for row in selected_rows
    )

    model, tokenizer = _load_model_and_tokenizer(config.model_name)
    output_path = prepare_results_file(config.output_path)
    print(f"Eligible rows: {eligible_count}", flush=True)
    print(
        "Per-pressure counts: "
        + ", ".join(
            f"{pressure_type}={pressure_counts.get(pressure_type, 0)}"
            for pressure_type in CODING_PROBE_PRESSURE_TYPES
        ),
        flush=True,
    )
    print(
        "Per-label counts: "
        + ", ".join(f"{label}={label_counts.get(label, 0)}" for label in (0, 1)),
        flush=True,
    )
    print("Layers: " + ", ".join(str(layer) for layer in config.layers), flush=True)
    print("Representations: " + ", ".join(config.representations), flush=True)
    print("Model device placement: " + _summarize_model_devices(model), flush=True)

    rows_written = 0
    started_at = time.perf_counter()
    with torch.no_grad():
        for index, row in enumerate(selected_rows, start=1):
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
                        "base_task_id": row["base_task_id"],
                        "pressure_type": row["pressure_type"],
                        "route_label": row["route_label"],
                        "binary_label": 1 if str(row["route_label"]) == "shortcut_success" else 0,
                        "layer": layer,
                        "representation": representation,
                        "model_name": row["model_name"],
                        "prompt": row["prompt"],
                        "archetype": row["archetype"],
                        "source_family": row.get("source_family"),
                        "source_task_name": row.get("source_task_name"),
                        "hidden_state": _flatten_hidden_state(projected),
                        "metadata": {
                            **dict(row.get("metadata", {})),
                            "thinking_mode": row.get("thinking_mode", config.thinking_mode),
                            "prompt_token_count": int(model_inputs["input_ids"].shape[-1]),
                            "extractor": "coding_hidden_states",
                            "layer": layer,
                            "representation": representation,
                        },
                    }
                    append_jsonl(output_path, output_row)
                    rows_written += 1

            if (
                config.progress_every > 0
                and (index % config.progress_every == 0 or index == eligible_count)
            ):
                elapsed_minutes = (time.perf_counter() - started_at) / 60
                print(
                    f"[{index}/{eligible_count}] hidden-state extraction "
                    f"{row['task_id']} -> {rows_written} rows "
                    f"({elapsed_minutes:.1f}m elapsed)",
                    flush=True,
                )

    print(f"Hidden-state rows written: {rows_written}", flush=True)
    return output_path


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the coding hidden-state extractor."""

    parser = argparse.ArgumentParser(
        description="Extract hidden states for the frozen PressureTrace coding paper slice.",
    )
    parser.add_argument("--manifest-path", type=Path, default=coding_probe_manifest_path())
    parser.add_argument("--results-path", type=Path, default=coding_probe_results_path())
    parser.add_argument("--output-path", type=Path, default=coding_probe_hidden_states_path())
    parser.add_argument("--model-name", type=str, default=CODING_V1_MODEL_NAME)
    parser.add_argument("--thinking-mode", type=str, default=CODING_V1_THINKING_MODE)
    args = parser.parse_args(argv)
    config = CodingProbeExtractionConfig(
        manifest_path=args.manifest_path,
        results_path=args.results_path,
        output_path=args.output_path,
        model_name=args.model_name,
        thinking_mode=args.thinking_mode,
    )
    return extract_coding_hidden_states(config)


if __name__ == "__main__":
    main()
