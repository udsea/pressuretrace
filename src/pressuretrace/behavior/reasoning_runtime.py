"""Shared runtime helpers for reasoning benchmark runners."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from pressuretrace.paths import results_dir
from pressuretrace.utils.io import ensure_directory

SYSTEM_PROMPT = (
    "You solve grade-school math word problems. Answer with just the final number."
)
ThinkingMode = Literal["default", "on", "off"]


@dataclass(frozen=True)
class ReasoningGenerationProfile:
    """Model-specific decoding settings shared across reasoning runners."""

    backend: Literal["pipeline_chat", "manual_qwen3"]
    do_sample: bool
    max_new_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    enable_thinking: bool | None = None


@dataclass(frozen=True)
class ManualReasoningGenerator:
    """Loaded model bundle for manual generation backends."""

    model: Any
    tokenizer: Any
    profile: ReasoningGenerationProfile


def slugify_model_name(model_name: str) -> str:
    """Convert a model identifier into a filename-safe slug."""

    return re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()


def default_results_path(
    prefix: str,
    split: str,
    pressure_type: str,
    model_name: str,
    thinking_mode: str | None = None,
) -> Path:
    """Build a stable default output path for reasoning pilot runs."""

    model_slug = slugify_model_name(model_name)
    if thinking_mode is None:
        filename = f"{prefix}_{split}_{pressure_type}_{model_slug}.jsonl"
    else:
        filename = f"{prefix}_{split}_{pressure_type}_{thinking_mode}_{model_slug}.jsonl"
    return results_dir() / filename


def count_rows_by_pressure_type(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count selected benchmark rows by pressure type."""

    counts: Counter[str] = Counter(str(row["pressure_type"]) for row in rows)
    ordered_pressure_types = sorted(counts, key=lambda value: (value != "control", value))
    return {pressure_type: counts[pressure_type] for pressure_type in ordered_pressure_types}


def count_base_tasks(rows: list[dict[str, Any]]) -> int:
    """Count distinct latent base tasks represented in the selected rows."""

    return len({str(row["metadata"]["base_task_id"]) for row in rows})


def prepare_results_file(path: Path) -> Path:
    """Create or truncate the output JSONL file before streaming rows."""

    ensure_directory(path.parent)
    path.write_bytes(b"")
    return path


def filter_manifest_rows(
    rows: list[dict[str, Any]],
    pressure_type: str,
    include_control: bool,
) -> list[dict[str, Any]]:
    """Select one pressure condition, optionally with the control rows."""

    if pressure_type == "all":
        selected_pressure_types = {
            str(row["pressure_type"]) for row in rows if str(row["pressure_type"]) != "control"
        }
    else:
        selected_pressure_types = {pressure_type}

    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        row_pressure_type = str(row["pressure_type"])
        if row_pressure_type == "control":
            if include_control:
                filtered_rows.append(row)
            continue
        if row_pressure_type in selected_pressure_types:
            filtered_rows.append(row)
    return filtered_rows


def is_qwen3_model(model_name: str) -> bool:
    """Return whether the model should use the dedicated Qwen3 path."""

    return model_name.startswith("Qwen/Qwen3-")


def strip_qwen3_thinking_content(text: str) -> str:
    """Remove Qwen3 thinking blocks, keeping only the final visible answer."""

    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _build_generation_config(
    tokenizer: Any,
    eos_token_id: int | list[int] | None,
    profile: ReasoningGenerationProfile,
) -> GenerationConfig:
    """Create a generation config without overlapping per-call kwargs."""

    generation_config = GenerationConfig(
        do_sample=profile.do_sample,
        max_new_tokens=profile.max_new_tokens,
    )
    generation_config.max_length = None

    if tokenizer is not None and tokenizer.pad_token_id is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
    if eos_token_id is not None:
        generation_config.eos_token_id = eos_token_id
    if profile.temperature is not None:
        generation_config.temperature = profile.temperature
    if profile.top_p is not None:
        generation_config.top_p = profile.top_p
    if profile.top_k is not None:
        generation_config.top_k = profile.top_k
    if profile.min_p is not None:
        generation_config.min_p = profile.min_p

    return generation_config


def _pipeline_load_kwargs() -> dict[str, Any]:
    """Choose conservative pipeline load settings for the current device."""

    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if torch.backends.mps.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        return {"device": torch.device("mps"), "model_kwargs": model_kwargs}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        return {"device_map": "auto", "model_kwargs": model_kwargs}
    model_kwargs["torch_dtype"] = torch.float32
    return {"device": torch.device("cpu"), "model_kwargs": model_kwargs}


def _manual_model_load_kwargs() -> dict[str, Any]:
    """Choose load settings for direct AutoModel generation paths."""

    load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if torch.backends.mps.is_available():
        load_kwargs["torch_dtype"] = torch.float16
        return load_kwargs
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"
        return load_kwargs
    load_kwargs["torch_dtype"] = torch.float32
    return load_kwargs


def _model_input_device(model: Any) -> torch.device:
    """Select a device for tokenizer outputs before generation."""

    if hasattr(model, "hf_device_map"):
        for raw_device in model.hf_device_map.values():
            if isinstance(raw_device, int):
                return torch.device(f"cuda:{raw_device}")
            if isinstance(raw_device, str) and raw_device not in {"cpu", "disk"}:
                return torch.device(raw_device)
    return next(model.parameters()).device


@lru_cache(maxsize=16)
def load_reasoning_generator(
    model_name: str,
    profile: ReasoningGenerationProfile,
) -> Any:
    """Load and cache a text-generation backend for reasoning pilots."""

    if profile.backend == "manual_qwen3":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **_manual_model_load_kwargs(),
        )
        return ManualReasoningGenerator(
            model=model,
            tokenizer=tokenizer,
            profile=profile,
        )

    generator = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=model_name,
        **_pipeline_load_kwargs(),
    )
    pipeline_tokenizer: Any = generator.tokenizer
    if (
        pipeline_tokenizer is not None
        and pipeline_tokenizer.pad_token_id is None
        and generator.model.config.eos_token_id is not None
    ):
        pipeline_tokenizer.pad_token_id = generator.model.config.eos_token_id

    generator.generation_config = _build_generation_config(
        tokenizer=pipeline_tokenizer,
        eos_token_id=generator.model.config.eos_token_id,
        profile=profile,
    )
    return generator


def infer_reasoning_response(
    prompt: str,
    model_name: str,
    profile: ReasoningGenerationProfile,
    *,
    strip_qwen3_thinking: bool,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Run a single reasoning prompt through the selected model backend."""

    generator = load_reasoning_generator(model_name, profile)
    if isinstance(generator, ManualReasoningGenerator):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = generator.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=generator.profile.enable_thinking,
        )
        model_inputs = generator.tokenizer([text], return_tensors="pt")
        input_device = _model_input_device(generator.model)
        model_inputs = {name: tensor.to(input_device) for name, tensor in model_inputs.items()}
        with torch.inference_mode():
            generated_ids = generator.model.generate(
                **model_inputs,
                do_sample=generator.profile.do_sample,
                max_new_tokens=generator.profile.max_new_tokens,
                temperature=generator.profile.temperature,
                top_p=generator.profile.top_p,
                top_k=generator.profile.top_k,
                min_p=generator.profile.min_p,
                pad_token_id=generator.tokenizer.pad_token_id,
                eos_token_id=generator.tokenizer.eos_token_id,
            )
        output_ids = generated_ids[0][model_inputs["input_ids"].shape[1] :]
        response = generator.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if strip_qwen3_thinking:
            return strip_qwen3_thinking_content(response)
        return response

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    outputs = generator(messages)
    generated = outputs[0]["generated_text"]
    if isinstance(generated, list):
        final_message = generated[-1]
        if isinstance(final_message, dict):
            return str(final_message.get("content", "")).strip()
    return str(generated).strip()
