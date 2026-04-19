"""Extract layer-wise last-token activations from two models on identical prompts.

For each prompt in a manifest, runs both models forward once and records the
final-prompt-token hidden state at a chosen layer. The output is a JSONL of
paired activations suitable for cross-model alignment analysis.

Usage:
    uv run python -m pressuretrace.analysis.extract_paired_activations \\
        --manifest-path data/manifests/reasoning_paper_slice_qwen-qwen3-14b_off.jsonl \\
        --pressure-type control \\
        --limit 100 \\
        --layer -4 \\
        --model-a Qwen/Qwen3-14B \\
        --model-b google/gemma-3-27b-it \\
        --output-path results/cross_model_paired_activations_layer-4.jsonl
"""

from __future__ import annotations

import argparse
import gc
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pressuretrace.patching.reasoning_patching_core import (
    final_token_activation_for_layer,
    get_prompt_hidden_states,
    load_model_and_tokenizer,
)
from pressuretrace.utils.io import append_jsonl, prepare_results_file, read_jsonl


def _select_prompts(
    manifest_path: Path,
    pressure_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest_path)
    selected = [r for r in rows if r.get("pressure_type") == pressure_type]
    if limit:
        selected = selected[:limit]
    return selected


def _extract_activations_for_model(
    model_name: str,
    thinking_mode: str,
    prompt_rows: Sequence[dict[str, Any]],
    layer: int,
) -> list[np.ndarray]:
    """Load a model, iterate prompts, return a list of final-token activations."""

    print(f"\nLoading {model_name}...")
    bundle = load_model_and_tokenizer(model_name, thinking_mode=thinking_mode)

    activations: list[np.ndarray] = []
    for i, row in enumerate(prompt_rows, 1):
        t0 = time.perf_counter()
        prompt = row["prompt"]
        prompt_inputs, outputs = get_prompt_hidden_states(bundle, prompt)
        activation = final_token_activation_for_layer(
            outputs, prompt_inputs, model=bundle.model, layer=layer,
        )
        activations.append(activation.detach().cpu().float().numpy())
        elapsed = time.perf_counter() - t0
        print(
            f"  [{i}/{len(prompt_rows)}] {row.get('task_id', '?')} "
            f"dim={activations[-1].shape[0]} ({elapsed:.1f}s)"
        )

    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return activations


def run_paired_extraction(
    *,
    manifest_path: Path,
    pressure_type: str,
    limit: int,
    layer: int,
    model_a: str,
    model_b: str,
    thinking_mode: str,
    output_path: Path,
) -> Path:
    """Extract paired activations from two models and stream pairs to output JSONL."""

    prompt_rows = _select_prompts(manifest_path, pressure_type, limit)
    print(f"Selected {len(prompt_rows)} prompts (pressure_type={pressure_type})")

    acts_a = _extract_activations_for_model(model_a, thinking_mode, prompt_rows, layer)
    acts_b = _extract_activations_for_model(model_b, thinking_mode, prompt_rows, layer)

    prepare_results_file(output_path)
    for row, act_a, act_b in zip(prompt_rows, acts_a, acts_b):
        record = {
            "task_id": row.get("task_id"),
            "base_task_id": row.get("metadata", {}).get("base_task_id"),
            "pressure_type": row.get("pressure_type"),
            "layer": layer,
            "model_a": model_a,
            "model_b": model_b,
            "activation_a": act_a.tolist(),
            "activation_b": act_b.tolist(),
            "dim_a": int(act_a.shape[0]),
            "dim_b": int(act_b.shape[0]),
        }
        append_jsonl(output_path, record)

    print(f"\nWrote {len(prompt_rows)} paired activations to {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract paired final-prompt-token activations from two models.",
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--pressure-type", type=str, default="control")
    parser.add_argument("--limit", type=int, default=229)
    parser.add_argument("--layer", type=int, default=-4)
    parser.add_argument("--model-a", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--model-b", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--thinking-mode", type=str, default="off")
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    return run_paired_extraction(
        manifest_path=args.manifest_path,
        pressure_type=args.pressure_type,
        limit=args.limit,
        layer=args.layer,
        model_a=args.model_a,
        model_b=args.model_b,
        thinking_mode=args.thinking_mode,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
