"""Generation verification: patch layer activation, then decode, to confirm route rescue."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from pressuretrace.patching.build_reasoning_patch_pairs import load_reasoning_patch_pairs
from pressuretrace.patching.reasoning_patching_core import (
    final_token_activation_for_layer,
    generate_with_patched_activation,
    get_prompt_hidden_states,
    load_model_and_tokenizer,
)
from pressuretrace.patching.reasoning_patching_metrics import (
    FIRST_PASS_PRESSURE_TYPES,
    build_answer_token_sequence,
)
from pressuretrace.utils.io import append_jsonl, ensure_directory, prepare_results_file


def _slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()


def classify_output(output: str, gold: str, shortcut: str) -> str:
    """Classify a generated output as 'gold', 'shortcut', or 'other' via numeric containment."""

    output_norm = output.strip().lower().replace(",", "")
    gold_norm = gold.strip().lower().replace(",", "")
    shortcut_norm = shortcut.strip().lower().replace(",", "")

    if output_norm == gold_norm:
        return "gold"
    if output_norm == shortcut_norm:
        return "shortcut"

    output_numbers = re.findall(r"\b\d+\b", output_norm)
    if gold_norm in output_numbers:
        return "gold"
    if shortcut_norm in output_numbers:
        return "shortcut"
    return "other"


def _generate_unpatched(bundle: Any, prompt_inputs: Any, *, max_new_tokens: int = 48) -> str:
    """Generate from a prompt without any patching."""

    with torch.no_grad():
        generated_ids = bundle.model.generate(
            input_ids=prompt_inputs.input_ids,
            attention_mask=prompt_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
            use_cache=True,
        )
    new_ids = generated_ids[0, prompt_inputs.input_ids.shape[-1]:]
    return bundle.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_generation_patching_experiment(
    *,
    frozen_root: Path,
    model_name: str,
    thinking_mode: str,
    layer: int,
    pressure_types: tuple[str, ...],
    output_path: Path,
    summary_path: Path,
    max_pairs: int | None = None,
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    """Run generation verification: patch layer at final prompt token, then generate."""

    slug = _slug(model_name)
    manifest_path = frozen_root / "data" / "manifests" / f"reasoning_paper_slice_{slug}_{thinking_mode}.jsonl"
    results_path = frozen_root / "results" / f"reasoning_paper_slice_{slug}_{thinking_mode}.jsonl"
    patch_pairs_path = frozen_root / "results" / f"reasoning_patch_pairs_{slug}_{thinking_mode}.jsonl"

    pairs = load_reasoning_patch_pairs(
        patch_pairs_path=patch_pairs_path,
        manifest_path=manifest_path,
        results_path=results_path,
        pressure_types=pressure_types,
    )

    bundle = load_model_and_tokenizer(model_name, thinking_mode=thinking_mode)

    eligible = []
    for pair in pairs:
        answer_tokens = build_answer_token_sequence(
            bundle.tokenizer,
            gold_answer=pair.gold_answer,
            shortcut_answer=pair.shortcut_answer,
        )
        if answer_tokens is not None:
            eligible.append((pair, answer_tokens))

    if max_pairs is not None:
        eligible = eligible[:max_pairs]

    print(f"Eligible pairs: {len(eligible)}")
    prepare_results_file(output_path)

    rows: list[dict[str, Any]] = []
    for i, (pair, _answer_tokens) in enumerate(eligible, start=1):
        t0 = time.perf_counter()

        pressure_inputs, pressure_outputs = get_prompt_hidden_states(bundle, pair.pressure_prompt)
        control_inputs, control_outputs = get_prompt_hidden_states(bundle, pair.control_prompt)

        control_activation = final_token_activation_for_layer(
            control_outputs, control_inputs, model=bundle.model, layer=layer,
        )

        baseline_output = _generate_unpatched(bundle, pressure_inputs, max_new_tokens=max_new_tokens)
        rescue_output = generate_with_patched_activation(
            bundle,
            pressure_inputs,
            layer=layer,
            donor_final_token_activation=control_activation,
            max_new_tokens=max_new_tokens,
        )

        baseline_route = classify_output(baseline_output, pair.gold_answer, pair.shortcut_answer)
        rescue_route = classify_output(rescue_output, pair.gold_answer, pair.shortcut_answer)
        route_changed = baseline_route != rescue_route
        rescued = baseline_route == "shortcut" and rescue_route == "gold"

        row = {
            "base_task_id": pair.base_task_id,
            "pressure_type": pair.pressure_type,
            "layer": layer,
            "gold_answer": pair.gold_answer,
            "shortcut_answer": pair.shortcut_answer,
            "baseline_output": baseline_output,
            "rescue_output": rescue_output,
            "baseline_route": baseline_route,
            "rescue_route": rescue_route,
            "route_changed": route_changed,
            "rescued": rescued,
        }
        rows.append(row)
        append_jsonl(output_path, row)

        elapsed = time.perf_counter() - t0
        print(
            f"[{i}/{len(eligible)}] {pair.base_task_id} {pair.pressure_type} "
            f"baseline={baseline_route} rescue={rescue_route} rescued={rescued} "
            f"({elapsed:.1f}s)"
        )

    total = len(rows)
    shortcut_baseline = sum(1 for r in rows if r["baseline_route"] == "shortcut")
    rescued_count = sum(1 for r in rows if r["rescued"])
    rescue_rate = rescued_count / shortcut_baseline if shortcut_baseline > 0 else 0.0

    by_pressure: dict[str, dict[str, Any]] = {}
    for pt in pressure_types:
        pt_rows = [r for r in rows if r["pressure_type"] == pt]
        pt_shortcut = sum(1 for r in pt_rows if r["baseline_route"] == "shortcut")
        pt_rescued = sum(1 for r in pt_rows if r["rescued"])
        by_pressure[pt] = {
            "total": len(pt_rows),
            "shortcut_baseline": pt_shortcut,
            "rescued": pt_rescued,
            "rescue_rate": pt_rescued / pt_shortcut if pt_shortcut > 0 else 0.0,
        }

    summary = {
        "model": model_name,
        "layer": layer,
        "total_pairs": total,
        "shortcut_baseline_count": shortcut_baseline,
        "rescued_count": rescued_count,
        "rescue_rate": rescue_rate,
        "by_pressure_type": by_pressure,
    }

    ensure_directory(summary_path.parent)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Generation Verification Summary ===")
    print(f"Total pairs: {total}")
    if total:
        print(f"Shortcut at baseline: {shortcut_baseline} ({shortcut_baseline / total:.1%})")
    print(f"Rescued by patch: {rescued_count} ({rescue_rate:.1%} of shortcut-baseline pairs)")
    for pt, stats in by_pressure.items():
        print(
            f"  {pt}: rescued {stats['rescued']}/{stats['shortcut_baseline']} "
            f"({stats['rescue_rate']:.1%})"
        )

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generation verification for reasoning route patching.",
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=Path("pressuretrace-frozen/reasoning_v2_qwen3_14b_seq_off"),
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--thinking-mode", type=str, default="off")
    parser.add_argument("--layer", type=int, default=-6)
    parser.add_argument(
        "--pressure-types",
        type=str,
        default=",".join(FIRST_PASS_PRESSURE_TYPES),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "pressuretrace-frozen/reasoning_v2_qwen3_14b_seq_off/results/"
            "reasoning_generation_patching_qwen-qwen3-14b_off.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path(
            "pressuretrace-frozen/reasoning_v2_qwen3_14b_seq_off/results/"
            "reasoning_generation_patching_summary_qwen-qwen3-14b_off.json"
        ),
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    pressure_types = tuple(p.strip() for p in args.pressure_types.split(",") if p.strip())
    return run_generation_patching_experiment(
        frozen_root=args.frozen_root,
        model_name=args.model_name,
        thinking_mode=args.thinking_mode,
        layer=args.layer,
        pressure_types=pressure_types,
        output_path=args.output_path,
        summary_path=args.summary_path,
        max_pairs=args.max_pairs,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
