"""Preemptive control pipeline: detect risky route state, intervene, measure outcomes.

For each adversarial prompt:
  1. Extract final prompt-token hidden state at the target layer.
  2. Project onto probe direction -> risk score.
  3. If risk > threshold, apply rescue patch at target layer and generate.
  4. Else generate normally.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pressuretrace.adversarial.run_detector_robustness import classify_output
from pressuretrace.analysis.direction_characterization import (
    compute_mean_difference_direction,
    load_hidden_states_for_layer,
)
from pressuretrace.patching.reasoning_patching_core import (
    final_token_activation_for_layer,
    generate_with_patched_activation,
    get_prompt_hidden_states,
    load_model_and_tokenizer,
)
from pressuretrace.utils.io import (
    append_jsonl,
    ensure_directory,
    prepare_results_file,
    read_jsonl,
)


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


def run_control_pipeline(
    *,
    adversarial_results_path: Path,
    static_hidden_states_path: Path,
    model_name: str,
    thinking_mode: str,
    control_prompts_path: Path,
    layer: int = -6,
    threshold: float = 0.0,
    output_path: Path,
    max_tasks: int | None = None,
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    """Run the detect -> intervene -> generate control pipeline."""

    X_static, y_static = load_hidden_states_for_layer(
        static_hidden_states_path, layer=layer, representation="last_token",
    )
    induction_dir, _rescue_dir = compute_mean_difference_direction(X_static, y_static)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_static)
    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    probe.fit(X_scaled, y_static)

    adv_rows = read_jsonl(adversarial_results_path)
    if max_tasks is not None:
        adv_rows = adv_rows[:max_tasks]

    control_rows = read_jsonl(control_prompts_path)
    control_by_base_task: dict[str, dict[str, Any]] = {}
    for row in control_rows:
        if row.get("pressure_type") != "control":
            continue
        metadata = row.get("metadata") or {}
        base_task_id = metadata.get("base_task_id")
        if base_task_id is not None:
            control_by_base_task[base_task_id] = row

    bundle = load_model_and_tokenizer(model_name, thinking_mode=thinking_mode)
    prepare_results_file(output_path)

    rows: list[dict[str, Any]] = []
    for i, adv_row in enumerate(adv_rows, start=1):
        t0 = time.perf_counter()
        adv_prompt = adv_row["adv_prompt"]
        base_task_id = adv_row["base_task_id"]
        gold = str(adv_row.get("gold_answer", ""))
        shortcut = str(adv_row.get("shortcut_answer", ""))

        prompt_inputs, outputs = get_prompt_hidden_states(bundle, adv_prompt)
        activation = final_token_activation_for_layer(
            outputs, prompt_inputs, model=bundle.model, layer=layer,
        )
        activation_np = activation.detach().cpu().float().numpy()

        risk_score = float(np.dot(activation_np, induction_dir))
        activation_scaled = scaler.transform(activation_np.reshape(1, -1))
        risk_prob = float(probe.predict_proba(activation_scaled)[0, 1])

        should_intervene = risk_score > threshold
        intervention_applied = False

        if should_intervene:
            control_row = control_by_base_task.get(base_task_id)
            if control_row is not None:
                control_prompt = control_row.get("prompt", "")
                control_inputs, control_outputs = get_prompt_hidden_states(
                    bundle, control_prompt,
                )
                control_activation = final_token_activation_for_layer(
                    control_outputs, control_inputs, model=bundle.model, layer=layer,
                )
                output = generate_with_patched_activation(
                    bundle,
                    prompt_inputs,
                    layer=layer,
                    donor_final_token_activation=control_activation,
                    max_new_tokens=max_new_tokens,
                )
                intervention_applied = True
            else:
                output = _generate_unpatched(
                    bundle, prompt_inputs, max_new_tokens=max_new_tokens,
                )
        else:
            output = _generate_unpatched(
                bundle, prompt_inputs, max_new_tokens=max_new_tokens,
            )

        route = classify_output(output, gold, shortcut)

        row = {
            "base_task_id": base_task_id,
            "risk_score": risk_score,
            "risk_prob": risk_prob,
            "should_intervene": should_intervene,
            "intervention_applied": intervention_applied,
            "output": output,
            "route": route,
            "gold_answer": gold,
            "shortcut_answer": shortcut,
        }
        rows.append(row)
        append_jsonl(output_path, row)

        elapsed = time.perf_counter() - t0
        print(
            f"[{i}/{len(adv_rows)}] {base_task_id} risk={risk_score:.3f} "
            f"intervened={intervention_applied} route={route} ({elapsed:.1f}s)"
        )

    total = len(rows)
    intervened = sum(1 for r in rows if r["intervention_applied"])
    robust = sum(1 for r in rows if r["route"] == "gold")
    shortcut_count = sum(1 for r in rows if r["route"] == "shortcut")

    summary = {
        "model": model_name,
        "layer": layer,
        "threshold": threshold,
        "total": total,
        "intervention_rate": intervened / total if total > 0 else 0.0,
        "robust_rate_with_pipeline": robust / total if total > 0 else 0.0,
        "shortcut_rate_with_pipeline": shortcut_count / total if total > 0 else 0.0,
    }

    summary_path = output_path.parent / output_path.name.replace(
        ".jsonl", "_summary.json"
    )
    ensure_directory(summary_path.parent)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Control Pipeline Summary ===")
    print(f"Threshold: {threshold}")
    print(f"Intervention rate: {summary['intervention_rate']:.1%}")
    print(f"Robust rate WITH pipeline: {summary['robust_rate_with_pipeline']:.1%}")
    print(f"Shortcut rate WITH pipeline: {summary['shortcut_rate_with_pipeline']:.1%}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preemptive control pipeline for adversarial PressureTrace.",
    )
    parser.add_argument(
        "--adversarial-results-path",
        type=Path,
        default=Path("results/adversarial_detector_robustness_qwen-qwen3-14b_off.jsonl"),
    )
    parser.add_argument(
        "--static-hidden-states-path",
        type=Path,
        default=Path(
            "pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/"
            "reasoning_probe_hidden_states_qwen-qwen3-14b_off.jsonl"
        ),
    )
    parser.add_argument(
        "--control-prompts-path",
        type=Path,
        default=Path(
            "pressuretrace-frozen/reasoning_v2_qwen3_14b_off/data/manifests/"
            "reasoning_paper_slice_qwen-qwen3-14b_off.jsonl"
        ),
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--thinking-mode", type=str, default="off")
    parser.add_argument("--layer", type=int, default=-6)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/adversarial_control_pipeline_qwen-qwen3-14b_off.jsonl"),
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    return run_control_pipeline(
        adversarial_results_path=args.adversarial_results_path,
        static_hidden_states_path=args.static_hidden_states_path,
        model_name=args.model_name,
        thinking_mode=args.thinking_mode,
        control_prompts_path=args.control_prompts_path,
        layer=args.layer,
        threshold=args.threshold,
        output_path=args.output_path,
        max_tasks=args.max_tasks,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
