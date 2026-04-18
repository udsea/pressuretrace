"""Measure probe AUC under adversarial pressure vs static pressure.

Trains a logistic-regression probe on static pressure hidden states,
then runs full generation on the best adversarial prompt per task to
collect (hidden_state, route_label) pairs, and measures probe AUC.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pressuretrace.analysis.direction_characterization import (
    load_hidden_states_for_layer,
)
from pressuretrace.behavior.reasoning_runtime import (
    generation_profile_for_reasoning_v2,
    infer_reasoning_response,
)
from pressuretrace.patching.reasoning_patching_core import (
    final_token_activation_for_layer,
    get_prompt_hidden_states,
    load_model_and_tokenizer,
)
from pressuretrace.utils.io import (
    append_jsonl,
    ensure_directory,
    prepare_results_file,
    read_jsonl,
)


def classify_output(output: str, gold: str, shortcut: str) -> str:
    """Classify a generated output as gold, shortcut, or other."""

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


def run_detector_robustness(
    *,
    attack_results_path: Path,
    hidden_states_path: Path,
    model_name: str,
    thinking_mode: str,
    layer: int = -6,
    output_path: Path,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Run the detector robustness experiment."""

    print("Loading static hidden states for probe training...")
    X_static, y_static = load_hidden_states_for_layer(
        hidden_states_path, layer=layer, representation="last_token",
    )

    scaler = StandardScaler()
    X_static_scaled = scaler.fit_transform(X_static)
    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    probe.fit(X_static_scaled, y_static)
    static_auc = roc_auc_score(
        y_static, probe.predict_proba(X_static_scaled)[:, 1]
    )
    print(f"Static probe train AUC (in-sample): {static_auc:.4f}")

    attack_rows = read_jsonl(attack_results_path)
    if max_tasks is not None:
        attack_rows = attack_rows[:max_tasks]
    print(f"Attack results loaded: {len(attack_rows)} tasks")

    bundle = load_model_and_tokenizer(model_name, thinking_mode=thinking_mode)
    profile = generation_profile_for_reasoning_v2(model_name, thinking_mode)

    prepare_results_file(output_path)
    adversarial_hidden_states: list[np.ndarray] = []
    adversarial_labels: list[int] = []

    for i, row in enumerate(attack_rows, start=1):
        t0 = time.perf_counter()
        best_adv = row.get("best_adversarial")
        if best_adv is None:
            continue

        adv_prompt = best_adv["prompt"]
        gold = str(row["gold_answer"])
        shortcut = str(row["shortcut_answer"])

        response = infer_reasoning_response(
            adv_prompt,
            model_name,
            profile,
            strip_qwen3_thinking=True,
        )
        route = classify_output(response, gold, shortcut)
        binary_label = 1 if route == "shortcut" else 0

        prompt_inputs, outputs = get_prompt_hidden_states(bundle, adv_prompt)
        activation = final_token_activation_for_layer(
            outputs, prompt_inputs, model=bundle.model, layer=layer,
        )
        hidden_state = activation.detach().cpu().float().numpy()

        adversarial_hidden_states.append(hidden_state)
        adversarial_labels.append(binary_label)

        result_row = {
            "base_task_id": row["base_task_id"],
            "adv_prompt": adv_prompt,
            "response": response,
            "route": route,
            "binary_label": binary_label,
            "attack_label": best_adv["label"],
            "attack_score": best_adv["score"],
            "gold_answer": gold,
            "shortcut_answer": shortcut,
        }
        append_jsonl(output_path, result_row)

        elapsed = time.perf_counter() - t0
        print(
            f"[{i}/{len(attack_rows)}] {row['base_task_id']} route={route} ({elapsed:.1f}s)"
        )

    X_adv = np.array(adversarial_hidden_states, dtype=np.float32)
    y_adv = np.array(adversarial_labels, dtype=np.int32)

    shortcut_rate_adv = float(y_adv.mean()) if len(y_adv) else 0.0
    print(f"\nAdversarial shortcut rate: {shortcut_rate_adv:.1%}")

    adversarial_auc: float | None
    if len(np.unique(y_adv)) < 2:
        print("WARNING: Only one class in adversarial labels. Cannot compute AUC.")
        adversarial_auc = None
    else:
        X_adv_scaled = scaler.transform(X_adv)
        adv_probs = probe.predict_proba(X_adv_scaled)[:, 1]
        adversarial_auc = float(roc_auc_score(y_adv, adv_probs))
        print(f"Adversarial probe AUC: {adversarial_auc:.4f}")
        print(f"Static probe AUC (reference): {static_auc:.4f}")
        print(f"AUC degradation: {static_auc - adversarial_auc:.4f}")

    summary = {
        "model": model_name,
        "layer": layer,
        "static_probe_train_auc": float(static_auc),
        "adversarial_probe_auc": adversarial_auc,
        "auc_degradation": (
            float(static_auc - adversarial_auc) if adversarial_auc is not None else None
        ),
        "n_adversarial_tasks": len(adversarial_labels),
        "adversarial_shortcut_rate": shortcut_rate_adv,
        "static_shortcut_rate": float(y_static.mean()),
    }

    summary_path = output_path.parent / output_path.name.replace(".jsonl", "_summary.json")
    ensure_directory(summary_path.parent)
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detector robustness: probe AUC under adversarial vs static pressure.",
    )
    parser.add_argument(
        "--attack-results-path",
        type=Path,
        default=Path("results/adversarial_attack_search_qwen-qwen3-14b_off.jsonl"),
    )
    parser.add_argument(
        "--hidden-states-path",
        type=Path,
        default=Path(
            "pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/"
            "reasoning_probe_hidden_states_qwen-qwen3-14b_off.jsonl"
        ),
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--thinking-mode", type=str, default="off")
    parser.add_argument("--layer", type=int, default=-6)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/adversarial_detector_robustness_qwen-qwen3-14b_off.jsonl"),
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    return run_detector_robustness(
        attack_results_path=args.attack_results_path,
        hidden_states_path=args.hidden_states_path,
        model_name=args.model_name,
        thinking_mode=args.thinking_mode,
        layer=args.layer,
        output_path=args.output_path,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
