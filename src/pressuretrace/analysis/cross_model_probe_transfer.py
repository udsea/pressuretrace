"""Transfer a probe direction from one model to another via linear alignment.

Given paired final-token activations (Qwen_act, Gemma_act) on the same prompts,
fit a least-squares map W such that Qwen_act @ W ~= Gemma_act. Then transfer
the Qwen induction direction into Gemma's hidden space, and test how well the
transferred direction predicts Gemma's shortcut-vs-robust routes.

Inputs:
  --paired-path  JSONL from extract_paired_activations (model_a = source, b = target)
  --source-hidden-states-path  frozen probe hidden states for model_a (for direction)
  --target-hidden-states-path  frozen probe hidden states for model_b (for evaluation)
  --layer  decoder layer index

Outputs a summary JSON with the native vs transferred AUC for the target model.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pressuretrace.analysis.direction_characterization import (
    compute_mean_difference_direction,
    load_hidden_states_for_layer,
)
from pressuretrace.utils.io import ensure_directory, read_jsonl


def load_paired_activations(
    paired_path: Path,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load paired activation matrices (X_source, X_target) filtered to one layer."""

    rows = read_jsonl(paired_path)
    filtered = [r for r in rows if r.get("layer") == layer]
    if not filtered:
        available = sorted({r.get("layer") for r in rows})
        raise ValueError(
            f"No paired rows at layer={layer}. Available layers: {available}"
        )
    X_source = np.array(
        [r["activation_a"] for r in filtered], dtype=np.float32,
    )
    X_target = np.array(
        [r["activation_b"] for r in filtered], dtype=np.float32,
    )
    return X_source, X_target


def fit_linear_alignment(
    X_source: np.ndarray,
    X_target: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Fit W such that X_source @ W ~= X_target via ridge regression.

    Returns W of shape (source_dim, target_dim).
    """

    n, d_source = X_source.shape
    _, d_target = X_target.shape
    gram = X_source.T @ X_source + ridge * np.eye(d_source, dtype=np.float32)
    W = np.linalg.solve(gram, X_source.T @ X_target)
    pred = X_source @ W
    residual = float(np.linalg.norm(X_target - pred) ** 2)
    total = float(np.linalg.norm(X_target - X_target.mean(axis=0, keepdims=True)) ** 2)
    r2 = 1.0 - residual / max(total, 1e-8)
    print(
        f"Linear alignment fit: n={n}, source_dim={d_source}, target_dim={d_target}, "
        f"R^2={r2:.4f}"
    )
    return W


def evaluate_direction_auc(
    X: np.ndarray,
    y: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Measure AUC of projecting X onto direction for predicting y=1 (shortcut)."""

    scores = X @ direction
    return float(roc_auc_score(y, scores))


def run_cross_model_probe_transfer(
    *,
    paired_path: Path,
    source_hidden_states_path: Path,
    target_hidden_states_path: Path,
    layer: int,
    output_path: Path,
) -> dict[str, Any]:
    """Compute linear alignment and evaluate transferred direction on target model."""

    print(f"Loading source hidden states: {source_hidden_states_path}")
    X_src_probe, y_src_probe = load_hidden_states_for_layer(
        source_hidden_states_path, layer=layer, representation="last_token",
    )
    induction_src, _ = compute_mean_difference_direction(X_src_probe, y_src_probe)
    print(
        f"Source direction: dim={induction_src.shape[0]}, "
        f"source probe n={len(y_src_probe)} "
        f"(shortcut={int(y_src_probe.sum())}, robust={int((y_src_probe == 0).sum())})"
    )

    print(f"\nLoading target hidden states: {target_hidden_states_path}")
    X_tgt_probe, y_tgt_probe = load_hidden_states_for_layer(
        target_hidden_states_path, layer=layer, representation="last_token",
    )
    induction_tgt_native, _ = compute_mean_difference_direction(X_tgt_probe, y_tgt_probe)
    print(
        f"Target direction: dim={induction_tgt_native.shape[0]}, "
        f"target probe n={len(y_tgt_probe)} "
        f"(shortcut={int(y_tgt_probe.sum())}, robust={int((y_tgt_probe == 0).sum())})"
    )

    print(f"\nLoading paired activations: {paired_path}")
    X_source_paired, X_target_paired = load_paired_activations(paired_path, layer=layer)
    print(
        f"Paired: n={X_source_paired.shape[0]}, "
        f"source_dim={X_source_paired.shape[1]}, target_dim={X_target_paired.shape[1]}"
    )

    if X_source_paired.shape[1] != induction_src.shape[0]:
        raise ValueError(
            "Source paired activation dim does not match source probe hidden dim."
        )
    if X_target_paired.shape[1] != induction_tgt_native.shape[0]:
        raise ValueError(
            "Target paired activation dim does not match target probe hidden dim."
        )

    print("\nFitting linear alignment source -> target...")
    W = fit_linear_alignment(X_source_paired, X_target_paired)

    transferred = induction_src @ W
    transferred_norm = float(np.linalg.norm(transferred))
    transferred_unit = transferred / (transferred_norm + 1e-8)

    native_auc = evaluate_direction_auc(X_tgt_probe, y_tgt_probe, induction_tgt_native)
    transferred_auc = evaluate_direction_auc(X_tgt_probe, y_tgt_probe, transferred_unit)
    random_direction = np.random.default_rng(0).standard_normal(induction_tgt_native.shape[0])
    random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8)
    random_auc = evaluate_direction_auc(X_tgt_probe, y_tgt_probe, random_direction)

    scaler = StandardScaler()
    X_tgt_scaled = scaler.fit_transform(X_tgt_probe)
    tgt_probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    tgt_probe.fit(X_tgt_scaled, y_tgt_probe)
    tgt_lr_auc = float(
        roc_auc_score(y_tgt_probe, tgt_probe.predict_proba(X_tgt_scaled)[:, 1])
    )

    cos_native_vs_transferred = float(
        induction_tgt_native @ transferred_unit
    )

    summary = {
        "layer": layer,
        "paired_path": str(paired_path),
        "source_hidden_states_path": str(source_hidden_states_path),
        "target_hidden_states_path": str(target_hidden_states_path),
        "n_paired": int(X_source_paired.shape[0]),
        "n_target_probe": int(len(y_tgt_probe)),
        "source_dim": int(X_source_paired.shape[1]),
        "target_dim": int(X_target_paired.shape[1]),
        "target_native_mean_diff_auc": native_auc,
        "target_lr_probe_in_sample_auc": tgt_lr_auc,
        "target_transferred_direction_auc": transferred_auc,
        "target_random_direction_auc": random_auc,
        "cosine_native_vs_transferred": cos_native_vs_transferred,
        "transferred_norm_before_unit": transferred_norm,
    }

    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Cross-Model Probe Transfer Summary ===")
    print(f"Layer: {layer}")
    print(f"Paired n: {summary['n_paired']}")
    print(f"Target native mean-diff AUC:      {native_auc:.4f}")
    print(f"Target LR probe AUC (in-sample):  {tgt_lr_auc:.4f}")
    print(f"Target transferred direction AUC: {transferred_auc:.4f}")
    print(f"Target random direction AUC:      {random_auc:.4f}")
    print(f"cos(native, transferred):         {cos_native_vs_transferred:.4f}")
    print(f"Summary written to: {output_path}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer probe direction across models via linear alignment.",
    )
    parser.add_argument("--paired-path", type=Path, required=True)
    parser.add_argument("--source-hidden-states-path", type=Path, required=True)
    parser.add_argument("--target-hidden-states-path", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=-4)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    return run_cross_model_probe_transfer(
        paired_path=args.paired_path,
        source_hidden_states_path=args.source_hidden_states_path,
        target_hidden_states_path=args.target_hidden_states_path,
        layer=args.layer,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
