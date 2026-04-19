"""Transfer a probe direction from one model to another via linear alignment.

Given paired final-token activations (X_source, X_target) on the same prompts,
reduce both sides with PCA, fit Ridge regression W such that
X_source_pca @ W ~= X_target_pca, then transfer the source induction direction
into the target space and measure how well it predicts the target model's
shortcut-vs-robust routes.

With high-dimensional activations and limited paired samples, naive least
squares overfits to R^2 ~= 1.0 while producing a useless transferred direction.
PCA + Ridge with an alpha sweep exposes the generalization story.

Inputs:
  --paired-path  JSONL from extract_paired_activations (model_a = source, b = target)
  --source-hidden-states-path  frozen probe hidden states for model_a (for direction)
  --target-hidden-states-path  frozen probe hidden states for model_b (for evaluation)
  --layer  decoder layer index
  --n-pca-components  PCA dimensionality used on each side before alignment
  --ridge-alpha  comma-separated Ridge alpha values to sweep

Outputs a summary JSON with native vs transferred AUC per alpha value.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
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


def pca_reduce(X: np.ndarray, n_components: int) -> tuple[np.ndarray, PCA]:
    """Fit PCA on X and return (X_reduced, fitted_pca)."""

    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def compute_linear_alignment(
    X_source: np.ndarray,
    X_target: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Fit W such that X_source @ W ~= X_target via Ridge regression.

    Returns (W, r2_on_training) where W has shape (source_dim, target_dim).
    """

    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X_source, X_target)
    W = reg.coef_.T
    pred = X_source @ W
    residual = float(np.linalg.norm(X_target - pred) ** 2)
    total = float(
        np.linalg.norm(X_target - X_target.mean(axis=0, keepdims=True)) ** 2
    )
    r2 = 1.0 - residual / max(total, 1e-8)
    return W, r2


def transfer_direction_through_pca(
    direction_source: np.ndarray,
    pca_source: PCA,
    W: np.ndarray,
    pca_target: PCA,
) -> np.ndarray:
    """Project a direction from source full-dim space to target full-dim space.

    Directions are displacements, so we only apply the PCA rotation (components_)
    and skip the mean-centering step that points in the full space would use.
    """

    d_source_pca = direction_source @ pca_source.components_.T
    d_target_pca = d_source_pca @ W
    d_target_full = d_target_pca @ pca_target.components_
    return d_target_full


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
    n_pca_components: int,
    ridge_alphas: Sequence[float],
) -> dict[str, Any]:
    """Run PCA + Ridge alignment with an alpha sweep."""

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
    n_paired = X_source_paired.shape[0]
    print(
        f"Paired: n={n_paired}, "
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

    effective_components = min(
        n_pca_components,
        n_paired,
        X_source_paired.shape[1],
        X_target_paired.shape[1],
    )
    if effective_components != n_pca_components:
        print(
            f"Clamping n_components {n_pca_components} -> {effective_components} "
            f"(limited by paired sample count or activation dim)"
        )
    print(f"\nFitting PCA(n_components={effective_components}) on both sides...")
    X_source_pca, pca_source = pca_reduce(X_source_paired, effective_components)
    X_target_pca, pca_target = pca_reduce(X_target_paired, effective_components)
    print(
        f"Source PCA explained var: {pca_source.explained_variance_ratio_.sum():.4f}, "
        f"target PCA explained var: {pca_target.explained_variance_ratio_.sum():.4f}"
    )

    native_auc = evaluate_direction_auc(X_tgt_probe, y_tgt_probe, induction_tgt_native)
    rng = np.random.default_rng(0)
    random_direction = rng.standard_normal(induction_tgt_native.shape[0])
    random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8)
    random_auc = evaluate_direction_auc(X_tgt_probe, y_tgt_probe, random_direction)

    scaler = StandardScaler()
    X_tgt_scaled = scaler.fit_transform(X_tgt_probe)
    tgt_probe_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    tgt_probe_lr.fit(X_tgt_scaled, y_tgt_probe)
    tgt_lr_auc = float(
        roc_auc_score(y_tgt_probe, tgt_probe_lr.predict_proba(X_tgt_scaled)[:, 1])
    )

    per_alpha_results: list[dict[str, Any]] = []
    for alpha in ridge_alphas:
        print(f"\n--- Ridge alignment (alpha={alpha}) ---")
        W, r2 = compute_linear_alignment(X_source_pca, X_target_pca, alpha=alpha)
        print(f"R^2 on paired (in-sample): {r2:.4f}")

        transferred = transfer_direction_through_pca(
            induction_src, pca_source, W, pca_target,
        )
        transferred_norm = float(np.linalg.norm(transferred))
        transferred_unit = transferred / (transferred_norm + 1e-8)

        transferred_auc = evaluate_direction_auc(
            X_tgt_probe, y_tgt_probe, transferred_unit,
        )
        cos_native_vs_transferred = float(
            induction_tgt_native @ transferred_unit
            / (np.linalg.norm(induction_tgt_native) + 1e-8)
        )

        print(f"  transferred AUC: {transferred_auc:.4f}")
        print(f"  cos(native, transferred): {cos_native_vs_transferred:.4f}")
        print(f"  transferred norm (pre-unit): {transferred_norm:.4f}")

        per_alpha_results.append({
            "alpha": float(alpha),
            "r2_in_sample": r2,
            "transferred_auc": transferred_auc,
            "cos_native_vs_transferred": cos_native_vs_transferred,
            "transferred_norm_before_unit": transferred_norm,
        })

    summary = {
        "layer": layer,
        "paired_path": str(paired_path),
        "source_hidden_states_path": str(source_hidden_states_path),
        "target_hidden_states_path": str(target_hidden_states_path),
        "n_paired": int(n_paired),
        "n_target_probe": int(len(y_tgt_probe)),
        "source_dim": int(X_source_paired.shape[1]),
        "target_dim": int(X_target_paired.shape[1]),
        "n_pca_components": int(effective_components),
        "source_pca_explained_var": float(pca_source.explained_variance_ratio_.sum()),
        "target_pca_explained_var": float(pca_target.explained_variance_ratio_.sum()),
        "target_native_mean_diff_auc": native_auc,
        "target_lr_probe_in_sample_auc": tgt_lr_auc,
        "target_random_direction_auc": random_auc,
        "ridge_alpha_sweep": per_alpha_results,
    }

    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Cross-Model Probe Transfer Summary ===")
    print(f"Layer: {layer}  Paired n: {n_paired}  PCA: {effective_components}")
    print(f"Target native mean-diff AUC:      {native_auc:.4f}")
    print(f"Target LR probe AUC (in-sample):  {tgt_lr_auc:.4f}")
    print(f"Target random direction AUC:      {random_auc:.4f}")
    print(f"{'alpha':>10} {'R^2':>8} {'xfer_auc':>10} {'cos_nt':>10}")
    for row in per_alpha_results:
        print(
            f"{row['alpha']:>10.2f} "
            f"{row['r2_in_sample']:>8.4f} "
            f"{row['transferred_auc']:>10.4f} "
            f"{row['cos_native_vs_transferred']:>10.4f}"
        )
    print(f"Summary written to: {output_path}")

    return summary


def _parse_alpha_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer probe direction across models via PCA + Ridge alignment.",
    )
    parser.add_argument("--paired-path", type=Path, required=True)
    parser.add_argument("--source-hidden-states-path", type=Path, required=True)
    parser.add_argument("--target-hidden-states-path", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=-4)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--n-pca-components", type=int, default=50)
    parser.add_argument(
        "--ridge-alpha",
        type=str,
        default="100,1000,10000,100000",
        help="Comma-separated list of Ridge alpha values to sweep.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    alphas = _parse_alpha_list(args.ridge_alpha)
    return run_cross_model_probe_transfer(
        paired_path=args.paired_path,
        source_hidden_states_path=args.source_hidden_states_path,
        target_hidden_states_path=args.target_hidden_states_path,
        layer=args.layer,
        output_path=args.output_path,
        n_pca_components=args.n_pca_components,
        ridge_alphas=alphas,
    )


if __name__ == "__main__":
    main()
