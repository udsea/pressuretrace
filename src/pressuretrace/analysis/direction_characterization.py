"""Characterize the route-choice direction via vocabulary projection and token analysis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pressuretrace.utils.io import ensure_directory, read_jsonl


def load_hidden_states_for_layer(
    hidden_states_path: Path,
    *,
    layer: int,
    representation: str = "last_token",
) -> tuple[np.ndarray, np.ndarray]:
    """Load hidden states and binary labels for one layer/representation combo.

    Returns (X, y) where X is [n_episodes, hidden_dim] and y is [n_episodes] int.
    binary_label: 0 = robust_correct, 1 = shortcut_followed.
    """

    rows = read_jsonl(hidden_states_path)
    filtered = [
        r for r in rows
        if r["layer"] == layer and r["representation"] == representation
    ]
    if not filtered:
        available = sorted({r["layer"] for r in rows})
        raise ValueError(
            f"No rows found for layer={layer}, representation={representation}. "
            f"Available layers: {available}"
        )
    X = np.array([r["hidden_state"] for r in filtered], dtype=np.float32)
    y = np.array([int(r["binary_label"]) for r in filtered], dtype=np.int32)
    return X, y


def compute_mean_difference_direction(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit-norm induction and rescue directions via class-mean difference.

    induction_direction: robust_correct centroid -> shortcut_followed centroid.
    rescue_direction: shortcut_followed centroid -> robust_correct centroid.
    """

    robust_mean = X[y == 0].mean(axis=0)
    shortcut_mean = X[y == 1].mean(axis=0)

    induction = shortcut_mean - robust_mean
    induction = induction / (np.linalg.norm(induction) + 1e-8)

    rescue = robust_mean - shortcut_mean
    rescue = rescue / (np.linalg.norm(rescue) + 1e-8)

    return induction, rescue


def project_direction_onto_vocabulary(
    direction: np.ndarray,
    model_name: str,
    *,
    top_k: int = 30,
    skip_special_tokens: bool = True,
    _lm_head_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a hidden-state direction onto vocabulary via the LM head weight matrix.

    Returns the top-k promoted and suppressed tokens. If _lm_head_cache is passed,
    the LM head weight and tokenizer will be reused across invocations.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = _lm_head_cache if _lm_head_cache is not None else {}
    if "lm_head_weight" not in cache or "tokenizer" not in cache:
        print(f"Loading {model_name} for vocabulary projection...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        lm_head_weight: np.ndarray | None = None
        for name, param in model.named_parameters():
            if "lm_head" in name and "weight" in name:
                lm_head_weight = param.detach().float().numpy()
                print(f"Found LM head: {name}, shape: {lm_head_weight.shape}")
                break
        if lm_head_weight is None:
            # Some models tie lm_head to input embeddings
            embed = model.get_input_embeddings()
            if embed is not None and hasattr(embed, "weight"):
                lm_head_weight = embed.weight.detach().float().numpy()
                print(f"Using tied input embeddings as LM head, shape: {lm_head_weight.shape}")
        if lm_head_weight is None:
            raise ValueError("Could not find lm_head.weight or tied embeddings in model.")
        cache["lm_head_weight"] = lm_head_weight
        cache["tokenizer"] = tokenizer

    lm_head_weight = cache["lm_head_weight"]
    tokenizer = cache["tokenizer"]

    direction_f32 = direction.astype(np.float32)
    scores = lm_head_weight @ direction_f32

    top_promoted_ids = np.argsort(scores)[-top_k:][::-1].tolist()
    top_suppressed_ids = np.argsort(scores)[:top_k].tolist()

    def decode_token(token_id: int) -> str:
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens)
            return decoded.strip() if decoded.strip() else f"<token_{token_id}>"
        except Exception:
            return f"<token_{token_id}>"

    promoted_tokens = [
        {"token_id": int(tid), "token_str": decode_token(tid), "score": float(scores[tid])}
        for tid in top_promoted_ids
    ]
    suppressed_tokens = [
        {"token_id": int(tid), "token_str": decode_token(tid), "score": float(scores[tid])}
        for tid in top_suppressed_ids
    ]

    return {
        "promoted": promoted_tokens,
        "suppressed": suppressed_tokens,
        "direction_norm": float(np.linalg.norm(direction)),
        "vocab_size": int(lm_head_weight.shape[0]),
        "hidden_dim": int(lm_head_weight.shape[1]),
    }


def run_direction_characterization(
    *,
    hidden_states_path: Path,
    model_name: str,
    layer: int = -6,
    representation: str = "last_token",
    top_k: int = 30,
    output_path: Path,
) -> dict[str, Any]:
    """Full direction characterization pipeline."""

    print(f"Loading hidden states from {hidden_states_path}")
    X, y = load_hidden_states_for_layer(
        hidden_states_path, layer=layer, representation=representation,
    )
    print(f"Loaded {len(X)} episodes: {(y == 0).sum()} robust, {(y == 1).sum()} shortcut")

    induction_dir, rescue_dir = compute_mean_difference_direction(X, y)
    print(f"Computed directions. Hidden dim: {induction_dir.shape[0]}")

    lm_head_cache: dict[str, Any] = {}
    print("Projecting induction direction onto vocabulary...")
    induction_vocab = project_direction_onto_vocabulary(
        induction_dir, model_name, top_k=top_k, _lm_head_cache=lm_head_cache,
    )

    print("Projecting rescue direction onto vocabulary...")
    rescue_vocab = project_direction_onto_vocabulary(
        rescue_dir, model_name, top_k=top_k, _lm_head_cache=lm_head_cache,
    )

    robust_proj = X[y == 0] @ induction_dir
    shortcut_proj = X[y == 1] @ induction_dir
    separation = {
        "robust_mean_projection": float(robust_proj.mean()),
        "shortcut_mean_projection": float(shortcut_proj.mean()),
        "projection_gap": float(shortcut_proj.mean() - robust_proj.mean()),
        "robust_std": float(robust_proj.std()),
        "shortcut_std": float(shortcut_proj.std()),
    }

    result = {
        "model_name": model_name,
        "layer": layer,
        "representation": representation,
        "n_robust": int((y == 0).sum()),
        "n_shortcut": int((y == 1).sum()),
        "separation": separation,
        "induction_direction_vocab": induction_vocab,
        "rescue_direction_vocab": rescue_vocab,
    }

    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Written to {output_path}")

    print("\n=== INDUCTION DIRECTION (robust -> shortcut) ===")
    print("Top promoted tokens (model moves TOWARD shortcut):")
    for item in induction_vocab["promoted"][:15]:
        print(f"  {repr(item['token_str']):20s}  score={item['score']:.4f}")
    print("\nTop suppressed tokens (model moves AWAY from):")
    for item in induction_vocab["suppressed"][:15]:
        print(f"  {repr(item['token_str']):20s}  score={item['score']:.4f}")

    print("\n=== RESCUE DIRECTION (shortcut -> robust) ===")
    print("Top promoted tokens (model moves TOWARD robust):")
    for item in rescue_vocab["promoted"][:15]:
        print(f"  {repr(item['token_str']):20s}  score={item['score']:.4f}")

    print(f"\nProjection gap (shortcut_mean - robust_mean): {separation['projection_gap']:.4f}")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Characterize the route-choice direction via vocabulary projection.",
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
    parser.add_argument("--layer", type=int, default=-6)
    parser.add_argument("--representation", type=str, default="last_token")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "pressuretrace-frozen/reasoning_v2_qwen3_14b_off/results/"
            "reasoning_direction_characterization_qwen-qwen3-14b_off.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    return run_direction_characterization(
        hidden_states_path=args.hidden_states_path,
        model_name=args.model_name,
        layer=args.layer,
        representation=args.representation,
        top_k=args.top_k,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
