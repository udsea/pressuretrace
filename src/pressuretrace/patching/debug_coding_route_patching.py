"""One-pair debugging utilities for coding route patching."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pressuretrace.config import (
    CODING_V1_MODEL_NAME,
    CODING_V1_THINKING_MODE,
    resolve_coding_frozen_root,
)
from pressuretrace.patching.build_coding_patch_pairs import (
    TARGET_PRESSURE_TYPES,
    CodingPatchPair,
    load_coding_patch_pairs,
)
from pressuretrace.patching.coding_patching_core import (
    build_model_inputs,
    capture_greedy_generation_trace,
    debug_generation_step_patch,
    generation_step_window_label,
    load_model_and_tokenizer,
    resolve_generation_step_window,
)
from pressuretrace.patching.run_coding_route_patching import _load_task_rows_by_task_id
from pressuretrace.utils.io import ensure_directory, write_jsonl

DEBUG_JSONL_FILENAME = "coding_route_patching_debug_qwen-qwen3-14b_off.jsonl"
DEBUG_TXT_FILENAME = "coding_route_patching_debug_qwen-qwen3-14b_off.txt"


@dataclass(frozen=True)
class CodingRoutePatchingDebugPaths:
    """Resolved frozen inputs and output paths for one-pair patch debugging."""

    frozen_root: Path
    manifest_path: Path
    results_path: Path
    patch_pairs_path: Path
    output_path: Path
    summary_txt_path: Path


@dataclass(frozen=True)
class CodingRoutePatchingDebugConfig:
    """Configuration for the one-pair coding patch debug runner."""

    frozen_root: Path
    manifest_path: Path
    results_path: Path
    patch_pairs_path: Path
    output_path: Path
    summary_txt_path: Path
    model_name: str = CODING_V1_MODEL_NAME
    thinking_mode: str = CODING_V1_THINKING_MODE
    pressure_types: tuple[str, ...] = TARGET_PRESSURE_TYPES
    layer: int = -8
    position_window: str = "gen_1"
    base_task_id: str | None = None
    pair_index: int = 0
    top_k: int = 10


@dataclass(frozen=True)
class CodingRoutePatchingDebugArtifacts:
    """Artifacts written by the coding route patch debug runner."""

    output_path: Path
    summary_txt_path: Path
    base_task_id: str
    pressure_type: str
    layer: int
    position_window: str
    rows_written: int


def resolve_debug_paths(
    frozen_root: Path | None = None,
) -> CodingRoutePatchingDebugPaths:
    """Resolve coding patch debug paths under a frozen root."""

    resolved_root = frozen_root or resolve_coding_frozen_root()
    results_root = resolved_root / "results"
    return CodingRoutePatchingDebugPaths(
        frozen_root=resolved_root,
        manifest_path=(
            resolved_root
            / "data"
            / "manifests"
            / "coding_paper_slice_qwen-qwen3-14b_off.jsonl"
        ),
        results_path=results_root / "coding_paper_slice_qwen-qwen3-14b_off.jsonl",
        patch_pairs_path=results_root / "coding_patch_pairs_qwen-qwen3-14b_off.jsonl",
        output_path=results_root / DEBUG_JSONL_FILENAME,
        summary_txt_path=results_root / DEBUG_TXT_FILENAME,
    )


def build_debug_config(
    *,
    frozen_root: Path | None = None,
    manifest_path: Path | None = None,
    results_path: Path | None = None,
    patch_pairs_path: Path | None = None,
    output_path: Path | None = None,
    summary_txt_path: Path | None = None,
    model_name: str = CODING_V1_MODEL_NAME,
    thinking_mode: str = CODING_V1_THINKING_MODE,
    pressure_types: tuple[str, ...] = TARGET_PRESSURE_TYPES,
    layer: int = -8,
    position_window: str = "gen_1",
    base_task_id: str | None = None,
    pair_index: int = 0,
    top_k: int = 10,
) -> CodingRoutePatchingDebugConfig:
    """Build a typed coding patch debug config from explicit overrides."""

    defaults = resolve_debug_paths(frozen_root)
    return CodingRoutePatchingDebugConfig(
        frozen_root=defaults.frozen_root,
        manifest_path=manifest_path or defaults.manifest_path,
        results_path=results_path or defaults.results_path,
        patch_pairs_path=patch_pairs_path or defaults.patch_pairs_path,
        output_path=output_path or defaults.output_path,
        summary_txt_path=summary_txt_path or defaults.summary_txt_path,
        model_name=model_name,
        thinking_mode=thinking_mode,
        pressure_types=pressure_types,
        layer=layer,
        position_window=position_window,
        base_task_id=base_task_id,
        pair_index=pair_index,
        top_k=top_k,
    )


def _select_debug_pair(
    pairs: list[CodingPatchPair],
    *,
    base_task_id: str | None,
    pair_index: int,
) -> CodingPatchPair:
    """Select one coding patch pair by explicit base task id or stable index."""

    if base_task_id is not None:
        for pair in pairs:
            if pair.base_task_id == base_task_id:
                return pair
        raise ValueError(f"No coding patch pair found for base_task_id={base_task_id}.")
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(f"pair_index={pair_index} is out of range for {len(pairs)} pairs.")
    return pairs[pair_index]


def _top_tokens_text(tokens: list[dict[str, Any]]) -> str:
    """Render compact top-k token diagnostics."""

    return ", ".join(
        f"{token['rank']}:{token['token_str']!r}@{token['probability']:.4f}"
        for token in tokens
    )


def _write_summary(
    *,
    config: CodingRoutePatchingDebugConfig,
    pair: CodingPatchPair,
    rows: list[dict[str, Any]],
) -> Path:
    """Write a human-readable summary for one-pair patch debugging."""

    ensure_directory(config.summary_txt_path.parent)
    lines = [
        "PressureTrace coding route patching debug",
        "",
        f"Frozen root: {config.frozen_root}",
        f"Patch pairs: {config.patch_pairs_path}",
        f"Base task: {pair.base_task_id}",
        f"Pressure type: {pair.pressure_type}",
        f"Archetype: {pair.archetype}",
        f"Layer: {config.layer}",
        f"Position window: {generation_step_window_label(config.position_window)}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                (
                    f"{row['direction']} step={row['generation_step'] + 1} "
                    f"top1_changed={row['top1_changed']}"
                ),
                (
                    f"  hidden_delta_l2={row['hidden_delta_l2']:.6f} "
                    f"hidden_delta_max_abs={row['hidden_delta_max_abs']:.6f}"
                ),
                (
                    f"  donor_target_hidden_delta_l2={row['donor_target_hidden_delta_l2']:.6f} "
                    f"donor_target_hidden_delta_max_abs="
                    f"{row['donor_target_hidden_delta_max_abs']:.6f}"
                ),
                (
                    f"  logit_delta_l2={row['logit_delta_l2']:.6f} "
                    f"logit_delta_max_abs={row['logit_delta_max_abs']:.6f}"
                ),
                (
                    f"  baseline_top1={row['baseline_top1_token_str']!r} "
                    f"patched_top1={row['patched_top1_token_str']!r}"
                ),
                "  baseline_top_tokens=" + _top_tokens_text(row["baseline_top_tokens"]),
                "  patched_top_tokens=" + _top_tokens_text(row["patched_top_tokens"]),
                "",
            ]
        )
    config.summary_txt_path.write_text("\n".join(lines), encoding="utf-8")
    return config.summary_txt_path


def run_coding_route_patching_debug(
    config: CodingRoutePatchingDebugConfig,
) -> CodingRoutePatchingDebugArtifacts:
    """Run one-pair coding patch debugging with hidden/logit delta diagnostics."""

    pairs = load_coding_patch_pairs(
        patch_pairs_path=config.patch_pairs_path,
        manifest_path=config.manifest_path,
        results_path=config.results_path,
        pressure_types=config.pressure_types,
    )
    if not pairs:
        raise ValueError(
            "No coding patch pairs were available for debug. Build pairs first for the "
            f"requested pressure types: {config.pressure_types}."
        )
    pair = _select_debug_pair(
        pairs,
        base_task_id=config.base_task_id,
        pair_index=config.pair_index,
    )
    _, results_by_task_id = _load_task_rows_by_task_id(
        manifest_path=config.manifest_path,
        results_path=config.results_path,
    )
    control_result_row = results_by_task_id[str(pair.control_task_id)]
    pressure_result_row = results_by_task_id[str(pair.pressure_task_id)]

    bundle = load_model_and_tokenizer(
        config.model_name,
        thinking_mode=config.thinking_mode,
    )
    control_prompt_inputs = build_model_inputs(bundle, pair.control_prompt)
    pressure_prompt_inputs = build_model_inputs(bundle, pair.pressure_prompt)
    max_required_steps = max(resolve_generation_step_window(config.position_window)) + 1
    control_trace = capture_greedy_generation_trace(
        bundle,
        control_prompt_inputs,
        layer=config.layer,
        max_new_tokens=max_required_steps,
    )
    pressure_trace = capture_greedy_generation_trace(
        bundle,
        pressure_prompt_inputs,
        layer=config.layer,
        max_new_tokens=max_required_steps,
    )

    debug_rows: list[dict[str, Any]] = []
    direction_specs = (
        (
            "rescue",
            pressure_prompt_inputs,
            pressure_trace,
            control_trace,
            pressure_result_row,
        ),
        (
            "induction",
            control_prompt_inputs,
            control_trace,
            pressure_trace,
            control_result_row,
        ),
    )
    for direction, target_inputs, target_trace, donor_trace, target_result_row in direction_specs:
        for generation_step in resolve_generation_step_window(config.position_window):
            debug = debug_generation_step_patch(
                bundle,
                target_inputs,
                target_trace=target_trace,
                donor_trace=donor_trace,
                generation_step=generation_step,
                layer=config.layer,
                top_k=config.top_k,
            )
            debug_rows.append(
                {
                    "base_task_id": pair.base_task_id,
                    "pressure_type": pair.pressure_type,
                    "archetype": pair.archetype,
                    "layer": config.layer,
                    "position_window": generation_step_window_label(config.position_window),
                    "direction": direction,
                    "generation_step": debug.generation_step,
                    "control_task_id": pair.control_task_id,
                    "pressure_task_id": pair.pressure_task_id,
                    "target_task_id": str(target_result_row.get("task_id", "")),
                    "target_original_route_label": str(target_result_row.get("route_label", "")),
                    "prefix_token_ids": list(debug.prefix_token_ids),
                    "baseline_top1_token_id": debug.baseline_top1_token_id,
                    "baseline_top1_token_str": debug.baseline_top1_token_str,
                    "patched_top1_token_id": debug.patched_top1_token_id,
                    "patched_top1_token_str": debug.patched_top1_token_str,
                    "top1_changed": debug.top1_changed,
                    "hidden_delta_l2": debug.hidden_delta_l2,
                    "hidden_delta_max_abs": debug.hidden_delta_max_abs,
                    "donor_target_hidden_delta_l2": debug.donor_target_hidden_delta_l2,
                    "donor_target_hidden_delta_max_abs": (
                        debug.donor_target_hidden_delta_max_abs
                    ),
                    "logit_delta_l2": debug.logit_delta_l2,
                    "logit_delta_max_abs": debug.logit_delta_max_abs,
                    "baseline_top_tokens": [asdict(token) for token in debug.baseline_top_tokens],
                    "patched_top_tokens": [asdict(token) for token in debug.patched_top_tokens],
                }
            )

    write_jsonl(config.output_path, debug_rows)
    _write_summary(
        config=config,
        pair=pair,
        rows=debug_rows,
    )
    return CodingRoutePatchingDebugArtifacts(
        output_path=config.output_path,
        summary_txt_path=config.summary_txt_path,
        base_task_id=pair.base_task_id,
        pressure_type=pair.pressure_type,
        layer=config.layer,
        position_window=generation_step_window_label(config.position_window),
        rows_written=len(debug_rows),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for coding patch sanity debugging."""

    defaults = resolve_debug_paths()
    parser = argparse.ArgumentParser(
        description="Run one-pair debug diagnostics for coding route patching.",
    )
    parser.add_argument("--frozen-root", type=Path, default=defaults.frozen_root)
    parser.add_argument("--patch-pairs-path", type=Path, default=defaults.patch_pairs_path)
    parser.add_argument("--manifest-path", type=Path, default=defaults.manifest_path)
    parser.add_argument("--results-path", type=Path, default=defaults.results_path)
    parser.add_argument("--output-path", type=Path, default=defaults.output_path)
    parser.add_argument("--summary-txt-path", type=Path, default=defaults.summary_txt_path)
    parser.add_argument("--model-name", type=str, default=CODING_V1_MODEL_NAME)
    parser.add_argument("--thinking-mode", type=str, default=CODING_V1_THINKING_MODE)
    parser.add_argument(
        "--pressure-types",
        type=str,
        default=",".join(TARGET_PRESSURE_TYPES),
    )
    parser.add_argument("--layer", type=int, default=-8)
    parser.add_argument("--position-window", type=str, default="gen_1")
    parser.add_argument("--base-task-id", type=str, default=None)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> CodingRoutePatchingDebugArtifacts:
    """Run one-pair coding patch debugging from argparse."""

    args = build_arg_parser().parse_args(argv)
    pressure_types = tuple(
        part.strip() for part in str(args.pressure_types).split(",") if part.strip()
    )
    config = build_debug_config(
        frozen_root=args.frozen_root,
        patch_pairs_path=args.patch_pairs_path,
        manifest_path=args.manifest_path,
        results_path=args.results_path,
        output_path=args.output_path,
        summary_txt_path=args.summary_txt_path,
        model_name=args.model_name,
        thinking_mode=args.thinking_mode,
        pressure_types=pressure_types,
        layer=int(args.layer),
        position_window=str(args.position_window),
        base_task_id=args.base_task_id,
        pair_index=int(args.pair_index),
        top_k=int(args.top_k),
    )
    return run_coding_route_patching_debug(config)


if __name__ == "__main__":
    main()
