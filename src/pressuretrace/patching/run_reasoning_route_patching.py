"""First-pass logit-level route patching for frozen reasoning-family pairs."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pressuretrace.analysis.reasoning_probe_reports import get_frozen_reasoning_probe_paths
from pressuretrace.config import REASONING_V2_MODEL_NAME, REASONING_V2_THINKING_MODE
from pressuretrace.patching.build_reasoning_patch_pairs import (
    ReasoningPatchPair,
    load_reasoning_patch_pairs,
)
from pressuretrace.patching.reasoning_patching_core import (
    ReasoningPatchingBundle,
    final_token_activation_for_layer,
    get_next_token_logits_from_outputs,
    get_prompt_hidden_states,
    load_model_and_tokenizer,
    patch_final_token_activation,
)
from pressuretrace.patching.reasoning_patching_metrics import (
    FIRST_PASS_LAYERS,
    FIRST_PASS_PRESSURE_TYPES,
    AnswerTokenPair,
    aggregate_patch_rows,
    build_answer_token_pair,
    build_patch_comparison_row,
    compute_token_snapshot,
    highlight_summary_rows,
    patch_row_to_dict,
    plot_patch_summary,
    render_summary_text,
    write_summary_csv,
)
from pressuretrace.utils.io import append_jsonl, ensure_directory, prepare_results_file

RESULTS_FILENAME = "reasoning_route_patching_qwen-qwen3-14b_off.jsonl"
SUMMARY_TXT_FILENAME = "reasoning_route_patching_summary_qwen-qwen3-14b_off.txt"
SUMMARY_CSV_FILENAME = "reasoning_route_patching_summary_qwen-qwen3-14b_off.csv"
RESCUE_DELTA_GOLD_PROB_PLOT_FILENAME = (
    "reasoning_route_patching_rescue_delta_gold_prob_qwen-qwen3-14b_off.png"
)
RESCUE_DELTA_MARGIN_PLOT_FILENAME = (
    "reasoning_route_patching_rescue_delta_margin_qwen-qwen3-14b_off.png"
)
INDUCTION_DELTA_SHORTCUT_PROB_PLOT_FILENAME = (
    "reasoning_route_patching_induction_delta_shortcut_prob_qwen-qwen3-14b_off.png"
)


@dataclass(frozen=True)
class RoutePatchingPaths:
    """Resolved repo-local frozen paths for the first patching pass."""

    frozen_root: Path
    manifest_path: Path
    results_path: Path
    patch_pairs_path: Path
    output_path: Path
    summary_txt_path: Path
    summary_csv_path: Path
    rescue_delta_gold_prob_plot_path: Path
    rescue_delta_margin_plot_path: Path
    induction_delta_shortcut_prob_plot_path: Path


@dataclass(frozen=True)
class RoutePatchingConfig:
    """Configuration for a first-pass reasoning route-patching run."""

    frozen_root: Path
    manifest_path: Path
    results_path: Path
    patch_pairs_path: Path
    output_path: Path
    summary_txt_path: Path
    summary_csv_path: Path
    rescue_delta_gold_prob_plot_path: Path
    rescue_delta_margin_plot_path: Path
    induction_delta_shortcut_prob_plot_path: Path
    model_name: str = REASONING_V2_MODEL_NAME
    thinking_mode: str = REASONING_V2_THINKING_MODE
    layers: tuple[int, ...] = FIRST_PASS_LAYERS
    pressure_types: tuple[str, ...] = FIRST_PASS_PRESSURE_TYPES
    max_pairs: int | None = None


@dataclass(frozen=True)
class RoutePatchingArtifacts:
    """Paths and counts written by the route-patching runner."""

    output_path: Path
    summary_txt_path: Path
    summary_csv_path: Path
    rescue_delta_gold_prob_plot_path: Path
    rescue_delta_margin_plot_path: Path
    induction_delta_shortcut_prob_plot_path: Path
    total_pairs_loaded: int
    retained_pairs: int
    skipped_tokenization: int
    rows_written: int


@dataclass(frozen=True)
class EligiblePatchPair:
    """Matched control/pressure pair that survives single-token filtering."""

    pair: ReasoningPatchPair
    answer_tokens: AnswerTokenPair


def resolve_route_patching_paths(frozen_root: Path | None = None) -> RoutePatchingPaths:
    """Resolve the frozen reasoning input and derived output paths."""

    frozen_paths = get_frozen_reasoning_probe_paths(root=frozen_root)
    results_root = frozen_paths.frozen_root / "results"
    return RoutePatchingPaths(
        frozen_root=frozen_paths.frozen_root,
        manifest_path=frozen_paths.manifest_path,
        results_path=frozen_paths.paper_results_path,
        patch_pairs_path=frozen_paths.patch_pairs_path,
        output_path=results_root / RESULTS_FILENAME,
        summary_txt_path=results_root / SUMMARY_TXT_FILENAME,
        summary_csv_path=results_root / SUMMARY_CSV_FILENAME,
        rescue_delta_gold_prob_plot_path=(
            results_root / RESCUE_DELTA_GOLD_PROB_PLOT_FILENAME
        ),
        rescue_delta_margin_plot_path=results_root / RESCUE_DELTA_MARGIN_PLOT_FILENAME,
        induction_delta_shortcut_prob_plot_path=(
            results_root / INDUCTION_DELTA_SHORTCUT_PROB_PLOT_FILENAME
        ),
    )


def build_route_patching_config(
    *,
    frozen_root: Path | None = None,
    manifest_path: Path | None = None,
    results_path: Path | None = None,
    patch_pairs_path: Path | None = None,
    output_path: Path | None = None,
    summary_txt_path: Path | None = None,
    summary_csv_path: Path | None = None,
    rescue_delta_gold_prob_plot_path: Path | None = None,
    rescue_delta_margin_plot_path: Path | None = None,
    induction_delta_shortcut_prob_plot_path: Path | None = None,
    model_name: str = REASONING_V2_MODEL_NAME,
    thinking_mode: str = REASONING_V2_THINKING_MODE,
    layers: tuple[int, ...] = FIRST_PASS_LAYERS,
    pressure_types: tuple[str, ...] = FIRST_PASS_PRESSURE_TYPES,
    max_pairs: int | None = None,
) -> RoutePatchingConfig:
    """Build a route-patching config from explicit overrides."""

    defaults = resolve_route_patching_paths(frozen_root)
    return RoutePatchingConfig(
        frozen_root=defaults.frozen_root,
        manifest_path=manifest_path or defaults.manifest_path,
        results_path=results_path or defaults.results_path,
        patch_pairs_path=patch_pairs_path or defaults.patch_pairs_path,
        output_path=output_path or defaults.output_path,
        summary_txt_path=summary_txt_path or defaults.summary_txt_path,
        summary_csv_path=summary_csv_path or defaults.summary_csv_path,
        rescue_delta_gold_prob_plot_path=(
            rescue_delta_gold_prob_plot_path or defaults.rescue_delta_gold_prob_plot_path
        ),
        rescue_delta_margin_plot_path=(
            rescue_delta_margin_plot_path or defaults.rescue_delta_margin_plot_path
        ),
        induction_delta_shortcut_prob_plot_path=(
            induction_delta_shortcut_prob_plot_path
            or defaults.induction_delta_shortcut_prob_plot_path
        ),
        model_name=model_name,
        thinking_mode=thinking_mode,
        layers=layers,
        pressure_types=pressure_types,
        max_pairs=max_pairs,
    )


def default_route_patching_config() -> RoutePatchingConfig:
    """Return the default repo-local route-patching config."""

    return build_route_patching_config()


def _parse_comma_separated_strings(raw_value: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI argument into a stable tuple."""

    values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not values:
        raise ValueError("At least one comma-separated value is required.")
    return values


def _parse_comma_separated_layers(raw_value: str) -> tuple[int, ...]:
    """Parse a comma-separated layer list into integers."""

    return tuple(int(value) for value in _parse_comma_separated_strings(raw_value))


def _format_counter(counter: Counter[str], order: Sequence[str]) -> str:
    """Render counts in a stable order."""

    return ", ".join(f"{key}={counter.get(key, 0)}" for key in order)


def _summarize_model_devices(model: Any) -> str:
    """Render a compact summary of where the model layers live."""

    if hasattr(model, "hf_device_map"):
        device_counts: Counter[str] = Counter(
            str(device) for device in model.hf_device_map.values()
        )
        return ", ".join(
            f"{device}={count}" for device, count in sorted(device_counts.items())
        )
    try:
        return str(next(model.parameters()).device)
    except StopIteration:  # pragma: no cover - defensive fallback
        return "unknown"


def select_eligible_patch_pairs(
    pairs: Sequence[ReasoningPatchPair],
    tokenizer: Any,
) -> tuple[list[EligiblePatchPair], int]:
    """Keep only pairs whose gold and shortcut answers are single-token strings."""

    retained: list[EligiblePatchPair] = []
    skipped = 0
    for pair in pairs:
        answer_tokens = build_answer_token_pair(
            tokenizer,
            gold_answer=pair.gold_answer,
            shortcut_answer=pair.shortcut_answer,
        )
        if answer_tokens is None:
            skipped += 1
            continue
        retained.append(EligiblePatchPair(pair=pair, answer_tokens=answer_tokens))
    return retained, skipped


def _build_baseline_snapshots(
    bundle: ReasoningPatchingBundle,
    eligible_pair: EligiblePatchPair,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Run baseline forwards for one matched control/pressure pair."""

    control_inputs, control_outputs = get_prompt_hidden_states(
        bundle,
        eligible_pair.pair.control_prompt,
    )
    pressure_inputs, pressure_outputs = get_prompt_hidden_states(
        bundle,
        eligible_pair.pair.pressure_prompt,
    )
    answer_tokens = eligible_pair.answer_tokens
    control_snapshot = compute_token_snapshot(
        get_next_token_logits_from_outputs(control_outputs),
        gold_token_id=answer_tokens.gold_token_id,
        shortcut_token_id=answer_tokens.shortcut_token_id,
        tokenizer=bundle.tokenizer,
    )
    pressure_snapshot = compute_token_snapshot(
        get_next_token_logits_from_outputs(pressure_outputs),
        gold_token_id=answer_tokens.gold_token_id,
        shortcut_token_id=answer_tokens.shortcut_token_id,
        tokenizer=bundle.tokenizer,
    )
    return (
        control_inputs,
        control_outputs,
        pressure_inputs,
        pressure_outputs,
        control_snapshot,
        pressure_snapshot,
    )


def _run_pair_layer_patches(
    bundle: ReasoningPatchingBundle,
    eligible_pair: EligiblePatchPair,
    *,
    layers: Sequence[int],
) -> list[dict[str, Any]]:
    """Run rescue and induction patching for one matched pair across layers."""

    pair = eligible_pair.pair
    answer_tokens = eligible_pair.answer_tokens
    (
        control_inputs,
        control_outputs,
        pressure_inputs,
        pressure_outputs,
        control_snapshot,
        pressure_snapshot,
    ) = _build_baseline_snapshots(bundle, eligible_pair)

    rows: list[dict[str, Any]] = []
    for layer in layers:
        control_activation = final_token_activation_for_layer(
            control_outputs,
            control_inputs,
            model=bundle.model,
            layer=layer,
        )
        pressure_activation = final_token_activation_for_layer(
            pressure_outputs,
            pressure_inputs,
            model=bundle.model,
            layer=layer,
        )

        rescue_logits = patch_final_token_activation(
            bundle,
            pressure_inputs,
            layer=layer,
            donor_final_token_activation=control_activation,
        )
        induction_logits = patch_final_token_activation(
            bundle,
            control_inputs,
            layer=layer,
            donor_final_token_activation=pressure_activation,
        )

        rescue_snapshot = compute_token_snapshot(
            rescue_logits,
            gold_token_id=answer_tokens.gold_token_id,
            shortcut_token_id=answer_tokens.shortcut_token_id,
            tokenizer=bundle.tokenizer,
        )
        induction_snapshot = compute_token_snapshot(
            induction_logits,
            gold_token_id=answer_tokens.gold_token_id,
            shortcut_token_id=answer_tokens.shortcut_token_id,
            tokenizer=bundle.tokenizer,
        )

        rows.append(
            patch_row_to_dict(
                build_patch_comparison_row(
                    base_task_id=pair.base_task_id,
                    pressure_type=pair.pressure_type,
                    layer=layer,
                    direction="rescue",
                    control_task_id=pair.control_task_id,
                    pressure_task_id=pair.pressure_task_id,
                    gold_answer=pair.gold_answer,
                    shortcut_answer=pair.shortcut_answer,
                    answer_tokens=answer_tokens,
                    control_snapshot=control_snapshot,
                    pressure_snapshot=pressure_snapshot,
                    patched_snapshot=rescue_snapshot,
                    metadata=pair.metadata,
                )
            )
        )
        rows.append(
            patch_row_to_dict(
                build_patch_comparison_row(
                    base_task_id=pair.base_task_id,
                    pressure_type=pair.pressure_type,
                    layer=layer,
                    direction="induction",
                    control_task_id=pair.control_task_id,
                    pressure_task_id=pair.pressure_task_id,
                    gold_answer=pair.gold_answer,
                    shortcut_answer=pair.shortcut_answer,
                    answer_tokens=answer_tokens,
                    control_snapshot=control_snapshot,
                    pressure_snapshot=pressure_snapshot,
                    patched_snapshot=induction_snapshot,
                    metadata=pair.metadata,
                )
            )
        )
    return rows


def _write_summary_text(
    *,
    config: RoutePatchingConfig,
    summary_rows: Sequence[Any],
    total_pairs_loaded: int,
    retained_pairs: int,
    skipped_tokenization: int,
    rows_written: int,
) -> Path:
    """Write the human-readable route-patching summary."""

    ensure_directory(config.summary_txt_path.parent)
    grouped_lines = render_summary_text(summary_rows).strip().splitlines()
    grouped_summary = "\n".join(
        grouped_lines[2:]
        if grouped_lines and grouped_lines[0] == "PressureTrace reasoning route patching summary"
        else grouped_lines
    )
    config.summary_txt_path.write_text(
        "\n".join(
            [
                "PressureTrace reasoning route patching summary",
                "",
                f"Frozen root: {config.frozen_root}",
                f"Manifest: {config.manifest_path}",
                f"Results: {config.results_path}",
                f"Patch pairs: {config.patch_pairs_path}",
                f"Model: {config.model_name}",
                f"Thinking mode: {config.thinking_mode}",
                (
                    "First-pass design: prompt-only logit-level patching over the final "
                    "prompt token; rescue and induction directions; no generation."
                ),
                (
                    "Filters: pressure_type in "
                    f"{tuple(config.pressure_types)}; control=robust_correct; "
                    "pressure=shortcut_followed; gold and shortcut answers must each "
                    "tokenize to exactly one token."
                ),
                f"Pairs loaded: {total_pairs_loaded}",
                f"Retained single-token pairs: {retained_pairs}",
                f"Skipped for tokenization: {skipped_tokenization}",
                f"Rows written: {rows_written}",
                "",
                grouped_summary,
                "",
                highlight_summary_rows(summary_rows).rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config.summary_txt_path


def run_reasoning_route_patching(config: RoutePatchingConfig) -> RoutePatchingArtifacts:
    """Run the first logit-level route-patching sweep on frozen reasoning pairs."""

    started_at = time.perf_counter()
    pairs = load_reasoning_patch_pairs(
        patch_pairs_path=config.patch_pairs_path,
        manifest_path=config.manifest_path,
        results_path=config.results_path,
        pressure_types=config.pressure_types,
    )
    total_pairs_loaded = len(pairs)
    bundle = load_model_and_tokenizer(
        config.model_name,
        thinking_mode=config.thinking_mode,
    )
    eligible_pairs, skipped_tokenization = select_eligible_patch_pairs(pairs, bundle.tokenizer)
    if config.max_pairs is not None:
        eligible_pairs = eligible_pairs[: config.max_pairs]

    pressure_counts = Counter(pair.pair.pressure_type for pair in eligible_pairs)
    print(f"Retained pairs: {len(eligible_pairs)}")
    print("Per-pressure counts: " + _format_counter(pressure_counts, config.pressure_types))
    print(f"Skipped for tokenization: {skipped_tokenization}")
    print("Layers tested: " + ", ".join(str(layer) for layer in config.layers))
    print("Model device placement: " + _summarize_model_devices(bundle.model))

    prepare_results_file(config.output_path)
    written_rows: list[dict[str, Any]] = []
    total_pairs = len(eligible_pairs)
    for index, eligible_pair in enumerate(eligible_pairs, start=1):
        pair_started_at = time.perf_counter()
        pair_rows = _run_pair_layer_patches(
            bundle,
            eligible_pair,
            layers=config.layers,
        )
        written_rows.extend(pair_rows)
        for row in pair_rows:
            append_jsonl(config.output_path, row)
        elapsed = time.perf_counter() - pair_started_at
        total_elapsed = time.perf_counter() - started_at
        print(
            f"[{index}/{total_pairs}] {eligible_pair.pair.base_task_id} "
            f"{eligible_pair.pair.pressure_type} "
            f"wrote {len(pair_rows)} rows in {elapsed:.1f}s "
            f"(total {total_elapsed / 60:.1f}m)"
        )

    summary_rows = aggregate_patch_rows(written_rows)
    write_summary_csv(summary_rows, config.summary_csv_path)
    _write_summary_text(
        config=config,
        summary_rows=summary_rows,
        total_pairs_loaded=total_pairs_loaded,
        retained_pairs=len(eligible_pairs),
        skipped_tokenization=skipped_tokenization,
        rows_written=len(written_rows),
    )

    plot_patch_summary(
        summary_rows,
        direction="rescue",
        metric_name="mean_delta_gold_prob",
        output_path=config.rescue_delta_gold_prob_plot_path,
    )
    plot_patch_summary(
        summary_rows,
        direction="rescue",
        metric_name="mean_delta_margin",
        output_path=config.rescue_delta_margin_plot_path,
    )
    plot_patch_summary(
        summary_rows,
        direction="induction",
        metric_name="mean_delta_shortcut_prob",
        output_path=config.induction_delta_shortcut_prob_plot_path,
    )

    print(f"Output rows written: {len(written_rows)}")
    print(f"Results: {config.output_path}")
    print(f"Summary TXT: {config.summary_txt_path}")
    print(f"Summary CSV: {config.summary_csv_path}")
    print(f"Elapsed: {(time.perf_counter() - started_at) / 60:.1f}m")

    return RoutePatchingArtifacts(
        output_path=config.output_path,
        summary_txt_path=config.summary_txt_path,
        summary_csv_path=config.summary_csv_path,
        rescue_delta_gold_prob_plot_path=config.rescue_delta_gold_prob_plot_path,
        rescue_delta_margin_plot_path=config.rescue_delta_margin_plot_path,
        induction_delta_shortcut_prob_plot_path=config.induction_delta_shortcut_prob_plot_path,
        total_pairs_loaded=total_pairs_loaded,
        retained_pairs=len(eligible_pairs),
        skipped_tokenization=skipped_tokenization,
        rows_written=len(written_rows),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the first-pass route-patching runner."""

    defaults = resolve_route_patching_paths()
    parser = argparse.ArgumentParser(
        description="Run first-pass logit-level activation patching over frozen reasoning pairs.",
    )
    parser.add_argument("--frozen-root", type=Path, default=defaults.frozen_root)
    parser.add_argument("--patch-pairs-path", type=Path, default=defaults.patch_pairs_path)
    parser.add_argument("--manifest-path", type=Path, default=defaults.manifest_path)
    parser.add_argument(
        "--results-path",
        "--paper-results-path",
        dest="results_path",
        type=Path,
        default=defaults.results_path,
        help="Frozen reasoning paper-slice results used to reconstruct prompts.",
    )
    parser.add_argument("--output-path", type=Path, default=defaults.output_path)
    parser.add_argument("--summary-txt-path", type=Path, default=defaults.summary_txt_path)
    parser.add_argument("--summary-csv-path", type=Path, default=defaults.summary_csv_path)
    parser.add_argument(
        "--rescue-delta-gold-prob-plot-path",
        type=Path,
        default=defaults.rescue_delta_gold_prob_plot_path,
    )
    parser.add_argument(
        "--rescue-delta-margin-plot-path",
        type=Path,
        default=defaults.rescue_delta_margin_plot_path,
    )
    parser.add_argument(
        "--induction-delta-shortcut-prob-plot-path",
        type=Path,
        default=defaults.induction_delta_shortcut_prob_plot_path,
    )
    parser.add_argument("--model-name", type=str, default=REASONING_V2_MODEL_NAME)
    parser.add_argument("--thinking-mode", type=str, default=REASONING_V2_THINKING_MODE)
    parser.add_argument(
        "--layers",
        type=str,
        default=",".join(str(layer) for layer in FIRST_PASS_LAYERS),
    )
    parser.add_argument(
        "--pressure-types",
        type=str,
        default=",".join(FIRST_PASS_PRESSURE_TYPES),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on retained pairs for debugging.",
    )
    return parser


def build_config_from_args(args: argparse.Namespace) -> RoutePatchingConfig:
    """Convert parsed argparse values into a typed route-patching config."""

    return build_route_patching_config(
        frozen_root=args.frozen_root,
        patch_pairs_path=args.patch_pairs_path,
        manifest_path=args.manifest_path,
        results_path=args.results_path,
        output_path=args.output_path,
        summary_txt_path=args.summary_txt_path,
        summary_csv_path=args.summary_csv_path,
        rescue_delta_gold_prob_plot_path=args.rescue_delta_gold_prob_plot_path,
        rescue_delta_margin_plot_path=args.rescue_delta_margin_plot_path,
        induction_delta_shortcut_prob_plot_path=args.induction_delta_shortcut_prob_plot_path,
        model_name=args.model_name,
        thinking_mode=args.thinking_mode,
        layers=_parse_comma_separated_layers(args.layers),
        pressure_types=_parse_comma_separated_strings(args.pressure_types),
        max_pairs=args.max_pairs,
    )


def main(argv: Sequence[str] | None = None) -> RoutePatchingArtifacts:
    """Run the first-pass reasoning route-patching experiment."""

    args = build_arg_parser().parse_args(argv)
    return run_reasoning_route_patching(build_config_from_args(args))


if __name__ == "__main__":
    main()
