"""Early-generation route patching for the coding-family benchmark."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pressuretrace.behavior.run_coding_paper_slice import CODING_MAX_NEW_TOKENS
from pressuretrace.config import (
    CODING_V1_MODEL_NAME,
    CODING_V1_THINKING_MODE,
    resolve_coding_frozen_root,
)
from pressuretrace.evaluation.coding_eval import evaluate_coding_response
from pressuretrace.patching.build_coding_patch_pairs import (
    CodingPatchPair,
    load_coding_patch_pairs,
)
from pressuretrace.patching.coding_patching_core import (
    CodingPatchingBundle,
    build_model_inputs,
    capture_greedy_generation_trace,
    generation_step_window_label,
    greedy_generate_with_generation_window_patch,
    load_model_and_tokenizer,
)
from pressuretrace.patching.coding_patching_metrics import (
    EARLY_GENERATION_LAYERS,
    EARLY_GENERATION_POSITION_WINDOWS,
    EARLY_GENERATION_PRESSURE_TYPES,
    CodingRoutePatchingRow,
    aggregate_route_patching_rows,
    highlight_route_patching_rows,
    plot_route_patching_position_window_comparison,
    plot_route_patching_success_by_layer,
    render_route_patching_summary_text,
    route_patching_row_to_dict,
    write_route_patching_summary_csv,
)
from pressuretrace.utils.io import (
    append_jsonl,
    ensure_directory,
    prepare_results_file,
    read_jsonl,
)

RESULTS_FILENAME = "coding_route_patching_qwen-qwen3-14b_off.jsonl"
SUMMARY_TXT_FILENAME = "coding_route_patching_summary_qwen-qwen3-14b_off.txt"
SUMMARY_CSV_FILENAME = "coding_route_patching_summary_qwen-qwen3-14b_off.csv"
RESCUE_SUCCESS_PLOT_FILENAME = (
    "coding_route_patching_rescue_success_by_layer_qwen-qwen3-14b_off.png"
)
INDUCTION_SUCCESS_PLOT_FILENAME = (
    "coding_route_patching_induction_success_by_layer_qwen-qwen3-14b_off.png"
)
POSITION_WINDOW_COMPARISON_PLOT_FILENAME = (
    "coding_route_patching_position_window_comparison_qwen-qwen3-14b_off.png"
)


@dataclass(frozen=True)
class CodingRoutePatchingPaths:
    """Resolved repo-local frozen paths for early-generation coding patching."""

    frozen_root: Path
    manifest_path: Path
    results_path: Path
    patch_pairs_path: Path
    output_path: Path
    summary_txt_path: Path
    summary_csv_path: Path
    rescue_success_plot_path: Path
    induction_success_plot_path: Path
    position_window_comparison_plot_path: Path


@dataclass(frozen=True)
class CodingRoutePatchingConfig:
    """Configuration for early-generation coding route patching."""

    frozen_root: Path
    manifest_path: Path
    results_path: Path
    patch_pairs_path: Path
    output_path: Path
    summary_txt_path: Path
    summary_csv_path: Path
    rescue_success_plot_path: Path
    induction_success_plot_path: Path
    position_window_comparison_plot_path: Path
    model_name: str = CODING_V1_MODEL_NAME
    thinking_mode: str = CODING_V1_THINKING_MODE
    layers: tuple[int, ...] = EARLY_GENERATION_LAYERS
    pressure_types: tuple[str, ...] = EARLY_GENERATION_PRESSURE_TYPES
    position_windows: tuple[str, ...] = EARLY_GENERATION_POSITION_WINDOWS
    max_new_tokens: int = CODING_MAX_NEW_TOKENS
    max_pairs: int | None = None


@dataclass(frozen=True)
class CodingRoutePatchingArtifacts:
    """Paths and counts written by the coding route patching runner."""

    output_path: Path
    summary_txt_path: Path
    summary_csv_path: Path
    rescue_success_plot_path: Path
    induction_success_plot_path: Path
    position_window_comparison_plot_path: Path
    total_pairs_loaded: int
    retained_pairs: int
    skipped_tokenization: int
    skipped_missing_task_rows: int
    rows_written: int


@dataclass(frozen=True)
class EligiblePatchPair:
    """Matched control/pressure pair with tokenized donor traces and task rows."""

    pair: CodingPatchPair
    control_task_row: dict[str, Any]
    pressure_task_row: dict[str, Any]
    route_control: str
    route_pressure: str


def resolve_route_patching_paths(
    frozen_root: Path | None = None,
) -> CodingRoutePatchingPaths:
    """Resolve the frozen coding inputs and route-patching output paths."""

    resolved_root = frozen_root or resolve_coding_frozen_root()
    results_root = resolved_root / "results"
    return CodingRoutePatchingPaths(
        frozen_root=resolved_root,
        manifest_path=(
            resolved_root
            / "data"
            / "manifests"
            / "coding_paper_slice_qwen-qwen3-14b_off.jsonl"
        ),
        results_path=results_root / "coding_paper_slice_qwen-qwen3-14b_off.jsonl",
        patch_pairs_path=results_root / "coding_patch_pairs_qwen-qwen3-14b_off.jsonl",
        output_path=results_root / RESULTS_FILENAME,
        summary_txt_path=results_root / SUMMARY_TXT_FILENAME,
        summary_csv_path=results_root / SUMMARY_CSV_FILENAME,
        rescue_success_plot_path=results_root / RESCUE_SUCCESS_PLOT_FILENAME,
        induction_success_plot_path=results_root / INDUCTION_SUCCESS_PLOT_FILENAME,
        position_window_comparison_plot_path=(
            results_root / POSITION_WINDOW_COMPARISON_PLOT_FILENAME
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
    rescue_success_plot_path: Path | None = None,
    induction_success_plot_path: Path | None = None,
    position_window_comparison_plot_path: Path | None = None,
    model_name: str = CODING_V1_MODEL_NAME,
    thinking_mode: str = CODING_V1_THINKING_MODE,
    layers: tuple[int, ...] = EARLY_GENERATION_LAYERS,
    pressure_types: tuple[str, ...] = EARLY_GENERATION_PRESSURE_TYPES,
    position_windows: tuple[str, ...] = EARLY_GENERATION_POSITION_WINDOWS,
    max_new_tokens: int = CODING_MAX_NEW_TOKENS,
    max_pairs: int | None = None,
) -> CodingRoutePatchingConfig:
    """Build a route-patching config from explicit overrides."""

    defaults = resolve_route_patching_paths(frozen_root)
    return CodingRoutePatchingConfig(
        frozen_root=defaults.frozen_root,
        manifest_path=manifest_path or defaults.manifest_path,
        results_path=results_path or defaults.results_path,
        patch_pairs_path=patch_pairs_path or defaults.patch_pairs_path,
        output_path=output_path or defaults.output_path,
        summary_txt_path=summary_txt_path or defaults.summary_txt_path,
        summary_csv_path=summary_csv_path or defaults.summary_csv_path,
        rescue_success_plot_path=(
            rescue_success_plot_path or defaults.rescue_success_plot_path
        ),
        induction_success_plot_path=(
            induction_success_plot_path or defaults.induction_success_plot_path
        ),
        position_window_comparison_plot_path=(
            position_window_comparison_plot_path
            or defaults.position_window_comparison_plot_path
        ),
        model_name=model_name,
        thinking_mode=thinking_mode,
        layers=layers,
        pressure_types=pressure_types,
        position_windows=position_windows,
        max_new_tokens=max_new_tokens,
        max_pairs=max_pairs,
    )


def _parse_comma_separated_strings(raw_value: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI argument into a stable tuple."""

    values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not values:
        raise ValueError("At least one comma-separated value is required.")
    return values


def _parse_comma_separated_layers(raw_value: str) -> tuple[int, ...]:
    """Parse a comma-separated layer list into integers."""

    return tuple(int(value) for value in _parse_comma_separated_strings(raw_value))


def _load_results_row_index(results_path: Path) -> dict[str, dict[str, Any]]:
    """Index frozen coding results rows by task id."""

    return {str(row["task_id"]): row for row in read_jsonl(results_path)}


def _load_task_rows_by_task_id(
    *,
    manifest_path: Path,
    results_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index evaluator task rows and frozen result rows by task id."""

    results_by_task_id = _load_results_row_index(results_path)
    task_rows_by_task_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(manifest_path):
        task_id = str(row["task_id"])
        manifest_row = dict(row)
        result_row = results_by_task_id.get(task_id)
        if result_row is not None:
            manifest_prompt = str(manifest_row.get("prompt", ""))
            result_prompt = str(result_row.get("prompt", ""))
            if manifest_prompt and result_prompt and manifest_prompt != result_prompt:
                raise ValueError(
                    f"Prompt mismatch between manifest and results for task_id={task_id}."
                )
        task_rows_by_task_id[task_id] = manifest_row
    return task_rows_by_task_id, results_by_task_id


def _format_counter(counter: Counter[str], order: Sequence[str]) -> str:
    """Render counts in a stable order."""

    return ", ".join(f"{key}={counter.get(key, 0)}" for key in order)


def select_eligible_patch_pairs(
    pairs: Sequence[CodingPatchPair],
    *,
    task_rows_by_task_id: dict[str, dict[str, Any]],
    results_by_task_id: dict[str, dict[str, Any]],
) -> tuple[list[EligiblePatchPair], int]:
    """Keep only patch pairs with the evaluator task rows needed for patching."""

    retained_pairs: list[EligiblePatchPair] = []
    skipped_missing_task_rows = 0
    for pair in pairs:
        control_task_row = task_rows_by_task_id.get(pair.control_task_id)
        pressure_task_row = task_rows_by_task_id.get(pair.pressure_task_id)
        control_result_row = results_by_task_id.get(pair.control_task_id)
        pressure_result_row = results_by_task_id.get(pair.pressure_task_id)
        if (
            control_task_row is None
            or pressure_task_row is None
            or control_result_row is None
            or pressure_result_row is None
        ):
            skipped_missing_task_rows += 1
            continue

        retained_pairs.append(
            EligiblePatchPair(
                pair=pair,
                control_task_row=control_task_row,
                pressure_task_row=pressure_task_row,
                route_control=str(control_result_row.get("route_label", "")),
                route_pressure=str(pressure_result_row.get("route_label", "")),
            )
        )
    return retained_pairs, skipped_missing_task_rows


def _build_route_patching_row(
    *,
    eligible_pair: EligiblePatchPair,
    direction: str,
    layer: int,
    position_window: str,
    patch_applied: bool,
    top1_changed: bool,
    patched_step_count: int,
    patched_completion: str,
    evaluation: Any,
) -> CodingRoutePatchingRow:
    """Build one flat row for a patched completion and evaluator outcome."""

    rescue_success = direction == "rescue" and evaluation.route_label == "robust_success"
    induction_success = direction == "induction" and evaluation.route_label == "shortcut_success"
    return CodingRoutePatchingRow(
        base_task_id=eligible_pair.pair.base_task_id,
        archetype=eligible_pair.pair.archetype,
        pressure_type=eligible_pair.pair.pressure_type,
        layer=layer,
        position_window=position_window,
        direction=direction,
        control_task_id=eligible_pair.pair.control_task_id,
        pressure_task_id=eligible_pair.pair.pressure_task_id,
        route_control=eligible_pair.route_control,
        route_pressure=eligible_pair.route_pressure,
        patched_route_label=evaluation.route_label,
        patched_visible_pass=evaluation.passed_visible_tests,
        patched_hidden_pass=evaluation.passed_hidden_tests,
        patched_failure_subtype=evaluation.failure_subtype,
        patched_visible_failure_names=list(evaluation.visible_failure_names),
        patched_hidden_failure_names=list(evaluation.hidden_failure_names),
        rescue_success=rescue_success,
        induction_success=induction_success,
        visible_pass_hidden_fail_after_patch=(
            evaluation.passed_visible_tests and not evaluation.passed_hidden_tests
        ),
        patch_applied=patch_applied,
        top1_changed=top1_changed,
        patched_step_count=patched_step_count,
        patched_completion=patched_completion,
        patched_extracted_code=evaluation.extracted_code,
        metadata={**eligible_pair.pair.metadata, "evaluation_notes": list(evaluation.notes)},
    )


def _run_pair_layer_window(
    *,
    bundle: CodingPatchingBundle,
    eligible_pair: EligiblePatchPair,
    layer: int,
    position_window: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run rescue and induction for one pair/layer/window cell."""

    control_prompt_inputs = build_model_inputs(bundle, eligible_pair.pair.control_prompt)
    pressure_prompt_inputs = build_model_inputs(bundle, eligible_pair.pair.pressure_prompt)
    control_trace = capture_greedy_generation_trace(
        bundle,
        control_prompt_inputs,
        layer=layer,
        max_new_tokens=max_new_tokens,
    )
    pressure_trace = capture_greedy_generation_trace(
        bundle,
        pressure_prompt_inputs,
        layer=layer,
        max_new_tokens=max_new_tokens,
    )
    rows: list[dict[str, Any]] = []
    direction_specs = (
        (
            "rescue",
            pressure_prompt_inputs,
            control_trace,
            eligible_pair.pressure_task_row,
        ),
        (
            "induction",
            control_prompt_inputs,
            pressure_trace,
            eligible_pair.control_task_row,
        ),
    )
    for direction, target_prompt_inputs, donor_trace, target_task_row in direction_specs:
        generation_result = greedy_generate_with_generation_window_patch(
            bundle,
            target_prompt_inputs,
            layer=layer,
            patch_window=position_window,
            donor_trace=donor_trace,
            max_new_tokens=max_new_tokens,
        )
        evaluation = evaluate_coding_response(
            target_task_row,
            generation_result.generated_text,
        )
        rows.append(
            route_patching_row_to_dict(
                _build_route_patching_row(
                    eligible_pair=eligible_pair,
                    direction=direction,
                    layer=layer,
                    position_window=generation_step_window_label(position_window),
                    patch_applied=bool(generation_result.step_diagnostics),
                    top1_changed=any(
                        diagnostic.baseline_top1_token_id != diagnostic.patched_top1_token_id
                        for diagnostic in generation_result.step_diagnostics
                    ),
                    patched_step_count=len(generation_result.step_diagnostics),
                    patched_completion=generation_result.generated_text,
                    evaluation=evaluation,
                )
            )
        )
    return rows


def _write_summary_text(
    *,
    config: CodingRoutePatchingConfig,
    summary_rows: Sequence[Any],
    total_pairs_loaded: int,
    retained_pairs: int,
    skipped_tokenization: int,
    skipped_missing_task_rows: int,
    rows_written: int,
) -> Path:
    """Write the human-readable route patching summary."""

    ensure_directory(config.summary_txt_path.parent)
    grouped_lines = render_route_patching_summary_text(summary_rows).strip().splitlines()
    grouped_summary = "\n".join(
        grouped_lines[2:]
        if grouped_lines and grouped_lines[0] == "PressureTrace coding route patching summary"
        else grouped_lines
    )
    config.summary_txt_path.write_text(
        "\n".join(
            [
                "PressureTrace coding route patching summary",
                "",
                f"Frozen root: {config.frozen_root}",
                f"Manifest: {config.manifest_path}",
                f"Results: {config.results_path}",
                f"Patch pairs: {config.patch_pairs_path}",
                f"Model: {config.model_name}",
                f"Thinking mode: {config.thinking_mode}",
                (
                    "Design: early-generation hidden-state patching over coding completions; "
                    "patch aligned generation steps in windows gen_1 / gen_1_3 / gen_1_5, "
                    "then greedily complete the code and re-evaluate the full completion."
                ),
                (
                    "Filters: archetype=visible_test_hardcoding; pressure_type in "
                    f"{tuple(config.pressure_types)}; control=robust_success; "
                    "pressure=shortcut_success."
                ),
                f"Pairs loaded: {total_pairs_loaded}",
                f"Retained pairs: {retained_pairs}",
                f"Skipped for tokenization: {skipped_tokenization}",
                f"Skipped for missing task rows: {skipped_missing_task_rows}",
                f"Rows written: {rows_written}",
                "",
                grouped_summary,
                "",
                highlight_route_patching_rows(summary_rows).rstrip(),
                "",
                (
                    "Interpretation note: this setup is stronger than the previous "
                    "final-prompt-only patching attempt if rescue/induction success "
                    "appears non-null in these early-generation windows."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config.summary_txt_path


def run_coding_route_patching(
    config: CodingRoutePatchingConfig,
) -> CodingRoutePatchingArtifacts:
    """Run early-generation coding route patching over frozen matched pairs."""

    started_at = time.perf_counter()
    pairs = load_coding_patch_pairs(
        patch_pairs_path=config.patch_pairs_path,
        manifest_path=config.manifest_path,
        results_path=config.results_path,
        pressure_types=config.pressure_types,
    )
    total_pairs_loaded = len(pairs)
    task_rows_by_task_id, results_by_task_id = _load_task_rows_by_task_id(
        manifest_path=config.manifest_path,
        results_path=config.results_path,
    )
    bundle = load_model_and_tokenizer(
        config.model_name,
        thinking_mode=config.thinking_mode,
    )
    eligible_pairs, skipped_missing_task_rows = select_eligible_patch_pairs(
        pairs,
        task_rows_by_task_id=task_rows_by_task_id,
        results_by_task_id=results_by_task_id,
    )
    skipped_tokenization = 0
    if config.max_pairs is not None:
        eligible_pairs = eligible_pairs[: config.max_pairs]

    pressure_counts = Counter(pair.pair.pressure_type for pair in eligible_pairs)
    print(f"Retained pairs: {len(eligible_pairs)}")
    print("Per-pressure counts: " + _format_counter(pressure_counts, config.pressure_types))
    print(f"Skipped for tokenization: {skipped_tokenization}")
    print(f"Skipped for missing task rows: {skipped_missing_task_rows}")
    print("Layers tested: " + ", ".join(str(layer) for layer in config.layers))
    print("Position windows: " + ", ".join(config.position_windows))

    prepare_results_file(config.output_path)
    written_rows: list[dict[str, Any]] = []
    total_pairs = len(eligible_pairs)
    for index, eligible_pair in enumerate(eligible_pairs, start=1):
        pair_started_at = time.perf_counter()
        for layer in config.layers:
            for position_window in config.position_windows:
                pair_rows = _run_pair_layer_window(
                    bundle=bundle,
                    eligible_pair=eligible_pair,
                    layer=layer,
                    position_window=position_window,
                    max_new_tokens=config.max_new_tokens,
                )
                written_rows.extend(pair_rows)
                for row in pair_rows:
                    append_jsonl(config.output_path, row)
        elapsed = time.perf_counter() - pair_started_at
        total_elapsed = time.perf_counter() - started_at
        print(
            f"[{index}/{total_pairs}] {eligible_pair.pair.base_task_id} "
            f"{eligible_pair.pair.pressure_type} wrote "
            f"{len(config.layers) * len(config.position_windows) * 2} cells "
            f"in {elapsed:.1f}s (total {total_elapsed / 60:.1f}m)"
        )

    summary_rows = aggregate_route_patching_rows(written_rows)
    write_route_patching_summary_csv(summary_rows, config.summary_csv_path)
    _write_summary_text(
        config=config,
        summary_rows=summary_rows,
        total_pairs_loaded=total_pairs_loaded,
        retained_pairs=len(eligible_pairs),
        skipped_tokenization=skipped_tokenization,
        skipped_missing_task_rows=skipped_missing_task_rows,
        rows_written=len(written_rows),
    )
    plot_route_patching_success_by_layer(
        summary_rows,
        direction="rescue",
        output_path=config.rescue_success_plot_path,
    )
    plot_route_patching_success_by_layer(
        summary_rows,
        direction="induction",
        output_path=config.induction_success_plot_path,
    )
    plot_route_patching_position_window_comparison(
        summary_rows,
        output_path=config.position_window_comparison_plot_path,
    )

    print(f"Output rows written: {len(written_rows)}")
    print(f"Results: {config.output_path}")
    print(f"Summary TXT: {config.summary_txt_path}")
    print(f"Summary CSV: {config.summary_csv_path}")
    print(f"Elapsed: {(time.perf_counter() - started_at) / 60:.1f}m")

    return CodingRoutePatchingArtifacts(
        output_path=config.output_path,
        summary_txt_path=config.summary_txt_path,
        summary_csv_path=config.summary_csv_path,
        rescue_success_plot_path=config.rescue_success_plot_path,
        induction_success_plot_path=config.induction_success_plot_path,
        position_window_comparison_plot_path=config.position_window_comparison_plot_path,
        total_pairs_loaded=total_pairs_loaded,
        retained_pairs=len(eligible_pairs),
        skipped_tokenization=skipped_tokenization,
        skipped_missing_task_rows=skipped_missing_task_rows,
        rows_written=len(written_rows),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for coding route patching."""

    defaults = resolve_route_patching_paths()
    parser = argparse.ArgumentParser(
        description="Run early-generation route patching over frozen coding pairs.",
    )
    parser.add_argument("--frozen-root", type=Path, default=defaults.frozen_root)
    parser.add_argument("--patch-pairs-path", type=Path, default=defaults.patch_pairs_path)
    parser.add_argument("--manifest-path", type=Path, default=defaults.manifest_path)
    parser.add_argument("--results-path", type=Path, default=defaults.results_path)
    parser.add_argument("--output-path", type=Path, default=defaults.output_path)
    parser.add_argument("--summary-txt-path", type=Path, default=defaults.summary_txt_path)
    parser.add_argument("--summary-csv-path", type=Path, default=defaults.summary_csv_path)
    parser.add_argument(
        "--rescue-success-plot-path",
        type=Path,
        default=defaults.rescue_success_plot_path,
    )
    parser.add_argument(
        "--induction-success-plot-path",
        type=Path,
        default=defaults.induction_success_plot_path,
    )
    parser.add_argument(
        "--position-window-comparison-plot-path",
        type=Path,
        default=defaults.position_window_comparison_plot_path,
    )
    parser.add_argument("--model-name", type=str, default=CODING_V1_MODEL_NAME)
    parser.add_argument("--thinking-mode", type=str, default=CODING_V1_THINKING_MODE)
    parser.add_argument(
        "--layers",
        type=str,
        default=",".join(str(layer) for layer in EARLY_GENERATION_LAYERS),
    )
    parser.add_argument(
        "--pressure-types",
        type=str,
        default=",".join(EARLY_GENERATION_PRESSURE_TYPES),
    )
    parser.add_argument(
        "--position-windows",
        type=str,
        default=",".join(EARLY_GENERATION_POSITION_WINDOWS),
    )
    parser.add_argument("--max-new-tokens", type=int, default=CODING_MAX_NEW_TOKENS)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser


def build_config_from_args(args: argparse.Namespace) -> CodingRoutePatchingConfig:
    """Convert parsed argparse values into a typed route patching config."""

    return build_route_patching_config(
        frozen_root=args.frozen_root,
        patch_pairs_path=args.patch_pairs_path,
        manifest_path=args.manifest_path,
        results_path=args.results_path,
        output_path=args.output_path,
        summary_txt_path=args.summary_txt_path,
        summary_csv_path=args.summary_csv_path,
        rescue_success_plot_path=args.rescue_success_plot_path,
        induction_success_plot_path=args.induction_success_plot_path,
        position_window_comparison_plot_path=args.position_window_comparison_plot_path,
        model_name=args.model_name,
        thinking_mode=args.thinking_mode,
        layers=_parse_comma_separated_layers(args.layers),
        pressure_types=_parse_comma_separated_strings(args.pressure_types),
        position_windows=_parse_comma_separated_strings(args.position_windows),
        max_new_tokens=int(args.max_new_tokens),
        max_pairs=args.max_pairs,
    )


def main(argv: Sequence[str] | None = None) -> CodingRoutePatchingArtifacts:
    """Run the early-generation coding route patching experiment."""

    args = build_arg_parser().parse_args(argv)
    return run_coding_route_patching(build_config_from_args(args))


if __name__ == "__main__":
    main()
