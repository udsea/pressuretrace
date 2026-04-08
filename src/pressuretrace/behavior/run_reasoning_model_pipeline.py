"""Orchestrate the full reasoning-family replication pipeline for one model."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from pressuretrace.behavior.build_control_robust_slice import build_control_robust_slice
from pressuretrace.behavior.materialize_reasoning_slice import materialize_reasoning_slice
from pressuretrace.behavior.reasoning_runtime import slugify_model_name
from pressuretrace.behavior.run_reasoning_benchmark_v2 import run_reasoning_manifest_v2
from pressuretrace.behavior.run_reasoning_control_only import run_reasoning_control_only
from pressuretrace.generation.reasoning.make_reasoning_tasks_v2 import (
    build_reasoning_all_valid_transforms_v2,
)
from pressuretrace.patching.build_reasoning_patch_pairs import build_reasoning_patch_pairs
from pressuretrace.patching.run_reasoning_route_patching import (
    build_route_patching_config,
    run_reasoning_route_patching,
)
from pressuretrace.paths import manifests_dir, repo_root
from pressuretrace.probes.build_reasoning_probe_dataset import build_reasoning_probe_dataset
from pressuretrace.probes.extract_hidden_states_reasoning import (
    ReasoningProbeExtractionConfig,
    extract_reasoning_hidden_states,
)
from pressuretrace.probes.train_reasoning_probes import ProbeTrainingConfig, train_reasoning_probes
from pressuretrace.utils.io import ensure_directory, read_jsonl

SUPPORTED_REPLICATION_MODELS: tuple[str, ...] = (
    "google/gemma-3-27b-it",
    "sarvamai/sarvam-30b",
)


@dataclass(frozen=True)
class ReasoningReplicationPaths:
    """Materialized file layout for one model-specific frozen replication root."""

    frozen_root: Path
    pool_manifest: Path
    control_results: Path
    robust_slice: Path
    paper_manifest: Path
    paper_results: Path
    probe_hidden_states: Path
    probe_dataset: Path
    probe_metrics_jsonl: Path
    probe_metrics_csv: Path
    probe_summary_txt: Path
    patch_pairs: Path
    route_patching_results: Path
    route_patching_summary_txt: Path
    route_patching_summary_csv: Path
    route_patching_rescue_delta_gold_prob_plot: Path
    route_patching_rescue_delta_margin_plot: Path
    route_patching_induction_delta_shortcut_prob_plot: Path
    run_info_txt: Path


@dataclass(frozen=True)
class ReasoningModelPipelineConfig:
    """Configuration for one end-to-end reasoning replication run."""

    model_name: str
    frozen_root: Path
    thinking_mode: str = "off"
    split: str = "test"
    limit: int | None = None
    batch_size: int = 1
    resume: bool = True
    skip_probes: bool = False
    skip_patching: bool = False
    reuse_pool: bool = True
    show_progress: bool = True


def default_replication_frozen_root(model_name: str, thinking_mode: str) -> Path:
    """Return the default frozen root for a replicated reasoning run."""

    model_slug = slugify_model_name(model_name)
    return repo_root() / "pressuretrace-frozen" / f"reasoning_v2_{model_slug}_{thinking_mode}"


def reasoning_replication_paths(
    model_name: str,
    frozen_root: Path,
    thinking_mode: str,
) -> ReasoningReplicationPaths:
    """Build the model-specific frozen file layout under one root."""

    model_slug = slugify_model_name(model_name)
    manifests_root = frozen_root / "data" / "manifests"
    splits_root = frozen_root / "data" / "splits"
    results_root = frozen_root / "results"
    suffix = f"{model_slug}_{thinking_mode}"
    return ReasoningReplicationPaths(
        frozen_root=frozen_root,
        pool_manifest=manifests_root / "reasoning_all_valid_transforms.jsonl",
        control_results=results_root / f"reasoning_control_only_{suffix}.jsonl",
        robust_slice=splits_root / f"reasoning_control_robust_slice_{suffix}.jsonl",
        paper_manifest=manifests_root / f"reasoning_paper_slice_{suffix}.jsonl",
        paper_results=results_root / f"reasoning_paper_slice_{suffix}.jsonl",
        probe_hidden_states=results_root / f"reasoning_probe_hidden_states_{suffix}.jsonl",
        probe_dataset=results_root / f"reasoning_probe_dataset_{suffix}.jsonl",
        probe_metrics_jsonl=results_root / f"reasoning_probe_metrics_{suffix}.jsonl",
        probe_metrics_csv=results_root / f"reasoning_probe_metrics_{suffix}.csv",
        probe_summary_txt=results_root / f"reasoning_probe_summary_{suffix}.txt",
        patch_pairs=results_root / f"reasoning_patch_pairs_{suffix}.jsonl",
        route_patching_results=results_root / f"reasoning_route_patching_{suffix}.jsonl",
        route_patching_summary_txt=(
            results_root / f"reasoning_route_patching_summary_{suffix}.txt"
        ),
        route_patching_summary_csv=(
            results_root / f"reasoning_route_patching_summary_{suffix}.csv"
        ),
        route_patching_rescue_delta_gold_prob_plot=(
            results_root / f"reasoning_route_patching_rescue_delta_gold_prob_{suffix}.png"
        ),
        route_patching_rescue_delta_margin_plot=(
            results_root / f"reasoning_route_patching_rescue_delta_margin_{suffix}.png"
        ),
        route_patching_induction_delta_shortcut_prob_plot=(
            results_root / f"reasoning_route_patching_induction_delta_shortcut_prob_{suffix}.png"
        ),
        run_info_txt=frozen_root / "RUN_INFO.txt",
    )


def _prepare_frozen_directories(paths: ReasoningReplicationPaths) -> None:
    """Ensure the model-specific frozen directory layout exists."""

    ensure_directory(paths.frozen_root / "data" / "manifests")
    ensure_directory(paths.frozen_root / "data" / "splits")
    ensure_directory(paths.frozen_root / "results")


def _shared_pool_manifest_path() -> Path:
    """Return the shared repository-level transform pool path."""

    return manifests_dir() / "reasoning_all_valid_transforms.jsonl"


def _resolve_pool_manifest(
    config: ReasoningModelPipelineConfig,
    paths: ReasoningReplicationPaths,
) -> Path:
    """Reuse or rebuild the transformed reasoning pool for this replication run."""

    if config.limit is not None:
        return build_reasoning_all_valid_transforms_v2(
            split=config.split,
            limit=config.limit,
            output_path=paths.pool_manifest,
        )

    if config.reuse_pool and paths.pool_manifest.exists():
        print(f"Reusing model-local transform pool: {paths.pool_manifest}")
        return paths.pool_manifest

    shared_pool = _shared_pool_manifest_path()
    if config.reuse_pool and shared_pool.exists():
        ensure_directory(paths.pool_manifest.parent)
        shutil.copy2(shared_pool, paths.pool_manifest)
        print(f"Copied shared transform pool into frozen root: {paths.pool_manifest}")
        return paths.pool_manifest

    return build_reasoning_all_valid_transforms_v2(
        split=config.split,
        limit=config.limit,
        output_path=paths.pool_manifest,
    )


def _write_probe_metrics_csv(metrics_path: Path, csv_path: Path) -> Path:
    """Flatten probe metrics JSONL into a compact CSV file."""

    rows = [dict(row) for row in read_jsonl(metrics_path)]
    if not rows:
        raise ValueError("Probe metrics JSONL is empty.")

    fieldnames = [
        "kind",
        "feature_set",
        "layer",
        "representation",
        "roc_auc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "inverse_auc",
        "train_size",
        "test_size",
        "train_pos",
        "train_neg",
        "test_pos",
        "test_neg",
        "mean_prob_y1",
        "mean_prob_y0",
    ]
    ensure_directory(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return csv_path


def _path_is_ready(path: Path) -> bool:
    """Return whether an output path exists and is non-empty."""

    return path.exists() and path.stat().st_size > 0


def _count_jsonl_rows(path: Path) -> int | None:
    """Count JSONL rows for progress reporting, returning None for unreadable files."""

    try:
        return len(read_jsonl(path))
    except Exception:
        return None


def _expected_control_row_count(pool_manifest_path: Path) -> int:
    """Count the expected number of control rows in the transformed pool."""

    return sum(
        1
        for row in read_jsonl(pool_manifest_path)
        if str(row.get("pressure_type", "")) == "control"
    )


def _expected_manifest_row_count(manifest_path: Path) -> int | None:
    """Count the expected number of rows in a manifest JSONL when it exists."""

    if not manifest_path.exists():
        return None
    return len(read_jsonl(manifest_path))


def _has_expected_rows(path: Path, expected_rows: int | None) -> bool:
    """Return whether a JSONL artifact exists and matches the expected row count."""

    if expected_rows is None:
        return False
    actual_rows = _count_jsonl_rows(path)
    return actual_rows == expected_rows and actual_rows is not None


def _log(message: str, *, console: Console | None = None) -> None:
    """Print a line immediately, using rich when available."""

    if console is not None:
        console.print(message)
        return
    print(message, flush=True)


def _print_stage_header(
    stage_number: int,
    total_stages: int,
    label: str,
    destination: Path,
    *,
    console: Console | None = None,
) -> None:
    """Print a stable stage header with its main output path."""

    _log(f"Stage {stage_number}/{total_stages}: {label}", console=console)
    _log(f"  output: {destination}", console=console)


def _print_skip_existing(path: Path, *, console: Console | None = None) -> None:
    """Print a concise skip-existing message."""

    row_count = _count_jsonl_rows(path)
    if row_count is None:
        _log(f"  skip existing: {path}", console=console)
    else:
        _log(f"  skip existing: {path} ({row_count} rows)", console=console)


def _print_rebuild_existing(
    path: Path,
    *,
    expected_rows: int | None,
    console: Console | None = None,
) -> None:
    """Explain why an existing stage output is being recomputed."""

    actual_rows = _count_jsonl_rows(path)
    if actual_rows is None:
        return
    if expected_rows is None:
        _log(f"  existing output not trusted yet: {path} ({actual_rows} rows)", console=console)
        return
    if actual_rows != expected_rows:
        _log(
            "  existing output incomplete or stale: "
            f"{path} ({actual_rows}/{expected_rows} rows); recomputing",
            console=console,
        )


def _print_rebuild_due_to_upstream(path: Path, *, console: Console | None = None) -> None:
    """Explain that a stage is being recomputed because an upstream stage changed."""

    if path.exists():
        _log(f"  upstream changed; recomputing {path}", console=console)


def _write_run_info(
    config: ReasoningModelPipelineConfig,
    paths: ReasoningReplicationPaths,
    pool_manifest_path: Path,
) -> Path:
    """Write a compact run-info manifest for the replicated frozen root."""

    lines = [
        f"model={config.model_name}",
        f"thinking_mode={config.thinking_mode}",
        f"split={config.split}",
        f"limit={config.limit if config.limit is not None else 'all'}",
        f"batch_size={config.batch_size}",
        f"resume={config.resume}",
        f"transformed_pool={pool_manifest_path}",
        f"control_only_results={paths.control_results}",
        f"control_robust_slice={paths.robust_slice}",
        f"paper_slice_manifest={paths.paper_manifest}",
        f"paper_slice_results={paths.paper_results}",
        f"probe_hidden_states={paths.probe_hidden_states}",
        f"probe_dataset={paths.probe_dataset}",
        f"probe_metrics_jsonl={paths.probe_metrics_jsonl}",
        f"probe_metrics_csv={paths.probe_metrics_csv}",
        f"probe_summary={paths.probe_summary_txt}",
        f"patch_pairs={paths.patch_pairs}",
        f"route_patching_results={paths.route_patching_results}",
        f"route_patching_summary_txt={paths.route_patching_summary_txt}",
        f"route_patching_summary_csv={paths.route_patching_summary_csv}",
        (
            "route_patching_rescue_delta_gold_prob_plot="
            f"{paths.route_patching_rescue_delta_gold_prob_plot}"
        ),
        f"route_patching_rescue_delta_margin_plot={paths.route_patching_rescue_delta_margin_plot}",
        (
            "route_patching_induction_delta_shortcut_prob_plot="
            f"{paths.route_patching_induction_delta_shortcut_prob_plot}"
        ),
        f"skip_probes={config.skip_probes}",
        f"skip_patching={config.skip_patching}",
        f"reuse_pool={config.reuse_pool}",
    ]
    paths.run_info_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths.run_info_txt


def run_reasoning_model_pipeline(config: ReasoningModelPipelineConfig) -> ReasoningReplicationPaths:
    """Run the full reasoning-family replication pipeline for one model."""

    if config.thinking_mode != "off":
        raise ValueError(
            "Reasoning replication pipeline currently supports "
            "thinking_mode='off' only."
        )
    if config.model_name not in SUPPORTED_REPLICATION_MODELS:
        supported = ", ".join(SUPPORTED_REPLICATION_MODELS)
        raise ValueError(
            f"Unsupported replication model '{config.model_name}'. "
            f"Supported models: {supported}."
        )

    _prepare_frozen_directories(
        paths := reasoning_replication_paths(
            model_name=config.model_name,
            frozen_root=config.frozen_root,
            thinking_mode=config.thinking_mode,
        )
    )
    console = Console() if config.show_progress else None
    pool_manifest_path = _resolve_pool_manifest(config, paths)

    _log(f"Model: {config.model_name}", console=console)
    _log(f"Thinking mode: {config.thinking_mode}", console=console)
    _log(f"Batch size: {config.batch_size}", console=console)
    _log(f"Resume enabled: {config.resume}", console=console)
    _log(f"Frozen root: {paths.frozen_root}", console=console)
    _log(f"Transformed pool: {pool_manifest_path}", console=console)

    control_refreshed = False
    _print_stage_header(1, 6, "control-only run", paths.control_results, console=console)
    expected_control_rows = _expected_control_row_count(pool_manifest_path)
    if config.resume and _has_expected_rows(paths.control_results, expected_control_rows):
        _print_skip_existing(paths.control_results, console=console)
    else:
        if config.resume and paths.control_results.exists():
            _print_rebuild_existing(
                paths.control_results,
                expected_rows=expected_control_rows,
                console=console,
            )
        run_reasoning_control_only(
            manifest_path=pool_manifest_path,
            model_name=config.model_name,
            output_path=paths.control_results,
            thinking_mode=config.thinking_mode,
            batch_size=config.batch_size,
            console=console,
            show_progress=config.show_progress,
        )
        control_refreshed = True

    robust_slice_refreshed = False
    _print_stage_header(2, 6, "build control-robust slice", paths.robust_slice, console=console)
    if not control_refreshed and config.resume and _path_is_ready(paths.robust_slice):
        _print_skip_existing(paths.robust_slice, console=console)
    else:
        if control_refreshed:
            _print_rebuild_due_to_upstream(paths.robust_slice, console=console)
        build_control_robust_slice(
            control_results_path=paths.control_results,
            output_path=paths.robust_slice,
        )
        robust_slice_refreshed = True

    paper_manifest_refreshed = False
    _print_stage_header(3, 6, "materialize paper slice", paths.paper_manifest, console=console)
    if not robust_slice_refreshed and config.resume and _path_is_ready(paths.paper_manifest):
        _print_skip_existing(paths.paper_manifest, console=console)
    else:
        if robust_slice_refreshed:
            _print_rebuild_due_to_upstream(paths.paper_manifest, console=console)
        materialize_reasoning_slice(
            manifest_path=pool_manifest_path,
            slice_path=paths.robust_slice,
            output_path=paths.paper_manifest,
        )
        paper_manifest_refreshed = True

    paper_results_refreshed = False
    _print_stage_header(4, 6, "run paper slice", paths.paper_results, console=console)
    expected_paper_rows = _expected_manifest_row_count(paths.paper_manifest)
    if (
        not paper_manifest_refreshed
        and config.resume
        and _has_expected_rows(paths.paper_results, expected_paper_rows)
    ):
        _print_skip_existing(paths.paper_results, console=console)
    else:
        if paper_manifest_refreshed:
            _print_rebuild_due_to_upstream(paths.paper_results, console=console)
        elif config.resume and paths.paper_results.exists():
            _print_rebuild_existing(
                paths.paper_results,
                expected_rows=expected_paper_rows,
                console=console,
            )
        run_reasoning_manifest_v2(
            manifest_path=paths.paper_manifest,
            model_name=config.model_name,
            pressure_type="all",
            output_path=paths.paper_results,
            include_control=True,
            thinking_mode=config.thinking_mode,
            batch_size=config.batch_size,
            console=console,
            show_progress=config.show_progress,
        )
        paper_results_refreshed = True

    if config.skip_probes:
        _log("Stage 5/6: probe pipeline skipped", console=console)
    else:
        _log("Stage 5/6: run probe pipeline", console=console)
        probe_hidden_states_refreshed = False
        if (
            not paper_results_refreshed
            and config.resume
            and _path_is_ready(paths.probe_hidden_states)
        ):
            _print_skip_existing(paths.probe_hidden_states, console=console)
        else:
            if paper_results_refreshed:
                _print_rebuild_due_to_upstream(paths.probe_hidden_states, console=console)
            _log(f"  hidden-state extraction -> {paths.probe_hidden_states}", console=console)
            extract_reasoning_hidden_states(
                ReasoningProbeExtractionConfig(
                    manifest_path=paths.paper_manifest,
                    results_path=paths.paper_results,
                    output_path=paths.probe_hidden_states,
                    model_name=config.model_name,
                    thinking_mode=config.thinking_mode,
                )
            )
            probe_hidden_states_refreshed = True

        probe_dataset_refreshed = False
        if (
            not paper_results_refreshed
            and not probe_hidden_states_refreshed
            and config.resume
            and _path_is_ready(paths.probe_dataset)
        ):
            _print_skip_existing(paths.probe_dataset, console=console)
        else:
            if paper_results_refreshed or probe_hidden_states_refreshed:
                _print_rebuild_due_to_upstream(paths.probe_dataset, console=console)
            _log(f"  probe dataset -> {paths.probe_dataset}", console=console)
            build_reasoning_probe_dataset(
                input_path=paths.probe_hidden_states,
                output_path=paths.probe_dataset,
            )
            probe_dataset_refreshed = True

        probe_metrics_ready = _path_is_ready(paths.probe_metrics_jsonl) and _path_is_ready(
            paths.probe_summary_txt
        )
        probe_upstream_changed = (
            paper_results_refreshed or probe_hidden_states_refreshed or probe_dataset_refreshed
        )
        if (
            not probe_upstream_changed
            and config.resume
            and probe_metrics_ready
            and _path_is_ready(paths.probe_metrics_csv)
        ):
            _print_skip_existing(paths.probe_metrics_jsonl, console=console)
        else:
            if probe_upstream_changed:
                _print_rebuild_due_to_upstream(paths.probe_metrics_jsonl, console=console)
            if config.resume and probe_metrics_ready and not probe_upstream_changed:
                _log(
                    f"  regenerate probe metrics CSV -> {paths.probe_metrics_csv}",
                    console=console,
                )
            else:
                _log(f"  probe training -> {paths.probe_metrics_jsonl}", console=console)
                train_reasoning_probes(
                    ProbeTrainingConfig(
                        input_path=paths.probe_dataset,
                        metrics_path=paths.probe_metrics_jsonl,
                        summary_path=paths.probe_summary_txt,
                    )
                )
            _write_probe_metrics_csv(paths.probe_metrics_jsonl, paths.probe_metrics_csv)

    if config.skip_patching:
        _log("Stage 6/6: route patching skipped", console=console)
    else:
        _log("Stage 6/6: run route patching", console=console)
        patch_pairs_refreshed = False
        if not paper_results_refreshed and config.resume and _path_is_ready(paths.patch_pairs):
            _print_skip_existing(paths.patch_pairs, console=console)
        else:
            if paper_results_refreshed:
                _print_rebuild_due_to_upstream(paths.patch_pairs, console=console)
            _log(f"  patch pairs -> {paths.patch_pairs}", console=console)
            build_reasoning_patch_pairs(
                results_path=paths.paper_results,
                control_slice_path=paths.robust_slice,
                output_path=paths.patch_pairs,
            )
            patch_pairs_refreshed = True

        route_patching_ready = (
            _path_is_ready(paths.route_patching_results)
            and _path_is_ready(paths.route_patching_summary_txt)
            and _path_is_ready(paths.route_patching_summary_csv)
            and _path_is_ready(paths.route_patching_rescue_delta_gold_prob_plot)
            and _path_is_ready(paths.route_patching_rescue_delta_margin_plot)
            and _path_is_ready(paths.route_patching_induction_delta_shortcut_prob_plot)
        )
        if (
            not paper_results_refreshed
            and not patch_pairs_refreshed
            and config.resume
            and route_patching_ready
        ):
            _print_skip_existing(paths.route_patching_results, console=console)
        else:
            if paper_results_refreshed or patch_pairs_refreshed:
                _print_rebuild_due_to_upstream(paths.route_patching_results, console=console)
            _log(f"  route patching -> {paths.route_patching_results}", console=console)
            run_reasoning_route_patching(
                build_route_patching_config(
                    frozen_root=paths.frozen_root,
                    manifest_path=paths.paper_manifest,
                    results_path=paths.paper_results,
                    patch_pairs_path=paths.patch_pairs,
                    output_path=paths.route_patching_results,
                    summary_txt_path=paths.route_patching_summary_txt,
                    summary_csv_path=paths.route_patching_summary_csv,
                    rescue_delta_gold_prob_plot_path=(
                        paths.route_patching_rescue_delta_gold_prob_plot
                    ),
                    rescue_delta_margin_plot_path=paths.route_patching_rescue_delta_margin_plot,
                    induction_delta_shortcut_prob_plot_path=(
                        paths.route_patching_induction_delta_shortcut_prob_plot
                    ),
                    model_name=config.model_name,
                    thinking_mode=config.thinking_mode,
                )
            )

    _write_run_info(config, paths, pool_manifest_path)

    _log("", console=console)
    _log("Reasoning replication pipeline complete.", console=console)
    _log(f"Model: {config.model_name}", console=console)
    _log(f"Frozen root: {paths.frozen_root}", console=console)
    _log(f"Control-only results: {paths.control_results}", console=console)
    _log(f"Control-robust slice: {paths.robust_slice}", console=console)
    _log(f"Paper-slice manifest: {paths.paper_manifest}", console=console)
    _log(f"Paper-slice results: {paths.paper_results}", console=console)
    if config.skip_probes:
        _log("Probe outputs: skipped", console=console)
    else:
        _log(f"Probe hidden states: {paths.probe_hidden_states}", console=console)
        _log(f"Probe dataset: {paths.probe_dataset}", console=console)
        _log(f"Probe metrics JSONL: {paths.probe_metrics_jsonl}", console=console)
        _log(f"Probe metrics CSV: {paths.probe_metrics_csv}", console=console)
        _log(f"Probe summary: {paths.probe_summary_txt}", console=console)
    if config.skip_patching:
        _log("Route patching outputs: skipped", console=console)
    else:
        _log(f"Patch pairs: {paths.patch_pairs}", console=console)
        _log(f"Route patching results: {paths.route_patching_results}", console=console)
        _log(f"Route patching summary TXT: {paths.route_patching_summary_txt}", console=console)
        _log(f"Route patching summary CSV: {paths.route_patching_summary_csv}", console=console)
    _log(f"Run info: {paths.run_info_txt}", console=console)
    return paths


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the model replication helper."""

    supported_models = ", ".join(SUPPORTED_REPLICATION_MODELS)
    parser = argparse.ArgumentParser(
        description=(
            "Run the PressureTrace reasoning-family replication pipeline for one model. "
            f"Validated target models: {supported_models}."
        ),
    )
    parser.add_argument("model_name", help="Model to replicate, e.g. google/gemma-3-27b-it.")
    parser.add_argument(
        "--thinking",
        dest="thinking_mode",
        default="off",
        help="Thinking mode for the replication run. Default: off.",
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit frozen root; defaults to "
            "pressuretrace-frozen/reasoning_v2_<slug>_off."
        ),
    )
    parser.add_argument("--split", default="test", help="Dataset split for the transformed pool.")
    parser.add_argument("--limit", type=int, default=None, help="Optional retained-task cap.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Prompt batch size for non-Qwen3 reasoning inference. Default: 1.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip stages whose expected outputs already exist and are non-empty.",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip hidden-state extraction and probe training.",
    )
    parser.add_argument(
        "--skip-patching",
        action="store_true",
        help="Skip patch-pair construction and route patching.",
    )
    parser.add_argument(
        "--reuse-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing transformed pool when possible.",
    )
    parser.add_argument(
        "--progress",
        dest="show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show benchmark progress while model inference runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    """Run the reasoning-family replication pipeline from the command line."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    frozen_root = args.frozen_root or default_replication_frozen_root(
        model_name=args.model_name,
        thinking_mode=args.thinking_mode,
    )
    paths = run_reasoning_model_pipeline(
        ReasoningModelPipelineConfig(
            model_name=args.model_name,
            frozen_root=frozen_root,
            thinking_mode=args.thinking_mode,
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            resume=args.resume,
            skip_probes=args.skip_probes,
            skip_patching=args.skip_patching,
            reuse_pool=args.reuse_pool,
            show_progress=args.show_progress,
        )
    )
    return paths.frozen_root


if __name__ == "__main__":
    main()
