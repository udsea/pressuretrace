"""Orchestrate the full reasoning-family replication pipeline for one model."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from pressuretrace.behavior.build_control_robust_slice import build_control_robust_slice
from pressuretrace.behavior.materialize_reasoning_slice import materialize_reasoning_slice
from pressuretrace.behavior.reasoning_runtime import slugify_model_name
from pressuretrace.behavior.run_reasoning_benchmark_v2 import run_reasoning_manifest_v2
from pressuretrace.behavior.run_reasoning_control_only import run_reasoning_control_only
from pressuretrace.generation.reasoning.make_reasoning_tasks_v2 import (
    build_reasoning_all_valid_transforms_v2,
)
from pressuretrace.paths import manifests_dir, repo_root
from pressuretrace.probes.build_reasoning_probe_dataset import build_reasoning_probe_dataset
from pressuretrace.probes.extract_hidden_states_reasoning import (
    ReasoningProbeExtractionConfig,
    extract_reasoning_hidden_states,
)
from pressuretrace.probes.train_reasoning_probes import (
    ProbeTrainingConfig,
    train_reasoning_probes,
)
from pressuretrace.utils.io import ensure_directory, read_jsonl

SUPPORTED_REPLICATION_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
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
    run_info_txt: Path


@dataclass(frozen=True)
class ReasoningModelPipelineConfig:
    """Configuration for one end-to-end reasoning replication run."""

    model_name: str
    frozen_root: Path
    thinking_mode: str = "off"
    split: str = "test"
    limit: int | None = None
    skip_probes: bool = False
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
        f"skip_probes={config.skip_probes}",
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

    _prepare_frozen_directories(
        paths := reasoning_replication_paths(
            model_name=config.model_name,
            frozen_root=config.frozen_root,
            thinking_mode=config.thinking_mode,
        )
    )
    pool_manifest_path = _resolve_pool_manifest(config, paths)

    print(f"Model: {config.model_name}")
    print(f"Thinking mode: {config.thinking_mode}")
    print(f"Frozen root: {paths.frozen_root}")
    print(f"Transformed pool: {pool_manifest_path}")

    print("Stage 1/5: control-only run")
    run_reasoning_control_only(
        manifest_path=pool_manifest_path,
        model_name=config.model_name,
        output_path=paths.control_results,
        thinking_mode=config.thinking_mode,
        show_progress=config.show_progress,
    )

    print("Stage 2/5: build control-robust slice")
    build_control_robust_slice(
        control_results_path=paths.control_results,
        output_path=paths.robust_slice,
    )

    print("Stage 3/5: materialize paper slice")
    materialize_reasoning_slice(
        manifest_path=pool_manifest_path,
        slice_path=paths.robust_slice,
        output_path=paths.paper_manifest,
    )

    print("Stage 4/5: run paper slice")
    run_reasoning_manifest_v2(
        manifest_path=paths.paper_manifest,
        model_name=config.model_name,
        pressure_type="all",
        output_path=paths.paper_results,
        include_control=True,
        thinking_mode=config.thinking_mode,
        show_progress=config.show_progress,
    )

    if config.skip_probes:
        print("Stage 5/5: probe pipeline skipped")
    else:
        print("Stage 5/5: run probe pipeline")
        extract_reasoning_hidden_states(
            ReasoningProbeExtractionConfig(
                manifest_path=paths.paper_manifest,
                results_path=paths.paper_results,
                output_path=paths.probe_hidden_states,
                model_name=config.model_name,
                thinking_mode=config.thinking_mode,
            )
        )
        build_reasoning_probe_dataset(
            input_path=paths.probe_hidden_states,
            output_path=paths.probe_dataset,
        )
        train_reasoning_probes(
            ProbeTrainingConfig(
                input_path=paths.probe_dataset,
                metrics_path=paths.probe_metrics_jsonl,
                summary_path=paths.probe_summary_txt,
            )
        )
        _write_probe_metrics_csv(paths.probe_metrics_jsonl, paths.probe_metrics_csv)

    _write_run_info(config, paths, pool_manifest_path)

    print("")
    print("Reasoning replication pipeline complete.")
    print(f"Model: {config.model_name}")
    print(f"Frozen root: {paths.frozen_root}")
    print(f"Control-only results: {paths.control_results}")
    print(f"Control-robust slice: {paths.robust_slice}")
    print(f"Paper-slice manifest: {paths.paper_manifest}")
    print(f"Paper-slice results: {paths.paper_results}")
    if config.skip_probes:
        print("Probe outputs: skipped")
    else:
        print(f"Probe hidden states: {paths.probe_hidden_states}")
        print(f"Probe dataset: {paths.probe_dataset}")
        print(f"Probe metrics JSONL: {paths.probe_metrics_jsonl}")
        print(f"Probe metrics CSV: {paths.probe_metrics_csv}")
        print(f"Probe summary: {paths.probe_summary_txt}")
    print(f"Run info: {paths.run_info_txt}")
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
    parser.add_argument("model_name", help="Model to replicate, e.g. Qwen/Qwen2.5-7B-Instruct.")
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
        "--skip-probes",
        action="store_true",
        help="Skip hidden-state extraction and probe training.",
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
            skip_probes=args.skip_probes,
            reuse_pool=args.reuse_pool,
            show_progress=args.show_progress,
        )
    )
    return paths.frozen_root


if __name__ == "__main__":
    main()
