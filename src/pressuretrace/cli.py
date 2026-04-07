"""Typer CLI for PressureTrace."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pressuretrace.behavior.build_control_robust_slice import build_control_robust_slice
from pressuretrace.behavior.materialize_reasoning_slice import materialize_reasoning_slice
from pressuretrace.behavior.run_coding_benchmark import run_coding_pilot
from pressuretrace.behavior.run_reasoning_benchmark import run_reasoning_pilot
from pressuretrace.behavior.run_reasoning_benchmark_v2 import (
    run_reasoning_manifest_v2,
    run_reasoning_pilot_v2,
)
from pressuretrace.behavior.run_reasoning_control_only import run_reasoning_control_only
from pressuretrace.behavior.summarize_behavior import print_behavior_summary
from pressuretrace.behavior.summarize_behavior_v2 import print_behavior_summary_v2
from pressuretrace.config import DEFAULT_MODELS, PRESSURE_PROFILES
from pressuretrace.generation.reasoning.make_reasoning_tasks_v2 import (
    build_reasoning_all_valid_transforms_v2,
)
from pressuretrace.paths import data_dir, manifests_dir, repo_root, results_dir, splits_dir

app = typer.Typer(help="PressureTrace command-line interface.", no_args_is_help=True)
console = Console()


@app.command("reasoning-pilot")
def reasoning_pilot_command(
    split: str = typer.Option("test", help="Dataset split to load."),
    limit: int = typer.Option(
        20,
        min=1,
        help="Maximum number of shortcut-clean GSM8K base tasks to retain.",
    ),
    model_name: str = typer.Option(
        DEFAULT_MODELS.reasoning_model,
        help="Model identifier recorded in output rows.",
    ),
    pressure_type: str = typer.Option(
        "authority_conflict",
        help="Pressure type to run from the manifest, or 'all' to run every pressure type.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit output JSONL path.",
    ),
    manifest_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--manifest-path",
        help="Optional explicit manifest JSONL path.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Build artifacts without running model inference.",
    ),
    include_control: bool = typer.Option(
        True,
        "--include-control/--pressure-only",
        help="Include control episodes alongside pressure episodes.",
    ),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show live manifest and inference progress while the pilot runs.",
    ),
) -> None:
    """Run a reasoning pilot."""

    artifacts = run_reasoning_pilot(
        split=split,
        limit=limit,
        model_name=model_name,
        pressure_type=pressure_type,
        output_path=output_path,
        manifest_path=manifest_path,
        dry_run=dry_run,
        include_control=include_control,
        console=console,
        show_progress=show_progress,
    )
    console.print(f"Manifest: [bold]{artifacts.manifest_path}[/bold]")
    console.print(f"Results: [bold]{artifacts.results_path}[/bold]")
    if not dry_run:
        print_behavior_summary(artifacts.results_path)


@app.command("reasoning-pilot-v2")
def reasoning_pilot_v2_command(
    split: str = typer.Option("test", help="Dataset split to load."),
    limit: int = typer.Option(
        20,
        min=1,
        help="Maximum number of shortcut-clean GSM8K base tasks to retain.",
    ),
    model_name: str = typer.Option(
        DEFAULT_MODELS.reasoning_model,
        help="Model identifier recorded in output rows.",
    ),
    pressure_type: str = typer.Option(
        "all",
        help="Pressure type to run from the v2 manifest, or 'all' to run every pressure type.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit output JSONL path.",
    ),
    manifest_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--manifest-path",
        help="Optional explicit manifest JSONL path.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Build artifacts without running model inference.",
    ),
    include_control: bool = typer.Option(
        True,
        "--include-control/--pressure-only",
        help="Include control episodes alongside pressure episodes.",
    ),
    thinking_mode: str = typer.Option(
        "default",
        "--thinking-mode",
        help="Thinking mode for supported models: default, on, or off.",
    ),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show live manifest and inference progress while the pilot runs.",
    ),
) -> None:
    """Run a reasoning v2 pilot."""

    artifacts = run_reasoning_pilot_v2(
        split=split,
        limit=limit,
        model_name=model_name,
        pressure_type=pressure_type,
        output_path=output_path,
        manifest_path=manifest_path,
        dry_run=dry_run,
        include_control=include_control,
        thinking_mode=thinking_mode,
        console=console,
        show_progress=show_progress,
    )
    console.print(f"Manifest: [bold]{artifacts.manifest_path}[/bold]")
    console.print(f"Results: [bold]{artifacts.results_path}[/bold]")
    if not dry_run:
        print_behavior_summary_v2(artifacts.results_path)


@app.command("reasoning-build-pool-v2")
def reasoning_build_pool_v2_command(
    split: str = typer.Option("test", help="Dataset split to load."),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap on retained latent tasks for the transform pool.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit manifest JSONL path.",
    ),
) -> None:
    """Build the frozen all-valid transform pool for reasoning v2."""

    manifest_path = build_reasoning_all_valid_transforms_v2(
        split=split,
        limit=limit,
        output_path=output_path,
    )
    console.print(f"Transform pool: [bold]{manifest_path}[/bold]")


@app.command("reasoning-control-only-v2")
def reasoning_control_only_v2_command(
    manifest_path: Path = typer.Option(  # noqa: B008
        ...,
        "--manifest-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Frozen reasoning v2 manifest to evaluate under control only.",
    ),
    model_name: str = typer.Option(
        DEFAULT_MODELS.reasoning_model,
        help="Model identifier recorded in output rows.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit output JSONL path.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Build artifacts without running model inference.",
    ),
    thinking_mode: str = typer.Option(
        "default",
        "--thinking-mode",
        help="Thinking mode for supported models: default, on, or off.",
    ),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show live inference progress while the control-only run executes.",
    ),
) -> None:
    """Run only control rows from a frozen reasoning v2 manifest."""

    artifacts = run_reasoning_control_only(
        manifest_path=manifest_path,
        model_name=model_name,
        output_path=output_path,
        dry_run=dry_run,
        thinking_mode=thinking_mode,
        console=console,
        show_progress=show_progress,
    )
    console.print(f"Manifest: [bold]{artifacts.manifest_path}[/bold]")
    console.print(f"Results: [bold]{artifacts.results_path}[/bold]")
    if not dry_run:
        print_behavior_summary_v2(artifacts.results_path)


@app.command("reasoning-freeze-slice-v2")
def reasoning_freeze_slice_v2_command(
    input_path: Path = typer.Option(  # noqa: B008
        ...,
        "--input-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Control-only reasoning v2 results used to freeze the robust slice.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit split JSONL path.",
    ),
) -> None:
    """Freeze the model-specific control-robust slice."""

    slice_path = build_control_robust_slice(
        control_results_path=input_path,
        output_path=output_path,
    )
    console.print(f"Control-robust slice: [bold]{slice_path}[/bold]")


@app.command("reasoning-materialize-slice-v2")
def reasoning_materialize_slice_v2_command(
    manifest_path: Path = typer.Option(  # noqa: B008
        ...,
        "--manifest-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Broad reasoning v2 transform pool manifest.",
    ),
    slice_path: Path = typer.Option(  # noqa: B008
        ...,
        "--slice-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Frozen control-robust slice JSONL.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit paper-slice manifest path.",
    ),
) -> None:
    """Materialize the paper-slice manifest from a frozen latent task slice."""

    materialized_path = materialize_reasoning_slice(
        manifest_path=manifest_path,
        slice_path=slice_path,
        output_path=output_path,
    )
    console.print(f"Paper-slice manifest: [bold]{materialized_path}[/bold]")


@app.command("reasoning-run-manifest-v2")
def reasoning_run_manifest_v2_command(
    manifest_path: Path = typer.Option(  # noqa: B008
        ...,
        "--manifest-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Frozen reasoning v2 manifest to run directly.",
    ),
    model_name: str = typer.Option(
        DEFAULT_MODELS.reasoning_model,
        help="Model identifier recorded in output rows.",
    ),
    pressure_type: str = typer.Option(
        "all",
        help="Pressure type to run from the manifest, or 'all' to run every pressure type.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit output JSONL path.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Build artifacts without running model inference.",
    ),
    include_control: bool = typer.Option(
        True,
        "--include-control/--pressure-only",
        help="Include control episodes alongside pressure episodes.",
    ),
    thinking_mode: str = typer.Option(
        "default",
        "--thinking-mode",
        help="Thinking mode for supported models: default, on, or off.",
    ),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show live inference progress while the manifest run executes.",
    ),
) -> None:
    """Run inference over an already frozen reasoning v2 manifest."""

    artifacts = run_reasoning_manifest_v2(
        manifest_path=manifest_path,
        model_name=model_name,
        pressure_type=pressure_type,
        output_path=output_path,
        dry_run=dry_run,
        include_control=include_control,
        thinking_mode=thinking_mode,
        console=console,
        show_progress=show_progress,
    )
    console.print(f"Manifest: [bold]{artifacts.manifest_path}[/bold]")
    console.print(f"Results: [bold]{artifacts.results_path}[/bold]")
    if not dry_run:
        print_behavior_summary_v2(artifacts.results_path)


@app.command("coding-pilot")
def coding_pilot_command(
    split: str = typer.Option("test", help="Dataset split to load."),
    limit_per_dataset: int = typer.Option(
        5,
        min=1,
        help="Maximum number of tasks to load from each coding base dataset.",
    ),
    model_name: str = typer.Option(
        DEFAULT_MODELS.coding_model,
        help="Model identifier recorded in output rows.",
    ),
    pressure_profile: str = typer.Option(
        "medium",
        help=f"Pressure profile to use. Available: {', '.join(sorted(PRESSURE_PROFILES))}.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit output JSONL path.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Write pending result rows without calling model inference.",
    ),
    include_control: bool = typer.Option(
        True,
        "--include-control/--pressure-only",
        help="Include control episodes alongside pressure episodes.",
    ),
    include_humaneval: bool = typer.Option(
        True,
        "--humaneval/--no-humaneval",
        help="Include HumanEval-style tasks.",
    ),
    include_mbpp: bool = typer.Option(
        True,
        "--mbpp/--no-mbpp",
        help="Include MBPP-style tasks.",
    ),
) -> None:
    """Run a coding pilot."""

    path = run_coding_pilot(
        split=split,
        limit_per_dataset=limit_per_dataset,
        model_name=model_name,
        pressure_profile=pressure_profile,
        output_path=output_path,
        dry_run=dry_run,
        include_control=include_control,
        include_humaneval=include_humaneval,
        include_mbpp=include_mbpp,
    )
    console.print(f"Wrote coding pilot rows to [bold]{path}[/bold]")


@app.command("summarize")
def summarize_command(
    input_path: Path = typer.Option(  # noqa: B008
        ...,
        "--input-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Result JSONL file to summarize.",
    ),
) -> None:
    """Print aggregate route summaries for a result file."""

    print_behavior_summary(input_path)


@app.command("summarize-v2")
def summarize_v2_command(
    input_path: Path = typer.Option(  # noqa: B008
        ...,
        "--input-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Reasoning v2 result JSONL file to summarize.",
    ),
) -> None:
    """Print aggregate route summaries for a reasoning v2 result file."""

    print_behavior_summary_v2(input_path)


@app.command("print-paths")
def print_paths_command() -> None:
    """Print key repository paths."""

    table = Table(title="PressureTrace Paths")
    table.add_column("Name")
    table.add_column("Path")
    table.add_row("repo_root", str(repo_root()))
    table.add_row("data_dir", str(data_dir()))
    table.add_row("results_dir", str(results_dir()))
    table.add_row("manifests_dir", str(manifests_dir()))
    table.add_row("splits_dir", str(splits_dir()))
    console.print(table)
