"""Typer CLI for PressureTrace."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pressuretrace.behavior.run_coding_benchmark import run_coding_pilot
from pressuretrace.behavior.run_reasoning_benchmark import run_reasoning_pilot
from pressuretrace.behavior.summarize_behavior import print_behavior_summary
from pressuretrace.config import DEFAULT_MODELS, PRESSURE_PROFILES
from pressuretrace.paths import data_dir, manifests_dir, repo_root, results_dir, splits_dir

app = typer.Typer(help="PressureTrace v1 command-line interface.", no_args_is_help=True)
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
