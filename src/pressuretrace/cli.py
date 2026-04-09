"""Typer CLI for PressureTrace."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pressuretrace.behavior.build_coding_control_robust_slice import (
    build_coding_control_robust_slice,
)
from pressuretrace.behavior.build_control_robust_slice import build_control_robust_slice
from pressuretrace.behavior.debug_coding_tasks import run_coding_debug_tasks
from pressuretrace.behavior.materialize_coding_paper_slice import (
    materialize_coding_paper_slice,
)
from pressuretrace.behavior.materialize_reasoning_slice import materialize_reasoning_slice
from pressuretrace.behavior.run_coding_benchmark import run_coding_pilot
from pressuretrace.behavior.run_coding_control_only import run_coding_control_only
from pressuretrace.behavior.run_coding_paper_slice import (
    run_coding_manifest,
    run_coding_paper_slice,
)
from pressuretrace.behavior.run_reasoning_benchmark import run_reasoning_pilot
from pressuretrace.behavior.run_reasoning_benchmark_v2 import (
    run_reasoning_manifest_v2,
    run_reasoning_pilot_v2,
)
from pressuretrace.behavior.run_reasoning_control_only import run_reasoning_control_only
from pressuretrace.behavior.summarize_behavior import print_behavior_summary
from pressuretrace.behavior.summarize_behavior_v2 import print_behavior_summary_v2
from pressuretrace.behavior.summarize_coding_behavior import (
    export_coding_behavior_summary,
    render_coding_behavior_summary_text,
)
from pressuretrace.config import (
    CODING_V1_MODEL_NAME,
    CODING_V1_THINKING_MODE,
    DEFAULT_MODELS,
    PRESSURE_PROFILES,
    coding_probe_dataset_path,
    coding_probe_hidden_states_path,
    coding_probe_manifest_path,
    coding_probe_metrics_path,
    coding_probe_results_path,
    coding_probe_summary_path,
)
from pressuretrace.evaluation.coding_eval_debug import run_coding_eval_debug
from pressuretrace.generation.coding.make_coding_tasks import build_coding_all_valid_transforms
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


@app.command("coding-build-pool-v1")
def coding_build_pool_v1_command(
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Optional cap on retained coding-family base tasks.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit manifest JSONL path.",
    ),
) -> None:
    """Build the full coding-family transform pool."""

    manifest_path = build_coding_all_valid_transforms(limit=limit, output_path=output_path)
    console.print(f"Transform pool: [bold]{manifest_path}[/bold]")


@app.command("coding-eval-debug-v1")
def coding_eval_debug_v1_command(
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit JSONL path for the evaluator fixture report.",
    ),
    require_pass: bool = typer.Option(
        True,
        "--require-pass/--allow-failures",
        help="Fail the command if any evaluator gold fixture is mislabeled.",
    ),
) -> None:
    """Run mandatory evaluator self-tests on representative coding fixtures."""

    artifacts = run_coding_eval_debug(
        output_path=output_path,
        console=console,
        require_pass=require_pass,
    )
    console.print(f"Evaluator debug report: [bold]{artifacts.output_path}[/bold]")
    console.print(
        f"Matched expected labels: [bold]{artifacts.passing_cases}/{artifacts.total_cases}[/bold]"
    )


@app.command("coding-debug-run-v1")
def coding_debug_run_v1_command(
    model_name: str = typer.Option(
        "Qwen/Qwen3-14B",
        help="Model identifier recorded in debug rows.",
    ),
    thinking_mode: str = typer.Option(
        "off",
        "--thinking-mode",
        help="Thinking mode for supported models: default, on, or off.",
    ),
    source: str = typer.Option(
        "fixtures",
        "--source",
        help="Debug source: fixtures or model.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit debug JSONL output path.",
    ),
    batch_size: int = typer.Option(1, min=1, help="Batch size for model-backed debug runs."),
    require_shortcut_signal: bool = typer.Option(
        True,
        "--require-shortcut-signal/--allow-no-shortcut-signal",
        help="Fail the command if no visible-pass / hidden-fail shortcut signal appears.",
    ),
) -> None:
    """Run a small coding-family debug pass on representative tasks."""

    if source not in {"fixtures", "model"}:
        raise typer.BadParameter("source must be one of: fixtures, model")
    artifacts = run_coding_debug_tasks(
        model_name=model_name,
        thinking_mode=thinking_mode,
        source=source,
        batch_size=batch_size,
        output_path=output_path,
        console=console,
        require_shortcut_signal=require_shortcut_signal,
    )
    console.print(f"Debug run: [bold]{artifacts.output_path}[/bold]")
    console.print(
        "Shortcut-signal rows: "
        f"[bold]{artifacts.shortcut_signal_count}/{artifacts.row_count}[/bold]"
    )


@app.command("coding-control-only-v1")
def coding_control_only_v1_command(
    manifest_path: Path = typer.Option(  # noqa: B008
        ...,
        "--manifest-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Frozen coding-family manifest to evaluate under control only.",
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen3-14B",
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
        "off",
        "--thinking-mode",
        help="Thinking mode for supported models: default, on, or off.",
    ),
    batch_size: int = typer.Option(1, min=1, help="Batch size for inference."),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show live inference progress while the control-only run executes.",
    ),
) -> None:
    """Run only control rows from a frozen coding-family manifest."""

    artifacts = run_coding_control_only(
        manifest_path=manifest_path,
        model_name=model_name,
        output_path=output_path,
        dry_run=dry_run,
        thinking_mode=thinking_mode,
        batch_size=batch_size,
        console=console,
        show_progress=show_progress,
    )
    console.print(f"Manifest: [bold]{artifacts.manifest_path}[/bold]")
    console.print(f"Results: [bold]{artifacts.results_path}[/bold]")


@app.command("coding-freeze-slice-v1")
def coding_freeze_slice_v1_command(
    input_path: Path = typer.Option(  # noqa: B008
        ...,
        "--input-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Control-only coding results used to freeze the robust slice.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit split JSONL path.",
    ),
) -> None:
    """Freeze the model-specific coding-family control-robust slice."""

    slice_path = build_coding_control_robust_slice(
        control_results_path=input_path,
        output_path=output_path,
    )
    console.print(f"Control-robust slice: [bold]{slice_path}[/bold]")


@app.command("coding-materialize-slice-v1")
def coding_materialize_slice_v1_command(
    manifest_path: Path = typer.Option(  # noqa: B008
        ...,
        "--manifest-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Broad coding-family transform pool manifest.",
    ),
    slice_path: Path = typer.Option(  # noqa: B008
        ...,
        "--slice-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Frozen coding-family control-robust slice JSONL.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit paper-slice manifest path.",
    ),
) -> None:
    """Materialize the coding-family paper-slice manifest from a frozen slice."""

    materialized_path = materialize_coding_paper_slice(
        manifest_path=manifest_path,
        slice_path=slice_path,
        output_path=output_path,
    )
    console.print(f"Paper-slice manifest: [bold]{materialized_path}[/bold]")


@app.command("coding-run-manifest-v1")
def coding_run_manifest_v1_command(
    manifest_path: Path = typer.Option(  # noqa: B008
        ...,
        "--manifest-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Frozen coding-family manifest to run directly.",
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen3-14B",
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
        "off",
        "--thinking-mode",
        help="Thinking mode for supported models: default, on, or off.",
    ),
    batch_size: int = typer.Option(1, min=1, help="Batch size for inference."),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show live inference progress while the manifest run executes.",
    ),
) -> None:
    """Run inference over an already frozen coding-family manifest."""

    artifacts = run_coding_manifest(
        manifest_path=manifest_path,
        model_name=model_name,
        pressure_type=pressure_type,
        output_path=output_path,
        dry_run=dry_run,
        include_control=include_control,
        thinking_mode=thinking_mode,
        batch_size=batch_size,
        console=console,
        show_progress=show_progress,
    )
    console.print(f"Manifest: [bold]{artifacts.manifest_path}[/bold]")
    console.print(f"Results: [bold]{artifacts.results_path}[/bold]")


@app.command("coding-paper-slice-v1")
def coding_paper_slice_v1_command(
    manifest_path: Path = typer.Option(  # noqa: B008
        ...,
        "--manifest-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Frozen coding-family paper-slice manifest.",
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen3-14B",
        help="Model identifier recorded in output rows.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit output JSONL path.",
    ),
    pressure_type: str = typer.Option(
        "all",
        help="Pressure type to run from the manifest, or 'all' to run every pressure type.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Build artifacts without running model inference.",
    ),
    thinking_mode: str = typer.Option(
        "off",
        "--thinking-mode",
        help="Thinking mode for supported models: default, on, or off.",
    ),
    batch_size: int = typer.Option(1, min=1, help="Batch size for inference."),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show live inference progress while the paper-slice run executes.",
    ),
) -> None:
    """Run the coding-family paper slice."""

    artifacts = run_coding_paper_slice(
        manifest_path=manifest_path,
        model_name=model_name,
        output_path=output_path,
        pressure_type=pressure_type,
        dry_run=dry_run,
        thinking_mode=thinking_mode,
        batch_size=batch_size,
        console=console,
        show_progress=show_progress,
    )
    console.print(f"Manifest: [bold]{artifacts.manifest_path}[/bold]")
    console.print(f"Results: [bold]{artifacts.results_path}[/bold]")


@app.command("coding-summarize-v1")
def coding_summarize_v1_command(
    input_path: Path = typer.Option(  # noqa: B008
        ...,
        "--input-path",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Coding-family result JSONL file to summarize.",
    ),
    text_output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--text-output-path",
        help="Optional summary text output path.",
    ),
    csv_output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--csv-output-path",
        help="Optional summary CSV output path.",
    ),
) -> None:
    """Print and optionally export coding-family behavior summaries."""

    summary_text = render_coding_behavior_summary_text(input_path)
    console.print(summary_text.rstrip())
    if text_output_path is not None and csv_output_path is not None:
        export_coding_behavior_summary(
            input_path=input_path,
            text_output_path=text_output_path,
            csv_output_path=csv_output_path,
        )
        console.print(f"Summary text: [bold]{text_output_path}[/bold]")
        console.print(f"Summary CSV: [bold]{csv_output_path}[/bold]")


@app.command("coding-probe-extract-v1")
def coding_probe_extract_v1_command(
    manifest_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--manifest-path",
        help="Frozen coding paper-slice manifest JSONL.",
    ),
    results_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--results-path",
        help="Frozen coding paper-slice results JSONL.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Coding hidden-state JSONL output path.",
    ),
    model_name: str = typer.Option(
        CODING_V1_MODEL_NAME,
        help="Model name recorded in frozen coding rows.",
    ),
    thinking_mode: str = typer.Option(
        CODING_V1_THINKING_MODE,
        "--thinking-mode",
        help="Thinking mode recorded in frozen coding rows.",
    ),
) -> None:
    """Extract hidden states for the frozen coding paper slice."""

    from pressuretrace.probes.extract_hidden_states_coding import (
        CodingProbeExtractionConfig,
        extract_coding_hidden_states,
    )

    output = extract_coding_hidden_states(
        CodingProbeExtractionConfig(
            manifest_path=manifest_path or coding_probe_manifest_path(),
            results_path=results_path or coding_probe_results_path(),
            output_path=output_path or coding_probe_hidden_states_path(),
            model_name=model_name,
            thinking_mode=thinking_mode,
        )
    )
    console.print(f"hidden_states: [bold]{output}[/bold]")


@app.command("coding-probe-dataset-v1")
def coding_probe_dataset_v1_command(
    input_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--input-path",
        help="Coding hidden-state JSONL input path.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Coding probe dataset JSONL output path.",
    ),
) -> None:
    """Build the compact coding probe dataset."""

    from pressuretrace.probes.build_coding_probe_dataset import build_coding_probe_dataset

    output = build_coding_probe_dataset(
        input_path=input_path or coding_probe_hidden_states_path(),
        output_path=output_path or coding_probe_dataset_path(),
    )
    console.print(f"probe_dataset: [bold]{output}[/bold]")


@app.command("coding-train-probes-v1")
def coding_train_probes_v1_command(
    input_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--input-path",
        help="Coding probe dataset JSONL input path.",
    ),
    metrics_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--metrics-path",
        help="Coding probe metrics JSONL output path.",
    ),
    summary_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--summary-path",
        help="Coding probe summary TXT output path.",
    ),
) -> None:
    """Train first-pass coding probes on frozen hidden states."""

    from pressuretrace.probes.train_coding_probes import (
        CodingProbeTrainingConfig,
        train_coding_probes,
    )

    output = train_coding_probes(
        CodingProbeTrainingConfig(
            input_path=input_path or coding_probe_dataset_path(),
            metrics_path=metrics_path or coding_probe_metrics_path(),
            summary_path=summary_path or coding_probe_summary_path(),
        )
    )
    console.print(f"probe_metrics: [bold]{output}[/bold]")
    console.print(
        f"probe_summary: [bold]{summary_path or coding_probe_summary_path()}[/bold]"
    )


@app.command("coding-probe-summary-export-v1")
def coding_probe_summary_export_v1_command(
    frozen_root: Path | None = typer.Option(  # noqa: B008
        None,
        "--frozen-root",
        help="Optional frozen coding probe root; defaults to the configured root.",
    ),
    artifact_index: bool = typer.Option(
        True,
        "--artifact-index/--no-artifact-index",
        help="Write ARTIFACTS.md under the frozen coding root.",
    ),
    metrics_csv: bool = typer.Option(
        True,
        "--metrics-csv/--no-metrics-csv",
        help="Write the flat coding probe metrics CSV under frozen results.",
    ),
    summary: bool = typer.Option(
        True,
        "--summary/--no-summary",
        help="Rewrite the human-readable coding probe summary under frozen results.",
    ),
) -> None:
    """Export derived frozen coding probe reports."""

    from pressuretrace.analysis.coding_probe_reports import export_probe_reports

    outputs = export_probe_reports(
        frozen_root=frozen_root,
        write_artifact_index=artifact_index,
        write_csv=metrics_csv,
        write_summary=summary,
    )
    for name, path in outputs.items():
        console.print(f"{name}: [bold]{path}[/bold]")


@app.command("coding-patch-pairs-v1")
def coding_patch_pairs_v1_command(
    results_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--results-path",
        help="Optional explicit frozen coding results path.",
    ),
    control_slice_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--control-slice-path",
        help="Optional explicit frozen coding control-robust slice path.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit coding patch-pairs JSONL path.",
    ),
    pressure_types: str = typer.Option(
        "neutral_wrong_answer_cue,teacher_anchor",
        "--pressure-types",
        help="Comma-separated pressure types to retain while building pairs.",
    ),
) -> None:
    """Build matched control/pressure patch pairs for coding route patching."""

    from pressuretrace.config import resolve_coding_frozen_root
    from pressuretrace.patching.build_coding_patch_pairs import build_coding_patch_pairs

    frozen_root = resolve_coding_frozen_root()
    patch_pairs_path = build_coding_patch_pairs(
        results_path=(
            results_path
            or frozen_root / "results" / "coding_paper_slice_qwen-qwen3-14b_off.jsonl"
        ),
        control_slice_path=(
            control_slice_path
            or frozen_root
            / "data"
            / "splits"
            / "coding_control_robust_slice_qwen-qwen3-14b_off.jsonl"
        ),
        output_path=(
            output_path
            or frozen_root / "results" / "coding_patch_pairs_qwen-qwen3-14b_off.jsonl"
        ),
        pressure_types=tuple(part.strip() for part in pressure_types.split(",") if part.strip()),
    )
    console.print(f"patch_pairs: [bold]{patch_pairs_path}[/bold]")


@app.command("coding-route-patching-v1")
def coding_route_patching_v1_command(
    frozen_root: Path | None = typer.Option(  # noqa: B008
        None,
        "--frozen-root",
        help="Optional explicit frozen coding root.",
    ),
    patch_pairs_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--patch-pairs-path",
        help="Optional explicit coding patch-pairs JSONL path.",
    ),
    manifest_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--manifest-path",
        help="Optional explicit frozen coding manifest path.",
    ),
    results_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--results-path",
        help="Optional explicit frozen coding paper-slice results path.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit coding patching-output JSONL path.",
    ),
    summary_txt_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--summary-txt-path",
        help="Optional explicit coding patching summary TXT path.",
    ),
    summary_csv_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--summary-csv-path",
        help="Optional explicit coding patching summary CSV path.",
    ),
    model_name: str = typer.Option(
        CODING_V1_MODEL_NAME,
        help="Model used for coding route patching.",
    ),
    thinking_mode: str = typer.Option(
        CODING_V1_THINKING_MODE,
        "--thinking-mode",
        help="Thinking mode for coding route patching.",
    ),
    layers: str = typer.Option(
        "-8,-10,-6",
        "--layers",
        help="Comma-separated patch layers.",
    ),
    pressure_types: str = typer.Option(
        "neutral_wrong_answer_cue",
        "--pressure-types",
        help="Comma-separated pressure types to retain.",
    ),
    max_pairs: int | None = typer.Option(
        None,
        "--max-pairs",
        help="Optional cap on retained coding patch pairs for debugging.",
    ),
) -> None:
    """Run first-pass continuation-level coding route patching on frozen pairs."""

    from pressuretrace.patching.run_coding_route_patching import (
        build_route_patching_config,
        run_coding_route_patching,
    )

    config = build_route_patching_config(
        frozen_root=frozen_root,
        patch_pairs_path=patch_pairs_path,
        manifest_path=manifest_path,
        results_path=results_path,
        output_path=output_path,
        summary_txt_path=summary_txt_path,
        summary_csv_path=summary_csv_path,
        model_name=model_name,
        thinking_mode=thinking_mode,
        layers=tuple(int(part.strip()) for part in layers.split(",") if part.strip()),
        pressure_types=tuple(part.strip() for part in pressure_types.split(",") if part.strip()),
        max_pairs=max_pairs,
    )
    artifacts = run_coding_route_patching(config)
    console.print(f"patching_results: [bold]{artifacts.output_path}[/bold]")
    console.print(f"summary_txt: [bold]{artifacts.summary_txt_path}[/bold]")
    console.print(f"summary_csv: [bold]{artifacts.summary_csv_path}[/bold]")
    console.print(
        "plots: "
        f"[bold]{artifacts.rescue_delta_robust_prob_plot_path}[/bold], "
        f"[bold]{artifacts.rescue_delta_margin_plot_path}[/bold], "
        f"[bold]{artifacts.induction_delta_shortcut_prob_plot_path}[/bold]"
    )


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


@app.command("probe-summary-export")
def probe_summary_export_command(
    frozen_root: Path | None = typer.Option(  # noqa: B008
        None,
        "--frozen-root",
        help="Optional frozen reasoning probe root; defaults to the configured root.",
    ),
    artifact_index: bool = typer.Option(
        True,
        "--artifact-index/--no-artifact-index",
        help="Write ARTIFACTS.md under the frozen root.",
    ),
    metrics_csv: bool = typer.Option(
        True,
        "--metrics-csv/--no-metrics-csv",
        help="Write the flat probe metrics CSV under frozen results.",
    ),
    summary: bool = typer.Option(
        True,
        "--summary/--no-summary",
        help="Rewrite the polished human-readable probe summary under frozen results.",
    ),
) -> None:
    """Export derived frozen reasoning probe reports."""

    from pressuretrace.analysis.reasoning_probe_reports import export_probe_reports

    outputs = export_probe_reports(
        frozen_root=frozen_root,
        write_artifact_index=artifact_index,
        write_csv=metrics_csv,
        write_summary=summary,
    )
    for name, path in outputs.items():
        console.print(f"{name}: [bold]{path}[/bold]")


@app.command("probe-plots")
def probe_plots_command(
    metrics_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--metrics-path",
        help="Optional explicit probe metrics JSONL path.",
    ),
    dataset_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--dataset-path",
        help="Optional explicit probe dataset JSONL path.",
    ),
    summary_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--summary-path",
        help="Optional explicit probe summary path.",
    ),
    output_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-dir",
        help="Optional plot output directory; defaults to the frozen results root.",
    ),
) -> None:
    """Generate frozen reasoning probe plots."""

    from pressuretrace.analysis.plot_reasoning_probe_results import generate_reasoning_probe_plots
    from pressuretrace.analysis.reasoning_probe_reports import get_frozen_reasoning_probe_paths

    paths = get_frozen_reasoning_probe_paths()
    outputs = generate_reasoning_probe_plots(
        metrics_path=metrics_path or paths.probe_metrics_path,
        dataset_path=dataset_path or paths.probe_dataset_path,
        summary_path=summary_path or paths.probe_summary_path,
        output_dir=output_dir or paths.frozen_root / "results",
    )
    for path in outputs:
        console.print(f"plot: [bold]{path}[/bold]")


@app.command("probe-table-export")
def probe_table_export_command(
    metrics_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--metrics-path",
        help="Optional explicit probe metrics JSONL path.",
    ),
    dataset_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--dataset-path",
        help="Optional explicit probe dataset JSONL path.",
    ),
    csv_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--csv-path",
        help="Optional explicit paper-table CSV path.",
    ),
    markdown_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--markdown-path",
        help="Optional explicit paper-table Markdown path.",
    ),
) -> None:
    """Export the frozen reasoning probe paper table."""

    from pressuretrace.analysis.export_reasoning_probe_table import export_reasoning_probe_table
    from pressuretrace.analysis.reasoning_probe_reports import get_frozen_reasoning_probe_paths

    paths = get_frozen_reasoning_probe_paths()
    outputs = export_reasoning_probe_table(
        metrics_path=metrics_path or paths.probe_metrics_path,
        dataset_path=dataset_path or paths.probe_dataset_path,
        csv_path=csv_path or paths.table_csv_path,
        markdown_path=markdown_path or paths.table_md_path,
    )
    for name, path in outputs.items():
        console.print(f"{name}: [bold]{path}[/bold]")


@app.command("reasoning-patch-pairs")
def reasoning_patch_pairs_command(
    results_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--results-path",
        help="Optional explicit frozen reasoning results path.",
    ),
    control_slice_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--control-slice-path",
        help="Optional explicit frozen control-robust slice path.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit patch-pairs JSONL path.",
    ),
) -> None:
    """Build matched control/pressure patch pairs for reasoning route patching."""

    from pressuretrace.analysis.reasoning_probe_reports import get_frozen_reasoning_probe_paths
    from pressuretrace.patching.build_reasoning_patch_pairs import build_reasoning_patch_pairs

    paths = get_frozen_reasoning_probe_paths()
    patch_pairs_path = build_reasoning_patch_pairs(
        results_path=results_path or paths.paper_results_path,
        control_slice_path=control_slice_path or paths.control_slice_path,
        output_path=output_path or paths.patch_pairs_path,
    )
    console.print(f"patch_pairs: [bold]{patch_pairs_path}[/bold]")


@app.command("reasoning-route-patching")
def reasoning_route_patching_command(
    frozen_root: Path | None = typer.Option(  # noqa: B008
        None,
        "--frozen-root",
        help="Optional explicit repo-local frozen reasoning root.",
    ),
    patch_pairs_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--patch-pairs-path",
        help="Optional explicit patch-pairs JSONL path.",
    ),
    manifest_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--manifest-path",
        help="Optional explicit frozen manifest path.",
    ),
    results_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--results-path",
        help="Optional explicit frozen results path.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output-path",
        help="Optional explicit patching-output JSONL path.",
    ),
    summary_txt_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--summary-txt-path",
        help="Optional explicit patching summary TXT path.",
    ),
    summary_csv_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--summary-csv-path",
        help="Optional explicit patching summary CSV path.",
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen3-14B",
        help="Model used for reasoning route patching.",
    ),
    thinking_mode: str = typer.Option(
        "off",
        "--thinking-mode",
        help="Thinking mode for route patching.",
    ),
    layers: str = typer.Option(
        "-10,-8,-6",
        "--layers",
        help="Comma-separated patch layers.",
    ),
    pressure_types: str = typer.Option(
        "neutral_wrong_answer_cue,teacher_anchor",
        "--pressure-types",
        help="Comma-separated pressure types to retain.",
    ),
    max_pairs: int | None = typer.Option(
        None,
        "--max-pairs",
        help="Optional cap on retained pairs for debugging.",
    ),
) -> None:
    """Run first-pass logit-level reasoning route patching on frozen pairs."""

    from pressuretrace.patching.run_reasoning_route_patching import (
        build_route_patching_config,
        run_reasoning_route_patching,
    )

    config = build_route_patching_config(
        frozen_root=frozen_root,
        patch_pairs_path=patch_pairs_path,
        manifest_path=manifest_path,
        results_path=results_path,
        output_path=output_path,
        summary_txt_path=summary_txt_path,
        summary_csv_path=summary_csv_path,
        model_name=model_name,
        thinking_mode=thinking_mode,
        layers=tuple(int(part.strip()) for part in layers.split(",") if part.strip()),
        pressure_types=tuple(part.strip() for part in pressure_types.split(",") if part.strip()),
        max_pairs=max_pairs,
    )
    artifacts = run_reasoning_route_patching(config)
    console.print(f"patching_results: [bold]{artifacts.output_path}[/bold]")
    console.print(f"summary_txt: [bold]{artifacts.summary_txt_path}[/bold]")
    console.print(f"summary_csv: [bold]{artifacts.summary_csv_path}[/bold]")
    console.print(
        "plots: "
        f"[bold]{artifacts.rescue_delta_gold_prob_plot_path}[/bold], "
        f"[bold]{artifacts.rescue_delta_margin_plot_path}[/bold], "
        f"[bold]{artifacts.induction_delta_shortcut_prob_plot_path}[/bold]"
    )


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
