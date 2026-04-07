"""Run control-only reasoning v2 evaluation from a frozen manifest."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from pressuretrace.behavior.reasoning_runtime import slugify_model_name
from pressuretrace.behavior.run_reasoning_benchmark_v2 import (
    ReasoningPilotArtifactsV2,
    run_reasoning_manifest_v2,
)
from pressuretrace.config import DEFAULT_MODELS
from pressuretrace.paths import results_dir


def _default_control_only_results_path(
    model_name: str,
    thinking_mode: str,
) -> Path:
    """Build the default JSONL output path for control-only runs."""

    model_slug = slugify_model_name(model_name)
    return results_dir() / f"reasoning_control_only_{model_slug}_{thinking_mode}.jsonl"


def run_reasoning_control_only(
    manifest_path: Path,
    model_name: str = DEFAULT_MODELS.reasoning_model,
    output_path: Path | None = None,
    dry_run: bool = False,
    thinking_mode: str = "default",
    console: Console | None = None,
    show_progress: bool = False,
) -> ReasoningPilotArtifactsV2:
    """Run only control rows from a frozen reasoning v2 manifest."""

    destination = output_path or _default_control_only_results_path(
        model_name=model_name,
        thinking_mode=thinking_mode,
    )
    return run_reasoning_manifest_v2(
        manifest_path=manifest_path,
        model_name=model_name,
        pressure_type="control",
        output_path=destination,
        dry_run=dry_run,
        include_control=True,
        thinking_mode=thinking_mode,
        console=console,
        show_progress=show_progress,
    )
