"""Run control-only coding-family evaluation from a frozen manifest."""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console

from pressuretrace.behavior.run_coding_paper_slice import (
    CodingBehaviorArtifacts,
    run_coding_manifest,
)
from pressuretrace.paths import results_dir


def _slugify_model_name(model_name: str) -> str:
    """Convert a model identifier into a filename-safe slug."""

    return re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()


def _default_control_only_results_path(
    *,
    model_name: str,
    thinking_mode: str,
) -> Path:
    """Build the default JSONL output path for control-only coding runs."""

    model_slug = _slugify_model_name(model_name)
    return results_dir() / f"coding_control_only_{model_slug}_{thinking_mode}.jsonl"


def run_coding_control_only(
    *,
    manifest_path: Path,
    model_name: str,
    output_path: Path | None = None,
    dry_run: bool = False,
    thinking_mode: str = "off",
    batch_size: int = 1,
    console: Console | None = None,
    show_progress: bool = False,
) -> CodingBehaviorArtifacts:
    """Run only control rows from a frozen coding-family manifest."""

    destination = output_path or _default_control_only_results_path(
        model_name=model_name,
        thinking_mode=thinking_mode,
    )
    return run_coding_manifest(
        manifest_path=manifest_path,
        model_name=model_name,
        pressure_type="control",
        output_path=destination,
        dry_run=dry_run,
        include_control=True,
        thinking_mode=thinking_mode,
        batch_size=batch_size,
        console=console,
        show_progress=show_progress,
    )
