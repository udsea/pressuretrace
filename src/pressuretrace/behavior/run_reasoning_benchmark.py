"""Run the first real reasoning pilot from a paired manifest."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pressuretrace.behavior.reasoning_runtime import (
    ReasoningGenerationProfile,
    default_results_path,
    infer_reasoning_response,
    load_reasoning_generator,
    strip_qwen3_thinking_content,
)
from pressuretrace.behavior.reasoning_runtime import (
    count_base_tasks as _count_base_tasks,
)
from pressuretrace.behavior.reasoning_runtime import (
    count_rows_by_pressure_type as _count_rows_by_pressure_type,
)
from pressuretrace.behavior.reasoning_runtime import (
    filter_manifest_rows as _filter_manifest_rows,
)
from pressuretrace.behavior.reasoning_runtime import (
    is_qwen3_model as _is_qwen3_model,
)
from pressuretrace.config import DEFAULT_MODELS
from pressuretrace.evaluation.reasoning_eval import evaluate_reasoning_response
from pressuretrace.generation.reasoning.make_reasoning_tasks import (
    build_reasoning_manifest,
    load_reasoning_manifest,
)
from pressuretrace.utils.io import append_jsonl
from pressuretrace.utils.io import prepare_results_file as _prepare_results_file


@dataclass(frozen=True)
class ReasoningPilotArtifacts:
    """Paths produced by a reasoning pilot run."""

    manifest_path: Path
    results_path: Path
    row_count: int
    pressure_type: str


REASONING_MAX_NEW_TOKENS = 64
QWEN3_MAX_NEW_TOKENS = 256


def _strip_qwen3_thinking_content(text: str) -> str:
    """Expose the shared Qwen3 post-processing helper for v1 tests."""

    return strip_qwen3_thinking_content(text)


def _default_output_path(split: str, pressure_type: str, model_name: str) -> Path:
    """Build a stable default output path for reasoning pilots."""

    return default_results_path(
        prefix="reasoning_pilot",
        split=split,
        pressure_type=pressure_type,
        model_name=model_name,
    )


def _generation_profile_for_model(model_name: str) -> ReasoningGenerationProfile:
    """Choose a generation profile for the selected model."""

    if _is_qwen3_model(model_name):
        return ReasoningGenerationProfile(
            backend="manual_qwen3",
            do_sample=True,
            max_new_tokens=QWEN3_MAX_NEW_TOKENS,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            enable_thinking=True,
        )

    return ReasoningGenerationProfile(
        backend="pipeline_chat",
        do_sample=False,
        max_new_tokens=REASONING_MAX_NEW_TOKENS,
    )


def _load_reasoning_generator(model_name: str) -> Any:
    """Load and cache the shared text-generation backend for reasoning pilots."""

    return load_reasoning_generator(
        model_name=model_name,
        profile=_generation_profile_for_model(model_name),
    )


def infer_reasoning_model(prompt: str, model_name: str) -> str:
    """Run a single reasoning prompt through the selected model."""

    return infer_reasoning_response(
        prompt=prompt,
        model_name=model_name,
        profile=_generation_profile_for_model(model_name),
        strip_qwen3_thinking=True,
    )


def run_reasoning_pilot(
    split: str = "test",
    limit: int = 20,
    model_name: str = DEFAULT_MODELS.reasoning_model,
    pressure_type: str = "authority_conflict",
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    dry_run: bool = False,
    include_control: bool = True,
    console: Console | None = None,
    show_progress: bool = False,
) -> ReasoningPilotArtifacts:
    """Build a paired manifest, run a model, and write result rows."""

    if show_progress and console is not None:
        console.rule("PressureTrace Reasoning Pilot")
        console.print(f"Split: [bold]{split}[/bold]")
        console.print(f"Target retained base tasks: [bold]{limit}[/bold]")
        console.print(f"Model: [bold]{model_name}[/bold]")
        console.print(f"Requested pressure type: [bold]{pressure_type}[/bold]")
        console.print(f"Include control: [bold]{include_control}[/bold]")
        console.print("[bold]Step 1/3[/bold] Building paired reasoning manifest from GSM8K.")

    manifest_destination = build_reasoning_manifest(
        split=split,
        limit=limit,
        output_path=manifest_path,
    )
    manifest_rows = _filter_manifest_rows(
        rows=load_reasoning_manifest(manifest_destination),
        pressure_type=pressure_type,
        include_control=include_control,
    )
    results_path = _prepare_results_file(
        output_path
        or _default_output_path(
            split=split,
            pressure_type=pressure_type,
            model_name=model_name,
        )
    )
    selected_counts = _count_rows_by_pressure_type(manifest_rows)
    selected_base_tasks = _count_base_tasks(manifest_rows)

    if show_progress and console is not None:
        count_summary = ", ".join(
            f"{pressure_name}={count}"
            for pressure_name, count in selected_counts.items()
        )
        console.print(
            f"[bold]Step 2/3[/bold] Prepared [bold]{len(manifest_rows)}[/bold] episodes "
            f"from [bold]{selected_base_tasks}[/bold] base tasks."
        )
        console.print(f"Conditions: {count_summary}")
        console.print(f"Results path: [bold]{results_path}[/bold]")
        if dry_run:
            console.print("[bold]Step 3/3[/bold] Writing placeholder rows.")
        else:
            with console.status(f"Loading model [bold]{model_name}[/bold]..."):
                _load_reasoning_generator(model_name)
            console.print("[bold]Step 3/3[/bold] Running model inference.")

    progress: Progress | None = None
    progress_task_id: TaskID | None = None
    if show_progress and console is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        progress.start()
        progress_task_id = progress.add_task(
            description="Preparing episodes...",
            total=len(manifest_rows),
        )

    row_count = 0
    total_duration_seconds = 0.0
    try:
        for index, row in enumerate(manifest_rows, start=1):
            if progress is not None and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    description=(
                        f"[{row['pressure_type']}] {row['task_id']} "
                        f"({index}/{len(manifest_rows)})"
                    ),
                )

            if dry_run:
                result_row = {
                    **row,
                    "model_name": model_name,
                    "model_response": "",
                    "parsed_answer": None,
                    "route_label": "pending_inference",
                    "is_correct": False,
                    "duration_seconds": None,
                }
                append_jsonl(results_path, result_row)
                row_count += 1
                if progress is not None and progress_task_id is not None:
                    progress.advance(progress_task_id)
                continue

            start_time = time.perf_counter()
            response = infer_reasoning_model(prompt=str(row["prompt"]), model_name=model_name)
            duration_seconds = time.perf_counter() - start_time
            evaluation = evaluate_reasoning_response(
                model_response=response,
                gold_answer=str(row["gold_answer"]),
                shortcut_answer=str(row["shortcut_answer"]),
            )
            result_row = {
                **row,
                "model_name": model_name,
                "model_response": response,
                "parsed_answer": evaluation.parsed_answer,
                "route_label": evaluation.route_label.value,
                "is_correct": evaluation.is_correct,
                "duration_seconds": duration_seconds,
            }
            append_jsonl(results_path, result_row)
            row_count += 1
            total_duration_seconds += duration_seconds
            if progress is not None and progress_task_id is not None:
                average_duration_seconds = total_duration_seconds / row_count
                progress.advance(progress_task_id)
                progress.update(
                    progress_task_id,
                    description=(
                        f"[{row['pressure_type']}] {row['task_id']} "
                        f"({index}/{len(manifest_rows)}, avg {average_duration_seconds:.1f}s/ep)"
                    ),
                )
    finally:
        if progress is not None:
            progress.stop()

    if show_progress and console is not None:
        console.print(
            f"Completed [bold]{row_count}[/bold] episodes. "
            f"Wrote JSONL rows to [bold]{results_path}[/bold]."
        )

    return ReasoningPilotArtifacts(
        manifest_path=manifest_destination,
        results_path=results_path,
        row_count=row_count,
        pressure_type=pressure_type,
    )
