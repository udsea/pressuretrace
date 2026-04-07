"""Run the PressureTrace reasoning v2 benchmark from a paired manifest."""

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
    STRICT_INTEGER_SYSTEM_PROMPT,
    ReasoningGenerationProfile,
    ThinkingMode,
    default_results_path,
    generation_profile_for_reasoning_v2,
    infer_reasoning_response,
    load_reasoning_generator,
    validate_thinking_mode,
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
from pressuretrace.config import DEFAULT_MODELS
from pressuretrace.evaluation.reasoning_eval_v2 import evaluate_reasoning_response_v2
from pressuretrace.generation.reasoning.make_reasoning_tasks_v2 import (
    build_reasoning_manifest_v2,
    load_reasoning_manifest_v2,
)
from pressuretrace.utils.io import append_jsonl
from pressuretrace.utils.io import prepare_results_file as _prepare_results_file


@dataclass(frozen=True)
class ReasoningPilotArtifactsV2:
    """Paths produced by a reasoning v2 pilot run."""

    manifest_path: Path
    results_path: Path
    row_count: int
    pressure_type: str
    thinking_mode: ThinkingMode


def _default_output_path(
    split: str,
    pressure_type: str,
    thinking_mode: ThinkingMode,
    model_name: str,
) -> Path:
    """Build a stable default output path for reasoning v2 pilots."""

    return default_results_path(
        prefix="reasoning_pilot_v2",
        split=split,
        pressure_type=pressure_type,
        model_name=model_name,
        thinking_mode=thinking_mode,
    )


def _validate_thinking_mode(thinking_mode: str) -> ThinkingMode:
    """Validate and normalize the requested thinking mode."""

    return validate_thinking_mode(thinking_mode)


def _generation_profile_for_model_v2(
    model_name: str,
    thinking_mode: str,
) -> ReasoningGenerationProfile:
    """Choose a v2 generation profile for the selected model."""

    return generation_profile_for_reasoning_v2(model_name, thinking_mode)


def _load_reasoning_generator_v2(model_name: str, thinking_mode: str) -> Any:
    """Load and cache the shared text-generation backend for reasoning v2 pilots."""

    return load_reasoning_generator(
        model_name=model_name,
        profile=_generation_profile_for_model_v2(model_name, thinking_mode),
    )


def infer_reasoning_model_v2(
    prompt: str,
    model_name: str,
    thinking_mode: str,
) -> str:
    """Run a single reasoning prompt through the selected v2 model backend."""

    return infer_reasoning_response(
        prompt=prompt,
        model_name=model_name,
        profile=_generation_profile_for_model_v2(model_name, thinking_mode),
        strip_qwen3_thinking=False,
        system_prompt=STRICT_INTEGER_SYSTEM_PROMPT,
    )


def _run_reasoning_manifest_rows_v2(
    manifest_destination: Path,
    manifest_rows: list[dict[str, Any]],
    model_name: str,
    pressure_type: str,
    output_path: Path | None,
    dry_run: bool,
    validated_thinking_mode: ThinkingMode,
    console: Console | None,
    show_progress: bool,
) -> ReasoningPilotArtifactsV2:
    """Run inference over an already selected set of reasoning v2 manifest rows."""

    manifest_split = (
        str(manifest_rows[0]["metadata"].get("split", "test")) if manifest_rows else "test"
    )
    results_path = _prepare_results_file(
        output_path
        or _default_output_path(
            split=manifest_split,
            pressure_type=pressure_type,
            thinking_mode=validated_thinking_mode,
            model_name=model_name,
        )
    )
    selected_counts = _count_rows_by_pressure_type(manifest_rows)
    selected_base_tasks = _count_base_tasks(manifest_rows)

    if show_progress and console is not None:
        count_summary = ", ".join(
            f"{pressure_name}={count}" for pressure_name, count in selected_counts.items()
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
                _load_reasoning_generator_v2(model_name, validated_thinking_mode)
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
                        f"[{row['pressure_type']}] {row['task_id']} ({index}/{len(manifest_rows)})"
                    ),
                )

            if dry_run:
                result_row = {
                    **row,
                    "model_name": model_name,
                    "model_response": "",
                    "normalized_response": "",
                    "answer_visible_response": "",
                    "parsed_answer": None,
                    "selected_candidate": None,
                    "parse_candidates": [],
                    "parse_status": "failed",
                    "parse_confidence": "none",
                    "thinking_block_detected": False,
                    "thinking_mode": validated_thinking_mode,
                    "response_length_chars": 0,
                    "route_label": "pending_inference",
                    "failure_subtype": None,
                    "is_correct": False,
                    "duration_seconds": None,
                }
                append_jsonl(results_path, result_row)
                row_count += 1
                if progress is not None and progress_task_id is not None:
                    progress.advance(progress_task_id)
                continue

            start_time = time.perf_counter()
            response = infer_reasoning_model_v2(
                prompt=str(row["prompt"]),
                model_name=model_name,
                thinking_mode=validated_thinking_mode,
            )
            duration_seconds = time.perf_counter() - start_time
            evaluation = evaluate_reasoning_response_v2(
                model_response=response,
                gold_answer=str(row["gold_answer"]),
                shortcut_answer=str(row["shortcut_answer"]),
                prompt_text=str(row.get("base_question", row["prompt"])),
            )
            result_row = {
                **row,
                "model_name": model_name,
                "model_response": response,
                "normalized_response": evaluation.normalized_response,
                "answer_visible_response": evaluation.answer_visible_response,
                "parsed_answer": evaluation.parsed_answer,
                "selected_candidate": evaluation.selected_candidate,
                "parse_candidates": evaluation.parse_candidates,
                "parse_status": evaluation.parse_status.value,
                "parse_confidence": evaluation.parse_confidence.value,
                "thinking_block_detected": evaluation.thinking_block_detected,
                "thinking_mode": validated_thinking_mode,
                "response_length_chars": evaluation.response_length_chars,
                "route_label": evaluation.route_label.value,
                "failure_subtype": (
                    evaluation.failure_subtype.value
                    if evaluation.failure_subtype is not None
                    else None
                ),
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

    return ReasoningPilotArtifactsV2(
        manifest_path=manifest_destination,
        results_path=results_path,
        row_count=row_count,
        pressure_type=pressure_type,
        thinking_mode=validated_thinking_mode,
    )


def run_reasoning_manifest_v2(
    manifest_path: Path,
    model_name: str = DEFAULT_MODELS.reasoning_model,
    pressure_type: str = "all",
    output_path: Path | None = None,
    dry_run: bool = False,
    include_control: bool = True,
    thinking_mode: str = "default",
    console: Console | None = None,
    show_progress: bool = False,
) -> ReasoningPilotArtifactsV2:
    """Run reasoning v2 inference from a prebuilt manifest."""

    validated_thinking_mode = _validate_thinking_mode(thinking_mode)
    manifest_rows = _filter_manifest_rows(
        rows=load_reasoning_manifest_v2(manifest_path),
        pressure_type=pressure_type,
        include_control=include_control,
    )
    if show_progress and console is not None:
        console.rule("PressureTrace Reasoning V2 Manifest Run")
        console.print(f"Manifest: [bold]{manifest_path}[/bold]")
        console.print(f"Model: [bold]{model_name}[/bold]")
        console.print(f"Requested pressure type: [bold]{pressure_type}[/bold]")
        console.print(f"Thinking mode: [bold]{validated_thinking_mode}[/bold]")
        console.print(f"Include control: [bold]{include_control}[/bold]")
        console.print("[bold]Step 1/2[/bold] Loaded frozen reasoning v2 manifest.")

    return _run_reasoning_manifest_rows_v2(
        manifest_destination=manifest_path,
        manifest_rows=manifest_rows,
        model_name=model_name,
        pressure_type=pressure_type,
        output_path=output_path,
        dry_run=dry_run,
        validated_thinking_mode=validated_thinking_mode,
        console=console,
        show_progress=show_progress,
    )


def run_reasoning_pilot_v2(
    split: str = "test",
    limit: int = 20,
    model_name: str = DEFAULT_MODELS.reasoning_model,
    pressure_type: str = "all",
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    dry_run: bool = False,
    include_control: bool = True,
    thinking_mode: str = "default",
    console: Console | None = None,
    show_progress: bool = False,
) -> ReasoningPilotArtifactsV2:
    """Build a paired v2 manifest, run a model, and write result rows."""

    validated_thinking_mode = _validate_thinking_mode(thinking_mode)
    if show_progress and console is not None:
        console.rule("PressureTrace Reasoning V2 Pilot")
        console.print(f"Split: [bold]{split}[/bold]")
        console.print(f"Target retained base tasks: [bold]{limit}[/bold]")
        console.print(f"Model: [bold]{model_name}[/bold]")
        console.print(f"Requested pressure type: [bold]{pressure_type}[/bold]")
        console.print(f"Thinking mode: [bold]{validated_thinking_mode}[/bold]")
        console.print(f"Include control: [bold]{include_control}[/bold]")
        console.print("[bold]Step 1/3[/bold] Building paired reasoning v2 manifest from GSM8K.")

    manifest_destination = build_reasoning_manifest_v2(
        split=split,
        limit=limit,
        output_path=manifest_path,
    )
    manifest_rows = _filter_manifest_rows(
        rows=load_reasoning_manifest_v2(manifest_destination),
        pressure_type=pressure_type,
        include_control=include_control,
    )
    return _run_reasoning_manifest_rows_v2(
        manifest_destination=manifest_destination,
        manifest_rows=manifest_rows,
        model_name=model_name,
        pressure_type=pressure_type,
        output_path=output_path,
        dry_run=dry_run,
        validated_thinking_mode=validated_thinking_mode,
        console=console,
        show_progress=show_progress,
    )
