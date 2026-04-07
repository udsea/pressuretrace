"""Runner scaffold for coding pilots."""

from __future__ import annotations

from pathlib import Path

from pressuretrace.config import DEFAULT_MODELS, get_pressure_profile
from pressuretrace.evaluation.route_labels import RouteLabel
from pressuretrace.generation.coding.coding_transform import (
    build_pressure_variant,
    build_robust_variant,
    describe_shortcut_route,
)
from pressuretrace.generation.coding.humaneval_loader import load_humaneval_tasks
from pressuretrace.generation.coding.mbpp_loader import load_mbpp_tasks
from pressuretrace.paths import results_dir
from pressuretrace.types import BehaviorResult, BenchmarkEpisode, CodingTask
from pressuretrace.utils.io import write_jsonl


def infer_coding_model(episode: BenchmarkEpisode) -> str:
    """Placeholder hook for real coding-model inference."""

    del episode
    raise NotImplementedError("Model inference is not implemented in the v1 scaffold.")


def _default_output_path(split: str, pressure_profile: str) -> Path:
    """Build a stable default output path for coding pilots."""

    return results_dir() / f"coding_pilot_{split}_{pressure_profile}.jsonl"


def _load_coding_tasks(
    split: str,
    limit_per_dataset: int,
    include_humaneval: bool,
    include_mbpp: bool,
) -> list[CodingTask]:
    """Load configured coding task families."""

    tasks: list[CodingTask] = []
    if include_humaneval:
        tasks.extend(load_humaneval_tasks(split=split, limit=limit_per_dataset))
    if include_mbpp:
        tasks.extend(load_mbpp_tasks(split=split, limit=limit_per_dataset))
    if not tasks:
        raise ValueError("At least one coding base dataset must be selected.")
    return tasks


def _build_pending_result(task: CodingTask, episode: BenchmarkEpisode) -> BehaviorResult:
    """Return a schema-complete placeholder result row for dry runs."""

    return BehaviorResult(
        episode_id=episode.episode_id,
        task_id=task.task_id,
        family=task.family,
        source_dataset=task.source_dataset,
        variant=episode.variant,
        pressure_level=episode.pressure_level,
        route_label=RouteLabel.UNKNOWN.value,
        model_name=episode.model_name,
        model_response="",
        status="pending_inference",
        metadata={
            "source_split": task.source_split,
            "shortcut_route_description": describe_shortcut_route(task),
        },
    )


def run_coding_pilot(
    split: str = "test",
    limit_per_dataset: int = 5,
    model_name: str = DEFAULT_MODELS.coding_model,
    pressure_profile: str = "medium",
    output_path: Path | None = None,
    dry_run: bool = True,
    include_control: bool = True,
    include_humaneval: bool = True,
    include_mbpp: bool = True,
) -> Path:
    """Run a coding pilot and write JSONL result rows."""

    profile = get_pressure_profile(pressure_profile)
    tasks = _load_coding_tasks(
        split=split,
        limit_per_dataset=limit_per_dataset,
        include_humaneval=include_humaneval,
        include_mbpp=include_mbpp,
    )
    destination = output_path or _default_output_path(
        split=split,
        pressure_profile=pressure_profile,
    )

    rows: list[BehaviorResult] = []
    for task in tasks:
        episodes = [build_pressure_variant(task, model_name=model_name, pressure_profile=profile)]
        if include_control:
            episodes.insert(0, build_robust_variant(task, model_name=model_name))

        for episode in episodes:
            if dry_run:
                rows.append(_build_pending_result(task, episode))
                continue

            response = infer_coding_model(episode)
            rows.append(
                BehaviorResult(
                    episode_id=episode.episode_id,
                    task_id=task.task_id,
                    family=task.family,
                    source_dataset=task.source_dataset,
                    variant=episode.variant,
                    pressure_level=episode.pressure_level,
                    route_label=RouteLabel.UNKNOWN.value,
                    model_name=episode.model_name,
                    model_response=response,
                    status="pending_evaluation",
                    metadata={
                        "source_split": task.source_split,
                        "shortcut_route_description": describe_shortcut_route(task),
                        "evaluation_status": "pending",
                    },
                )
            )

    return write_jsonl(destination, rows)
