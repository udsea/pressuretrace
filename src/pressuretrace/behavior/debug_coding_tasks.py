"""Small coding-family debug runner for benchmark sanity checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.table import Table

from pressuretrace.behavior.run_coding_paper_slice import infer_coding_model_batch
from pressuretrace.evaluation.coding_eval import evaluate_coding_response
from pressuretrace.evaluation.coding_eval_debug import (
    REPRESENTATIVE_TASK_IDS,
    build_coding_eval_fixtures,
)
from pressuretrace.generation.coding.load_coding_base_tasks import load_coding_base_tasks
from pressuretrace.generation.coding.transform_coding_tasks import build_coding_episode_family
from pressuretrace.paths import results_dir
from pressuretrace.utils.io import write_jsonl

DebugSource = Literal["fixtures", "model"]


@dataclass(frozen=True)
class CodingDebugArtifacts:
    """Artifacts produced by a small coding-family debug run."""

    output_path: Path
    row_count: int
    shortcut_signal_count: int


def _slugify_model_name(model_name: str) -> str:
    """Convert a model identifier into a filename-safe slug."""

    return re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()


def _representative_rows() -> list[dict[str, Any]]:
    """Return the three representative tasks across all three coding variants."""

    tasks_by_id = {task.base_task_id: task for task in load_coding_base_tasks()}
    rows: list[dict[str, Any]] = []
    for base_task_id in REPRESENTATIVE_TASK_IDS:
        rows.extend(build_coding_episode_family(tasks_by_id[base_task_id]))
    return rows


def _fixture_completion_map() -> dict[tuple[str, str], str]:
    """Map representative tasks and fixture kinds to gold completions."""

    return {
        (fixture.base_task_id, fixture.fixture_kind): fixture.completion.strip()
        for fixture in build_coding_eval_fixtures()
    }


def _fixture_completion_for_row(row: dict[str, Any]) -> tuple[str, str]:
    """Choose a deterministic debug completion for one coding row."""

    pressure_type = str(row["pressure_type"])
    fixture_kind = "robust" if pressure_type == "control" else "shortcut"
    completion = _fixture_completion_map()[(str(row["base_task_id"]), fixture_kind)]
    return fixture_kind, completion


def _default_output_path(model_name: str, thinking_mode: str, source: DebugSource) -> Path:
    """Build the default output path for a small coding debug run."""

    model_slug = _slugify_model_name(model_name)
    if source == "fixtures":
        return results_dir() / f"coding_debug_run_{model_slug}_{thinking_mode}_fixtures.jsonl"
    return results_dir() / f"coding_debug_run_{model_slug}_{thinking_mode}.jsonl"


def _print_debug_table(rows: list[dict[str, Any]], *, console: Console | None = None) -> None:
    """Print a concise table for the debug run."""

    active_console = console or Console()
    table = Table(title="Coding Debug Run")
    table.add_column("Task")
    table.add_column("Pressure")
    table.add_column("Source")
    table.add_column("Route")
    table.add_column("Visible")
    table.add_column("Hidden")
    table.add_column("Subtype")
    for row in rows:
        table.add_row(
            str(row["base_task_id"]),
            str(row["pressure_type"]),
            str(row["debug_source"]),
            str(row["route_label"]),
            "pass" if bool(row["passed_visible_tests"]) else "fail",
            "pass" if bool(row["passed_hidden_tests"]) else "fail",
            str(row.get("failure_subtype") or ""),
        )
    active_console.print(table)


def run_coding_debug_tasks(
    *,
    model_name: str = "Qwen/Qwen3-14B",
    thinking_mode: str = "off",
    source: DebugSource = "fixtures",
    batch_size: int = 1,
    output_path: Path | None = None,
    console: Console | None = None,
    require_shortcut_signal: bool = True,
) -> CodingDebugArtifacts:
    """Run a small coding-family debug pass on representative tasks."""

    rows = _representative_rows()
    destination = output_path or _default_output_path(model_name, thinking_mode, source)

    if source == "model":
        responses = infer_coding_model_batch(
            [str(row["prompt"]) for row in rows],
            model_name=model_name,
            thinking_mode=thinking_mode,
            batch_size=max(1, batch_size),
        )
        fixture_roles = ["model"] * len(rows)
    else:
        fixture_pairs = [_fixture_completion_for_row(row) for row in rows]
        fixture_roles = [fixture_kind for fixture_kind, _ in fixture_pairs]
        responses = [completion for _, completion in fixture_pairs]

    debug_rows: list[dict[str, Any]] = []
    for row, response, fixture_role in zip(rows, responses, fixture_roles, strict=True):
        evaluation = evaluate_coding_response(row, response)
        debug_rows.append(
            {
                "task_id": row["task_id"],
                "base_task_id": row["base_task_id"],
                "archetype": row["archetype"],
                "pressure_type": row["pressure_type"],
                "prompt": row["prompt"],
                "debug_source": source,
                "fixture_role": fixture_role,
                "model_name": model_name,
                "thinking_mode": thinking_mode,
                "raw_code_output": response,
                "route_label": evaluation.route_label,
                "failure_subtype": evaluation.failure_subtype,
                "parse_status": evaluation.parse_status,
                "passed_visible_tests": evaluation.passed_visible_tests,
                "passed_hidden_tests": evaluation.passed_hidden_tests,
                "visible_test_results": evaluation.visible_test_results,
                "hidden_test_results": evaluation.hidden_test_results,
                "visible_failure_names": evaluation.visible_failure_names,
                "hidden_failure_names": evaluation.hidden_failure_names,
                "extracted_code": evaluation.extracted_code,
                "evaluation_notes": evaluation.notes,
            }
        )

    write_jsonl(destination, debug_rows)
    _print_debug_table(debug_rows, console=console)

    shortcut_signal_count = sum(
        1
        for row in debug_rows
        if str(row["route_label"]) == "shortcut_success"
        or (bool(row["passed_visible_tests"]) and not bool(row["passed_hidden_tests"]))
    )
    if require_shortcut_signal and shortcut_signal_count == 0:
        raise ValueError(
            "Coding debug run did not show any shortcut signal "
            "(no shortcut_success and no visible-pass / hidden-fail rows)."
        )

    return CodingDebugArtifacts(
        output_path=destination,
        row_count=len(debug_rows),
        shortcut_signal_count=shortcut_signal_count,
    )
