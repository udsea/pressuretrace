"""Build and load matched reasoning patch pairs for route patching."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pressuretrace.analysis.reasoning_probe_reports import (
    get_frozen_reasoning_probe_paths,
    load_jsonl_rows,
)
from pressuretrace.probes.train_reasoning_probes import deduplicate_episode_rows
from pressuretrace.utils.io import write_jsonl

TARGET_PRESSURE_TYPES: tuple[str, ...] = (
    "neutral_wrong_answer_cue",
    "teacher_anchor",
)
REQUIRED_PATCH_PAIR_KEYS: tuple[str, ...] = (
    "base_task_id",
    "pressure_type",
    "control_task_id",
    "pressure_task_id",
    "gold_answer",
    "shortcut_answer",
    "metadata",
)


@dataclass(frozen=True)
class ReasoningPatchPair:
    """Matched control/pressure pair with reconstructed prompts."""

    base_task_id: str
    pressure_type: str
    control_task_id: str
    pressure_task_id: str
    control_prompt: str
    pressure_prompt: str
    gold_answer: str
    shortcut_answer: str
    metadata: dict[str, Any]


def _load_control_slice(control_slice_path: Path) -> dict[str, dict[str, Any]]:
    """Index the frozen control-robust slice by base_task_id."""

    control_rows = load_jsonl_rows(control_slice_path)
    control_by_base_task_id: dict[str, dict[str, Any]] = {}
    for row in control_rows:
        base_task_id = str(row["base_task_id"])
        if str(row["control_route_label"]) != "robust_correct":
            continue
        if base_task_id in control_by_base_task_id:
            raise ValueError(f"Duplicate control slice entry for base_task_id={base_task_id}.")
        control_by_base_task_id[base_task_id] = row
    return control_by_base_task_id


def _metadata_value(row: dict[str, Any], key: str, default: Any = "") -> Any:
    """Return a value from a row, falling back to nested metadata when needed."""

    if key in row and row[key] is not None:
        return row[key]
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get(key) is not None:
        return metadata[key]
    return default


def _load_pressure_rows(results_path: Path) -> list[dict[str, Any]]:
    """Select the pressure rows that followed the shortcut answer."""

    result_rows = load_jsonl_rows(results_path)
    pressure_rows = [
        row
        for row in result_rows
        if str(row["pressure_type"]) in TARGET_PRESSURE_TYPES
        and str(row["route_label"]) == "shortcut_followed"
    ]
    return deduplicate_episode_rows(pressure_rows)


def build_reasoning_patch_pairs(
    *,
    results_path: Path,
    control_slice_path: Path,
    output_path: Path,
) -> Path:
    """Write matched control/pressure patch pairs for the frozen reasoning slice."""

    control_by_base_task_id = _load_control_slice(control_slice_path)
    pressure_rows = _load_pressure_rows(results_path)

    pair_rows: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, str]] = set()
    for pressure_row in pressure_rows:
        base_task_id = str(pressure_row["metadata"]["base_task_id"])
        pressure_type = str(pressure_row["pressure_type"])
        if pressure_type not in TARGET_PRESSURE_TYPES:
            continue

        control_row = control_by_base_task_id.get(base_task_id)
        if control_row is None:
            continue

        pair_key = (base_task_id, pressure_type)
        if pair_key in matched_keys:
            raise ValueError(
                f"Duplicate patch pair encountered for base_task_id={base_task_id}, "
                f"pressure_type={pressure_type}."
            )
        matched_keys.add(pair_key)
        pair_rows.append(
            {
                "base_task_id": base_task_id,
                "pressure_type": pressure_type,
                "control_task_id": str(control_row["control_task_id"]),
                "pressure_task_id": str(pressure_row["task_id"]),
                "gold_answer": str(pressure_row["gold_answer"]),
                "shortcut_answer": str(pressure_row["shortcut_answer"]),
                "metadata": {
                    "source_dataset": _metadata_value(control_row, "source_dataset"),
                    "source_id": _metadata_value(control_row, "source_id"),
                    "split": _metadata_value(control_row, "split"),
                    "prompt_family": _metadata_value(control_row, "prompt_family"),
                    "transformation_version": _metadata_value(
                        control_row,
                        "transformation_version",
                    ),
                    "model_name": _metadata_value(control_row, "model_name"),
                    "thinking_mode": _metadata_value(control_row, "thinking_mode"),
                    "control_route_label": _metadata_value(control_row, "control_route_label"),
                    "pressure_route_label": _metadata_value(pressure_row, "route_label"),
                    "pairing_strategy": "control_robust_vs_pressure_shortcut",
                },
            }
        )

    pair_rows.sort(key=lambda row: (str(row["pressure_type"]), str(row["base_task_id"])))
    write_jsonl(output_path, pair_rows)
    print(f"Matched patch pairs: {len(pair_rows)}")
    pressure_counts = Counter(str(row["pressure_type"]) for row in pair_rows)
    print(
        "Pressure types: "
        + ", ".join(
            f"{pressure_type}={pressure_counts.get(pressure_type, 0)}"
            for pressure_type in TARGET_PRESSURE_TYPES
        )
    )
    return output_path


def _validate_patch_pair_row(row: dict[str, Any]) -> None:
    """Ensure a stored patch-pair row meets the expected contract."""

    missing = [key for key in REQUIRED_PATCH_PAIR_KEYS if key not in row]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Patch-pair row is missing required keys: {missing_text}.")


def _build_task_prompt_index(
    *,
    manifest_path: Path,
    results_path: Path,
) -> dict[str, str]:
    """Index prompts by task id, validating manifest/result agreement when both exist."""

    prompts_by_task_id: dict[str, str] = {}
    manifest_rows = load_jsonl_rows(manifest_path)
    for row in manifest_rows:
        task_id = str(row["task_id"])
        prompt = str(row.get("prompt", ""))
        if prompt:
            prompts_by_task_id[task_id] = prompt

    result_rows = load_jsonl_rows(results_path)
    for row in result_rows:
        task_id = str(row["task_id"])
        prompt = str(row.get("prompt", ""))
        if not prompt:
            continue
        existing = prompts_by_task_id.get(task_id)
        if existing and existing != prompt:
            raise ValueError(f"Prompt mismatch between manifest and results for task_id={task_id}.")
        prompts_by_task_id[task_id] = prompt
    return prompts_by_task_id


def load_reasoning_patch_pairs(
    *,
    patch_pairs_path: Path,
    manifest_path: Path,
    results_path: Path,
    pressure_types: Sequence[str] = TARGET_PRESSURE_TYPES,
) -> list[ReasoningPatchPair]:
    """Load stored patch pairs and reconstruct their control and pressure prompts."""

    stored_rows = load_jsonl_rows(patch_pairs_path)
    print(f"Total pairs loaded: {len(stored_rows)}")
    prompt_by_task_id = _build_task_prompt_index(
        manifest_path=manifest_path,
        results_path=results_path,
    )

    retained_pairs: list[ReasoningPatchPair] = []
    pressure_counts: Counter[str] = Counter()
    allowed_pressure_types = {str(value) for value in pressure_types}
    for row in stored_rows:
        _validate_patch_pair_row(row)
        pressure_type = str(row["pressure_type"])
        if pressure_type not in allowed_pressure_types:
            continue

        control_task_id = str(row["control_task_id"])
        pressure_task_id = str(row["pressure_task_id"])
        control_prompt = prompt_by_task_id.get(control_task_id)
        pressure_prompt = prompt_by_task_id.get(pressure_task_id)
        if not control_prompt or not pressure_prompt:
            raise ValueError(
                "Could not reconstruct control/pressure prompts for "
                f"base_task_id={row['base_task_id']}."
            )

        retained_pairs.append(
            ReasoningPatchPair(
                base_task_id=str(row["base_task_id"]),
                pressure_type=pressure_type,
                control_task_id=control_task_id,
                pressure_task_id=pressure_task_id,
                control_prompt=control_prompt,
                pressure_prompt=pressure_prompt,
                gold_answer=str(row["gold_answer"]),
                shortcut_answer=str(row["shortcut_answer"]),
                metadata=dict(row["metadata"]),
            )
        )
        pressure_counts[pressure_type] += 1

    print(f"Retained pairs after filtering: {len(retained_pairs)}")
    print(
        "Per-pressure counts: "
        + ", ".join(
            f"{pressure_type}={pressure_counts.get(pressure_type, 0)}"
            for pressure_type in sorted(allowed_pressure_types)
        )
    )
    return retained_pairs


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for reasoning patch-pair construction."""

    paths = get_frozen_reasoning_probe_paths()
    parser = argparse.ArgumentParser(
        description="Build matched control/pressure patch pairs for the frozen reasoning run.",
    )
    parser.add_argument("--results-path", type=Path, default=paths.paper_results_path)
    parser.add_argument("--control-slice-path", type=Path, default=paths.control_slice_path)
    parser.add_argument("--output-path", type=Path, default=paths.patch_pairs_path)
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the reasoning patch-pair builder."""

    args = build_arg_parser().parse_args(argv)
    return build_reasoning_patch_pairs(
        results_path=args.results_path,
        control_slice_path=args.control_slice_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
