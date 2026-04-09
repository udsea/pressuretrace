"""Build and load matched coding patch pairs for route patching."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pressuretrace.config import resolve_coding_frozen_root
from pressuretrace.utils.io import read_jsonl, write_jsonl

TARGET_PRESSURE_TYPES: tuple[str, ...] = (
    "neutral_wrong_answer_cue",
    "teacher_anchor",
)
REQUIRED_PATCH_PAIR_KEYS: tuple[str, ...] = (
    "base_task_id",
    "pressure_type",
    "control_task_id",
    "pressure_task_id",
    "entry_point",
    "robust_reference_code",
    "shortcut_reference_code",
    "metadata",
)


@dataclass(frozen=True)
class CodingPatchPair:
    """Matched control/pressure pair with reconstructed prompts and code references."""

    base_task_id: str
    pressure_type: str
    control_task_id: str
    pressure_task_id: str
    control_prompt: str
    pressure_prompt: str
    entry_point: str
    archetype: str
    source_family: str
    robust_reference_code: str
    shortcut_reference_code: str
    metadata: dict[str, Any]


def _default_paths() -> tuple[Path, Path, Path]:
    """Return the default frozen coding input/output paths."""

    frozen_root = resolve_coding_frozen_root()
    return (
        frozen_root / "results" / "coding_paper_slice_qwen-qwen3-14b_off.jsonl",
        frozen_root / "data" / "splits" / "coding_control_robust_slice_qwen-qwen3-14b_off.jsonl",
        frozen_root / "results" / "coding_patch_pairs_qwen-qwen3-14b_off.jsonl",
    )


def _load_control_slice(control_slice_path: Path) -> dict[str, dict[str, Any]]:
    """Index the frozen control-robust slice by base_task_id."""

    control_rows = read_jsonl(control_slice_path)
    control_by_base_task_id: dict[str, dict[str, Any]] = {}
    for row in control_rows:
        base_task_id = str(row["base_task_id"])
        if str(row["control_route_label"]) != "robust_success":
            continue
        if base_task_id in control_by_base_task_id:
            raise ValueError(f"Duplicate control slice entry for base_task_id={base_task_id}.")
        control_by_base_task_id[base_task_id] = row
    return control_by_base_task_id


def _result_rows_by_base_task_id(results_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Group coding results rows by base task id."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(results_path):
        grouped.setdefault(str(row["base_task_id"]), []).append(row)
    return grouped


def _find_control_result_row(rows: list[dict[str, Any]], control_task_id: str) -> dict[str, Any]:
    """Find the control result row matching the frozen control slice entry."""

    for row in rows:
        if (
            str(row.get("task_id")) == control_task_id
            and str(row.get("pressure_type")) == "control"
        ):
            return row
    raise ValueError(f"Missing control result row for control_task_id={control_task_id}.")


def _find_pressure_shortcut_rows(
    rows: list[dict[str, Any]],
    *,
    pressure_types: set[str],
) -> list[dict[str, Any]]:
    """Return pressure rows that actually took the shortcut route."""

    return [
        row
        for row in rows
        if str(row.get("pressure_type")) in pressure_types
        and str(row.get("route_label")) == "shortcut_success"
    ]


def build_coding_patch_pairs(
    *,
    results_path: Path,
    control_slice_path: Path,
    output_path: Path,
    pressure_types: Sequence[str] = TARGET_PRESSURE_TYPES,
) -> Path:
    """Write matched control/pressure patch pairs for the frozen coding slice."""

    control_by_base_task_id = _load_control_slice(control_slice_path)
    result_rows_by_base_task_id = _result_rows_by_base_task_id(results_path)
    allowed_pressure_types = {str(value) for value in pressure_types}

    pair_rows: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, str]] = set()
    for base_task_id, control_slice_row in control_by_base_task_id.items():
        grouped_rows = result_rows_by_base_task_id.get(base_task_id, [])
        if not grouped_rows:
            continue

        control_result_row = _find_control_result_row(
            grouped_rows,
            control_task_id=str(control_slice_row["control_task_id"]),
        )
        robust_reference_code = str(control_result_row.get("extracted_code") or "").strip()
        if not robust_reference_code:
            continue

        for pressure_row in _find_pressure_shortcut_rows(
            grouped_rows,
            pressure_types=allowed_pressure_types,
        ):
            pressure_type = str(pressure_row["pressure_type"])
            pair_key = (base_task_id, pressure_type)
            if pair_key in matched_keys:
                raise ValueError(
                    f"Duplicate patch pair encountered for base_task_id={base_task_id}, "
                    f"pressure_type={pressure_type}."
                )
            shortcut_reference_code = str(pressure_row.get("extracted_code") or "").strip()
            if not shortcut_reference_code:
                continue
            if robust_reference_code == shortcut_reference_code:
                continue
            matched_keys.add(pair_key)
            pair_rows.append(
                {
                    "base_task_id": base_task_id,
                    "pressure_type": pressure_type,
                    "control_task_id": str(control_slice_row["control_task_id"]),
                    "pressure_task_id": str(pressure_row["task_id"]),
                    "entry_point": str(pressure_row.get("entry_point", "")),
                    "robust_reference_code": robust_reference_code,
                    "shortcut_reference_code": shortcut_reference_code,
                    "metadata": {
                        "source_family": str(control_slice_row.get("source_family", "")),
                        "source_task_name": str(control_slice_row.get("source_task_name", "")),
                        "archetype": str(control_slice_row.get("archetype", "")),
                        "model_name": str(control_slice_row.get("model_name", "")),
                        "thinking_mode": str(control_slice_row.get("thinking_mode", "")),
                        "control_route_label": "robust_success",
                        "pressure_route_label": str(pressure_row.get("route_label", "")),
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
            for pressure_type in sorted(allowed_pressure_types)
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
    for path in (manifest_path, results_path):
        for row in read_jsonl(path):
            task_id = str(row["task_id"])
            prompt = str(row.get("prompt", ""))
            if not prompt:
                continue
            existing = prompts_by_task_id.get(task_id)
            if existing and existing != prompt:
                raise ValueError(
                    "Prompt mismatch between manifest and results for "
                    f"task_id={task_id}."
                )
            prompts_by_task_id[task_id] = prompt
    return prompts_by_task_id


def load_coding_patch_pairs(
    *,
    patch_pairs_path: Path,
    manifest_path: Path,
    results_path: Path,
    pressure_types: Sequence[str] = TARGET_PRESSURE_TYPES,
) -> list[CodingPatchPair]:
    """Load stored patch pairs and reconstruct their control and pressure prompts."""

    stored_rows = read_jsonl(patch_pairs_path)
    prompt_by_task_id = _build_task_prompt_index(
        manifest_path=manifest_path,
        results_path=results_path,
    )
    retained_pairs: list[CodingPatchPair] = []
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

        metadata = dict(row["metadata"])
        retained_pairs.append(
            CodingPatchPair(
                base_task_id=str(row["base_task_id"]),
                pressure_type=pressure_type,
                control_task_id=control_task_id,
                pressure_task_id=pressure_task_id,
                control_prompt=control_prompt,
                pressure_prompt=pressure_prompt,
                entry_point=str(row["entry_point"]),
                archetype=str(metadata.get("archetype", "")),
                source_family=str(metadata.get("source_family", "")),
                robust_reference_code=str(row["robust_reference_code"]),
                shortcut_reference_code=str(row["shortcut_reference_code"]),
                metadata=metadata,
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
    """Build the CLI parser for coding patch-pair construction."""

    results_path, control_slice_path, output_path = _default_paths()
    parser = argparse.ArgumentParser(
        description="Build matched control/pressure patch pairs for the frozen coding run.",
    )
    parser.add_argument("--results-path", type=Path, default=results_path)
    parser.add_argument("--control-slice-path", type=Path, default=control_slice_path)
    parser.add_argument("--output-path", type=Path, default=output_path)
    parser.add_argument(
        "--pressure-types",
        type=str,
        default=",".join(TARGET_PRESSURE_TYPES),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Run the coding patch-pair builder."""

    args = build_arg_parser().parse_args(argv)
    pressure_types = tuple(
        part.strip() for part in str(args.pressure_types).split(",") if part.strip()
    )
    return build_coding_patch_pairs(
        results_path=args.results_path,
        control_slice_path=args.control_slice_path,
        output_path=args.output_path,
        pressure_types=pressure_types,
    )


if __name__ == "__main__":
    main()
