"""MBPP-style loader utilities."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pyarrow.ipc as ipc
from datasets import load_dataset

from pressuretrace.config import BASE_DATASETS
from pressuretrace.types import CodingTask


def _cached_mbpp_arrow_path(split: str) -> Path | None:
    """Return the cached MBPP Arrow path when available."""

    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / "google-research-datasets___mbpp"
        / "full"
        / "0.0.0"
    )
    if not cache_root.exists():
        return None
    matches = sorted(cache_root.glob(f"*/mbpp-{split}.arrow"))
    if not matches:
        return None
    return matches[-1]


def _load_cached_mbpp_rows(split: str) -> list[dict[str, Any]] | None:
    """Load cached MBPP rows directly from Arrow when possible."""

    arrow_path = _cached_mbpp_arrow_path(split)
    if arrow_path is None:
        return None
    with ipc.open_stream(arrow_path.open("rb")) as reader:
        return [dict(row) for row in reader.read_all().to_pylist()]


def _extract_entry_point(code: str | None) -> str | None:
    """Return the first top-level function name defined in MBPP code."""

    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def load_mbpp_tasks(split: str = "test", limit: int | None = None) -> list[CodingTask]:
    """Load MBPP-style coding tasks from Hugging Face datasets."""

    dataset_ref = BASE_DATASETS.mbpp
    cached_rows = _load_cached_mbpp_rows(split)
    if cached_rows is None:
        dataset = load_dataset(dataset_ref.path, split=split)
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        rows = [dict(example) for example in dataset]
    else:
        rows = cached_rows[: limit or None]

    tasks: list[CodingTask] = []
    for index, example in enumerate(rows):
        test_field = example.get("test_list") or example.get("test")
        if isinstance(test_field, list):
            test_code = "\n".join(test_field)
        else:
            test_code = test_field
        challenge_tests = example.get("challenge_test_list")
        if isinstance(challenge_tests, list):
            challenge_test_code = "\n".join(challenge_tests)
        else:
            challenge_test_code = challenge_tests

        tasks.append(
            CodingTask(
                task_id=str(example.get("task_id", f"mbpp-{split}-{index}")),
                source_dataset=dataset_ref.path,
                source_split=split,
                prompt=example.get("text", example.get("prompt", "")),
                entry_point=_extract_entry_point(example.get("code")),
                canonical_solution=example.get("code"),
                test_code=test_code,
                metadata={
                    "hf_dataset_path": dataset_ref.path,
                    "original_index": index,
                    "test_list": list(example.get("test_list") or []),
                    "challenge_test_list": list(example.get("challenge_test_list") or []),
                    "test_setup_code": str(example.get("test_setup_code", "")),
                    "challenge_test_code": challenge_test_code,
                },
            )
        )
    return tasks
