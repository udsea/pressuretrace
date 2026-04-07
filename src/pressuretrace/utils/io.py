"""Small JSONL helpers used throughout the repository."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import orjson
from pydantic import BaseModel


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_ready(row: Any) -> Any:
    """Convert common project types into JSON-serializable objects."""

    if isinstance(row, BaseModel):
        return row.model_dump(mode="json")
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(cast(Any, row))
    if isinstance(row, Path):
        return str(row)
    return row


def _default_serializer(value: Any) -> Any:
    """Handle JSON serialization for objects not covered by orjson directly."""

    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def write_jsonl(path: Path, rows: Iterable[Any]) -> Path:
    """Write an iterable of rows to a JSONL file."""

    ensure_directory(path.parent)
    with path.open("wb") as handle:
        for row in rows:
            payload = _json_ready(row)
            handle.write(orjson.dumps(payload, default=_default_serializer))
            handle.write(b"\n")
    return path


def append_jsonl(path: Path, row: Any) -> Path:
    """Append a single row to an existing JSONL file."""

    ensure_directory(path.parent)
    with path.open("ab") as handle:
        handle.write(orjson.dumps(_json_ready(row), default=_default_serializer))
        handle.write(b"\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    with path.open("rb") as handle:
        return [orjson.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    """Iterate lazily over rows in a JSONL file."""

    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield orjson.loads(line)
