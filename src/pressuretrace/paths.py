"""Helpers for resolving repository-relative paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk upward from a starting location until the repository root is found."""

    candidates = (start, *start.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not locate PressureTrace repository root from current file location.")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the absolute repository root path."""

    return _find_repo_root(Path(__file__).resolve().parent)


def data_dir() -> Path:
    """Return the data directory."""

    return repo_root() / "data"


def results_dir() -> Path:
    """Return the results directory."""

    return repo_root() / "results"


def manifests_dir() -> Path:
    """Return the directory for JSONL manifests and task catalogs."""

    return data_dir() / "manifests"


def splits_dir() -> Path:
    """Return the directory for deterministic split definitions."""

    return data_dir() / "splits"


def raw_data_dir() -> Path:
    """Return the raw data directory."""

    return data_dir() / "raw"


def interim_data_dir() -> Path:
    """Return the intermediate data directory."""

    return data_dir() / "interim"


def processed_data_dir() -> Path:
    """Return the processed data directory."""

    return data_dir() / "processed"


def paper_dir() -> Path:
    """Return the paper/specification directory."""

    return repo_root() / "paper"
