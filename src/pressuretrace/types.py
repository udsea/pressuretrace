"""Typed records used across PressureTrace generation and evaluation code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ReasoningTask(BaseModel):
    """Normalized base reasoning task derived from a source dataset."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: Literal["reasoning"] = "reasoning"
    source_dataset: str
    source_id: str
    prompt: str
    reference_solution: str
    reference_answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodingTask(BaseModel):
    """Canonical representation for a coding task derived from a base dataset."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: Literal["coding"] = "coding"
    source_dataset: str
    source_split: str
    prompt: str
    entry_point: str | None = None
    canonical_solution: str | None = None
    test_code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkEpisode(BaseModel):
    """A single model-facing benchmark episode produced from a canonical task."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: str
    family: str
    source_dataset: str
    source_split: str
    variant: str
    pressure_level: str
    prompt: str
    model_name: str
    expected_route: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BehaviorResult(BaseModel):
    """Structured output row for a single benchmark episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: str
    family: str
    source_dataset: str
    variant: str
    pressure_level: str
    route_label: str
    model_name: str
    model_response: str
    status: str
    score: float | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProbeRow(BaseModel):
    """Row-level metadata used to train or analyze linear probes."""

    model_config = ConfigDict(extra="forbid")

    row_id: str
    episode_id: str
    layer_index: int
    token_index: int
    route_label: str
    activation_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
