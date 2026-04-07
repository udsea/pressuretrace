"""Coding evaluation interfaces and honest placeholders."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from pressuretrace.evaluation.route_labels import RouteLabel
from pressuretrace.types import CodingTask


class CodingEvaluationRecord(BaseModel):
    """Structured output produced by coding evaluators."""

    model_config = ConfigDict(extra="forbid")

    route_label: str = RouteLabel.UNKNOWN.value
    passed_public_tests: bool | None = None
    passed_hidden_tests: bool | None = None
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class CodingEvaluator(Protocol):
    """Interface for future coding evaluators."""

    def evaluate(self, task: CodingTask, completion: str) -> CodingEvaluationRecord:
        """Return structured coding-evaluation outputs for a single completion."""


def build_pending_coding_evaluation(task: CodingTask, completion: str) -> CodingEvaluationRecord:
    """Return an explicit placeholder record when evaluation is not implemented."""

    del completion
    return CodingEvaluationRecord(
        route_label=RouteLabel.UNKNOWN.value,
        notes=[
            (
                "Coding evaluation is not implemented in the v1 scaffold yet. "
                "This row exists to preserve result schemas and downstream aggregation."
            )
        ],
        metadata={"task_id": task.task_id, "evaluation_status": "pending"},
    )


def evaluate_coding_response(task: CodingTask, completion: str) -> CodingEvaluationRecord:
    """Evaluate a coding completion.

    TODO:
    1. Execute canonical and adversarial tests in an isolated sandbox.
    2. Distinguish robust correctness from visible-test spec gaming.
    3. Emit failure metadata that can feed probe training.
    """

    del task, completion
    raise NotImplementedError("Coding evaluation is not implemented in the v1 scaffold.")
