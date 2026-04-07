"""Central route-label definitions used across task families."""

from __future__ import annotations

from enum import StrEnum


class RouteLabel(StrEnum):
    """Supported route labels for v1 benchmark outputs."""

    ROBUST_CORRECT = "robust_correct"
    SHORTCUT_FOLLOWED = "shortcut_followed"
    OTHER_ERROR = "other_error"
    SPEC_GAMING_SUSPECTED = "spec_gaming_suspected"
    UNKNOWN = "unknown"


REASONING_ROUTE_LABELS: tuple[RouteLabel, ...] = (
    RouteLabel.ROBUST_CORRECT,
    RouteLabel.SHORTCUT_FOLLOWED,
    RouteLabel.OTHER_ERROR,
)

CODING_ROUTE_LABELS: tuple[RouteLabel, ...] = (
    RouteLabel.ROBUST_CORRECT,
    RouteLabel.SPEC_GAMING_SUSPECTED,
    RouteLabel.OTHER_ERROR,
    RouteLabel.UNKNOWN,
)
