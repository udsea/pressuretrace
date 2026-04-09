"""Metric and reporting helpers for coding route patching."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import torch
import torch.nn.functional as F

from pressuretrace.utils.io import ensure_directory

Direction: TypeAlias = Literal["rescue", "induction"]
FIRST_PASS_LAYERS: tuple[int, ...] = (-8, -10, -6)
FIRST_PASS_PRESSURE_TYPES: tuple[str, ...] = ("neutral_wrong_answer_cue",)
FIRST_PASS_DIRECTIONS: tuple[Direction, ...] = ("rescue", "induction")
EARLY_GENERATION_LAYERS: tuple[int, ...] = (-10, -8, -6)
EARLY_GENERATION_PRESSURE_TYPES: tuple[str, ...] = ("neutral_wrong_answer_cue",)
EARLY_GENERATION_POSITION_WINDOWS: tuple[str, ...] = ("gen_1", "gen_1_3", "gen_1_5")


class SupportsTokenToString(Protocol):
    """Tokenizer surface used by the metrics helpers."""

    def convert_ids_to_tokens(self, token_ids: int | Sequence[int]) -> str | list[str]:
        """Convert token ids to token strings."""

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        """Decode token ids to a string."""


@dataclass(frozen=True)
class ContinuationPair:
    """Tokenized robust and shortcut continuation sequences for one matched pair."""

    robust_token_ids: tuple[int, ...]
    robust_token_strs: tuple[str, ...]
    shortcut_token_ids: tuple[int, ...]
    shortcut_token_strs: tuple[str, ...]


@dataclass(frozen=True)
class ContinuationScore:
    """Teacher-forced log-prob score for one continuation sequence."""

    token_ids: tuple[int, ...]
    token_strs: tuple[str, ...]
    logprob_sum: float
    logprob_mean: float


@dataclass(frozen=True)
class ContinuationSnapshot:
    """Sequence-aware robust vs shortcut preference metrics for one prompt state."""

    robust_logit: float
    shortcut_logit: float
    robust_prob: float
    shortcut_prob: float
    robust_minus_shortcut_margin: float
    robust_sequence_logprob_sum: float
    shortcut_sequence_logprob_sum: float
    robust_sequence_logprob_mean: float
    shortcut_sequence_logprob_mean: float
    preferred_route: Literal["robust", "shortcut"]
    top1_token_id: int
    top1_token_str: str


@dataclass(frozen=True)
class CodingPatchComparisonRow:
    """Flat metrics row for a single control/pressure/patched coding comparison."""

    base_task_id: str
    pressure_type: str
    layer: int
    direction: Direction
    control_task_id: str
    pressure_task_id: str
    entry_point: str
    archetype: str
    source_family: str
    robust_reference_code: str
    shortcut_reference_code: str
    robust_token_ids: list[int]
    robust_token_strs: list[str]
    shortcut_token_ids: list[int]
    shortcut_token_strs: list[str]
    continuation_scoring_mode: str
    baseline_kind: Literal["control", "pressure"]
    control_robust_prob: float
    control_shortcut_prob: float
    control_margin: float
    pressure_robust_prob: float
    pressure_shortcut_prob: float
    pressure_margin: float
    patched_robust_prob: float
    patched_shortcut_prob: float
    patched_margin: float
    control_preferred_route: str
    pressure_preferred_route: str
    patched_preferred_route: str
    control_top1_token_id: int
    control_top1_token_str: str
    pressure_top1_token_id: int
    pressure_top1_token_str: str
    patched_top1_token_id: int
    patched_top1_token_str: str
    delta_robust_prob: float
    delta_shortcut_prob: float
    delta_margin: float
    top1_changed: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CodingPatchSummaryRow:
    """Aggregated patching summary for one pressure/layer/direction group."""

    pressure_type: str
    layer: int
    direction: Direction
    n_pairs: int
    mean_delta_robust_prob: float
    mean_delta_shortcut_prob: float
    mean_delta_margin: float
    top1_changed_rate: float


def _as_1d_tensor(logits: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Convert logits to a float tensor with one dimension."""

    tensor = torch.as_tensor(logits, dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError(f"Expected a 1D logits vector, got shape {tuple(tensor.shape)}.")
    return tensor


def _token_to_string(tokenizer: SupportsTokenToString | None, token_id: int) -> str:
    """Render a token id as a human-readable token string."""

    if tokenizer is None:
        return str(token_id)
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token, list):
            return token[0]
        return str(token)
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def build_continuation_pair(
    *,
    tokenizer: SupportsTokenToString,
    robust_token_ids: Sequence[int],
    shortcut_token_ids: Sequence[int],
) -> ContinuationPair | None:
    """Build tokenized robust/shortcut continuation sequences for scoring."""

    robust_ids = tuple(int(token_id) for token_id in robust_token_ids)
    shortcut_ids = tuple(int(token_id) for token_id in shortcut_token_ids)
    if not robust_ids or not shortcut_ids:
        return None
    return ContinuationPair(
        robust_token_ids=robust_ids,
        robust_token_strs=tuple(_token_to_string(tokenizer, token_id) for token_id in robust_ids),
        shortcut_token_ids=shortcut_ids,
        shortcut_token_strs=tuple(
            _token_to_string(tokenizer, token_id) for token_id in shortcut_ids
        ),
    )


def compute_continuation_snapshot(
    *,
    robust_score: ContinuationScore,
    shortcut_score: ContinuationScore,
    next_token_logits: Sequence[float] | torch.Tensor,
    tokenizer: SupportsTokenToString | None = None,
) -> ContinuationSnapshot:
    """Compute sequence-aware robust vs shortcut preference metrics for one prompt state."""

    scores = torch.tensor(
        [robust_score.logprob_mean, shortcut_score.logprob_mean],
        dtype=torch.float32,
    )
    pairwise_probabilities = F.softmax(scores, dim=0)
    next_token_tensor = _as_1d_tensor(next_token_logits)
    top1_token_id = int(torch.argmax(next_token_tensor).item())
    preferred_route: Literal["robust", "shortcut"] = (
        "robust" if robust_score.logprob_mean >= shortcut_score.logprob_mean else "shortcut"
    )
    return ContinuationSnapshot(
        robust_logit=robust_score.logprob_mean,
        shortcut_logit=shortcut_score.logprob_mean,
        robust_prob=float(pairwise_probabilities[0].item()),
        shortcut_prob=float(pairwise_probabilities[1].item()),
        robust_minus_shortcut_margin=float(
            robust_score.logprob_mean - shortcut_score.logprob_mean
        ),
        robust_sequence_logprob_sum=robust_score.logprob_sum,
        shortcut_sequence_logprob_sum=shortcut_score.logprob_sum,
        robust_sequence_logprob_mean=robust_score.logprob_mean,
        shortcut_sequence_logprob_mean=shortcut_score.logprob_mean,
        preferred_route=preferred_route,
        top1_token_id=top1_token_id,
        top1_token_str=_token_to_string(tokenizer, top1_token_id),
    )


def baseline_kind_for_direction(direction: Direction) -> Literal["control", "pressure"]:
    """Return the baseline prompt that a patch direction should be compared against."""

    return "pressure" if direction == "rescue" else "control"


def build_patch_comparison_row(
    *,
    base_task_id: str,
    pressure_type: str,
    layer: int,
    direction: Direction,
    control_task_id: str,
    pressure_task_id: str,
    entry_point: str,
    archetype: str,
    source_family: str,
    robust_reference_code: str,
    shortcut_reference_code: str,
    continuation_pair: ContinuationPair,
    control_snapshot: ContinuationSnapshot,
    pressure_snapshot: ContinuationSnapshot,
    patched_snapshot: ContinuationSnapshot,
    metadata: Mapping[str, Any],
) -> CodingPatchComparisonRow:
    """Build a flat row for a single patched run."""

    baseline_kind = baseline_kind_for_direction(direction)
    baseline_snapshot = pressure_snapshot if baseline_kind == "pressure" else control_snapshot
    delta_robust_prob = patched_snapshot.robust_prob - baseline_snapshot.robust_prob
    delta_shortcut_prob = patched_snapshot.shortcut_prob - baseline_snapshot.shortcut_prob
    delta_margin = (
        patched_snapshot.robust_minus_shortcut_margin
        - baseline_snapshot.robust_minus_shortcut_margin
    )
    top1_changed = patched_snapshot.top1_token_id != baseline_snapshot.top1_token_id

    return CodingPatchComparisonRow(
        base_task_id=base_task_id,
        pressure_type=pressure_type,
        layer=layer,
        direction=direction,
        control_task_id=control_task_id,
        pressure_task_id=pressure_task_id,
        entry_point=entry_point,
        archetype=archetype,
        source_family=source_family,
        robust_reference_code=robust_reference_code,
        shortcut_reference_code=shortcut_reference_code,
        robust_token_ids=list(continuation_pair.robust_token_ids),
        robust_token_strs=list(continuation_pair.robust_token_strs),
        shortcut_token_ids=list(continuation_pair.shortcut_token_ids),
        shortcut_token_strs=list(continuation_pair.shortcut_token_strs),
        continuation_scoring_mode="sequence_mean_logprob_pair_softmax",
        baseline_kind=baseline_kind,
        control_robust_prob=control_snapshot.robust_prob,
        control_shortcut_prob=control_snapshot.shortcut_prob,
        control_margin=control_snapshot.robust_minus_shortcut_margin,
        pressure_robust_prob=pressure_snapshot.robust_prob,
        pressure_shortcut_prob=pressure_snapshot.shortcut_prob,
        pressure_margin=pressure_snapshot.robust_minus_shortcut_margin,
        patched_robust_prob=patched_snapshot.robust_prob,
        patched_shortcut_prob=patched_snapshot.shortcut_prob,
        patched_margin=patched_snapshot.robust_minus_shortcut_margin,
        control_preferred_route=control_snapshot.preferred_route,
        pressure_preferred_route=pressure_snapshot.preferred_route,
        patched_preferred_route=patched_snapshot.preferred_route,
        control_top1_token_id=control_snapshot.top1_token_id,
        control_top1_token_str=control_snapshot.top1_token_str,
        pressure_top1_token_id=pressure_snapshot.top1_token_id,
        pressure_top1_token_str=pressure_snapshot.top1_token_str,
        patched_top1_token_id=patched_snapshot.top1_token_id,
        patched_top1_token_str=patched_snapshot.top1_token_str,
        delta_robust_prob=delta_robust_prob,
        delta_shortcut_prob=delta_shortcut_prob,
        delta_margin=delta_margin,
        top1_changed=top1_changed,
        metadata=dict(metadata),
    )


def patch_row_to_dict(row: CodingPatchComparisonRow | Mapping[str, Any]) -> dict[str, Any]:
    """Convert a patch comparison row to a plain dictionary."""

    if isinstance(row, CodingPatchComparisonRow):
        return asdict(row)
    return dict(row)


def aggregate_patch_rows(
    rows: Iterable[CodingPatchComparisonRow | Mapping[str, Any]],
) -> list[CodingPatchSummaryRow]:
    """Aggregate patch rows by pressure type, layer, and direction."""

    grouped_rows: dict[tuple[str, int, Direction], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_dict = patch_row_to_dict(row)
        key = (
            str(row_dict["pressure_type"]),
            int(row_dict["layer"]),
            row_dict["direction"],
        )
        grouped_rows[key].append(row_dict)

    summary_rows: list[CodingPatchSummaryRow] = []
    for (pressure_type, layer, direction), group_rows in sorted(grouped_rows.items()):
        n_pairs = len(group_rows)
        mean_delta_robust_prob = (
            sum(float(row["delta_robust_prob"]) for row in group_rows) / n_pairs
        )
        mean_delta_shortcut_prob = (
            sum(float(row["delta_shortcut_prob"]) for row in group_rows) / n_pairs
        )
        mean_delta_margin = sum(float(row["delta_margin"]) for row in group_rows) / n_pairs
        top1_changed_rate = sum(bool(row["top1_changed"]) for row in group_rows) / n_pairs
        summary_rows.append(
            CodingPatchSummaryRow(
                pressure_type=pressure_type,
                layer=layer,
                direction=direction,
                n_pairs=n_pairs,
                mean_delta_robust_prob=mean_delta_robust_prob,
                mean_delta_shortcut_prob=mean_delta_shortcut_prob,
                mean_delta_margin=mean_delta_margin,
                top1_changed_rate=top1_changed_rate,
            )
        )
    return summary_rows


def _format_float(value: float, digits: int = 4) -> str:
    """Format a floating-point summary value."""

    return f"{value:.{digits}f}"


def render_summary_text(rows: Iterable[CodingPatchSummaryRow | Mapping[str, Any]]) -> str:
    """Render a compact TXT summary for grouped patch metrics."""

    summary_rows = [
        asdict(row) if isinstance(row, CodingPatchSummaryRow) else dict(row)
        for row in rows
    ]
    lines = [
        "PressureTrace coding route patching summary",
        "",
        "Grouped by pressure_type, layer, direction.",
        "",
        "pressure_type | layer | direction | n_pairs | mean_delta_robust_prob | "
        "mean_delta_shortcut_prob | mean_delta_margin | top1_changed_rate",
    ]
    for row in summary_rows:
        lines.append(
            " | ".join(
                [
                    str(row["pressure_type"]),
                    str(row["layer"]),
                    str(row["direction"]),
                    str(row["n_pairs"]),
                    _format_float(float(row["mean_delta_robust_prob"])),
                    _format_float(float(row["mean_delta_shortcut_prob"])),
                    _format_float(float(row["mean_delta_margin"])),
                    _format_float(float(row["top1_changed_rate"])),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def highlight_summary_rows(rows: Iterable[CodingPatchSummaryRow | Mapping[str, Any]]) -> str:
    """Render concise highlights for the grouped summary."""

    summary_rows = [
        row if isinstance(row, CodingPatchSummaryRow) else CodingPatchSummaryRow(**dict(row))
        for row in rows
    ]
    if not summary_rows:
        return "No summary rows were produced.\n"

    def _best(direction: Direction, metric_name: str) -> CodingPatchSummaryRow:
        candidates = [row for row in summary_rows if row.direction == direction]
        return max(candidates, key=lambda row: getattr(row, metric_name))

    best_rescue_robust = _best("rescue", "mean_delta_robust_prob")
    best_rescue_margin = _best("rescue", "mean_delta_margin")
    strongest_induction = _best("induction", "mean_delta_shortcut_prob")

    lines = [
        "Highlights:",
        (
            "  Best rescue layer by mean delta_robust_prob: "
            f"{best_rescue_robust.pressure_type}, layer={best_rescue_robust.layer}, "
            f"value={_format_float(best_rescue_robust.mean_delta_robust_prob)}"
        ),
        (
            "  Best rescue layer by mean delta_margin: "
            f"{best_rescue_margin.pressure_type}, layer={best_rescue_margin.layer}, "
            f"value={_format_float(best_rescue_margin.mean_delta_margin)}"
        ),
        (
            "  Strongest induction layer by shortcut increase: "
            f"{strongest_induction.pressure_type}, layer={strongest_induction.layer}, "
            f"value={_format_float(strongest_induction.mean_delta_shortcut_prob)}"
        ),
    ]
    return "\n".join(lines) + "\n"


def patch_summary_to_csv_text(rows: Iterable[CodingPatchSummaryRow | Mapping[str, Any]]) -> str:
    """Render a compact CSV summary as text."""

    header = (
        "pressure_type,layer,direction,n_pairs,mean_delta_robust_prob,"
        "mean_delta_shortcut_prob,mean_delta_margin,top1_changed_rate"
    )
    body = [
        ",".join(
            [
                str(row["pressure_type"]),
                str(row["layer"]),
                str(row["direction"]),
                str(row["n_pairs"]),
                _format_float(float(row["mean_delta_robust_prob"])),
                _format_float(float(row["mean_delta_shortcut_prob"])),
                _format_float(float(row["mean_delta_margin"])),
                _format_float(float(row["top1_changed_rate"])),
            ]
        )
        for row in (
            asdict(item) if isinstance(item, CodingPatchSummaryRow) else dict(item)
            for item in rows
        )
    ]
    return "\n".join([header, *body]) + "\n"


def write_summary_csv(
    rows: Iterable[CodingPatchSummaryRow | Mapping[str, Any]],
    output_path: Path,
) -> Path:
    """Write grouped patch summary rows to CSV."""

    ensure_directory(output_path.parent)
    output_path.write_text(patch_summary_to_csv_text(rows), encoding="utf-8")
    return output_path


def plot_patch_summary(
    rows: Iterable[CodingPatchSummaryRow | Mapping[str, Any]],
    *,
    direction: Direction,
    metric_name: Literal["mean_delta_robust_prob", "mean_delta_margin", "mean_delta_shortcut_prob"],
    output_path: Path,
) -> Path:
    """Plot one grouped patch-summary metric across layers by pressure type."""

    from matplotlib import pyplot as plt

    summary_rows = [
        row if isinstance(row, CodingPatchSummaryRow) else CodingPatchSummaryRow(**dict(row))
        for row in rows
    ]
    filtered_rows = [row for row in summary_rows if row.direction == direction]
    ensure_directory(output_path.parent)
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for pressure_type in sorted({row.pressure_type for row in filtered_rows}):
        pressure_rows = sorted(
            (row for row in filtered_rows if row.pressure_type == pressure_type),
            key=lambda row: row.layer,
        )
        axis.plot(
            [row.layer for row in pressure_rows],
            [getattr(row, metric_name) for row in pressure_rows],
            marker="o",
            label=pressure_type,
        )
    axis.set_xlabel("Layer")
    axis.set_ylabel(metric_name)
    axis.set_title(f"Coding route patching: {direction} / {metric_name}")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def position_window_steps(position_window: str) -> tuple[int, ...]:
    """Return the generation-step indices covered by a named patch window."""

    if position_window == "gen_1":
        return (0,)
    if position_window == "gen_1_3":
        return (0, 1, 2)
    if position_window == "gen_1_5":
        return (0, 1, 2, 3, 4)
    raise ValueError(f"Unknown position_window={position_window!r}.")


@dataclass(frozen=True)
class CodingRoutePatchingRow:
    """One generation-time coding patching result row."""

    base_task_id: str
    archetype: str
    pressure_type: str
    layer: int
    position_window: str
    direction: Direction
    control_task_id: str
    pressure_task_id: str
    route_control: str
    route_pressure: str
    patched_route_label: str
    patched_visible_pass: bool
    patched_hidden_pass: bool
    patched_failure_subtype: str | None
    patched_visible_failure_names: list[str]
    patched_hidden_failure_names: list[str]
    rescue_success: bool
    induction_success: bool
    visible_pass_hidden_fail_after_patch: bool
    patch_applied: bool
    top1_changed: bool
    patched_step_count: int
    patched_completion: str
    patched_extracted_code: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CodingRoutePatchingSummaryRow:
    """Grouped summary for generation-time coding route patching."""

    pressure_type: str
    layer: int
    position_window: str
    direction: Direction
    n_pairs: int
    rescue_success_rate: float
    induction_success_rate: float
    robust_rate_after_patch: float
    shortcut_rate_after_patch: float
    visible_pass_hidden_fail_rate_after_patch: float
    patch_applied_rate: float
    top1_changed_rate: float


def route_patching_row_to_dict(
    row: CodingRoutePatchingRow | Mapping[str, Any],
) -> dict[str, Any]:
    """Convert a generation-time coding patching row to a plain dictionary."""

    if isinstance(row, CodingRoutePatchingRow):
        return asdict(row)
    return dict(row)


def aggregate_route_patching_rows(
    rows: Iterable[CodingRoutePatchingRow | Mapping[str, Any]],
) -> list[CodingRoutePatchingSummaryRow]:
    """Aggregate generation-time coding patching rows into grouped summaries."""

    grouped_rows: dict[tuple[str, int, str, Direction], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_dict = route_patching_row_to_dict(row)
        key = (
            str(row_dict["pressure_type"]),
            int(row_dict["layer"]),
            str(row_dict["position_window"]),
            row_dict["direction"],
        )
        grouped_rows[key].append(row_dict)

    summary_rows: list[CodingRoutePatchingSummaryRow] = []
    for (pressure_type, layer, position_window, direction), group_rows in sorted(
        grouped_rows.items()
    ):
        n_pairs = len(group_rows)

        summary_rows.append(
            CodingRoutePatchingSummaryRow(
                pressure_type=pressure_type,
                layer=layer,
                position_window=position_window,
                direction=direction,
                n_pairs=n_pairs,
                rescue_success_rate=sum(
                    bool(row["rescue_success"]) for row in group_rows
                )
                / n_pairs,
                induction_success_rate=sum(
                    bool(row["induction_success"]) for row in group_rows
                )
                / n_pairs,
                robust_rate_after_patch=sum(
                    str(row["patched_route_label"]) == "robust_success"
                    for row in group_rows
                )
                / n_pairs,
                shortcut_rate_after_patch=sum(
                    str(row["patched_route_label"]) == "shortcut_success"
                    for row in group_rows
                )
                / n_pairs,
                visible_pass_hidden_fail_rate_after_patch=sum(
                    bool(row["visible_pass_hidden_fail_after_patch"])
                    for row in group_rows
                )
                / n_pairs,
                patch_applied_rate=sum(bool(row["patch_applied"]) for row in group_rows) / n_pairs,
                top1_changed_rate=sum(bool(row["top1_changed"]) for row in group_rows) / n_pairs,
            )
        )
    return summary_rows


def render_route_patching_summary_text(
    rows: Iterable[CodingRoutePatchingSummaryRow | Mapping[str, Any]],
) -> str:
    """Render grouped generation-time route patching summaries as TXT."""

    summary_rows = [
        asdict(row) if isinstance(row, CodingRoutePatchingSummaryRow) else dict(row)
        for row in rows
    ]
    lines = [
        "PressureTrace coding route patching summary",
        "",
        "Grouped by pressure_type, layer, position_window, direction.",
        "",
        "pressure_type | layer | position_window | direction | n_pairs | "
        "rescue_success_rate | induction_success_rate | robust_rate_after_patch | "
        "shortcut_rate_after_patch | visible_pass_hidden_fail_rate_after_patch | "
        "patch_applied_rate | top1_changed_rate",
    ]
    for row in summary_rows:
        lines.append(
            " | ".join(
                [
                    str(row["pressure_type"]),
                    str(row["layer"]),
                    str(row["position_window"]),
                    str(row["direction"]),
                    str(row["n_pairs"]),
                    _format_float(float(row["rescue_success_rate"])),
                    _format_float(float(row["induction_success_rate"])),
                    _format_float(float(row["robust_rate_after_patch"])),
                    _format_float(float(row["shortcut_rate_after_patch"])),
                    _format_float(float(row["visible_pass_hidden_fail_rate_after_patch"])),
                    _format_float(float(row["patch_applied_rate"])),
                    _format_float(float(row["top1_changed_rate"])),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def highlight_route_patching_rows(
    rows: Iterable[CodingRoutePatchingSummaryRow | Mapping[str, Any]],
) -> str:
    """Render concise highlights for grouped route patching summaries."""

    summary_rows = [
        (
            row
            if isinstance(row, CodingRoutePatchingSummaryRow)
            else CodingRoutePatchingSummaryRow(**dict(row))
        )
        for row in rows
    ]
    if not summary_rows:
        return "No route patching summary rows were produced.\n"

    rescue_rows = [row for row in summary_rows if row.direction == "rescue"]
    induction_rows = [row for row in summary_rows if row.direction == "induction"]
    lines = ["Highlights:"]
    if rescue_rows:
        best_rescue = max(rescue_rows, key=lambda row: row.rescue_success_rate)
        lines.append(
            "  Best rescue configuration by rescue_success_rate: "
            f"{best_rescue.pressure_type}, layer={best_rescue.layer}, "
            f"position_window={best_rescue.position_window}, "
            f"value={_format_float(best_rescue.rescue_success_rate)}"
        )
    if induction_rows:
        best_induction = max(induction_rows, key=lambda row: row.induction_success_rate)
        lines.append(
            "  Best induction configuration by induction_success_rate: "
            f"{best_induction.pressure_type}, layer={best_induction.layer}, "
            f"position_window={best_induction.position_window}, "
            f"value={_format_float(best_induction.induction_success_rate)}"
        )
    if rescue_rows or induction_rows:
        strongest_boundary_row = max(
            summary_rows,
            key=lambda row: max(row.rescue_success_rate, row.induction_success_rate),
        )
        best_success = max(
            strongest_boundary_row.rescue_success_rate,
            strongest_boundary_row.induction_success_rate,
        )
        lines.append(
            "  Comparison to the previous final-prompt-only null setup: "
            + (
                "early-generation patching is stronger."
                if best_success > 0.0
                else "no stronger signal was detected."
            )
        )
        lines.append(
            "  Prompt-to-generation boundary interpretation: "
            + (
                "signal is concentrated at the boundary / first generated token."
                if strongest_boundary_row.position_window == "gen_1"
                else "signal appears to require a broader early-generation window."
            )
        )
        lines.append(
            "  Strongest early-generation cell: "
            f"layer={strongest_boundary_row.layer}, "
            f"position_window={strongest_boundary_row.position_window}, "
            f"direction={strongest_boundary_row.direction}"
        )
    return "\n".join(lines) + "\n"


def route_patching_summary_to_csv_text(
    rows: Iterable[CodingRoutePatchingSummaryRow | Mapping[str, Any]],
) -> str:
    """Render grouped route patching summaries as CSV text."""

    header = (
        "pressure_type,layer,position_window,direction,n_pairs,rescue_success_rate,"
        "induction_success_rate,robust_rate_after_patch,shortcut_rate_after_patch,"
        "visible_pass_hidden_fail_rate_after_patch,patch_applied_rate,top1_changed_rate"
    )
    body = [
        ",".join(
            [
                str(row["pressure_type"]),
                str(row["layer"]),
                str(row["position_window"]),
                str(row["direction"]),
                str(row["n_pairs"]),
                _format_float(float(row["rescue_success_rate"])),
                _format_float(float(row["induction_success_rate"])),
                _format_float(float(row["robust_rate_after_patch"])),
                _format_float(float(row["shortcut_rate_after_patch"])),
                _format_float(float(row["visible_pass_hidden_fail_rate_after_patch"])),
                _format_float(float(row["patch_applied_rate"])),
                _format_float(float(row["top1_changed_rate"])),
            ]
        )
        for row in (
            asdict(item) if isinstance(item, CodingRoutePatchingSummaryRow) else dict(item)
            for item in rows
        )
    ]
    return "\n".join([header, *body]) + "\n"


def write_route_patching_summary_csv(
    rows: Iterable[CodingRoutePatchingSummaryRow | Mapping[str, Any]],
    output_path: Path,
) -> Path:
    """Write grouped route patching summary rows to CSV."""

    ensure_directory(output_path.parent)
    output_path.write_text(route_patching_summary_to_csv_text(rows), encoding="utf-8")
    return output_path


def plot_route_patching_success_by_layer(
    rows: Iterable[CodingRoutePatchingSummaryRow | Mapping[str, Any]],
    *,
    direction: Direction,
    output_path: Path,
) -> Path:
    """Plot rescue or induction success by layer for each position window."""

    from matplotlib import pyplot as plt

    summary_rows = [
        (
            row
            if isinstance(row, CodingRoutePatchingSummaryRow)
            else CodingRoutePatchingSummaryRow(**dict(row))
        )
        for row in rows
    ]
    filtered_rows = [row for row in summary_rows if row.direction == direction]
    ensure_directory(output_path.parent)
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    metric_name = "rescue_success_rate" if direction == "rescue" else "induction_success_rate"
    for position_window in EARLY_GENERATION_POSITION_WINDOWS:
        window_rows = sorted(
            (row for row in filtered_rows if row.position_window == position_window),
            key=lambda row: row.layer,
        )
        if not window_rows:
            continue
        axis.plot(
            [row.layer for row in window_rows],
            [getattr(row, metric_name) for row in window_rows],
            marker="o",
            label=position_window,
        )
    axis.set_xlabel("Layer")
    axis.set_ylabel(metric_name)
    axis.set_title(f"Coding route patching {direction} success by layer")
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_route_patching_position_window_comparison(
    rows: Iterable[CodingRoutePatchingSummaryRow | Mapping[str, Any]],
    *,
    output_path: Path,
) -> Path:
    """Plot the best success rate attained by each position window."""

    from matplotlib import pyplot as plt

    summary_rows = [
        (
            row
            if isinstance(row, CodingRoutePatchingSummaryRow)
            else CodingRoutePatchingSummaryRow(**dict(row))
        )
        for row in rows
    ]
    ensure_directory(output_path.parent)
    best_by_window = []
    labels: list[str] = []
    for position_window in EARLY_GENERATION_POSITION_WINDOWS:
        matching_rows = [row for row in summary_rows if row.position_window == position_window]
        if not matching_rows:
            continue
        best_value = max(
            max(row.rescue_success_rate, row.induction_success_rate) for row in matching_rows
        )
        labels.append(position_window)
        best_by_window.append(best_value)

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar(labels, best_by_window)
    axis.set_ylabel("Best success rate")
    axis.set_title("Coding route patching position-window comparison")
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
