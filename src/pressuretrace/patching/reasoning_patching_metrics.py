"""Metric and reporting helpers for reasoning route patching."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from pressuretrace.utils.io import ensure_directory

Direction: TypeAlias = Literal["rescue", "induction"]
FIRST_PASS_LAYERS: tuple[int, ...] = (-10, -8, -6)
FIRST_PASS_PRESSURE_TYPES: tuple[str, ...] = (
    "neutral_wrong_answer_cue",
    "teacher_anchor",
)
FIRST_PASS_DIRECTIONS: tuple[Direction, ...] = ("rescue", "induction")


class SupportsTokenToString(Protocol):
    """Tokenizer surface used by the metrics helpers."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token ids."""

    def convert_ids_to_tokens(self, token_ids: int | Sequence[int]) -> str | list[str]:
        """Convert token ids to token strings."""

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        """Decode token ids to a string."""


@dataclass(frozen=True)
class AnswerTokenPair:
    """Single-token answer ids and token strings for a matched pair."""

    gold_token_id: int
    gold_token_str: str
    shortcut_token_id: int
    shortcut_token_str: str


@dataclass(frozen=True)
class TokenSnapshot:
    """Next-token metrics for a single prompt state."""

    gold_logit: float
    shortcut_logit: float
    gold_prob: float
    shortcut_prob: float
    gold_minus_shortcut_margin: float
    top1_token_id: int
    top1_token_str: str


@dataclass(frozen=True)
class PatchComparisonRow:
    """Flat metrics row for a single control/pressure/patched comparison."""

    base_task_id: str
    pressure_type: str
    layer: int
    direction: Direction
    control_task_id: str
    pressure_task_id: str
    gold_answer: str
    shortcut_answer: str
    gold_token_id: int
    gold_token_str: str
    shortcut_token_id: int
    shortcut_token_str: str
    baseline_kind: Literal["control", "pressure"]
    control_gold_logit: float
    control_shortcut_logit: float
    control_gold_prob: float
    control_shortcut_prob: float
    control_gold_minus_shortcut_margin: float
    control_top1_token_id: int
    control_top1_token_str: str
    pressure_gold_logit: float
    pressure_shortcut_logit: float
    pressure_gold_prob: float
    pressure_shortcut_prob: float
    pressure_gold_minus_shortcut_margin: float
    pressure_top1_token_id: int
    pressure_top1_token_str: str
    patched_gold_logit: float
    patched_shortcut_logit: float
    patched_gold_prob: float
    patched_shortcut_prob: float
    patched_gold_minus_shortcut_margin: float
    patched_top1_token_id: int
    patched_top1_token_str: str
    delta_gold_logit: float
    delta_shortcut_logit: float
    delta_gold_prob: float
    delta_shortcut_prob: float
    delta_margin: float
    top1_changed: bool
    delta_gold_logit_vs_control_baseline: float | None
    delta_gold_prob_vs_control_baseline: float | None
    delta_shortcut_logit_vs_control_baseline: float | None
    delta_shortcut_prob_vs_control_baseline: float | None
    delta_margin_vs_control_baseline: float | None
    top1_changed_vs_control_baseline: bool | None
    delta_gold_logit_vs_pressure_baseline: float | None
    delta_gold_prob_vs_pressure_baseline: float | None
    delta_shortcut_logit_vs_pressure_baseline: float | None
    delta_shortcut_prob_vs_pressure_baseline: float | None
    delta_margin_vs_pressure_baseline: float | None
    top1_changed_vs_pressure_baseline: bool | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PatchSummaryRow:
    """Aggregated patching summary for one pressure/layer/direction group."""

    pressure_type: str
    layer: int
    direction: Direction
    n_pairs: int
    mean_delta_gold_prob: float
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


def _single_token_id(tokenizer: SupportsTokenToString, text: str) -> int | None:
    """Return the token id when a text maps to exactly one token."""

    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) != 1:
        return None
    return int(encoded[0])


def build_answer_token_pair(
    tokenizer: SupportsTokenToString,
    *,
    gold_answer: str,
    shortcut_answer: str,
) -> AnswerTokenPair | None:
    """Build single-token ids and strings for the gold and shortcut answers."""

    gold_token_id = _single_token_id(tokenizer, gold_answer)
    shortcut_token_id = _single_token_id(tokenizer, shortcut_answer)
    if gold_token_id is None or shortcut_token_id is None:
        return None
    return AnswerTokenPair(
        gold_token_id=gold_token_id,
        gold_token_str=_token_to_string(tokenizer, gold_token_id),
        shortcut_token_id=shortcut_token_id,
        shortcut_token_str=_token_to_string(tokenizer, shortcut_token_id),
    )


def compute_token_snapshot(
    logits: Sequence[float] | torch.Tensor,
    *,
    gold_token_id: int,
    shortcut_token_id: int,
    tokenizer: SupportsTokenToString | None = None,
) -> TokenSnapshot:
    """Compute next-token logit and probability metrics for one prompt state."""

    tensor = _as_1d_tensor(logits)
    probabilities = F.softmax(tensor, dim=-1)
    top1_token_id = int(torch.argmax(tensor).item())
    return TokenSnapshot(
        gold_logit=float(tensor[gold_token_id].item()),
        shortcut_logit=float(tensor[shortcut_token_id].item()),
        gold_prob=float(probabilities[gold_token_id].item()),
        shortcut_prob=float(probabilities[shortcut_token_id].item()),
        gold_minus_shortcut_margin=float(
            tensor[gold_token_id].item() - tensor[shortcut_token_id].item()
        ),
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
    gold_answer: str,
    shortcut_answer: str,
    answer_tokens: AnswerTokenPair,
    control_snapshot: TokenSnapshot,
    pressure_snapshot: TokenSnapshot,
    patched_snapshot: TokenSnapshot,
    metadata: Mapping[str, Any],
) -> PatchComparisonRow:
    """Build a flat row for a single patched run."""

    baseline_kind = baseline_kind_for_direction(direction)
    baseline_snapshot = pressure_snapshot if baseline_kind == "pressure" else control_snapshot
    delta_gold_logit = patched_snapshot.gold_logit - baseline_snapshot.gold_logit
    delta_shortcut_logit = patched_snapshot.shortcut_logit - baseline_snapshot.shortcut_logit
    delta_gold_prob = patched_snapshot.gold_prob - baseline_snapshot.gold_prob
    delta_shortcut_prob = patched_snapshot.shortcut_prob - baseline_snapshot.shortcut_prob
    delta_margin = (
        patched_snapshot.gold_minus_shortcut_margin
        - baseline_snapshot.gold_minus_shortcut_margin
    )
    top1_changed = patched_snapshot.top1_token_id != baseline_snapshot.top1_token_id

    return PatchComparisonRow(
        base_task_id=base_task_id,
        pressure_type=pressure_type,
        layer=layer,
        direction=direction,
        control_task_id=control_task_id,
        pressure_task_id=pressure_task_id,
        gold_answer=gold_answer,
        shortcut_answer=shortcut_answer,
        gold_token_id=answer_tokens.gold_token_id,
        gold_token_str=answer_tokens.gold_token_str,
        shortcut_token_id=answer_tokens.shortcut_token_id,
        shortcut_token_str=answer_tokens.shortcut_token_str,
        baseline_kind=baseline_kind,
        control_gold_logit=control_snapshot.gold_logit,
        control_shortcut_logit=control_snapshot.shortcut_logit,
        control_gold_prob=control_snapshot.gold_prob,
        control_shortcut_prob=control_snapshot.shortcut_prob,
        control_gold_minus_shortcut_margin=control_snapshot.gold_minus_shortcut_margin,
        control_top1_token_id=control_snapshot.top1_token_id,
        control_top1_token_str=control_snapshot.top1_token_str,
        pressure_gold_logit=pressure_snapshot.gold_logit,
        pressure_shortcut_logit=pressure_snapshot.shortcut_logit,
        pressure_gold_prob=pressure_snapshot.gold_prob,
        pressure_shortcut_prob=pressure_snapshot.shortcut_prob,
        pressure_gold_minus_shortcut_margin=pressure_snapshot.gold_minus_shortcut_margin,
        pressure_top1_token_id=pressure_snapshot.top1_token_id,
        pressure_top1_token_str=pressure_snapshot.top1_token_str,
        patched_gold_logit=patched_snapshot.gold_logit,
        patched_shortcut_logit=patched_snapshot.shortcut_logit,
        patched_gold_prob=patched_snapshot.gold_prob,
        patched_shortcut_prob=patched_snapshot.shortcut_prob,
        patched_gold_minus_shortcut_margin=patched_snapshot.gold_minus_shortcut_margin,
        patched_top1_token_id=patched_snapshot.top1_token_id,
        patched_top1_token_str=patched_snapshot.top1_token_str,
        delta_gold_logit=delta_gold_logit,
        delta_shortcut_logit=delta_shortcut_logit,
        delta_gold_prob=delta_gold_prob,
        delta_shortcut_prob=delta_shortcut_prob,
        delta_margin=delta_margin,
        top1_changed=top1_changed,
        delta_gold_logit_vs_control_baseline=(
            delta_gold_logit if baseline_kind == "control" else None
        ),
        delta_gold_prob_vs_control_baseline=(
            delta_gold_prob if baseline_kind == "control" else None
        ),
        delta_shortcut_logit_vs_control_baseline=(
            delta_shortcut_logit if baseline_kind == "control" else None
        ),
        delta_shortcut_prob_vs_control_baseline=(
            delta_shortcut_prob if baseline_kind == "control" else None
        ),
        delta_margin_vs_control_baseline=delta_margin if baseline_kind == "control" else None,
        top1_changed_vs_control_baseline=top1_changed if baseline_kind == "control" else None,
        delta_gold_logit_vs_pressure_baseline=(
            delta_gold_logit if baseline_kind == "pressure" else None
        ),
        delta_gold_prob_vs_pressure_baseline=(
            delta_gold_prob if baseline_kind == "pressure" else None
        ),
        delta_shortcut_logit_vs_pressure_baseline=(
            delta_shortcut_logit if baseline_kind == "pressure" else None
        ),
        delta_shortcut_prob_vs_pressure_baseline=(
            delta_shortcut_prob if baseline_kind == "pressure" else None
        ),
        delta_margin_vs_pressure_baseline=delta_margin if baseline_kind == "pressure" else None,
        top1_changed_vs_pressure_baseline=top1_changed if baseline_kind == "pressure" else None,
        metadata=dict(metadata),
    )


def patch_row_to_dict(row: PatchComparisonRow | Mapping[str, Any]) -> dict[str, Any]:
    """Convert a patch comparison row to a plain dictionary."""

    if isinstance(row, PatchComparisonRow):
        return asdict(row)
    return dict(row)


def aggregate_patch_rows(
    rows: Iterable[PatchComparisonRow | Mapping[str, Any]],
) -> list[PatchSummaryRow]:
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

    summary_rows: list[PatchSummaryRow] = []
    for (pressure_type, layer, direction), group_rows in sorted(grouped_rows.items()):
        n_pairs = len(group_rows)
        mean_delta_gold_prob = sum(float(row["delta_gold_prob"]) for row in group_rows) / n_pairs
        mean_delta_shortcut_prob = (
            sum(float(row["delta_shortcut_prob"]) for row in group_rows) / n_pairs
        )
        mean_delta_margin = sum(float(row["delta_margin"]) for row in group_rows) / n_pairs
        top1_changed_rate = sum(bool(row["top1_changed"]) for row in group_rows) / n_pairs
        summary_rows.append(
            PatchSummaryRow(
                pressure_type=pressure_type,
                layer=layer,
                direction=direction,
                n_pairs=n_pairs,
                mean_delta_gold_prob=mean_delta_gold_prob,
                mean_delta_shortcut_prob=mean_delta_shortcut_prob,
                mean_delta_margin=mean_delta_margin,
                top1_changed_rate=top1_changed_rate,
            )
        )
    return summary_rows


def summary_rows_to_csv_rows(
    rows: Iterable[PatchSummaryRow | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert summary rows to compact CSV-ready dictionaries."""

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = asdict(row) if isinstance(row, PatchSummaryRow) else dict(row)
        csv_rows.append(
            {
                "pressure_type": row_dict["pressure_type"],
                "layer": row_dict["layer"],
                "direction": row_dict["direction"],
                "n_pairs": row_dict["n_pairs"],
                "mean_delta_gold_prob": row_dict["mean_delta_gold_prob"],
                "mean_delta_shortcut_prob": row_dict["mean_delta_shortcut_prob"],
                "mean_delta_margin": row_dict["mean_delta_margin"],
                "top1_changed_rate": row_dict["top1_changed_rate"],
            }
        )
    return csv_rows


def _format_float(value: float, digits: int = 4) -> str:
    """Format a floating-point summary value."""

    return f"{value:.{digits}f}"


def render_summary_text(rows: Iterable[PatchSummaryRow | Mapping[str, Any]]) -> str:
    """Render a compact TXT summary for grouped patch metrics."""

    summary_rows = [asdict(row) if isinstance(row, PatchSummaryRow) else dict(row) for row in rows]
    lines = [
        "PressureTrace reasoning route patching summary",
        "",
        "Grouped by pressure_type, layer, direction.",
        "",
        "pressure_type | layer | direction | n_pairs | mean_delta_gold_prob | "
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
                    _format_float(float(row["mean_delta_gold_prob"])),
                    _format_float(float(row["mean_delta_shortcut_prob"])),
                    _format_float(float(row["mean_delta_margin"])),
                    _format_float(float(row["top1_changed_rate"])),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def select_best_layer(
    rows: Iterable[PatchSummaryRow | Mapping[str, Any]],
    *,
    metric_name: str,
    direction: Direction,
    pressure_type: str,
) -> PatchSummaryRow | None:
    """Pick the best layer for a pressure type and direction by a named mean metric."""

    candidates: list[PatchSummaryRow] = []
    for row in rows:
        row_obj = row if isinstance(row, PatchSummaryRow) else PatchSummaryRow(**dict(row))
        if row_obj.pressure_type == pressure_type and row_obj.direction == direction:
            candidates.append(row_obj)
    if not candidates:
        return None
    return max(candidates, key=lambda item: getattr(item, metric_name))


def patch_summary_to_csv_text(rows: Iterable[PatchSummaryRow | Mapping[str, Any]]) -> str:
    """Render a compact CSV summary as text."""

    header = (
        "pressure_type,layer,direction,n_pairs,mean_delta_gold_prob,"
        "mean_delta_shortcut_prob,mean_delta_margin,top1_changed_rate"
    )
    body = [
        ",".join(
            [
                str(row["pressure_type"]),
                str(row["layer"]),
                str(row["direction"]),
                str(row["n_pairs"]),
                _format_float(float(row["mean_delta_gold_prob"])),
                _format_float(float(row["mean_delta_shortcut_prob"])),
                _format_float(float(row["mean_delta_margin"])),
                _format_float(float(row["top1_changed_rate"])),
            ]
        )
        for row in (
            asdict(item) if isinstance(item, PatchSummaryRow) else dict(item) for item in rows
        )
    ]
    return "\n".join([header, *body]) + "\n"


def highlight_summary_rows(rows: Iterable[PatchSummaryRow | Mapping[str, Any]]) -> str:
    """Render concise highlights for the grouped summary."""

    summary_rows = [
        row if isinstance(row, PatchSummaryRow) else PatchSummaryRow(**dict(row))
        for row in rows
    ]
    if not summary_rows:
        return "No summary rows were produced.\n"

    def _best(direction: Direction, metric_name: str) -> PatchSummaryRow:
        candidates = [row for row in summary_rows if row.direction == direction]
        return max(candidates, key=lambda row: getattr(row, metric_name))

    best_rescue_gold = _best("rescue", "mean_delta_gold_prob")
    best_rescue_margin = _best("rescue", "mean_delta_margin")
    strongest_induction = _best("induction", "mean_delta_shortcut_prob")

    lines = [
        "Highlights:",
        (
            "  Best rescue layer by mean delta_gold_prob: "
            f"{best_rescue_gold.pressure_type}, layer={best_rescue_gold.layer}, "
            f"value={_format_float(best_rescue_gold.mean_delta_gold_prob)}"
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
        "",
        "Per-pressure-type trends:",
    ]
    for pressure_type in sorted({row.pressure_type for row in summary_rows}):
        rescue_rows = [
            row
            for row in summary_rows
            if row.pressure_type == pressure_type and row.direction == "rescue"
        ]
        induction_rows = [
            row
            for row in summary_rows
            if row.pressure_type == pressure_type and row.direction == "induction"
        ]
        if rescue_rows:
            best_rescue = max(rescue_rows, key=lambda row: row.mean_delta_margin)
            lines.append(
                "  "
                + f"{pressure_type}: rescue best margin at layer={best_rescue.layer} "
                + f"({_format_float(best_rescue.mean_delta_margin)})"
            )
        if induction_rows:
            best_induction = max(
                induction_rows,
                key=lambda row: row.mean_delta_shortcut_prob,
            )
            lines.append(
                "  "
                + (
                    f"{pressure_type}: induction strongest shortcut increase at "
                    f"layer={best_induction.layer} "
                )
                + f"({_format_float(best_induction.mean_delta_shortcut_prob)})"
            )
    return "\n".join(lines) + "\n"


def write_summary_csv(
    rows: Iterable[PatchSummaryRow | Mapping[str, Any]],
    output_path: Path,
) -> Path:
    """Write grouped patch summary rows to CSV."""

    ensure_directory(output_path.parent)
    output_path.write_text(patch_summary_to_csv_text(rows), encoding="utf-8")
    return output_path


def write_summary_text(
    rows: Iterable[PatchSummaryRow | Mapping[str, Any]],
    output_path: Path,
) -> Path:
    """Write the human-readable patch summary text."""

    ensure_directory(output_path.parent)
    summary_rows = [
        row if isinstance(row, PatchSummaryRow) else PatchSummaryRow(**dict(row))
        for row in rows
    ]
    output_path.write_text(
        render_summary_text(summary_rows) + "\n" + highlight_summary_rows(summary_rows),
        encoding="utf-8",
    )
    return output_path


def plot_patch_summary(
    rows: Iterable[PatchSummaryRow | Mapping[str, Any]],
    *,
    direction: Direction,
    metric_name: Literal["mean_delta_gold_prob", "mean_delta_margin", "mean_delta_shortcut_prob"],
    output_path: Path,
) -> Path:
    """Plot one grouped patch-summary metric across layers by pressure type."""

    summary_rows = [
        row if isinstance(row, PatchSummaryRow) else PatchSummaryRow(**dict(row))
        for row in rows
    ]
    filtered_rows = [row for row in summary_rows if row.direction == direction]
    ensure_directory(output_path.parent)
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for pressure_type in sorted({row.pressure_type for row in filtered_rows}):
        pressure_rows = sorted(
            [row for row in filtered_rows if row.pressure_type == pressure_type],
            key=lambda row: row.layer,
        )
        axis.plot(
            [row.layer for row in pressure_rows],
            [getattr(row, metric_name) for row in pressure_rows],
            marker="o",
            linewidth=2,
            label=pressure_type,
        )
    axis.set_xlabel("Layer")
    axis.set_ylabel(metric_name)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
