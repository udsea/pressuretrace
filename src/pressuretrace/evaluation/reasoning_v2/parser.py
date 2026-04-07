"""Parser implementation for PressureTrace reasoning v2 responses."""

from __future__ import annotations

import re

from pressuretrace.evaluation.reasoning_v2.types import (
    ParseCandidate,
    ParseConfidence,
    ParseStatus,
    ReasoningParseResultV2,
)

INTEGER_PATTERN = r"[-+]?\d[\d,]*(?:\.\d+)?"
INTEGER_TOKEN_PATTERN = re.compile(INTEGER_PATTERN)
BOXED_PATTERN = re.compile(r"\\boxed\{\s*(?P<number>" + INTEGER_PATTERN + r")\s*\}")
INLINE_MATH_PATTERN = re.compile(r"\$(?P<number>" + INTEGER_PATTERN + r")\$")
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"</?think>", re.IGNORECASE)
FINAL_LINE_PATTERN = re.compile(
    r"^(?:[*_`\- ]*)?"
    r"(?:(?:the )?(?:final )?answer(?: is|:)?\s*)?"
    r"(?P<number>"
    + INTEGER_PATTERN
    + r")"
    r"(?:\s+[A-Za-z%]+)?"
    r"(?:[.)!?\s*_`\-]*)$",
    re.IGNORECASE,
)
EXPLICIT_CUE_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (
        re.compile(
            r"(?:final answer|the final answer is|the correct answer is)\s*[:=]?\s*"
            r"(?P<number>"
            + INTEGER_PATTERN
            + r")"
            r"(?:\s+[A-Za-z%]+)?",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        re.compile(
            r"(?:correct answer|answer is|answer)\s*[:=]?\s*"
            r"(?P<number>"
            + INTEGER_PATTERN
            + r")"
            r"(?:\s+[A-Za-z%]+)?",
            re.IGNORECASE,
        ),
        2,
    ),
    (
        re.compile(
            r"(?:therefore|thus|so)\s*"
            r"(?:the answer is|answer is|the result is|result is)?\s*[:=]?\s*"
            r"(?P<number>"
            + INTEGER_PATTERN
            + r")"
            r"(?:\s+[A-Za-z%]+)?",
            re.IGNORECASE,
        ),
        1,
    ),
)
LATE_RESPONSE_WINDOW_CHARS = 200


def _normalize_numeric_token(token: str) -> str | None:
    """Normalize an integer-like token into a canonical string."""

    stripped = token.strip().rstrip(".,!?;:")
    stripped = stripped.replace(",", "")
    if re.fullmatch(r"[-+]?\d+", stripped):
        return str(int(stripped))
    if re.fullmatch(r"[-+]?\d+\.\d+", stripped):
        numeric = float(stripped)
        if numeric.is_integer():
            return str(int(numeric))
    return None


def _normalize_response_text(text: str) -> str:
    """Normalize whitespace for diagnostics without changing raw output."""

    lines = [
        " ".join(line.strip().split())
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]
    return "\n".join(lines)


def _derive_answer_visible_response(text: str) -> tuple[str, bool]:
    """Strip known hidden-thinking wrappers while preserving visible answer text."""

    thinking_block_detected = "<think>" in text.lower() or "</think>" in text.lower()
    visible = THINK_BLOCK_PATTERN.sub(" ", text)
    if "</think>" in visible.lower():
        visible = re.split(r"</think>", visible, flags=re.IGNORECASE)[-1]
    visible = THINK_TAG_PATTERN.sub(" ", visible)
    visible = visible.strip()
    if visible:
        return visible, thinking_block_detected
    return text.strip(), thinking_block_detected


def _collect_explicit_candidates(text: str) -> list[ParseCandidate]:
    """Extract answer candidates from explicit final-answer cue phrases."""

    candidates: list[ParseCandidate] = []
    for pattern, priority in EXPLICIT_CUE_PATTERNS:
        for match in pattern.finditer(text):
            normalized = _normalize_numeric_token(match.group("number"))
            if normalized is None:
                continue
            start, end = match.span("number")
            candidates.append(
                ParseCandidate(
                    value=normalized,
                    start=start,
                    end=end,
                    source=ParseStatus.EXPLICIT_CUE,
                    priority=priority,
                )
            )
    return candidates


def _collect_structured_candidates(text: str) -> list[ParseCandidate]:
    """Extract answer candidates from boxed or inline-math wrappers."""

    candidates: list[ParseCandidate] = []
    for pattern in (BOXED_PATTERN, INLINE_MATH_PATTERN):
        for match in pattern.finditer(text):
            normalized = _normalize_numeric_token(match.group("number"))
            if normalized is None:
                continue
            start, end = match.span("number")
            candidates.append(
                ParseCandidate(
                    value=normalized,
                    start=start,
                    end=end,
                    source=ParseStatus.STRUCTURED_WRAPPER,
                    priority=2,
                )
            )
    return candidates


def _collect_final_line_candidates(text: str) -> list[ParseCandidate]:
    """Extract a candidate from the final non-empty line when it is answer-like."""

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    final_line = lines[-1].strip()
    match = FINAL_LINE_PATTERN.match(final_line)
    if match is None:
        return []

    normalized = _normalize_numeric_token(match.group("number"))
    if normalized is None:
        return []

    final_line_start = text.rfind(final_line)
    line_number_start = final_line_start + match.start("number")
    line_number_end = final_line_start + match.end("number")
    return [
        ParseCandidate(
            value=normalized,
            start=line_number_start,
            end=line_number_end,
            source=ParseStatus.FINAL_LINE,
            priority=1,
        )
    ]


def _collect_fallback_candidates(text: str) -> list[ParseCandidate]:
    """Extract all numeric candidates for the final low-confidence fallback."""

    candidates: list[ParseCandidate] = []
    for match in INTEGER_TOKEN_PATTERN.finditer(text):
        normalized = _normalize_numeric_token(match.group(0))
        if normalized is None:
            continue
        candidates.append(
            ParseCandidate(
                value=normalized,
                start=match.start(),
                end=match.end(),
                source=ParseStatus.FALLBACK_LAST_NUMBER,
                priority=0,
            )
        )
    return candidates


def _unique_candidate_values(candidates: list[ParseCandidate]) -> list[str]:
    """Preserve candidate values in the order they were observed."""

    values: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.value in seen:
            continue
        seen.add(candidate.value)
        values.append(candidate.value)
    return values


def _select_stage_candidate(
    candidates: list[ParseCandidate],
    text_length: int,
) -> tuple[ParseCandidate | None, bool]:
    """Pick the best candidate for a stage, flagging ambiguous late collisions."""

    if not candidates:
        return None, False

    best_by_value: dict[str, ParseCandidate] = {}
    for candidate in candidates:
        existing = best_by_value.get(candidate.value)
        if existing is None or (candidate.priority, candidate.start) > (
            existing.priority,
            existing.start,
        ):
            best_by_value[candidate.value] = candidate

    ranked = sorted(
        best_by_value.values(),
        key=lambda candidate: (candidate.priority, candidate.start),
        reverse=True,
    )
    best_candidate = ranked[0]
    late_cutoff = max(text_length - LATE_RESPONSE_WINDOW_CHARS, 0)
    for competitor in ranked[1:]:
        if (
            competitor.priority == best_candidate.priority
            and competitor.start >= late_cutoff
            and best_candidate.start >= late_cutoff
        ):
            return None, True
    return best_candidate, False


def _looks_like_numeric_final_line(text: str) -> bool:
    """Check whether the final non-empty line looks like a clean numeric answer."""

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    return FINAL_LINE_PATTERN.match(lines[-1].strip()) is not None


def parse_reasoning_answer_v2(model_response: str) -> ReasoningParseResultV2:
    """Parse a reasoning-model response into a structured v2 parse result."""

    normalized_response = _normalize_response_text(model_response)
    answer_visible_response, thinking_block_detected = _derive_answer_visible_response(
        model_response
    )
    visible_response = answer_visible_response or model_response

    explicit_candidates = _collect_explicit_candidates(visible_response)
    structured_candidates = _collect_structured_candidates(visible_response)
    final_line_candidates = _collect_final_line_candidates(visible_response)
    fallback_candidates = _collect_fallback_candidates(visible_response)
    parse_candidates = _unique_candidate_values(
        explicit_candidates
        + structured_candidates
        + final_line_candidates
        + fallback_candidates
    )

    stage_specs: tuple[tuple[list[ParseCandidate], ParseStatus, ParseConfidence], ...] = (
        (explicit_candidates, ParseStatus.EXPLICIT_CUE, ParseConfidence.HIGH),
        (structured_candidates, ParseStatus.STRUCTURED_WRAPPER, ParseConfidence.HIGH),
        (final_line_candidates, ParseStatus.FINAL_LINE, ParseConfidence.MEDIUM),
        (fallback_candidates, ParseStatus.FALLBACK_LAST_NUMBER, ParseConfidence.LOW),
    )

    for candidates, parse_status, parse_confidence in stage_specs:
        selected, is_ambiguous = _select_stage_candidate(candidates, len(visible_response))
        if is_ambiguous:
            return ReasoningParseResultV2(
                normalized_response=normalized_response,
                answer_visible_response=visible_response,
                parsed_answer=None,
                selected_candidate=None,
                parse_candidates=parse_candidates,
                parse_status=ParseStatus.AMBIGUOUS,
                parse_confidence=ParseConfidence.NONE,
                thinking_block_detected=thinking_block_detected,
                response_length_chars=len(model_response),
            )
        if selected is None:
            continue
        return ReasoningParseResultV2(
            normalized_response=normalized_response,
            answer_visible_response=visible_response,
            parsed_answer=selected.value,
            selected_candidate=selected.value,
            parse_candidates=parse_candidates,
            parse_status=parse_status,
            parse_confidence=parse_confidence,
            thinking_block_detected=thinking_block_detected,
            response_length_chars=len(model_response),
        )

    return ReasoningParseResultV2(
        normalized_response=normalized_response,
        answer_visible_response=visible_response,
        parsed_answer=None,
        selected_candidate=None,
        parse_candidates=parse_candidates,
        parse_status=ParseStatus.FAILED,
        parse_confidence=ParseConfidence.NONE,
        thinking_block_detected=thinking_block_detected,
        response_length_chars=len(model_response),
    )
