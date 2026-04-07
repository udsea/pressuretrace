"""Tests for reasoning benchmark runner helpers."""

from pressuretrace.behavior.run_reasoning_benchmark import (
    _generation_profile_for_model,
    _is_qwen3_model,
    _strip_qwen3_thinking_content,
)


def test_is_qwen3_model_detects_qwen3_family() -> None:
    """Qwen3 models should use the dedicated thinking-aware backend."""

    assert _is_qwen3_model("Qwen/Qwen3-32B")
    assert _is_qwen3_model("Qwen/Qwen3-8B")
    assert not _is_qwen3_model("Qwen/Qwen2.5-14B-Instruct")


def test_generation_profile_for_qwen3_uses_sampling() -> None:
    """Qwen3 should not run through the greedy generic path."""

    profile = _generation_profile_for_model("Qwen/Qwen3-14B")

    assert profile.backend == "manual_qwen3"
    assert profile.do_sample is True
    assert profile.enable_thinking is True
    assert profile.temperature == 0.6
    assert profile.top_p == 0.95
    assert profile.top_k == 20


def test_strip_qwen3_thinking_content_keeps_final_answer() -> None:
    """Only the visible post-think answer should remain."""

    response = "<think>\nTry 7 + 5 carefully.\n</think>\n12"

    assert _strip_qwen3_thinking_content(response) == "12"
