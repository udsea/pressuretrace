"""Tests for reasoning runtime pipeline-output normalization."""

from __future__ import annotations

import unittest

from pressuretrace.behavior.reasoning_runtime import (
    _extract_pipeline_generated,
    _extract_pipeline_response,
)


class ReasoningRuntimeTestCase(unittest.TestCase):
    def test_extract_pipeline_response_handles_chat_message_list(self) -> None:
        generated = [
            {"role": "system", "content": "ignore"},
            {"role": "assistant", "content": "42"},
        ]

        self.assertEqual(_extract_pipeline_response(generated), "42")

    def test_extract_pipeline_response_handles_nested_generated_text_dict(self) -> None:
        generated = {"generated_text": [{"role": "assistant", "content": "17"}]}

        self.assertEqual(_extract_pipeline_response(generated), "17")

    def test_extract_pipeline_generated_handles_batched_pipeline_shape(self) -> None:
        output = [[{"generated_text": [{"role": "assistant", "content": "9"}]}]]

        self.assertEqual(
            _extract_pipeline_response(_extract_pipeline_generated(output)),
            "9",
        )


if __name__ == "__main__":
    unittest.main()
