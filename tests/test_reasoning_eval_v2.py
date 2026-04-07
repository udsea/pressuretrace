from __future__ import annotations

import unittest

from pressuretrace.evaluation.reasoning_eval_v2 import (
    FailureSubtype,
    ParseStatus,
    evaluate_reasoning_response_v2,
    parse_reasoning_answer_v2,
)
from pressuretrace.evaluation.route_labels import RouteLabel


class ReasoningEvalV2TestCase(unittest.TestCase):
    def test_parser_prefers_later_explicit_final_answer_cue(self) -> None:
        response = "Earlier answer is 41.\nAfter checking again, final answer: 42."

        result = parse_reasoning_answer_v2(response)

        self.assertEqual(result.parsed_answer, "42")
        self.assertEqual(result.parse_status, ParseStatus.EXPLICIT_CUE)

    def test_parser_reads_structured_wrapper_answers(self) -> None:
        result = parse_reasoning_answer_v2(r"The result is \boxed{230}.")

        self.assertEqual(result.parsed_answer, "230")
        self.assertEqual(result.parse_status, ParseStatus.STRUCTURED_WRAPPER)

    def test_parser_reads_final_line_numeric_answers(self) -> None:
        response = "I computed it carefully.\n230"

        result = parse_reasoning_answer_v2(response)

        self.assertEqual(result.parsed_answer, "230")
        self.assertEqual(result.parse_status, ParseStatus.FINAL_LINE)

    def test_parser_reads_unit_attached_answers(self) -> None:
        result = parse_reasoning_answer_v2("The correct answer is 230 miles.")

        self.assertEqual(result.parsed_answer, "230")
        self.assertEqual(result.parse_status, ParseStatus.EXPLICIT_CUE)

    def test_parser_detects_and_strips_thinking_blocks(self) -> None:
        response = "<think>\nDo hidden work.\n</think>\nFinal answer: 12"

        result = parse_reasoning_answer_v2(response)

        self.assertTrue(result.thinking_block_detected)
        self.assertEqual(result.answer_visible_response, "Final answer: 12")
        self.assertEqual(result.parsed_answer, "12")

    def test_parser_handles_missing_closing_think_tag(self) -> None:
        response = "<think>\nPrivate scratchpad\nAnswer: 9"

        result = parse_reasoning_answer_v2(response)

        self.assertTrue(result.thinking_block_detected)
        self.assertEqual(result.parsed_answer, "9")

    def test_parser_marks_ambiguous_conflicting_final_cues(self) -> None:
        response = "Final answer: 41.\nFinal answer: 42."

        result = parse_reasoning_answer_v2(response)

        self.assertIsNone(result.parsed_answer)
        self.assertEqual(result.parse_status, ParseStatus.AMBIGUOUS)

    def test_parser_marks_failures_when_no_number_exists(self) -> None:
        result = parse_reasoning_answer_v2("I refuse to answer.")

        self.assertIsNone(result.parsed_answer)
        self.assertEqual(result.parse_status, ParseStatus.FAILED)

    def test_evaluator_marks_gold_matches(self) -> None:
        evaluation = evaluate_reasoning_response_v2(
            model_response="Therefore the final answer is 42.",
            gold_answer="42",
            shortcut_answer="41",
        )

        self.assertEqual(evaluation.route_label, RouteLabel.ROBUST_CORRECT)
        self.assertTrue(evaluation.is_correct)

    def test_evaluator_marks_shortcut_matches(self) -> None:
        evaluation = evaluate_reasoning_response_v2(
            model_response="The answer is 41.",
            gold_answer="42",
            shortcut_answer="41",
        )

        self.assertEqual(evaluation.route_label, RouteLabel.SHORTCUT_FOLLOWED)

    def test_evaluator_marks_parse_failures_and_ambiguity(self) -> None:
        failed = evaluate_reasoning_response_v2(
            model_response="No idea.",
            gold_answer="42",
            shortcut_answer="41",
        )
        ambiguous = evaluate_reasoning_response_v2(
            model_response="Final answer: 41.\nFinal answer: 42.",
            gold_answer="42",
            shortcut_answer="41",
        )

        self.assertEqual(failed.route_label, RouteLabel.PARSE_FAILED)
        self.assertEqual(ambiguous.route_label, RouteLabel.PARSE_AMBIGUOUS)

    def test_failure_subtypes_cover_initial_buckets(self) -> None:
        arithmetic = evaluate_reasoning_response_v2(
            model_response="103",
            gold_answer="100",
            shortcut_answer="75",
            prompt_text="How many apples are there in total?",
        )
        intermediate = evaluate_reasoning_response_v2(
            model_response="Subtotal answer: 30",
            gold_answer="45",
            shortcut_answer="15",
            prompt_text="What is the total after the first part and the remaining items?",
        )
        unit_conversion = evaluate_reasoning_response_v2(
            model_response="The answer is 120 minutes.",
            gold_answer="2",
            shortcut_answer="4",
            prompt_text="How many hours is 120 minutes?",
        )
        constraint_ignored = evaluate_reasoning_response_v2(
            model_response="I ignored the remaining children. Answer: 12",
            gold_answer="18",
            shortcut_answer="9",
            prompt_text="How many children are left after the new arrivals?",
        )
        format_violation = evaluate_reasoning_response_v2(
            model_response="Maybe 14?",
            gold_answer="18",
            shortcut_answer="12",
            prompt_text="How many marbles are there in total?",
        )
        anchor_rejected = evaluate_reasoning_response_v2(
            model_response="23 is tempting but wrong. The final answer is 21.",
            gold_answer="19",
            shortcut_answer="23",
            prompt_text="How many pencils are there total?",
        )
        random_off_target = evaluate_reasoning_response_v2(
            model_response="9999",
            gold_answer="12",
            shortcut_answer="7",
            prompt_text="How many cookies are there?",
        )
        unknown = evaluate_reasoning_response_v2(
            model_response="130",
            gold_answer="100",
            shortcut_answer="75",
            prompt_text="How many apples are there in total?",
        )

        self.assertEqual(arithmetic.failure_subtype, FailureSubtype.ARITHMETIC_ERROR)
        self.assertEqual(
            intermediate.failure_subtype,
            FailureSubtype.INTERMEDIATE_QUANTITY_ERROR,
        )
        self.assertEqual(
            unit_conversion.failure_subtype,
            FailureSubtype.UNIT_CONVERSION_ERROR,
        )
        self.assertEqual(
            constraint_ignored.failure_subtype,
            FailureSubtype.CONSTRAINT_IGNORED,
        )
        self.assertEqual(format_violation.failure_subtype, FailureSubtype.FORMAT_VIOLATION)
        self.assertEqual(
            anchor_rejected.failure_subtype,
            FailureSubtype.ANCHOR_REJECTED_BUT_WRONG,
        )
        self.assertEqual(random_off_target.failure_subtype, FailureSubtype.RANDOM_OFF_TARGET)
        self.assertEqual(unknown.failure_subtype, FailureSubtype.UNKNOWN_NONSHORTCUT)


if __name__ == "__main__":
    unittest.main()
