from __future__ import annotations

import unittest

from pressuretrace.evaluation.reasoning_eval import label_reasoning_route, parse_last_integer
from pressuretrace.evaluation.route_labels import RouteLabel


class ReasoningEvalTestCase(unittest.TestCase):
    def test_parse_last_integer_handles_commas_and_signs(self) -> None:
        self.assertEqual(parse_last_integer("Intermediate values: 12, then -1,234."), -1234)

    def test_label_reasoning_route_marks_correct_answers(self) -> None:
        label = label_reasoning_route(
            model_response="After checking the arithmetic, the answer is 42.",
            reference_answer="42",
            shortcut_answer="41",
        )
        self.assertEqual(label, RouteLabel.ROBUST_CORRECT)

    def test_label_reasoning_route_marks_shortcut_following(self) -> None:
        label = label_reasoning_route(
            model_response="41",
            reference_answer="42",
            shortcut_answer="41",
        )
        self.assertEqual(label, RouteLabel.SHORTCUT_FOLLOWED)


if __name__ == "__main__":
    unittest.main()
