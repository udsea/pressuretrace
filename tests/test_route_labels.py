from __future__ import annotations

import unittest

from pressuretrace.evaluation.route_labels import (
    CODING_ROUTE_LABELS,
    REASONING_ROUTE_LABELS,
    REASONING_V2_ROUTE_LABELS,
    RouteLabel,
)


class RouteLabelsTestCase(unittest.TestCase):
    def test_route_label_values_are_unique(self) -> None:
        values = [label.value for label in RouteLabel]
        self.assertEqual(len(values), len(set(values)))

    def test_family_label_sets_include_expected_members(self) -> None:
        self.assertIn(RouteLabel.ROBUST_CORRECT, REASONING_ROUTE_LABELS)
        self.assertIn(RouteLabel.SHORTCUT_FOLLOWED, REASONING_ROUTE_LABELS)
        self.assertIn(RouteLabel.WRONG_NONSHORTCUT, REASONING_V2_ROUTE_LABELS)
        self.assertIn(RouteLabel.PARSE_FAILED, REASONING_V2_ROUTE_LABELS)
        self.assertIn(RouteLabel.SPEC_GAMING_SUSPECTED, CODING_ROUTE_LABELS)


if __name__ == "__main__":
    unittest.main()
