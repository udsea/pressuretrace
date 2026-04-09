from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.evaluation.coding_eval_debug import (
    run_coding_eval_debug,
    summarize_coding_eval_debug_report,
)
from pressuretrace.utils.io import read_jsonl


class CodingEvalDebugTestCase(unittest.TestCase):
    def test_coding_eval_debug_fixtures_all_match_expected_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "coding_eval_debug_report.jsonl"
            artifacts = run_coding_eval_debug(
                output_path=output_path,
                require_pass=True,
            )
            rows = read_jsonl(output_path)

        self.assertEqual(artifacts.total_cases, 12)
        self.assertEqual(artifacts.passing_cases, 12)
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(bool(row["matched_expected_label"]) for row in rows))

        summary = summarize_coding_eval_debug_report(rows)
        self.assertEqual(summary["shortcut_possible_task_count"], 3)
        self.assertEqual(summary["robust_and_shortcut_fixture_task_count"], 3)


if __name__ == "__main__":
    unittest.main()
