from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.behavior.debug_coding_tasks import run_coding_debug_tasks
from pressuretrace.utils.io import read_jsonl


class DebugCodingTasksTestCase(unittest.TestCase):
    def test_fixture_debug_run_shows_shortcut_signal_on_pressure_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "coding_debug_run.jsonl"
            artifacts = run_coding_debug_tasks(
                source="fixtures",
                output_path=output_path,
                require_shortcut_signal=True,
            )
            rows = read_jsonl(output_path)

        self.assertEqual(artifacts.row_count, 9)
        self.assertGreaterEqual(artifacts.shortcut_signal_count, 6)
        control_rows = [row for row in rows if row["pressure_type"] == "control"]
        pressure_rows = [row for row in rows if row["pressure_type"] != "control"]
        self.assertTrue(all(row["route_label"] == "robust_success" for row in control_rows))
        self.assertTrue(all(row["route_label"] == "shortcut_success" for row in pressure_rows))


if __name__ == "__main__":
    unittest.main()
