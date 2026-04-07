from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressuretrace.behavior.run_reasoning_control_only import run_reasoning_control_only


class RunReasoningControlOnlyTestCase(unittest.TestCase):
    def test_control_only_runner_delegates_to_manifest_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "manifest.jsonl"
            manifest_path.write_text("", encoding="utf-8")

            with patch(
                "pressuretrace.behavior.run_reasoning_control_only.run_reasoning_manifest_v2"
            ) as mock_run:
                run_reasoning_control_only(
                    manifest_path=manifest_path,
                    model_name="Qwen/Qwen3-14B",
                    thinking_mode="off",
                )

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["manifest_path"], manifest_path)
        self.assertEqual(kwargs["pressure_type"], "control")
        self.assertEqual(kwargs["thinking_mode"], "off")
        self.assertTrue(str(kwargs["output_path"]).endswith("qwen-qwen3-14b_off.jsonl"))


if __name__ == "__main__":
    unittest.main()
