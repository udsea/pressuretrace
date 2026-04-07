from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.probes.build_reasoning_probe_dataset import build_reasoning_probe_dataset
from pressuretrace.utils.io import read_jsonl, write_jsonl


class BuildReasoningProbeDatasetTestCase(unittest.TestCase):
    def test_build_reasoning_probe_dataset_normalizes_rows(self) -> None:
        rows = [
            {
                "task_id": "task_1",
                "base_task_id": "base_1",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_followed",
                "binary_label": 1,
                "layer": -1,
                "representation": "last_token",
                "hidden_state": [0.1, 0.2, 0.3],
                "prompt": "Prompt text 42",
            }
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "hidden.jsonl"
            output_path = Path(tempdir) / "dataset.jsonl"
            write_jsonl(input_path, rows)

            dataset_path = build_reasoning_probe_dataset(
                input_path=input_path,
                output_path=output_path,
            )
            normalized_rows = read_jsonl(dataset_path)

        self.assertEqual(len(normalized_rows), 1)
        self.assertEqual(normalized_rows[0]["binary_label"], 1)
        self.assertEqual(normalized_rows[0]["hidden_state"], [0.1, 0.2, 0.3])
        self.assertEqual(
            set(normalized_rows[0]),
            {
                "task_id",
                "base_task_id",
                "pressure_type",
                "route_label",
                "binary_label",
                "layer",
                "representation",
                "hidden_state",
                "prompt",
            },
        )

    def test_build_reasoning_probe_dataset_rejects_bad_hidden_state(self) -> None:
        rows = [
            {
                "task_id": "task_1",
                "base_task_id": "base_1",
                "pressure_type": "teacher_anchor",
                "route_label": "shortcut_followed",
                "binary_label": 1,
                "layer": -1,
                "representation": "last_token",
                "hidden_state": ["not-a-number"],
                "prompt": "Prompt text",
            }
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            input_path = Path(tempdir) / "hidden.jsonl"
            output_path = Path(tempdir) / "dataset.jsonl"
            write_jsonl(input_path, rows)

            with self.assertRaises(ValueError):
                build_reasoning_probe_dataset(
                    input_path=input_path,
                    output_path=output_path,
                )


if __name__ == "__main__":
    unittest.main()
