from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.behavior.materialize_reasoning_slice import materialize_reasoning_slice
from pressuretrace.utils.io import read_jsonl, write_jsonl


class MaterializeReasoningSliceTestCase(unittest.TestCase):
    def test_materialize_reasoning_slice_filters_manifest_by_base_task_id(self) -> None:
        manifest_rows = [
            {
                "task_id": "gsm8k_reasoning_v2_000001_control",
                "pressure_type": "control",
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000001"},
            },
            {
                "task_id": "gsm8k_reasoning_v2_000001_teacher_anchor",
                "pressure_type": "teacher_anchor",
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000001"},
            },
            {
                "task_id": "gsm8k_reasoning_v2_000002_control",
                "pressure_type": "control",
                "metadata": {"base_task_id": "gsm8k_reasoning_v2_000002"},
            },
        ]
        slice_rows = [
            {
                "base_task_id": "gsm8k_reasoning_v2_000001",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "robust_correct",
            }
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "manifest.jsonl"
            slice_path = Path(tempdir) / "slice.jsonl"
            output_path = Path(tempdir) / "paper_slice.jsonl"
            write_jsonl(manifest_path, manifest_rows)
            write_jsonl(slice_path, slice_rows)

            materialized_path = materialize_reasoning_slice(
                manifest_path=manifest_path,
                slice_path=slice_path,
                output_path=output_path,
            )
            filtered_rows = read_jsonl(materialized_path)

        self.assertEqual(len(filtered_rows), 2)
        self.assertEqual(
            {row["task_id"] for row in filtered_rows},
            {
                "gsm8k_reasoning_v2_000001_control",
                "gsm8k_reasoning_v2_000001_teacher_anchor",
            },
        )


if __name__ == "__main__":
    unittest.main()
