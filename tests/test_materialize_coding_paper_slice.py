from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.behavior.materialize_coding_paper_slice import materialize_coding_paper_slice
from pressuretrace.utils.io import read_jsonl, write_jsonl


class MaterializeCodingPaperSliceTestCase(unittest.TestCase):
    def test_materialize_coding_paper_slice_filters_by_base_task_id(self) -> None:
        manifest_rows = [
            {
                "task_id": "coding_v1_visible_test_hardcoding_001_control",
                "base_task_id": "coding_v1_visible_test_hardcoding_001",
                "pressure_type": "control",
            },
            {
                "task_id": "coding_v1_visible_test_hardcoding_001_teacher_anchor",
                "base_task_id": "coding_v1_visible_test_hardcoding_001",
                "pressure_type": "teacher_anchor",
            },
            {
                "task_id": "coding_v1_visible_test_hardcoding_002_control",
                "base_task_id": "coding_v1_visible_test_hardcoding_002",
                "pressure_type": "control",
            },
        ]
        slice_rows = [
            {
                "base_task_id": "coding_v1_visible_test_hardcoding_001",
                "model_name": "Qwen/Qwen3-14B",
                "thinking_mode": "off",
                "control_route_label": "robust_success",
            }
        ]

        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "manifest.jsonl"
            slice_path = Path(tempdir) / "slice.jsonl"
            output_path = Path(tempdir) / "paper_slice.jsonl"
            write_jsonl(manifest_path, manifest_rows)
            write_jsonl(slice_path, slice_rows)

            materialized_path = materialize_coding_paper_slice(
                manifest_path=manifest_path,
                slice_path=slice_path,
                output_path=output_path,
            )
            filtered_rows = read_jsonl(materialized_path)

        self.assertEqual(len(filtered_rows), 2)
        self.assertEqual(
            {row["task_id"] for row in filtered_rows},
            {
                "coding_v1_visible_test_hardcoding_001_control",
                "coding_v1_visible_test_hardcoding_001_teacher_anchor",
            },
        )

    def test_materialize_raises_for_empty_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            manifest_path = Path(tempdir) / "manifest.jsonl"
            slice_path = Path(tempdir) / "slice.jsonl"
            write_jsonl(manifest_path, [])
            write_jsonl(slice_path, [])

            with self.assertRaisesRegex(
                ValueError,
                "control-robust slice is empty",
            ):
                materialize_coding_paper_slice(
                    manifest_path=manifest_path,
                    slice_path=slice_path,
                )


if __name__ == "__main__":
    unittest.main()
