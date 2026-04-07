from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from pressuretrace.analysis.export_reasoning_probe_table import export_reasoning_probe_table


class ExportReasoningProbeTableTestCase(unittest.TestCase):
    def test_export_reasoning_probe_table_includes_top_hidden_rows_and_baselines(self) -> None:
        frozen_root = Path("pressuretrace-frozen/reasoning_v2_qwen3_14b_off")
        metrics_path = frozen_root / "results" / "reasoning_probe_metrics_qwen-qwen3-14b_off.jsonl"
        dataset_path = (
            frozen_root / "results" / "reasoning_probe_dataset_qwen-qwen3-14b_off.jsonl"
        )

        with tempfile.TemporaryDirectory() as tempdir:
            csv_path = Path(tempdir) / "table.csv"
            md_path = Path(tempdir) / "table.md"
            outputs = export_reasoning_probe_table(
                metrics_path=metrics_path,
                dataset_path=dataset_path,
                csv_path=csv_path,
                markdown_path=md_path,
            )
            csv_text = outputs["csv"].read_text(encoding="utf-8")
            md_text = outputs["markdown"].read_text(encoding="utf-8")
            with outputs["csv"].open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(csv_rows), 8)
        self.assertEqual(csv_rows[0]["method"], "hidden_state_probe")
        self.assertEqual(csv_rows[-2]["method"], "prompt_length")
        self.assertEqual(csv_rows[-1]["method"], "tfidf_prompt")
        self.assertIn("hidden_state_probe", csv_text)
        self.assertIn("prompt_length", csv_text)
        self.assertIn("tfidf_prompt", md_text)


if __name__ == "__main__":
    unittest.main()
