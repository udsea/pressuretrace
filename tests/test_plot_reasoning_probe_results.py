from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.analysis.plot_reasoning_probe_results import generate_reasoning_probe_plots
from pressuretrace.utils.io import write_jsonl


class PlotReasoningProbeResultsTestCase(unittest.TestCase):
    def test_generate_reasoning_probe_plots_writes_three_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            metrics_path = root / "metrics.jsonl"
            dataset_path = root / "dataset.jsonl"
            summary_path = root / "summary.txt"
            output_dir = root / "plots"

            metrics_rows = []
            for layer in (-10, -8):
                metrics_rows.extend(
                    [
                        {
                            "kind": "hidden_state_probe",
                            "layer": layer,
                            "representation": "last_token",
                            "roc_auc": 0.8,
                            "accuracy": 0.75,
                        },
                        {
                            "kind": "hidden_state_probe",
                            "layer": layer,
                            "representation": "mean_pool",
                            "roc_auc": 0.6,
                            "accuracy": 0.62,
                        },
                    ]
                )
            write_jsonl(metrics_path, metrics_rows)
            write_jsonl(
                dataset_path,
                [
                    {
                        "task_id": "task_1",
                        "base_task_id": "base_1",
                        "pressure_type": "teacher_anchor",
                        "binary_label": 0,
                    },
                    {
                        "task_id": "task_2",
                        "base_task_id": "base_2",
                        "pressure_type": "neutral_wrong_answer_cue",
                        "binary_label": 1,
                    },
                ],
            )
            summary_path.write_text(
                "\n".join(
                    [
                        "Baseline comparison:",
                        "  prompt_length: roc_auc=0.6413, accuracy=0.6063",
                        "  tfidf_prompt: roc_auc=0.6290, accuracy=0.5669",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            outputs = generate_reasoning_probe_plots(
                metrics_path=metrics_path,
                dataset_path=dataset_path,
                summary_path=summary_path,
                output_dir=output_dir,
            )
            self.assertEqual(len(outputs), 3)
            self.assertTrue(all(path.suffix == ".png" for path in outputs))
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
