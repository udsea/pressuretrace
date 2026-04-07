from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.analysis.reasoning_probe_reports import export_probe_reports
from pressuretrace.utils.io import write_jsonl


class ReasoningProbeReportsTestCase(unittest.TestCase):
    def test_export_probe_reports_writes_artifact_index_csv_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            frozen_root = Path(tempdir)
            (frozen_root / "data" / "manifests").mkdir(parents=True)
            (frozen_root / "data" / "splits").mkdir(parents=True)
            (frozen_root / "results").mkdir(parents=True)

            write_jsonl(
                frozen_root
                / "data"
                / "manifests"
                / "reasoning_paper_slice_qwen-qwen3-14b_off.jsonl",
                [{"task_id": "control_1"}],
            )
            write_jsonl(
                frozen_root
                / "data"
                / "splits"
                / "reasoning_control_robust_slice_qwen-qwen3-14b_off.jsonl",
                [{"base_task_id": "base_1"}],
            )
            write_jsonl(
                frozen_root / "results" / "reasoning_paper_slice_qwen-qwen3-14b_off.jsonl",
                [{"task_id": "control_1"}],
            )
            hidden_rows = [
                {"task_id": "task_1", "base_task_id": "base_1", "binary_label": 0},
                {"task_id": "task_2", "base_task_id": "base_2", "binary_label": 1},
            ]
            write_jsonl(
                frozen_root / "results" / "reasoning_probe_hidden_states_qwen-qwen3-14b_off.jsonl",
                hidden_rows,
            )
            dataset_rows = [
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
            ]
            write_jsonl(
                frozen_root / "results" / "reasoning_probe_dataset_qwen-qwen3-14b_off.jsonl",
                dataset_rows,
            )
            metrics_rows = [
                {
                    "kind": "hidden_state_probe",
                    "feature_set": "hidden_state",
                    "layer": -10,
                    "representation": "last_token",
                    "accuracy": 0.8,
                    "roc_auc": 0.81,
                    "inverse_auc": 0.19,
                    "precision": 0.8,
                    "recall": 0.8,
                    "f1": 0.8,
                    "train_size": 1,
                    "test_size": 1,
                    "train_pos": 1,
                    "train_neg": 0,
                    "test_pos": 0,
                    "test_neg": 1,
                    "mean_prob_y1": 0.9,
                    "mean_prob_y0": 0.1,
                }
            ]
            write_jsonl(
                frozen_root / "results" / "reasoning_probe_metrics_qwen-qwen3-14b_off.jsonl",
                metrics_rows,
            )
            (
                frozen_root / "results" / "reasoning_probe_summary_qwen-qwen3-14b_off.txt"
            ).write_text(
                "\n".join(
                    [
                        "PressureTrace reasoning probe summary",
                        "",
                        "Baseline comparison:",
                        "  prompt_length: roc_auc=0.6413, accuracy=0.6063",
                        "  tfidf_prompt: roc_auc=0.6290, accuracy=0.5669",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            outputs = export_probe_reports(frozen_root=frozen_root)

            artifact_text = outputs["artifact_index"].read_text(encoding="utf-8")
            summary_text = outputs["summary"].read_text(encoding="utf-8")
            csv_text = outputs["metrics_csv"].read_text(encoding="utf-8")

        self.assertIn("Frozen root", artifact_text)
        self.assertIn("teacher_anchor=1", summary_text)
        self.assertIn("prompt_length: roc_auc=0.6413", summary_text)
        self.assertIn("kind,feature_set,layer", csv_text)


if __name__ == "__main__":
    unittest.main()
