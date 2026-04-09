from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.analysis.coding_probe_reports import export_probe_reports
from pressuretrace.utils.io import write_jsonl


class CodingProbeReportsTestCase(unittest.TestCase):
    def test_export_probe_reports_writes_artifact_index_csv_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            frozen_root = Path(tempdir)
            (frozen_root / "data" / "manifests").mkdir(parents=True)
            (frozen_root / "data" / "splits").mkdir(parents=True)
            (frozen_root / "results").mkdir(parents=True)

            write_jsonl(
                frozen_root / "data" / "manifests" / "coding_paper_slice_qwen-qwen3-14b_off.jsonl",
                [{"task_id": "task_1"}],
            )
            write_jsonl(
                frozen_root
                / "data"
                / "splits"
                / "coding_control_robust_slice_qwen-qwen3-14b_off.jsonl",
                [{"base_task_id": "base_1"}],
            )
            write_jsonl(
                frozen_root / "results" / "coding_paper_slice_qwen-qwen3-14b_off.jsonl",
                [{"task_id": "task_1"}],
            )
            hidden_rows = [
                {"task_id": "task_1", "base_task_id": "base_1", "binary_label": 0},
                {"task_id": "task_2", "base_task_id": "base_2", "binary_label": 1},
            ]
            write_jsonl(
                frozen_root / "results" / "coding_probe_hidden_states_qwen-qwen3-14b_off.jsonl",
                hidden_rows,
            )
            dataset_rows = []
            for index in range(6):
                dataset_rows.append(
                    {
                        "task_id": f"task_{index}",
                        "base_task_id": f"base_{index}",
                        "pressure_type": (
                            "teacher_anchor" if index >= 3 else "neutral_wrong_answer_cue"
                        ),
                        "binary_label": 1 if index >= 3 else 0,
                        "archetype": "spec_omission" if index % 2 else "weak_checker_exploit",
                        "prompt": f"Prompt {index}",
                        "route_label": "shortcut_success" if index >= 3 else "robust_success",
                    }
                )
            write_jsonl(
                frozen_root / "results" / "coding_probe_dataset_qwen-qwen3-14b_off.jsonl",
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
                    "train_size": 4,
                    "test_size": 2,
                    "train_pos": 2,
                    "train_neg": 2,
                    "test_pos": 1,
                    "test_neg": 1,
                    "mean_prob_y1": 0.9,
                    "mean_prob_y0": 0.1,
                }
            ]
            write_jsonl(
                frozen_root / "results" / "coding_probe_metrics_qwen-qwen3-14b_off.jsonl",
                metrics_rows,
            )

            outputs = export_probe_reports(frozen_root=frozen_root)

            artifact_text = outputs["artifact_index"].read_text(encoding="utf-8")
            summary_text = outputs["summary"].read_text(encoding="utf-8")
            csv_text = outputs["metrics_csv"].read_text(encoding="utf-8")

        self.assertIn("Frozen root", artifact_text)
        self.assertIn("Per-archetype counts", summary_text)
        self.assertIn("teacher_anchor=3", summary_text)
        self.assertIn("kind,feature_set,layer", csv_text)


if __name__ == "__main__":
    unittest.main()
