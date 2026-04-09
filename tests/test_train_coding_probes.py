from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.config import CODING_PROBE_LAYERS, CODING_PROBE_REPRESENTATIONS
from pressuretrace.probes.train_coding_probes import (
    CodingProbeTrainingConfig,
    train_coding_probes,
)
from pressuretrace.utils.io import read_jsonl, write_jsonl


class TrainCodingProbesTestCase(unittest.TestCase):
    def test_train_coding_probes_writes_probe_metrics_and_summary(self) -> None:
        rows: list[dict[str, object]] = []
        for base_index in range(10):
            label = 1 if base_index >= 5 else 0
            base_task_id = f"coding_base_{base_index:03d}"
            prompt = f"Prompt {base_index} shortcut {label}"
            pressure_type = "teacher_anchor" if label else "neutral_wrong_answer_cue"
            route_label = "shortcut_success" if label else "robust_success"
            archetype = (
                "visible_test_hardcoding" if base_index % 3 == 0 else "weak_checker_exploit"
            )
            for layer in CODING_PROBE_LAYERS:
                for representation in CODING_PROBE_REPRESENTATIONS:
                    rows.append(
                        {
                            "task_id": f"{base_task_id}_{pressure_type}",
                            "base_task_id": base_task_id,
                            "pressure_type": pressure_type,
                            "route_label": route_label,
                            "binary_label": label,
                            "layer": layer,
                            "representation": representation,
                            "hidden_state": [
                                float(label),
                                float(label) + 0.5,
                                float(base_index),
                            ],
                            "prompt": prompt,
                            "archetype": archetype,
                            "source_family": "humaneval_like",
                            "source_task_name": f"humaneval_like/task_{base_index}",
                        }
                    )

        with tempfile.TemporaryDirectory() as tempdir:
            dataset_path = Path(tempdir) / "dataset.jsonl"
            metrics_path = Path(tempdir) / "metrics.jsonl"
            summary_path = Path(tempdir) / "summary.txt"
            write_jsonl(dataset_path, rows)

            output_path = train_coding_probes(
                CodingProbeTrainingConfig(
                    input_path=dataset_path,
                    metrics_path=metrics_path,
                    summary_path=summary_path,
                    seed=42,
                    test_size=0.3,
                )
            )
            metrics_rows = read_jsonl(output_path)
            summary_text = summary_path.read_text(encoding="utf-8")

        self.assertEqual(
            len(metrics_rows),
            len(CODING_PROBE_LAYERS) * len(CODING_PROBE_REPRESENTATIONS),
        )
        self.assertTrue(all(row["kind"] == "hidden_state_probe" for row in metrics_rows))
        self.assertIn("Dataset filtering logic", summary_text)
        self.assertIn("Per-archetype counts", summary_text)
        self.assertIn("Best hidden-state probe result", summary_text)
        self.assertIn("Baseline results", summary_text)


if __name__ == "__main__":
    unittest.main()
