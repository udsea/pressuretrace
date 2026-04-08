"""Tests for the reasoning-family model replication pipeline helper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pressuretrace.behavior.run_reasoning_model_pipeline import (
    ReasoningModelPipelineConfig,
    default_replication_frozen_root,
    run_reasoning_model_pipeline,
)
from pressuretrace.utils.io import write_jsonl


class RunReasoningModelPipelineTestCase(unittest.TestCase):
    def test_default_replication_frozen_root_uses_model_slug(self) -> None:
        path = default_replication_frozen_root("google/gemma-3-27b-it", "off")

        self.assertTrue(str(path).endswith("reasoning_v2_google-gemma-3-27b-it_off"))

    def test_pipeline_runs_all_stages_and_writes_run_info(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            frozen_root = Path(tempdir) / "reasoning_v2_google-gemma-3-27b-it_off"
            pool_manifest = (
                frozen_root / "data" / "manifests" / "reasoning_all_valid_transforms.jsonl"
            )
            metrics_path = (
                frozen_root
                / "results"
                / "reasoning_probe_metrics_google-gemma-3-27b-it_off.jsonl"
            )

            def fake_resolve_pool(*args, **kwargs):  # type: ignore[no-untyped-def]
                del args, kwargs
                write_jsonl(
                    pool_manifest,
                    [
                        {
                            "task_id": "task_1_control",
                            "pressure_type": "control",
                            "metadata": {"base_task_id": "base_1"},
                        }
                    ],
                )
                return pool_manifest

            config = ReasoningModelPipelineConfig(
                model_name="google/gemma-3-27b-it",
                frozen_root=frozen_root,
                thinking_mode="off",
                show_progress=False,
            )

            with patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline._resolve_pool_manifest",
                side_effect=fake_resolve_pool,
            ), patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.run_reasoning_control_only"
            ) as mock_control, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.build_control_robust_slice"
            ) as mock_slice, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.materialize_reasoning_slice"
            ) as mock_materialize, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.run_reasoning_manifest_v2"
            ) as mock_manifest, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.extract_reasoning_hidden_states"
            ) as mock_extract, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.build_reasoning_probe_dataset"
            ) as mock_dataset, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.train_reasoning_probes"
            ) as mock_train, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline._write_probe_metrics_csv"
            ) as mock_csv, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.build_reasoning_patch_pairs"
            ) as mock_patch_pairs, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.run_reasoning_route_patching"
            ) as mock_route_patching:
                def train_side_effect(config_arg):  # type: ignore[no-untyped-def]
                    write_jsonl(
                        metrics_path,
                        [
                            {
                                "kind": "hidden_state_probe",
                                "feature_set": "hidden_state",
                                "layer": -10,
                                "representation": "last_token",
                                "roc_auc": 0.75,
                                "accuracy": 0.7,
                                "f1": 0.74,
                            }
                        ],
                    )
                    return config_arg.metrics_path

                mock_train.side_effect = train_side_effect

                paths = run_reasoning_model_pipeline(config)

                mock_control.assert_called_once()
                self.assertEqual(mock_control.call_args.kwargs["manifest_path"], pool_manifest)
                self.assertEqual(mock_control.call_args.kwargs["thinking_mode"], "off")
                mock_slice.assert_called_once()
                mock_materialize.assert_called_once()
                mock_manifest.assert_called_once()
                mock_extract.assert_called_once()
                self.assertEqual(
                    mock_extract.call_args.args[0].model_name,
                    "google/gemma-3-27b-it",
                )
                self.assertEqual(mock_extract.call_args.args[0].thinking_mode, "off")
                mock_dataset.assert_called_once()
                mock_train.assert_called_once()
                mock_csv.assert_called_once()
                mock_patch_pairs.assert_called_once()
                mock_route_patching.assert_called_once()
                self.assertEqual(
                    mock_patch_pairs.call_args.kwargs["output_path"],
                    paths.patch_pairs,
                )
                self.assertTrue(paths.run_info_txt.exists())
                run_info = paths.run_info_txt.read_text(encoding="utf-8")
                self.assertIn("model=google/gemma-3-27b-it", run_info)
                self.assertIn(f"probe_metrics_jsonl={paths.probe_metrics_jsonl}", run_info)
                self.assertIn(f"route_patching_results={paths.route_patching_results}", run_info)

    def test_pipeline_skip_probes_skips_probe_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            frozen_root = Path(tempdir) / "reasoning_v2_sarvamai-sarvam-30b_off"
            pool_manifest = (
                frozen_root / "data" / "manifests" / "reasoning_all_valid_transforms.jsonl"
            )

            def fake_resolve_pool(*args, **kwargs):  # type: ignore[no-untyped-def]
                del args, kwargs
                write_jsonl(
                    pool_manifest,
                    [
                        {
                            "task_id": "task_1_control",
                            "pressure_type": "control",
                            "metadata": {"base_task_id": "base_1"},
                        }
                    ],
                )
                return pool_manifest

            config = ReasoningModelPipelineConfig(
                model_name="sarvamai/sarvam-30b",
                frozen_root=frozen_root,
                thinking_mode="off",
                skip_probes=True,
                skip_patching=True,
                show_progress=False,
            )

            with patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline._resolve_pool_manifest",
                side_effect=fake_resolve_pool,
            ), patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.run_reasoning_control_only"
            ), patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.build_control_robust_slice"
            ), patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.materialize_reasoning_slice"
            ), patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.run_reasoning_manifest_v2"
            ), patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.extract_reasoning_hidden_states"
            ) as mock_extract, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.build_reasoning_probe_dataset"
            ) as mock_dataset, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.train_reasoning_probes"
            ) as mock_train, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.build_reasoning_patch_pairs"
            ) as mock_patch_pairs, patch(
                "pressuretrace.behavior.run_reasoning_model_pipeline.run_reasoning_route_patching"
            ) as mock_route_patching:
                paths = run_reasoning_model_pipeline(config)

                mock_extract.assert_not_called()
                mock_dataset.assert_not_called()
                mock_train.assert_not_called()
                mock_patch_pairs.assert_not_called()
                mock_route_patching.assert_not_called()
                self.assertTrue(paths.run_info_txt.exists())
                self.assertIn(
                    "skip_probes=True",
                    paths.run_info_txt.read_text(encoding="utf-8"),
                )
                self.assertIn(
                    "skip_patching=True",
                    paths.run_info_txt.read_text(encoding="utf-8"),
                )

    def test_pipeline_rejects_models_outside_replication_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = ReasoningModelPipelineConfig(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                frozen_root=Path(tempdir) / "reasoning_v2_qwen-qwen2-5-7b-instruct_off",
                thinking_mode="off",
                show_progress=False,
            )

            with self.assertRaises(ValueError):
                run_reasoning_model_pipeline(config)


if __name__ == "__main__":
    unittest.main()
