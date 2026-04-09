from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pressuretrace.patching.coding_patching_metrics import (
    CodingRoutePatchingRow,
    aggregate_route_patching_rows,
    plot_route_patching_position_window_comparison,
    plot_route_patching_success_by_layer,
    render_route_patching_summary_text,
    route_patching_summary_to_csv_text,
)


class CodingRoutePatchingMetricsTestCase(unittest.TestCase):
    """Focused tests for coding route-patching reporting helpers."""

    def _make_row(
        self,
        *,
        pressure_type: str,
        layer: int,
        position_window: str,
        direction: str,
        patched_route_label: str,
        rescue_success: bool,
        induction_success: bool,
        visible_pass_hidden_fail_after_patch: bool,
        patch_applied: bool = True,
        top1_changed: bool = False,
    ) -> CodingRoutePatchingRow:
        return CodingRoutePatchingRow(
            base_task_id="base_1",
            archetype="visible_test_hardcoding",
            pressure_type=pressure_type,
            layer=layer,
            position_window=position_window,
            direction=direction,
            control_task_id="control_1",
            pressure_task_id="pressure_1",
            route_control="robust_success",
            route_pressure="shortcut_success",
            patched_route_label=patched_route_label,
            patched_visible_pass=True,
            patched_hidden_pass=patched_route_label == "robust_success",
            patched_failure_subtype=(
                "hidden_test_failure" if patched_route_label == "shortcut_success" else None
            ),
            patched_visible_failure_names=[],
            patched_hidden_failure_names=[],
            rescue_success=rescue_success,
            induction_success=induction_success,
            visible_pass_hidden_fail_after_patch=visible_pass_hidden_fail_after_patch,
            patch_applied=patch_applied,
            top1_changed=top1_changed,
            patched_step_count=3,
            patched_completion="def solve(): pass",
            patched_extracted_code="def solve(): pass",
            metadata={"base_task_id": "base_1"},
        )

    def test_aggregate_and_render_route_patching_rows(self) -> None:
        rows = [
            self._make_row(
                pressure_type="neutral_wrong_answer_cue",
                layer=-10,
                position_window="gen_1",
                direction="rescue",
                patched_route_label="robust_success",
                rescue_success=True,
                induction_success=False,
                visible_pass_hidden_fail_after_patch=False,
            ),
            self._make_row(
                pressure_type="neutral_wrong_answer_cue",
                layer=-10,
                position_window="gen_1",
                direction="rescue",
                patched_route_label="shortcut_success",
                rescue_success=False,
                induction_success=False,
                visible_pass_hidden_fail_after_patch=True,
                top1_changed=True,
            ),
            self._make_row(
                pressure_type="teacher_anchor",
                layer=-8,
                position_window="gen_1_3",
                direction="induction",
                patched_route_label="shortcut_success",
                rescue_success=False,
                induction_success=True,
                visible_pass_hidden_fail_after_patch=True,
            ),
        ]

        summary_rows = aggregate_route_patching_rows(rows)

        self.assertEqual(len(summary_rows), 2)
        rescue_row = next(
            row
            for row in summary_rows
            if row.pressure_type == "neutral_wrong_answer_cue" and row.direction == "rescue"
        )
        self.assertEqual(rescue_row.position_window, "gen_1")
        self.assertAlmostEqual(rescue_row.rescue_success_rate, 0.5)
        self.assertAlmostEqual(rescue_row.robust_rate_after_patch, 0.5)
        self.assertAlmostEqual(rescue_row.shortcut_rate_after_patch, 0.5)
        self.assertAlmostEqual(rescue_row.visible_pass_hidden_fail_rate_after_patch, 0.5)
        self.assertAlmostEqual(rescue_row.top1_changed_rate, 0.5)

        induction_row = next(
            row
            for row in summary_rows
            if row.pressure_type == "teacher_anchor" and row.direction == "induction"
        )
        self.assertEqual(induction_row.position_window, "gen_1_3")
        self.assertAlmostEqual(induction_row.induction_success_rate, 1.0)
        self.assertAlmostEqual(induction_row.shortcut_rate_after_patch, 1.0)
        self.assertAlmostEqual(induction_row.visible_pass_hidden_fail_rate_after_patch, 1.0)

        text = render_route_patching_summary_text(summary_rows)
        csv_text = route_patching_summary_to_csv_text(summary_rows)
        self.assertIn("Grouped by pressure_type, layer, position_window, direction.", text)
        self.assertIn("rescue_success_rate", text)
        self.assertIn("induction_success_rate", text)
        self.assertIn("position_window", csv_text.splitlines()[0])
        self.assertIn("visible_pass_hidden_fail_rate_after_patch", csv_text)

    def test_route_patching_plots_write_files(self) -> None:
        rows = [
            self._make_row(
                pressure_type="neutral_wrong_answer_cue",
                layer=-10,
                position_window="gen_1",
                direction="rescue",
                patched_route_label="robust_success",
                rescue_success=True,
                induction_success=False,
                visible_pass_hidden_fail_after_patch=False,
            ),
            self._make_row(
                pressure_type="neutral_wrong_answer_cue",
                layer=-8,
                position_window="gen_1_3",
                direction="rescue",
                patched_route_label="shortcut_success",
                rescue_success=False,
                induction_success=False,
                visible_pass_hidden_fail_after_patch=True,
            ),
            self._make_row(
                pressure_type="teacher_anchor",
                layer=-10,
                position_window="gen_1",
                direction="induction",
                patched_route_label="shortcut_success",
                rescue_success=False,
                induction_success=True,
                visible_pass_hidden_fail_after_patch=True,
            ),
        ]
        summary_rows = aggregate_route_patching_rows(rows)

        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            rescue_plot = temp_path / "rescue.png"
            induction_plot = temp_path / "induction.png"
            window_plot = temp_path / "window.png"

            plot_route_patching_success_by_layer(
                summary_rows,
                direction="rescue",
                output_path=rescue_plot,
            )
            plot_route_patching_success_by_layer(
                summary_rows,
                direction="induction",
                output_path=induction_plot,
            )
            plot_route_patching_position_window_comparison(
                summary_rows,
                output_path=window_plot,
            )

            self.assertTrue(rescue_plot.exists())
            self.assertTrue(induction_plot.exists())
            self.assertTrue(window_plot.exists())


if __name__ == "__main__":
    unittest.main()
