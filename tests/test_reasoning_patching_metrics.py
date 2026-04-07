from __future__ import annotations

import unittest
from dataclasses import asdict

from pressuretrace.patching.reasoning_patching_metrics import (
    AnswerTokenPair,
    PatchSummaryRow,
    TokenSnapshot,
    aggregate_patch_rows,
    build_answer_token_pair,
    build_patch_comparison_row,
    compute_token_snapshot,
    patch_summary_to_csv_text,
    render_summary_text,
    summary_rows_to_csv_rows,
)


class DummyTokenizer:
    """Minimal tokenizer stub for metrics tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        mapping = {
            "7": [7],
            "42": [42],
            "125": [125],
            "96": [96],
            "multi token": [1, 2],
        }
        return mapping[text]

    def convert_ids_to_tokens(self, token_ids: int | list[int]) -> str | list[str]:
        if isinstance(token_ids, list):
            return [self.convert_ids_to_tokens(token_id) for token_id in token_ids]
        return f"tok_{token_ids}"


class ReasoningPatchingMetricsTestCase(unittest.TestCase):
    def test_build_answer_token_pair_requires_single_token_answers(self) -> None:
        tokenizer = DummyTokenizer()

        pair = build_answer_token_pair(
            tokenizer,
            gold_answer="125",
            shortcut_answer="96",
        )

        self.assertEqual(
            pair,
            AnswerTokenPair(
                gold_token_id=125,
                gold_token_str="tok_125",
                shortcut_token_id=96,
                shortcut_token_str="tok_96",
            ),
        )
        self.assertIsNone(
            build_answer_token_pair(
                tokenizer,
                gold_answer="multi token",
                shortcut_answer="96",
            )
        )

    def test_compute_token_snapshot_uses_softmax_and_top1(self) -> None:
        tokenizer = DummyTokenizer()
        snapshot = compute_token_snapshot(
            [1.0, 3.0, 2.0],
            gold_token_id=0,
            shortcut_token_id=2,
            tokenizer=tokenizer,
        )

        self.assertEqual(snapshot.top1_token_id, 1)
        self.assertEqual(snapshot.top1_token_str, "tok_1")
        self.assertTrue(snapshot.shortcut_prob > snapshot.gold_prob)
        self.assertAlmostEqual(
            snapshot.gold_minus_shortcut_margin,
            -1.0,
            places=6,
        )

    def test_build_patch_comparison_row_uses_directional_baseline(self) -> None:
        gold = TokenSnapshot(1.0, 0.5, 0.6, 0.4, 0.5, 1, "tok_1")
        pressure = TokenSnapshot(0.8, 0.7, 0.55, 0.35, 0.1, 2, "tok_2")
        patched = TokenSnapshot(1.2, 0.2, 0.7, 0.2, 1.0, 3, "tok_3")
        tokenizer = DummyTokenizer()
        pair = build_answer_token_pair(tokenizer, gold_answer="125", shortcut_answer="96")
        assert pair is not None

        rescue_row = build_patch_comparison_row(
            base_task_id="base_1",
            pressure_type="neutral_wrong_answer_cue",
            layer=-10,
            direction="rescue",
            control_task_id="control_1",
            pressure_task_id="pressure_1",
            gold_answer="125",
            shortcut_answer="96",
            answer_tokens=pair,
            control_snapshot=gold,
            pressure_snapshot=pressure,
            patched_snapshot=patched,
            metadata={"split": "test"},
        )

        self.assertEqual(rescue_row.baseline_kind, "pressure")
        self.assertAlmostEqual(rescue_row.delta_gold_prob, 0.15)
        self.assertAlmostEqual(rescue_row.delta_shortcut_prob, -0.15)
        self.assertAlmostEqual(rescue_row.delta_margin, 0.9)
        self.assertTrue(rescue_row.top1_changed)

        induction_row = build_patch_comparison_row(
            base_task_id="base_1",
            pressure_type="neutral_wrong_answer_cue",
            layer=-10,
            direction="induction",
            control_task_id="control_1",
            pressure_task_id="pressure_1",
            gold_answer="125",
            shortcut_answer="96",
            answer_tokens=pair,
            control_snapshot=gold,
            pressure_snapshot=pressure,
            patched_snapshot=patched,
            metadata={"split": "test"},
        )

        self.assertEqual(induction_row.baseline_kind, "control")
        self.assertAlmostEqual(induction_row.delta_gold_prob, 0.10)
        self.assertAlmostEqual(induction_row.delta_shortcut_prob, -0.20)
        self.assertAlmostEqual(induction_row.delta_margin, 0.5)

    def test_aggregate_and_render_summary_rows(self) -> None:
        rows = [
            {
                "pressure_type": "neutral_wrong_answer_cue",
                "layer": -10,
                "direction": "rescue",
                "delta_gold_prob": 0.2,
                "delta_shortcut_prob": -0.1,
                "delta_margin": 0.3,
                "top1_changed": True,
            },
            {
                "pressure_type": "neutral_wrong_answer_cue",
                "layer": -10,
                "direction": "rescue",
                "delta_gold_prob": 0.0,
                "delta_shortcut_prob": -0.2,
                "delta_margin": 0.1,
                "top1_changed": False,
            },
            {
                "pressure_type": "teacher_anchor",
                "layer": -8,
                "direction": "induction",
                "delta_gold_prob": -0.1,
                "delta_shortcut_prob": 0.4,
                "delta_margin": -0.5,
                "top1_changed": True,
            },
        ]

        summary_rows = aggregate_patch_rows(rows)
        self.assertEqual(len(summary_rows), 2)

        rescue = next(
            row for row in summary_rows if row.pressure_type == "neutral_wrong_answer_cue"
        )
        self.assertEqual(rescue.n_pairs, 2)
        self.assertAlmostEqual(rescue.mean_delta_gold_prob, 0.1)
        self.assertAlmostEqual(rescue.top1_changed_rate, 0.5)

        csv_rows = summary_rows_to_csv_rows(summary_rows)
        self.assertEqual(csv_rows[0]["pressure_type"], "neutral_wrong_answer_cue")
        self.assertIn("mean_delta_margin", csv_rows[0])

        text = render_summary_text(summary_rows)
        csv_text = patch_summary_to_csv_text(summary_rows)
        self.assertIn("PressureTrace reasoning route patching summary", text)
        self.assertIn("pressure_type,layer,direction", csv_text)

    def test_summary_rows_round_trip_with_dataclass(self) -> None:
        row = PatchSummaryRow(
            pressure_type="teacher_anchor",
            layer=-6,
            direction="induction",
            n_pairs=3,
            mean_delta_gold_prob=0.1,
            mean_delta_shortcut_prob=0.2,
            mean_delta_margin=0.05,
            top1_changed_rate=1.0,
        )
        self.assertEqual(asdict(row)["direction"], "induction")
        self.assertAlmostEqual(
            patch_summary_to_csv_text([row]).count("\n"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
