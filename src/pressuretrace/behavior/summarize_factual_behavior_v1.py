"""Summarize route-label rates for the factual family benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from pressuretrace.utils.io import read_jsonl

PRESSURE_TYPE_ORDER = [
    "control",
    "authority_conflict",
    "encyclopedia_anchor",
    "neutral_wrong_answer_cue",
    "expert_citation",
]


def summarize_factual_results(results_path: Path) -> None:
    rows = read_jsonl(results_path)
    by_pressure: dict[str, list] = {}
    for row in rows:
        pt = row["pressure_type"]
        by_pressure.setdefault(pt, []).append(row)

    print(f"\nFactual Family Results: {results_path.name}")
    print(f"{'Pressure Type':<30} {'N':>5} {'Robust':>8} {'Shortcut':>10} {'Wrong':>8}")
    print("-" * 65)

    for pt in PRESSURE_TYPE_ORDER:
        if pt not in by_pressure:
            continue
        pt_rows = by_pressure[pt]
        n = len(pt_rows)
        robust = sum(1 for r in pt_rows if r["route_label"] == "robust_correct")
        shortcut = sum(1 for r in pt_rows if r["route_label"] == "shortcut_followed")
        wrong = sum(1 for r in pt_rows if r["route_label"] == "wrong_nonshortcut")
        print(
            f"{pt:<30} {n:>5} {robust/n:>7.1%} {shortcut/n:>9.1%} {wrong/n:>7.1%}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize_factual_results(args.results_path)


if __name__ == "__main__":
    main()
