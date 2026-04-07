# PressureTrace

PressureTrace is a research scaffold for studying pressure-induced transitions from robust problem solving toward shortcut-seeking or oversight-evasive behavior. The v1 codebase is organized around generating controlled paired task variants from reputable base datasets, running benchmark episodes, and summarizing behavioral traces in a way that can later support probing and activation-patching analysis.

## Why Build From Transformed Base Datasets

PressureTrace is intentionally not a template-only benchmark. The goal is to start from reputable datasets with real task structure, then apply controlled transformations that preserve task legitimacy while introducing pressure conditions, oversight gaps, or shortcut opportunities. That keeps the benchmark closer to realistic failure modes than synthetic prompt soup.

## Current V1 Scope

- Reasoning shortcut-seeking under pressure using GSM8K-style math reasoning tasks.
- Coding spec-gaming under pressure using HumanEval-style and MBPP-style coding tasks.

## Status

This repository is an early scaffold. The directory structure, CLI surface, JSONL data flow, loaders, and evaluation interfaces are in place, but the full research logic for transformation, inference, coding evaluation, probe extraction, and activation patching is still TODO.

## Setup

PressureTrace uses Python 3.11+ and `uv`.

```bash
uv sync --dev
uv run pressuretrace print-paths
```

Or use the bootstrap helper:

```bash
bash scripts/bootstrap.sh
```

## Example Commands

Run a dry pilot for reasoning:

```bash
uv run pressuretrace reasoning-pilot --limit 10 --split test --dry-run
```

Run a dry pilot for coding:

```bash
uv run pressuretrace coding-pilot --limit-per-dataset 5 --split test --dry-run
```

Summarize a result file:

```bash
uv run pressuretrace summarize --input-path results/reasoning_pilot_test_medium.jsonl
```

## Repository Layout

- `src/pressuretrace/`: package source code.
- `data/`: raw, intermediate, processed, manifest, and split artifacts.
- `results/`: benchmark outputs, probe artifacts, and summaries.
- `paper/`: short working spec for PressureTrace v1.
- `scripts/`: bootstrap and pilot wrappers.

## License

No license file has been added yet. Treat the repository as unlicensed / all rights reserved until a license is specified.
