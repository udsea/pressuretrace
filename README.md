# PressureTrace

PressureTrace is a research scaffold for studying pressure-induced transitions from robust problem solving toward shortcut-seeking or oversight-evasive behavior. The v1 codebase is organized around generating controlled paired task variants from reputable base datasets, running benchmark episodes, and summarizing behavioral traces in a way that can later support probing and activation-patching analysis.

## Why Build From Transformed Base Datasets

PressureTrace is intentionally not a template-only benchmark. The goal is to start from reputable datasets with real task structure, then apply controlled transformations that preserve task legitimacy while introducing pressure conditions, oversight gaps, or shortcut opportunities. That keeps the benchmark closer to realistic failure modes than synthetic prompt soup.

## Current Scope

- Reasoning shortcut-seeking under pressure using GSM8K-style math reasoning tasks.
- Reasoning v2 paper-slice workflow with frozen artifacts, probes, and route patching.
- Coding spec-gaming under pressure using HumanEval-style and MBPP-style coding tasks.

## Status

This repository now has an active reasoning-family v2 path with:

- paired manifest generation
- control-robust slice construction
- frozen paper-slice artifacts
- hidden-state probe training
- first-pass route patching

The coding-family side is still lighter-weight and remains closer to scaffold status.

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
- `pressuretrace-frozen/`: frozen reasoning-family bundles that the current probe and patching workflows read.
- `results/`: ignored local run outputs.
- `archive/`: archived legacy snapshots that are kept for provenance but are not part of the active runtime path.
- `paper/`: working notes and specs.
- `scripts/`: bootstrap, benchmark, sweep, replication, and analysis wrappers.

## License

No license file has been added yet. Treat the repository as unlicensed / all rights reserved until a license is specified.
