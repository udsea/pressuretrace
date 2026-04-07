.PHONY: install sync fmt lint test run-reasoning-pilot run-coding-pilot run-reasoning-frontier-sweep run-reasoning-5090-sweep run-reasoning-pilot-v2 run-reasoning-thinking-ablation-v2

install:
	uv sync --dev

sync:
	uv sync --dev

fmt:
	uv run ruff format src tests

lint:
	uv run ruff check src tests
	uv run mypy src

test:
	PYTHONPATH=src uv run python -m unittest discover -s tests -p 'test_*.py'

run-reasoning-pilot:
	uv run pressuretrace reasoning-pilot --limit 10 --split test --dry-run

run-reasoning-pilot-v2:
	uv run pressuretrace reasoning-pilot-v2 --limit 10 --split test --dry-run

run-reasoning-frontier-sweep:
	bash scripts/run_reasoning_frontier_sweep.sh

run-reasoning-5090-sweep:
	bash scripts/run_reasoning_5090_sweep.sh

run-reasoning-thinking-ablation-v2:
	bash scripts/run_reasoning_thinking_ablation_v2.sh

run-coding-pilot:
	uv run pressuretrace coding-pilot --limit-per-dataset 5 --split test --dry-run
