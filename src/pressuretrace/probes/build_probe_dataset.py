"""Scaffold for building probe-ready datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProbeDatasetConfig:
    """Configuration for constructing probe datasets."""

    label_field: str = "route_label"
    min_examples_per_label: int = 50
    train_fraction: float = 0.8


def build_probe_dataset(
    activations_dir: Path,
    behavior_results_path: Path,
    output_path: Path,
    config: ProbeDatasetConfig,
) -> Path:
    """Join behavior labels with extracted activations into a probe dataset."""

    del activations_dir, behavior_results_path, output_path, config
    raise NotImplementedError("Probe-dataset construction is not implemented in the v1 scaffold.")
