"""Scaffold for probe training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProbeTrainingConfig:
    """Configuration for linear-probe training."""

    regularization_strength: float = 1.0
    max_iter: int = 1000
    random_seed: int = 23


def train_probes(
    probe_dataset_path: Path,
    output_dir: Path,
    config: ProbeTrainingConfig,
) -> Path:
    """Train route-classification probes from activation features."""

    del probe_dataset_path, output_dir, config
    raise NotImplementedError("Probe training is not implemented in the v1 scaffold.")
