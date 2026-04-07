"""Scaffold for activation patching experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActivationPatchingConfig:
    """Configuration for causal activation-patching runs."""

    source_route_label: str
    target_route_label: str
    layers: tuple[int, ...]
    patch_strategy: str = "tokenwise_residual"


def run_activation_patching(
    behavior_results_path: Path,
    activations_dir: Path,
    output_dir: Path,
    config: ActivationPatchingConfig,
) -> Path:
    """Run activation patching between matched source and target episodes."""

    del behavior_results_path, activations_dir, output_dir, config
    raise NotImplementedError("Activation patching is not implemented in the v1 scaffold.")
