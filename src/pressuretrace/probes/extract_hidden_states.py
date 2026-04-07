"""Scaffold for hidden-state extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HiddenStateExtractionConfig:
    """Configuration for extracting hidden states from model runs."""

    model_name: str
    layers: tuple[int, ...]
    token_pooling: str = "last_token"
    batch_size: int = 1


def extract_hidden_states(
    results_path: Path,
    output_dir: Path,
    config: HiddenStateExtractionConfig,
) -> Path:
    """Extract and persist hidden states for benchmark episodes.

    TODO:
    - Rehydrate episode prompts and responses.
    - Run the model with forward hooks on requested layers.
    - Persist aligned activation tensors and index metadata.
    """

    del results_path, output_dir, config
    raise NotImplementedError("Hidden-state extraction is not implemented in the v1 scaffold.")
