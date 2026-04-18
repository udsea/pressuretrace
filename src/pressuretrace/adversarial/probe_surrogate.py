"""Fast probe-based surrogate scorer for adversarial attack search.

Instead of running full generation to check if the model shortcutted,
project the final-prompt-token hidden state onto the induction direction.
High projection = model is in shortcut route state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pressuretrace.patching.reasoning_patching_core import (
    ReasoningPatchingBundle,
    final_token_activation_for_layer,
    get_prompt_hidden_states,
)


class ProbeDirectionScorer:
    """Scores prompts by projecting their final-token hidden state onto the induction direction.

    Higher score = more shortcut-like (closer to shortcut_followed centroid).
    """

    def __init__(
        self,
        induction_direction: np.ndarray,
        bundle: ReasoningPatchingBundle,
        layer: int = -6,
    ):
        self.direction = torch.tensor(induction_direction, dtype=torch.float32)
        self.bundle = bundle
        self.layer = layer

    @classmethod
    def from_hidden_states_file(
        cls,
        hidden_states_path: Path,
        bundle: ReasoningPatchingBundle,
        *,
        layer: int = -6,
        representation: str = "last_token",
    ) -> "ProbeDirectionScorer":
        """Build scorer from a frozen probe hidden-states file."""

        from pressuretrace.analysis.direction_characterization import (
            compute_mean_difference_direction,
            load_hidden_states_for_layer,
        )

        X, y = load_hidden_states_for_layer(
            hidden_states_path, layer=layer, representation=representation,
        )
        induction_dir, _ = compute_mean_difference_direction(X, y)
        return cls(induction_dir, bundle, layer=layer)

    def score_prompt(self, prompt: str) -> float:
        """Return the induction-direction projection score for a prompt."""

        prompt_inputs, outputs = get_prompt_hidden_states(self.bundle, prompt)
        activation = final_token_activation_for_layer(
            outputs, prompt_inputs, model=self.bundle.model, layer=self.layer,
        )
        activation_np = activation.detach().cpu().float().numpy()
        return float(np.dot(activation_np, self.direction.numpy()))

    def score_prompts_batch(self, prompts: list[str]) -> list[float]:
        """Score multiple prompts sequentially."""

        return [self.score_prompt(p) for p in prompts]
