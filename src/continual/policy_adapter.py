"""Policy boundary shared by trajectory collection and continual evaluation.

Adapters return only an action, hiding framework-specific prediction state.
Model loading and legacy checkpoint conversion deliberately remain outside
this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from continual_manipulation.environments import ACTION_SHAPE, OBSERVATION_SHAPE
from numpy.typing import NDArray

if TYPE_CHECKING:
    from stable_baselines3 import SAC


class SB3SACPolicy:
    """Adapt an already loaded SB3 SAC policy to the common policy interface.

    Construction verifies only the observation/action shapes. Checkpoint
    loading, device placement, and task selection remain the caller's job;
    DiffCRL uses this adapter for its frozen current-task experts.
    """

    def __init__(self, model: SAC) -> None:
        if model.observation_space.shape != OBSERVATION_SHAPE:
            raise ValueError("Model observation shape does not match MetaWorld v3.")
        if model.action_space.shape != ACTION_SHAPE:
            raise ValueError("Model action shape does not match MetaWorld v3.")
        self.model = model

    def act(
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> NDArray[np.floating]:
        """Return one finite 4D action and discard SB3's recurrent-state slot."""
        observation = np.asarray(observation)
        if observation.shape != OBSERVATION_SHAPE or not np.isfinite(observation).all():
            raise ValueError("Expected one finite MetaWorld observation.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        action = np.asarray(action)
        if action.shape != ACTION_SHAPE or not np.isfinite(action).all():
            raise ValueError("Expected one finite MetaWorld action.")
        return action.copy()
