"""Minimal structural interfaces accepted by benchmark protocols."""

from typing import Protocol
import numpy as np
from numpy.typing import NDArray


class Policy(Protocol):
    def act(
        self, observation: NDArray[np.float32], deterministic: bool = True
    ) -> NDArray[np.floating]:
        """Return one action for an unbatched observation."""
        ...
