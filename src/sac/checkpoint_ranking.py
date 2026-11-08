"""Pure checkpoint-ranking rules shared by SAC evaluation and exports."""

from __future__ import annotations

import numpy as np


def best_return_index(mean_returns) -> int:
    """Return NumPy's first maximum, preserving existing NaN behavior."""
    return int(np.argmax(mean_returns))


def success_score(success_rate, mean_return) -> tuple:
    """Rank success first and mean return second."""
    values = np.asarray((success_rate, mean_return), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Checkpoint ranking metrics must be finite.")
    return float(values[0]), float(values[1])


def best_success_index(success_rates, mean_returns) -> int:
    """Return the first lexicographic success/return maximum."""
    if len(success_rates) != len(mean_returns) or not len(success_rates):
        raise ValueError("Checkpoint ranking inputs must be nonempty and aligned.")
    return max(
        range(len(success_rates)),
        key=lambda index: success_score(success_rates[index], mean_returns[index]),
    )
