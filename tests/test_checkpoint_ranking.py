import numpy as np
import pytest

from train_sac import best_success_index


def test_checkpoint_ranking_ties_and_nan_behavior() -> None:
    # Success takes priority over return; exact ties keep the first checkpoint.
    assert best_success_index([0.5, 0.5, 0.4], [2.0, 2.0, 100.0]) == 0
    # Return breaks a success-rate tie.
    assert best_success_index([0.5, 0.5], [1.0, 2.0]) == 1
    with pytest.raises(ValueError, match="finite"):
        best_success_index([0.5, np.nan, 1.0], [1.0, 100.0, 0.0])


def test_checkpoint_ranking_rejects_nonfinite_or_unaligned_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        best_success_index([0.5, 1.0], [1.0, np.inf])
    with pytest.raises(ValueError, match="nonempty and aligned"):
        best_success_index([], [])
    with pytest.raises(ValueError, match="nonempty and aligned"):
        best_success_index([0.5], [1.0, 2.0])
