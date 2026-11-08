import numpy as np
import pytest

from src.sac.checkpoint_ranking import best_return_index, best_success_index


def test_checkpoint_ranking_ties_and_nan_behavior() -> None:
    assert best_return_index([2.0, 2.0, 1.0]) == 0
    assert best_success_index([0.5, 0.5, 0.4], [2.0, 2.0, 100.0]) == 0
    assert best_success_index([0.5, 0.5], [1.0, 2.0]) == 1
    assert best_return_index([1.0, np.nan, 3.0]) == 1
    with pytest.raises(ValueError, match="finite"):
        best_success_index([0.5, np.nan, 1.0], [1.0, 100.0, 0.0])
