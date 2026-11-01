"""Compatibility exports for DiffCRL stage-dataset preparation.

New code should import from :mod:`src.continual.stage_dataset`.
"""

from src.continual.stage_dataset import (
    PreparedGroupedTrajectoryDataset,
    TrajectoryGroupPreparation,
    bc_dataset,
    concatenate_trajectory_groups,
    prepare_trajectory_group,
    split_indices,
    validate_trajectory_group,
)

__all__ = [
    "PreparedGroupedTrajectoryDataset",
    "TrajectoryGroupPreparation",
    "bc_dataset",
    "concatenate_trajectory_groups",
    "prepare_trajectory_group",
    "split_indices",
    "validate_trajectory_group",
]
