"""Deterministic DiffCRL stage-dataset preparation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.continual.diffcrl import (
    CurrentTaskData,
    DiffCRLTrainer,
    StageContext,
    StageRawData,
)
from src.continual.task_sequence import TaskSequence
from src.support.dataset import (
    concatenate_trajectory_groups,
    prepare_trajectory_group,
    split_indices,
    validate_trajectory_group,
)
from src.support.reproducibility import file_hash


def fixture_groups():
    generator = torch.Generator().manual_seed(20260929)
    groups = {
        0: torch.randn((5, 200, 43), generator=generator),
        1: torch.randn((5, 200, 43), generator=generator),
    }
    for values in groups.values():
        values[..., 18:36] = torch.cat(
            (values[:, :1, :18], values[:, :-1, :18]), dim=1
        )
        values[..., 36:39] = values[:, :1, 36:39]
    groups[0][..., 39:] = torch.tensor([-1.0, -0.9998, 0.25, 1.0])
    groups[1][..., 39:] = torch.tensor([-1.2, -0.5, 0.99995, 1.1])
    return groups


class StageDatasetTests(unittest.TestCase):
    def test_group_preparation_preserves_split_and_clip_diagnostic(self):
        values = fixture_groups()[1]
        result = prepare_trajectory_group(
            values,
            count=5,
            seed=1018,
            action_low=-torch.ones(4),
            action_high=torch.ones(4),
            clamp_epsilon=1e-4,
        )
        train_ids, validation_ids = split_indices(5, 1018)
        self.assertEqual(result.train_ids, train_ids)
        self.assertEqual(result.validation_ids, validation_ids)
        self.assertEqual(result.bc_target_clip_fraction, 0.75)

    def test_group_concatenation_preserves_mapping_order_and_exact_labels(self):
        groups = fixture_groups()
        preparations = {
            index: prepare_trajectory_group(
                values,
                count=5,
                seed=17 + 1000 + index,
                action_low=-torch.ones(4),
                action_high=torch.ones(4),
                clamp_epsilon=1e-4,
            )
            for index, values in groups.items()
        }
        result = concatenate_trajectory_groups(groups, preparations)
        expected_train = torch.cat(
            [groups[i][preparations[i].train_ids] for i in groups]
        )
        expected_validation = torch.cat(
            [groups[i][preparations[i].validation_ids] for i in groups]
        )
        torch.testing.assert_close(result.train, expected_train, rtol=0, atol=0)
        torch.testing.assert_close(
            result.validation, expected_validation, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.train_labels, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        )
        torch.testing.assert_close(result.validation_labels, torch.tensor([0, 1]))

    def test_stage_preparation_preserves_files_hashes_and_provenance(self):
        groups = fixture_groups()
        sequence = TaskSequence.from_names(["reach-v3", "hammer-v3"])
        trainer = object.__new__(DiffCRLTrainer)
        trainer.config = SimpleNamespace(runtime=SimpleNamespace(seed=17))
        trainer.sequence = sequence
        trainer.policy = SimpleNamespace(
            action_low=-torch.ones(4), action_high=torch.ones(4)
        )
        trainer.train_configurations = {}
        trainer.horizon = 200
        raw = StageRawData(
            groups=groups,
            sources={name: {"kind": "fixture"} for name in sequence.task_names},
        )
        current = CurrentTaskData(
            group=groups[1],
            source={"kind": "fixture"},
            collected_configurations=[
                {"configuration_index": 30 + i} for i in range(5)
            ],
            real_encode_clip_fraction=0.125,
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            context = StageContext(
                index=1,
                task=sequence.task(1),
                trajectory_count=5,
                directory=directory,
                policy_before_hash="policy-before",
                diffusion_before_hash="diffusion-before",
            )
            data = trainer._prepare_stage_data(context, raw, current)
            self.assertEqual(
                sorted(path.name for path in directory.glob("*.npy")),
                ["data_task_0.npy", "data_task_1.npy"],
            )
            for index, values in groups.items():
                reference = directory / f"reference_{index}.npy"
                np.save(reference, values.numpy(), allow_pickle=False)
                actual = directory / f"data_task_{index}.npy"
                self.assertEqual(actual.read_bytes(), reference.read_bytes())
                self.assertEqual(
                    raw.sources[sequence.task(index).task_name]["data_sha256"],
                    file_hash(actual),
                )
            self.assertEqual(
                data.splits,
                {
                    "reach-v3": {
                        "train_ids": [4, 0, 2, 3],
                        "validation_ids": [1],
                        "trajectories": 5,
                        "train_samples": 800,
                        "validation_samples": 200,
                    },
                    "hammer-v3": {
                        "train_ids": [1, 2, 0, 4],
                        "validation_ids": [3],
                        "trajectories": 5,
                        "train_samples": 800,
                        "validation_samples": 200,
                    },
                },
            )
            self.assertEqual(raw.sources["reach-v3"]["bc_target_clip_fraction"], 0.5)
            self.assertEqual(raw.sources["hammer-v3"]["bc_target_clip_fraction"], 0.75)
            self.assertEqual(
                trainer.train_configurations,
                {
                    "hammer-v3": {
                        "indices": [31, 32, 30, 34],
                        "source_trajectory_ids": [1, 2, 0, 4],
                    }
                },
            )

    def test_group_validation_preserves_existing_error(self):
        with self.assertRaisesRegex(ValueError, "Invalid physical stage data"):
            validate_trajectory_group(torch.zeros(5, 199, 43), count=5)


if __name__ == "__main__":
    unittest.main()
