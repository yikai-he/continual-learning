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
from src.continual.stage_dataset import (
    bc_dataset,
    concatenate_trajectory_groups,
    prepare_stage_groups,
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
    def test_stateless_stage_group_preparation_matches_exact_legacy_products(self):
        groups = fixture_groups()
        sources = {
            "reach-v3": {"kind": "fixture"},
            "hammer-v3": {"kind": "fixture"},
        }
        sources_before = {name: dict(value) for name, value in sources.items()}

        result = prepare_stage_groups(
            groups,
            sources,
            ["reach-v3", "hammer-v3"],
            stage=1,
            horizon=200,
            runtime_seed=17,
            action_low=-torch.ones(4),
            action_high=torch.ones(4),
            clamp_epsilon=1e-4,
            historical_replay_training_only=False,
        )

        self.assertEqual(list(result.preparations), [0, 1])
        self.assertEqual(list(result.splits), ["reach-v3", "hammer-v3"])
        self.assertEqual(result.preparations[0].train_ids, [4, 0, 2, 3])
        self.assertEqual(result.preparations[1].train_ids, [1, 2, 0, 4])
        self.assertEqual(result.preparations[0].validation_ids, [1])
        self.assertEqual(result.preparations[1].validation_ids, [3])
        self.assertEqual(sources, sources_before)
        expected_train = torch.cat(
            [groups[index][result.preparations[index].train_ids] for index in groups]
        )
        expected_validation = torch.cat(
            [
                groups[index][result.preparations[index].validation_ids]
                for index in groups
            ]
        )
        torch.testing.assert_close(result.dataset.train, expected_train, rtol=0, atol=0)
        torch.testing.assert_close(
            result.dataset.validation, expected_validation, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.dataset.train_labels, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        )
        torch.testing.assert_close(
            result.dataset.validation_labels, torch.tensor([0, 1])
        )
        train_bc = bc_dataset(result.dataset.train).tensors
        expected_bc = bc_dataset(expected_train).tensors
        for actual, expected in zip(train_bc, expected_bc):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_historical_training_only_preserves_order_and_validation_contract(self):
        groups = fixture_groups()
        sources = {
            "reach-v3": {"kind": "fixture"},
            "hammer-v3": {"kind": "fixture"},
        }
        result = prepare_stage_groups(
            groups,
            sources,
            ["reach-v3", "hammer-v3"],
            stage=1,
            horizon=200,
            runtime_seed=17,
            action_low=-torch.ones(4),
            action_high=torch.ones(4),
            clamp_epsilon=1e-4,
            historical_replay_training_only=True,
        )
        self.assertEqual(result.preparations[0].train_ids, [0, 1, 2, 3, 4])
        self.assertEqual(result.preparations[0].validation_ids, [])
        self.assertEqual(result.preparations[1].validation_ids, [3])
        self.assertEqual(result.splits["reach-v3"]["train_samples"], 1000)
        self.assertEqual(result.splits["reach-v3"]["validation_samples"], 0)

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

    def test_single_allocated_replay_trajectory_is_training_only(self):
        result = prepare_trajectory_group(
            fixture_groups()[0][:1],
            count=1,
            seed=17,
            action_low=-torch.ones(4),
            action_high=torch.ones(4),
            clamp_epsilon=1e-4,
        )
        self.assertEqual(result.train_ids, [0])
        self.assertEqual(result.validation_ids, [])

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

    def test_stage_preparation_accepts_different_group_sizes(self):
        groups = fixture_groups()
        groups[0] = groups[0][:3]
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
            real_encode_clip_fraction=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            data = trainer._prepare_stage_data(
                StageContext(
                    index=1,
                    task=sequence.task(1),
                    trajectory_count=5,
                    directory=Path(tmp),
                    policy_before_hash=None,
                    diffusion_before_hash=None,
                ),
                raw,
                current,
            )
        self.assertEqual(data.splits["reach-v3"]["trajectories"], 3)
        self.assertEqual(data.splits["hammer-v3"]["trajectories"], 5)

    def test_later_invalid_group_preserves_earlier_partial_artifact(self):
        groups = fixture_groups()
        groups[1] = groups[1][:, :-1]
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
                {"configuration_index": 30 + index} for index in range(5)
            ],
            real_encode_clip_fraction=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            context = StageContext(
                index=1,
                task=sequence.task(1),
                trajectory_count=5,
                directory=directory,
                policy_before_hash=None,
                diffusion_before_hash=None,
            )
            with self.assertRaisesRegex(ValueError, "Invalid physical stage data"):
                trainer._prepare_stage_data(context, raw, current)
            self.assertTrue((directory / "data_task_0.npy").is_file())
            self.assertFalse((directory / "data_task_1.npy").exists())
            self.assertIn("data_sha256", raw.sources["reach-v3"])
            self.assertNotIn("data_sha256", raw.sources["hammer-v3"])


if __name__ == "__main__":
    unittest.main()
