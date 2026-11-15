from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_replay_comparison import comparison_configs, run_comparison
from src.config import default_config


def test_comparison_arms_are_matched_and_isolated(tmp_path):
    base = default_config("diffcrl")
    base = replace(
        base,
        evaluation=replace(base.evaluation, episodes=50, mode="fixed-tasks"),
        continual=replace(
            base.continual,
            replay_mode="diffusion",
            replay_allocation="fixed-total",
            replay_budget=50,
        ),
    )
    configs = comparison_configs(base, tmp_path)

    assert set(configs) == {"no_replay", "expert_recollection", "ddpm_replay"}
    assert len({cfg.runtime.run_name for cfg in configs.values()}) == 3
    shared = configs["ddpm_replay"]
    for cfg in configs.values():
        assert cfg.bc == shared.bc
        assert cfg.evaluation == shared.evaluation
        assert cfg.continual.tasks == shared.continual.tasks
        assert cfg.continual.trajectories_per_task == 200
    assert configs["expert_recollection"].continual.replay_mode == "expert-recollection"
    assert configs["expert_recollection"].continual.replay_budget == 50
    assert configs["ddpm_replay"].continual.replay_budget == 50
    assert not configs["ddpm_replay"].continual.historical_replay_training_only
    assert not configs["expert_recollection"].continual.historical_replay_training_only
    assert not configs["no_replay"].continual.historical_replay_training_only


def test_comparison_arms_inherit_base_replay_budget(tmp_path):
    base = default_config("diffcrl")
    base = replace(
        base,
        evaluation=replace(base.evaluation, episodes=50, mode="fixed-tasks"),
        continual=replace(
            base.continual,
            replay_mode="diffusion",
            replay_allocation="fixed-total",
            replay_budget=25,
        ),
    )

    configs = comparison_configs(base, tmp_path)

    assert configs["expert_recollection"].continual.replay_budget == 25
    assert configs["ddpm_replay"].continual.replay_budget == 25
    assert configs["no_replay"].continual.replay_mode == "none"


def test_comparison_interleaves_each_stage_once_then_finalizes_each_arm(tmp_path):
    base = default_config("diffcrl")
    base = replace(
        base,
        evaluation=replace(base.evaluation, episodes=50, mode="fixed-tasks"),
        continual=replace(
            base.continual,
            tasks=base.continual.tasks[:2],
            replay_mode="diffusion",
            replay_allocation="fixed-total",
        ),
    )
    events = []

    class FakePolicy:
        def to(self, _device):
            return self

    class FakeMatrix:
        cells = [
            [{"mean_return": 1.0}],
            [{"mean_return": 1.0}, {"mean_return": 2.0}],
        ]

        @staticmethod
        def to_dict():
            return {"metrics_by_stage": {"1": {"average_performance": 0.5}}}

    class FakeTrainer:
        instances = []

        def __init__(self, config, *, progress):
            del progress
            self.name = config.runtime.run_name
            self.config = config
            self.output = Path(config.runtime.output) / self.name
            self.output.mkdir()
            self.sequence = SimpleNamespace(
                num_tasks=2,
                task=lambda stage: SimpleNamespace(
                    task_name=config.continual.tasks[stage]
                ),
            )
            self.policy = FakePolicy()
            self.policy_optimizer = object()
            self.matrix = FakeMatrix()
            self.trained = []
            self.finalized = 0
            self.instances.append(self)

        def train_task(self, stage):
            self.trained.append(stage)
            events.append(("train", stage, self.name))

        def finalize_run(self, *, process_started):
            assert process_started > 0
            self.finalized += 1
            events.append(("finalize", self.name))
            (self.output / "evaluation_matrix.json").write_text("{}\n")
            (self.output / "timing_summary.json").write_text("{}\n")
            (self.output / "RUN_COMPLETE").write_text("complete\n")
            return self.matrix

    def current_hash(trainer, stage):
        assert all(
            candidate.trained == list(range(stage + 1))
            for candidate in FakeTrainer.instances
        )
        events.append(("compare-data", stage, trainer.name))
        return f"shared-{stage}"

    def stage_report(trainer, stage):
        assert stage == 1
        assert trainer.trained == [0, 1]
        return {
            "splits": {
                base.continual.tasks[0]: {
                    "trajectories": 2,
                    "train_ids": [0],
                    "validation_ids": [1],
                },
                base.continual.tasks[1]: {
                    "trajectories": 2,
                    "train_ids": [0],
                    "validation_ids": [1],
                },
            }
        }

    with patch("run_replay_comparison.DiffCRLTrainer", FakeTrainer), patch(
        "run_replay_comparison._current_data_hash", side_effect=current_hash
    ), patch("run_replay_comparison._stage_report", side_effect=stage_report), patch(
        "run_replay_comparison.state_hash", return_value="shared-policy"
    ), patch(
        "run_replay_comparison.GeneralPolicy.load_training",
        return_value=(FakePolicy(), object(), {"stage": 0}),
    ):
        summary = run_comparison(base, tmp_path, progress=False)

    assert [trainer.trained for trainer in FakeTrainer.instances] == [[0, 1]] * 3
    assert [trainer.finalized for trainer in FakeTrainer.instances] == [1, 1, 1]
    assert events[:3] == [
        ("train", 0, "no_replay"),
        ("train", 0, "expert_recollection"),
        ("train", 0, "ddpm_replay"),
    ]
    first_finalization = next(
        index for index, event in enumerate(events) if event[0] == "finalize"
    )
    assert all(event[0] != "train" for event in events[first_finalization:])
    assert summary["shared_stage_zero_policy_state_sha256"] == "shared-policy"
    assert (tmp_path / "comparison_summary.json").is_file()
    for trainer in FakeTrainer.instances:
        assert (trainer.output / "evaluation_matrix.json").is_file()
        assert (trainer.output / "timing_summary.json").is_file()
        assert (trainer.output / "RUN_COMPLETE").is_file()
