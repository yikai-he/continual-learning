"""Training entry points consume typed YAML settings and persist runtime overrides."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

import train_diffcrl
import train_sac
from src.config import config_from_dict, config_to_dict, default_config, load_config
from src.continual.bc_policy import GeneralPolicy, fit_bc
from src.continual.trajectory_diffusion import (
    DiffusionConfig,
    FeatureNormalizer,
    TrajectoryDiffusion,
)
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig,
    TrajectoryFlowMatching,
)
from src.support.expert_manifest import validate_expert_checkpoint


class TrainingConfigTests(unittest.TestCase):
    def test_six_cw10_expert_configs_share_the_formal_sac_protocol(self):
        expected = (
            ("push_wall_v3.yaml", "push-wall-v3"),
            ("handle_press_side_v3.yaml", "handle-press-side-v3"),
            ("push_v3.yaml", "push-v3"),
            ("shelf_place_v3.yaml", "shelf-place-v3"),
            ("window_close_v3.yaml", "window-close-v3"),
            ("peg_unplug_side_v3.yaml", "peg-unplug-side-v3"),
        )
        for filename, task in expected:
            config = load_config(
                Path("configs/sac/cw10") / filename,
                expected_experiment="sac",
            )
            self.assertEqual(config.continual.tasks, [task])
            self.assertEqual(config.environment.backend, "metaworld-v3")
            self.assertEqual(config.environment.reward_function_version, "v2")
            self.assertEqual(config.sac.steps, 1_000_000)
            self.assertEqual(config.sac.learning_rate, 1e-3)
            self.assertEqual(config.sac.batch_size, 128)
            self.assertEqual(config.sac.buffer_size, 1_000_000)
            self.assertEqual(config.sac.learning_starts, 1_000)
            self.assertEqual(config.sac.random_exploration_steps, 10_000)
            self.assertEqual(config.sac.train_freq, 50)
            self.assertEqual(config.sac.gradient_steps, 50)
            self.assertEqual(config.evaluation.frequency, 40_000)
            self.assertEqual(config.evaluation.mode, "fixed-tasks")
            self.assertEqual(config.runtime.seed, 0)
            self.assertEqual(
                config.runtime.output, "runs/cw10_experts/stable_tanh_v2"
            )

    def test_sac_defaults_to_low_disk_with_separate_final_banks(self):
        config = default_config("sac")
        self.assertEqual(config.evaluation.task_set_seed, 10_000)
        self.assertEqual(config.evaluation.final_task_set_seeds, [20_000, 20_001])
        self.assertEqual(config.evaluation.final_reset_seed, 30_000)

    def test_sac_final_evaluation_uses_independent_banks_and_reset_seeds(self):
        config = default_config("sac")
        config = replace(
            config,
            evaluation=replace(
                config.evaluation,
                episodes=2,
                final_episodes=3,
                final_task_set_seeds=[20_000, 20_001],
                final_reset_seed=30_000,
            ),
        )
        with patch.object(
            train_sac, "evaluate_task", side_effect=[{"bank": 0}, {"bank": 1}]
        ) as evaluate:
            model = MagicMock()
            model.observation_space.shape = (39,)
            model.action_space.shape = (4,)
            self.assertEqual(
                train_sac.final_test(config, model),
                [{"bank": 0}, {"bank": 1}],
            )
        self.assertEqual(
            [call.kwargs["task_set_seed"] for call in evaluate.call_args_list],
            [20_000, 20_001],
        )
        self.assertEqual(
            [call.kwargs["seed"] for call in evaluate.call_args_list],
            [30_000, 30_003],
        )
        self.assertEqual(
            [call.kwargs["episodes"] for call in evaluate.call_args_list],
            [3, 3],
        )
        self.assertTrue(
            all(
                call.kwargs["evaluation_mode"] == "fixed-tasks"
                for call in evaluate.call_args_list
            )
        )

    def test_sac_early_stopping_defaults_disabled_and_parses_optional_block(self):
        default = default_config("sac")
        self.assertFalse(default.sac.early_stopping.enabled)
        values = config_to_dict(default)
        values["sac"]["early_stopping"] = {
            "enabled": True,
            "min_steps": 100_000,
            "success_threshold": 0.95,
            "patience": 3,
        }
        parsed = config_from_dict(values, expected_experiment="sac")
        self.assertTrue(parsed.sac.early_stopping.enabled)
        self.assertEqual(parsed.sac.early_stopping.min_steps, 100_000)

    def test_diffcrl_entrypoint_passes_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.yaml"
            path.write_text(
                yaml.safe_dump(
                    {"experiment": "diffcrl", "runtime": {"output": tmp, "seed": 9}}
                )
            )
            with patch.object(train_diffcrl, "DiffCRLTrainer") as trainer:
                train_diffcrl.main(
                    [
                        "--config",
                        str(path),
                        "--seed",
                        "0",
                        "--device",
                        "cpu",
                        "--run-name",
                        "trial",
                    ]
                )
                config = trainer.call_args.args[0]
                self.assertEqual(
                    (
                        config.runtime.seed,
                        config.runtime.device,
                        config.runtime.run_name,
                    ),
                    (0, "cpu", "trial"),
                )
                trainer.return_value.run.assert_called_once_with()

    def test_sac_wires_hyperparameters_and_saves_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = default_config("sac")
            c = replace(
                c,
                sac=replace(
                    c.sac,
                    steps=7,
                    learning_rate=0.002,
                    gamma=0.95,
                    batch_size=17,
                ),
                evaluation=replace(
                    c.evaluation, frequency=4, episodes=2, seed_offset=123
                ),
                runtime=replace(c.runtime, output=tmp),
            )
            path = Path(tmp) / "input.yaml"
            path.write_text(yaml.safe_dump(config_to_dict(c)))
            with patch.object(train_sac, "LegacyScheduleSAC") as sac, patch.object(
                train_sac.SAC, "load"
            ) as load, patch.object(
                train_sac, "make_env"
            ) as env, patch.object(
                train_sac, "Monitor", side_effect=lambda e: e
            ), patch.object(train_sac, "FixedSeedEvalCallback") as evaluation, patch.object(
                train_sac, "final_test", return_value=[{"success_rate": 1.0}]
            ), patch.object(
                train_sac,
                "initialize_sac_actor",
                return_value={"schema_version": 1, "mode": "scratch"},
            ):
                sac.return_value.num_timesteps = 7
                load.return_value = sac.return_value
                env.return_value.action_space.shape = (4,)
                evaluation.return_value.results = [[1.0, 2.0]]
                evaluation.return_value.successes = [[True, False]]
                evaluation.return_value.return_stddevs = [0.5]
                evaluation.return_value.task_hashes = ("a", "b")

                def learn(**_kwargs):
                    selected = (
                        Path(tmp)
                        / "trial/best_model.zip"
                    )
                    selected.write_bytes(b"selected")

                sac.return_value.learn.side_effect = learn
                train_sac.main(
                    [
                        "--config",
                        str(path),
                        "--seed",
                        "3",
                        "--device",
                        "cpu",
                        "--run-name",
                        "trial",
                    ]
                )
                kwargs = sac.call_args.kwargs
                self.assertEqual(
                    (kwargs["gamma"], kwargs["learning_rate"], kwargs["batch_size"]),
                    (0.95, 0.002, 17),
                )
                from src.sac_policy import LegacySACPolicy

                self.assertIs(sac.call_args.args[0], LegacySACPolicy)
                self.assertNotIn("policy_kwargs", kwargs)
                self.assertEqual((kwargs["seed"], kwargs["device"]), (3, "cpu"))
                self.assertEqual(kwargs["verbose"], 0)
                self.assertEqual(kwargs["random_exploration_steps"], 10_000)
                self.assertEqual(kwargs["update_interval"], 50)
                self.assertEqual(kwargs["gradient_steps"], 50)
                self.assertAlmostEqual(kwargs["target_entropy"], -4.000721504181298)
                self.assertEqual(
                    sac.return_value.learn.call_args.kwargs["total_timesteps"], 7
                )
                self.assertFalse(sac.return_value.learn.call_args.kwargs["progress_bar"])
                self.assertEqual(
                    [call.args[2] for call in env.call_args_list], [3, 126]
                )
                self.assertEqual(
                    [call.args[0] for call in env.call_args_list],
                    ["metaworld-v3", "metaworld-v3"],
                )
                self.assertEqual(
                    [
                        call.kwargs["reward_function_version"]
                        for call in env.call_args_list
                    ],
                    ["v2", "v2"],
                )
                self.assertEqual(evaluation.call_args.kwargs["eval_freq"], 4)
                self.assertEqual(evaluation.call_args.kwargs["seed"], 126)
                self.assertEqual(
                    evaluation.call_args.kwargs["best_success_model_save_path"],
                    Path(tmp) / "trial",
                )
                self.assertIs(
                    sac.return_value.learn.call_args.kwargs["callback"],
                    evaluation.return_value,
                )
                self.assertNotIn(
                    "reset_num_timesteps", sac.return_value.learn.call_args.kwargs
                )
            resolved = load_config(Path(tmp) / "trial/config.yaml")
            self.assertEqual(
                (
                    resolved.runtime.seed,
                    resolved.runtime.device,
                    resolved.runtime.run_name,
                ),
                (3, "cpu", "trial"),
            )
            self.assertEqual(resolved.runtime.run_dir, str(Path(tmp) / "trial"))
            results = json.loads((Path(tmp) / "trial/results.json").read_text())
            self.assertEqual(
                results["selected_expert"]["path"],
                "best_model.zip",
            )
            self.assertEqual(results["final_evaluation"], [{"success_rate": 1.0}])
            self.assertFalse((Path(tmp) / "trial/config.json").exists())
            selected = Path(tmp) / "trial/best_model.zip"
            identity = validate_expert_checkpoint(
                selected,
                expected_task="reach-v3",
                expected_backend="metaworld-v3",
                expected_reward_function_version="v2",
                expected_observation_shape=(39,),
                expected_action_shape=(4,),
                expected_horizon=200,
                allow_legacy=False,
            )
            self.assertFalse(identity.legacy)
            for removed in (
                "evaluation_history.csv",
                "run_summary.json",
                "EXPERIMENT_SUMMARY.md",
            ):
                self.assertFalse((Path(tmp) / "trial" / removed).exists())
            self.assertFalse((Path(tmp) / "trial/final_model.zip.manifest.json").exists())
            self.assertFalse((Path(tmp) / "trial/final_model.zip").exists())
            self.assertFalse((Path(tmp) / "trial/checkpoints").exists())

    def test_sac_directory_collision_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = default_config("sac")
            c = replace(c, runtime=replace(c.runtime, output=tmp))
            first = train_sac.create_run_directory(c)
            second = train_sac.create_run_directory(c)
            self.assertEqual(first, Path(tmp) / "reward_v2/reach-v3/seed_0")
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, second.parent)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    @pytest.mark.integration
    def test_cuda_diffcrl_execution_and_gradients(self):
        torch.manual_seed(0)
        model = TrajectoryDiffusion(
            DiffusionConfig(horizon=4, steps=2, width=8, num_tasks=1)
        )
        model.denoiser.to("cuda")
        data = torch.randn(2, 4, 43)
        ids = torch.zeros(2, dtype=torch.long)
        goals = torch.randn(2, 3)
        loss = model.loss(
            data, generator=torch.Generator().manual_seed(1), task_ids=ids, goals=goals
        )
        loss.backward()
        self.assertEqual(next(model.denoiser.parameters()).grad.device.type, "cuda")
        sample = model.sample(2, task_ids=ids, goals=goals)
        self.assertEqual((sample.device.type, sample.shape), ("cpu", (2, 4, 43)))
        policy = GeneralPolicy(-np.ones(4), np.ones(4)).to("cuda")
        dataset = TensorDataset(torch.randn(4, 39), torch.zeros(4, 4))
        result = fit_bc(policy, dataset, dataset, epochs=1, batch_size=2)
        self.assertTrue(np.isfinite(result["train_loss"]))
        self.assertEqual(policy.act(np.zeros(39)).shape, (4,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    @pytest.mark.integration
    def test_cuda_flow_matching_backbones_training_checkpoint_and_sampling(self):
        for backbone, width in (("temporal_cnn", 64), ("temporal_unet1d", 32)):
            with self.subTest(backbone=backbone), tempfile.TemporaryDirectory() as tmp:
                config = FlowMatchingConfig(
                    horizon=200,
                    width=width,
                    num_tasks=4,
                    integration_steps=10,
                    backbone=backbone,
                )
                model = TrajectoryFlowMatching(config)
                model.velocity_model.to("cuda")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                data = torch.randn(2, 200, 43)
                task_ids = torch.tensor([0, 1])
                goals = torch.randn(2, 3)
                optimizer.zero_grad()
                loss = model.loss(
                    data,
                    generator=torch.Generator().manual_seed(1),
                    task_ids=task_ids,
                    goals=goals,
                )
                loss.backward()
                optimizer.step()
                self.assertEqual(next(model.parameters()).device.type, "cuda")
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        and parameter.grad.device.type == "cuda"
                        for parameter in model.parameters()
                    )
                )
                sample = model.sample(2, seed=2, task_ids=task_ids, goals=goals)
                self.assertEqual(
                    (sample.device.type, sample.dtype, sample.shape),
                    ("cpu", torch.float32, (2, 200, 43)),
                )
                self.assertEqual(model.last_sample_nfe, 10)

                path = Path(tmp) / "flow.pt"
                model.save(
                    path,
                    FeatureNormalizer(torch.zeros(43), torch.ones(43)),
                    FeatureNormalizer(torch.zeros(3), torch.ones(3)),
                    {
                        "task_order": ["a", "b", "c", "d"],
                        "training_banks": {},
                        "train_configurations": {},
                        "generated_action_projection": "none",
                        "action_space": "raw",
                        "action_clamp_epsilon": 1e-4,
                    },
                )
                loaded, _, _, _ = TrajectoryFlowMatching.load(
                    path, expected_config=config
                )
                loaded.velocity_model.to("cuda")
                loaded_sample = loaded.sample(
                    2, seed=2, task_ids=task_ids, goals=goals
                )
                self.assertEqual(
                    (loaded_sample.device.type, loaded_sample.shape),
                    ("cpu", (2, 200, 43)),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    @pytest.mark.integration
    def test_cuda_diffusion_unet_training_and_sampling(self):
        model = TrajectoryDiffusion(
            DiffusionConfig(
                horizon=4,
                steps=2,
                backbone="temporal_unet1d",
                width=4,
                num_tasks=1,
            )
        )
        model.denoiser.to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = torch.randn(1, 4, 43)
        task_ids = torch.zeros(1, dtype=torch.long)
        goals = torch.zeros(1, 3)
        optimizer.zero_grad()
        loss = model.loss(
            data,
            generator=torch.Generator().manual_seed(1),
            task_ids=task_ids,
            goals=goals,
        )
        loss.backward()
        optimizer.step()
        self.assertEqual(next(model.parameters()).device.type, "cuda")
        sample = model.sample(
            1,
            seed=2,
            task_ids=task_ids,
            goals=goals,
            execution_mode="device-native",
        )
        self.assertEqual((sample.device.type, sample.shape), ("cpu", (1, 4, 43)))


if __name__ == "__main__":
    unittest.main()
