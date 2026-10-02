"""Training entry points consume typed YAML settings and persist runtime overrides."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset

import train_continual_sac
import train_diffcrl
import train_sac
from src.config import config_to_dict, default_config, load_config
from src.continual import sequential_sac
from src.continual.bc_policy import GeneralPolicy, fit_bc
from src.continual.trajectory_diffusion import DiffusionConfig, TrajectoryDiffusion


class TrainingConfigTests(unittest.TestCase):
    def test_continual_sac_requires_explicit_config(self):
        with self.assertRaises(SystemExit) as error:
            train_continual_sac.parse_args([])
        self.assertEqual(error.exception.code, 2)

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
                    net_arch=[16, 8],
                    checkpoint_freq=3,
                ),
                evaluation=replace(
                    c.evaluation, frequency=4, episodes=2, seed_offset=123
                ),
                runtime=replace(c.runtime, output=tmp),
            )
            path = Path(tmp) / "input.yaml"
            path.write_text(yaml.safe_dump(config_to_dict(c)))
            with patch.object(train_sac, "SAC") as sac, patch.object(
                train_sac, "make_env"
            ) as env, patch.object(
                train_sac, "Monitor", side_effect=lambda e: e
            ), patch.object(train_sac, "CallbackList"), patch.object(
                train_sac, "FixedSeedEvalCallback"
            ) as evaluation, patch.object(
                train_sac, "CheckpointCallback"
            ) as checkpoint:
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
                self.assertEqual(kwargs["policy_kwargs"], {"net_arch": [16, 8]})
                self.assertEqual((kwargs["seed"], kwargs["device"]), (3, "cpu"))
                self.assertEqual(kwargs["verbose"], 0)
                self.assertEqual(
                    sac.return_value.learn.call_args.kwargs["total_timesteps"], 7
                )
                self.assertTrue(sac.return_value.learn.call_args.kwargs["progress_bar"])
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
                    Path(tmp) / "trial/best_success_model",
                )
                self.assertEqual(checkpoint.call_args.kwargs["save_freq"], 3)
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

    def test_continual_sac_wires_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = default_config("continual_sac")
            c = replace(
                c,
                sac=replace(
                    c.sac, learning_rate=0.002, gamma=0.9, batch_size=7, net_arch=[8, 8]
                ),
                continual=replace(c.continual, tasks=["reach-v3"], steps_per_task=3),
                evaluation=replace(c.evaluation, episodes=1),
                runtime=replace(
                    c.runtime, output=str(Path(tmp) / "run"), device="cpu", seed=4
                ),
            )
            model = MagicMock()
            model.num_timesteps = 0
            model.observation_space.shape = (39,)
            model.action_space.shape = (4,)
            model.policy.parameters.return_value = [torch.tensor([1.0])]
            model.log_ent_coef = torch.tensor([0.0])
            model.learning_starts = c.sac.learning_starts
            model.replay_buffer.size.return_value = 3

            def learn(**kwargs):
                model.num_timesteps += kwargs["total_timesteps"]

            def save(path):
                Path(path).write_bytes(b"checkpoint")

            model.learn.side_effect = learn
            model.save.side_effect = save
            with patch.object(
                sequential_sac, "SAC", return_value=model
            ) as sac, patch.object(
                sequential_sac, "make_metaworld_env"
            ), patch.object(
                sequential_sac, "Monitor", side_effect=lambda e: e
            ), patch.object(
                sequential_sac, "evaluate_stage"
            ) as evaluation:
                train_continual_sac.run(c)
                kwargs = sac.call_args.kwargs
                self.assertEqual(
                    (kwargs["gamma"], kwargs["learning_rate"], kwargs["batch_size"]),
                    (0.9, 0.002, 7),
                )
                self.assertEqual(kwargs["policy_kwargs"], {"net_arch": [8, 8]})
                self.assertEqual((kwargs["seed"], kwargs["device"]), (4, "cpu"))
                self.assertTrue(model.learn.call_args.kwargs["progress_bar"])
                self.assertEqual(
                    evaluation.call_args.kwargs["evaluation_mode"], "sampled"
                )
                self.assertEqual(
                    evaluation.call_args.kwargs["reward_function_version"], "v2"
                )
                self.assertTrue(evaluation.call_args.kwargs["progress"])
                evaluation.assert_called_once()
            resolved = load_config(Path(tmp) / "run/config.yaml")
            self.assertEqual(resolved.runtime.seed, 4)
            self.assertEqual(resolved.continual.steps_per_task, 3)

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
    def test_cuda_diffcrl_execution_and_gradients(self):
        torch.manual_seed(0)
        model = TrajectoryDiffusion(
            DiffusionConfig(horizon=4, steps=2, width=8, num_tasks=1)
        )
        model.denoiser.to("cuda")
        data = torch.randn(2, 4, 22)
        ids = torch.zeros(2, dtype=torch.long)
        goals = torch.randn(2, 3)
        loss = model.loss(
            data, generator=torch.Generator().manual_seed(1), task_ids=ids, goals=goals
        )
        loss.backward()
        self.assertEqual(next(model.denoiser.parameters()).grad.device.type, "cuda")
        sample = model.sample(2, task_ids=ids, goals=goals)
        self.assertEqual((sample.device.type, sample.shape), ("cpu", (2, 4, 22)))
        policy = GeneralPolicy(-np.ones(4), np.ones(4), hidden_sizes=(8, 8)).to("cuda")
        dataset = TensorDataset(torch.randn(4, 39), torch.zeros(4, 4))
        result = fit_bc(policy, dataset, dataset, epochs=1, batch_size=2)
        self.assertTrue(np.isfinite(result["train_loss"]))
        self.assertEqual(policy.act(np.zeros(39)).shape, (4,))


if __name__ == "__main__":
    unittest.main()
