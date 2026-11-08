"""Configuration defaults, strict input validation, overrides, and persistence."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from src.config import (
    BC_HIDDEN_SIZES,
    FLOW_CHECKPOINT_SELECTION,
    FLOW_PROBABILITY_PATH,
    FLOW_RESIDUAL_BLOCKS,
    FLOW_SOURCE_DISTRIBUTION,
    FLOW_TIME_EMBEDDING,
    GENERATOR_GRADIENT_CLIP_NORM,
    GENERATOR_VALIDATION_DRAWS,
    REPLAY_PRIORITY_STRATEGY,
    SAC_ENT_COEF,
    SAC_NET_ARCH,
    config_from_dict,
    config_to_dict,
    load_config,
    output_directory,
    parse_config,
    resolve_device,
    save_resolved_config,
    validate_config,
)
from src.continual.resume import _scientific_config
from src.continual.trajectory_flow_matching import (
    FlowMatchingConfig as FlowMatchingModelConfig,
)

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def config(self, **sections):
        return config_from_dict(
            {"experiment": "diffcrl", "runtime": {"output": "/tmp/run"}, **sections}
        )

    def test_presets_preserve_distinct_defaults(self):
        diff = load_config(ROOT / "configs/diffcrl/diffcrl.yaml")
        self.assertEqual(
            (diff.continual.tasks, diff.continual.trajectories_per_task),
            (
                [
                    "button-press-v3",
                    "faucet-close-v3",
                    "hammer-v3",
                    "handle-press-side-v3",
                    "window-close-v3",
                ],
                200,
            ),
        )
        self.assertEqual(
            (diff.diffusion.epochs, diff.diffusion.steps, diff.diffusion.width),
            (300, 100, 64),
        )
        self.assertEqual(
            (diff.diffusion.batch_size, diff.diffusion.learning_rate), (8, 0.001)
        )
        self.assertEqual(
            (diff.bc.epochs, diff.bc.batch_size, diff.bc.learning_rate),
            (100, 256, 0.001),
        )
        self.assertEqual(
            (diff.evaluation.mode, diff.evaluation.episodes, diff.runtime.device),
            ("fixed-tasks", 50, "cpu"),
        )
        sac = load_config(ROOT / "configs/sac/sac.yaml")
        self.assertEqual(
            (sac.sac.steps, sac.sac.learning_rate, sac.sac.batch_size),
            (1000000, 0.001, 128),
        )
        self.assertEqual(SAC_NET_ARCH, (256,) * 4)
        self.assertEqual(SAC_ENT_COEF, "auto")
        self.assertEqual(
            (sac.evaluation.frequency, sac.evaluation.episodes, sac.runtime.device),
            (20000, 10, "auto"),
        )
        seq = load_config(ROOT / "configs/continual_sac/continual_sac.yaml")
        self.assertEqual(
            (seq.sac.learning_rate, seq.sac.batch_size),
            (0.001, 128),
        )
        self.assertEqual(
            (seq.continual.steps_per_task, seq.continual.boundary_warmup_steps),
            (1000000, 10000),
        )
        self.assertEqual(
            (seq.evaluation.mode, seq.evaluation.episodes, seq.runtime.device),
            ("fixed-tasks", 50, "cpu"),
        )
        for config in (diff, sac, seq):
            self.assertEqual(
                (config.sac.gamma, config.sac.tau, config.sac.learning_starts),
                (0.99, 0.005, 10000),
            )

    def test_all_maintained_presets_load_and_smokes_are_small(self):
        paths = {
            "sac": "configs/sac/sac.yaml",
            "sac_smoke": "configs/sac/sac_smoke.yaml",
            "diffcrl": "configs/diffcrl/diffcrl.yaml",
            "diffcrl_smoke": "configs/diffcrl/diffcrl_smoke.yaml",
            "no_replay": "configs/diffcrl/diffcrl_no_replay.yaml",
            "no_replay_smoke": "configs/diffcrl/diffcrl_no_replay_smoke.yaml",
            "continual_sac": "configs/continual_sac/continual_sac.yaml",
            "continual_sac_smoke": "configs/continual_sac/continual_sac_smoke.yaml",
        }
        loaded = {name: load_config(ROOT / path) for name, path in paths.items()}
        self.assertLess(loaded["sac_smoke"].sac.steps, loaded["sac"].sac.steps)
        self.assertLess(
            loaded["diffcrl_smoke"].continual.trajectories_per_task,
            loaded["diffcrl"].continual.trajectories_per_task,
        )
        self.assertLess(
            loaded["no_replay_smoke"].bc.epochs, loaded["no_replay"].bc.epochs
        )
        self.assertLess(
            loaded["continual_sac_smoke"].continual.steps_per_task,
            loaded["continual_sac"].continual.steps_per_task,
        )
        self.assertEqual(loaded["diffcrl"].continual.replay_mode, "diffusion")
        self.assertEqual(loaded["no_replay"].continual.replay_mode, "none")
        self.assertEqual(loaded["continual_sac"].experiment, "continual_sac")

    def test_reward_version_default_validation_and_persistence(self):
        self.assertEqual(self.config().environment.reward_function_version, "v2")
        for version in ("v1", "v2"):
            config = self.config(environment={"reward_function_version": version})
            with tempfile.TemporaryDirectory() as tmp:
                save_resolved_config(config, Path(tmp))
                self.assertEqual(
                    load_config(
                        Path(tmp) / "config.yaml"
                    ).environment.reward_function_version,
                    version,
                )
        for invalid in ("v4", "", 2, None):
            with self.subTest(version=invalid), self.assertRaisesRegex(
                ValueError, "environment.reward_function_version"
            ):
                self.config(environment={"reward_function_version": invalid})

    def test_optional_defaults_and_independent_instances(self):
        first, second = self.config(), self.config()
        self.assertEqual(first.bc.epochs, 100)
        self.assertEqual(second.bc.epochs, 100)

    def test_cli_overrides_yaml_including_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.yaml"
            path.write_text(
                "experiment: diffcrl\nruntime:\n  output: runs/example\n  seed: 9\n  device: auto\n  run_name: yaml-name\n"
            )
            config = parse_config(
                "diffcrl",
                "test",
                [
                    "--config",
                    str(path),
                    "--seed",
                    "0",
                    "--device",
                    "cpu",
                    "--run-name",
                    "cli-name",
                ],
            )
            self.assertEqual(
                (config.runtime.seed, config.runtime.device, config.runtime.run_name),
                (0, "cpu", "cli-name"),
            )
            preserved = parse_config(
                "diffcrl", "test", ["--config", str(path), "--device", "cpu"]
            )
            self.assertEqual(
                (preserved.runtime.seed, preserved.runtime.run_name), (9, "yaml-name")
            )
            self.assertEqual(
                output_directory(config), (Path.cwd() / "runs/example/cli-name")
            )

    def test_missing_unknown_and_wrong_types(self):
        cases = [
            ({}, "experiment"),
            ({"experiment": "diffcrl"}, "runtime.output"),
            ({"experiment": "other"}, "experiment"),
            ({"experiment": "diffcrl", "bc": {"epohs": 3}}, "bc.epohs"),
            ({"experiment": "diffcrl", "bc": None}, "bc"),
            ({"experiment": "diffcrl", "runtime": {"seed": True}}, "runtime.seed"),
            (
                {"experiment": "diffcrl", "diffusion": {"epochs": "3"}},
                "diffusion.epochs",
            ),
            (
                {"experiment": "diffcrl", "bc": {"learning_rate": float("nan")}},
                "bc.learning_rate",
            ),
            (
                {"experiment": "diffcrl", "continual": {"tasks": "reach-v3"}},
                "continual.tasks",
            ),
        ]
        for values, message in cases:
            with self.subTest(values=values), self.assertRaisesRegex(
                ValueError, message
            ):
                config_from_dict(values)

    def test_required_expert_mapping_and_no_default_path_replacement(self):
        with self.assertRaisesRegex(ValueError, "continual.experts.window-open-v3"):
            self.config(continual={"tasks": ["window-open-v3"]})
        c = self.config(
            continual={
                "tasks": ["button-press-v3"],
                "experts": {"button-press-v3": "expert.zip"},
            }
        )
        self.assertEqual(c.continual.experts, {"button-press-v3": "expert.zip"})
        self.assertEqual(
            self.config().continual.experts["button-press-v3"],
            "runs/experts/metaworld-v3/button-press-v3/policy.zip",
        )

    def test_removed_kuka_v2_backend_is_rejected(self):
        tasks = [
            "kuka-reach-v2",
            "kuka-push-v2",
            "kuka-hammer-v2",
            "kuka-handle-press-side-v2",
            "kuka-button-press-v2",
        ]
        experts = {task: f"/tmp/{task}.zip" for task in tasks}
        with self.assertRaisesRegex(ValueError, "environment.backend must be one of"):
            self.config(
                environment={"backend": "kuka-v2"},
                continual={"tasks": tasks, "experts": experts},
                evaluation={"mode": "sampled"},
            )

    def test_invalid_ranges_and_choices(self):
        cases = [
            ("bc", "epochs", 0),
            ("bc", "loss", "unknown"),
            ("diffusion", "width", 5),
            ("diffusion", "steps", 1),
            ("diffusion", "action_clamp_epsilon", 0),
            ("diffusion", "generated_action_projection", "invalid"),
            ("evaluation", "episodes", 51),
            ("continual", "trajectories_per_task", 1),
            ("continual", "tasks", []),
            ("continual", "tasks", ["reach-v3", "reach-v3"]),
            ("runtime", "device", "cdua"),
            ("runtime", "seed", -1),
            ("runtime", "run_name", "../escape"),
        ]
        for section, name, value in cases:
            values = config_to_dict(self.config())
            values[section][name] = value
            with self.subTest(field=f"{section}.{name}"), self.assertRaises(ValueError):
                config_from_dict(values)

    def test_removed_fixed_fields_are_not_public_configuration(self):
        fixed = {
            "sac": {"ent_coef": SAC_ENT_COEF, "net_arch": list(SAC_NET_ARCH)},
            "bc": {"hidden_sizes": list(BC_HIDDEN_SIZES), "normalization_mode": "layer-norm", "use_layer_norm": True},
            "flow_matching": {
                "residual_blocks": FLOW_RESIDUAL_BLOCKS,
                "time_embedding": FLOW_TIME_EMBEDDING,
                "gradient_clip_norm": GENERATOR_GRADIENT_CLIP_NORM,
                "validation_samples": GENERATOR_VALIDATION_DRAWS,
                "checkpoint_selection": FLOW_CHECKPOINT_SELECTION,
                "probability_path": FLOW_PROBABILITY_PATH,
                "source_distribution": FLOW_SOURCE_DISTRIBUTION,
            },
            "continual": {"replay_priority_strategy": REPLAY_PRIORITY_STRATEGY},
        }
        for section, fields in fixed.items():
            for name, value in fields.items():
                with self.subTest(field=f"{section}.{name}"), self.assertRaises(ValueError):
                    config_from_dict({"experiment": "diffcrl", section: {name: value}})

        legacy_values = {
            "experiment": "diffcrl",
            "runtime": {"output": "/tmp/run"},
            **{key: value for key, value in fixed.items() if key != "sac"},
        }
        legacy_values["continual"] = {
            **legacy_values["continual"],
            "replay_mode": "flow-matching",
        }
        legacy = config_from_dict(
            legacy_values,
            allow_legacy_fixed_fields=True,
        )
        self.assertEqual(
            legacy,
            config_from_dict(
                {"experiment": "diffcrl", "continual": {"replay_mode": "flow-matching"}, "flow_matching": {}, "runtime": {"output": "/tmp/run"}}
            ),
        )
        legacy_sac = config_from_dict(
            {"experiment": "sac", "sac": fixed["sac"]},
            allow_legacy_fixed_fields=True,
        )
        self.assertEqual(legacy_sac, config_from_dict({"experiment": "sac"}))
        bad = {"experiment": "diffcrl", "sac": {"ent_coef": 0.2}}
        with self.assertRaisesRegex(ValueError, "Legacy config field"):
            config_from_dict(bad, allow_legacy_fixed_fields=True)
        for name, value in [
            ("gamma", 1.1),
            ("batch_size", -1),
            ("learning_starts", -1),
        ]:
            values = {
                "experiment": "sac",
                "sac": {name: value},
                "runtime": {"output": "/tmp/run"},
            }
            with self.subTest(field=f"sac.{name}"), self.assertRaises(ValueError):
                config_from_dict(values)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.config(runtime={"output": "/tmp/run", "seed": 10000})

    def test_invalid_yaml_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            for contents in (
                "",
                "[]",
                "runtime: [",
                "experiment: sac\nexperiment: diffcrl\n",
                "experiment: sac\nruntime:\n  seed: 1\n  seed: 2\n",
            ):
                path.write_text(contents)
                with self.subTest(contents=contents), self.assertRaises(ValueError):
                    load_config(path)
            with self.assertRaisesRegex(ValueError, "Cannot load config"):
                load_config(Path(tmp) / "missing.yaml")

    def test_wrong_entrypoint(self):
        with self.assertRaisesRegex(ValueError, "must be sac"):
            load_config(
                ROOT / "configs/diffcrl/diffcrl.yaml", expected_experiment="sac"
            )

    def test_experiment_specific_fields_reject_irrelevant_settings(self):
        cases = [
            ({"experiment": "continual_sac", "bc": {"epochs": 1}}, "section bc"),
            (
                {"experiment": "continual_sac", "continual": {"experts": {}}},
                "continual.experts",
            ),
            ({"experiment": "continual_sac", "sac": {"steps": 1}}, "sac.steps"),
            ({"experiment": "diffcrl", "sac": {"gamma": 0.99}}, "section sac"),
            (
                {"experiment": "diffcrl", "evaluation": {"frequency": 1}},
                "evaluation.frequency",
            ),
            ({"experiment": "sac", "diffusion": {"epochs": 1}}, "section diffusion"),
            (
                {"experiment": "sac", "evaluation": {"seed": 10000}},
                "evaluation.seed",
            ),
        ]
        for values, message in cases:
            values.setdefault("runtime", {"output": "/tmp/run"})
            with self.subTest(values=values), self.assertRaisesRegex(
                ValueError, message
            ):
                config_from_dict(values)

    def test_cuda_validation_and_auto_resolution(self):
        c = self.config(runtime={"output": "/tmp/run", "device": "auto"})
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device(c).runtime.device, "cpu")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                resolve_device(replace(c, runtime=replace(c.runtime, device="cuda")))
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=1
        ):
            self.assertEqual(resolve_device(c).runtime.device, "cuda")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                resolve_device(replace(c, runtime=replace(c.runtime, device="cuda:1")))

    def test_resolved_config_roundtrip_and_no_overwrite(self):
        c = self.config(
            runtime={
                "output": "/tmp/run",
                "seed": 7,
                "device": "cpu",
                "run_name": "test",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_resolved_config(c, Path(tmp))
            resolved = load_config(Path(tmp) / "config.yaml")
            self.assertEqual(
                resolved,
                replace(
                    c, runtime=replace(c.runtime, run_dir=str(Path(tmp).resolve()))
                ),
            )
            saved = yaml.safe_load((Path(tmp) / "config.yaml").read_text())
            self.assertEqual(
                saved,
                config_to_dict(
                    replace(
                        c, runtime=replace(c.runtime, run_dir=str(Path(tmp).resolve()))
                    )
                ),
            )
            self.assertNotIn("sac", saved)
            with self.assertRaises(FileExistsError):
                save_resolved_config(c, Path(tmp))

    def test_direct_dataclass_validation(self):
        c = self.config()
        with self.assertRaisesRegex(ValueError, "bc.epochs"):
            validate_config(replace(c, bc=replace(c.bc, epochs="one")))

    def test_flow_matching_configuration_is_typed_serialized_and_model_compatible(self):
        config = self.config(
            continual={"replay_mode": "flow-matching"},
            flow_matching={
                "epochs": 12,
                "batch_size": 4,
                "learning_rate": 0.0005,
                "width": 32,
                "time_embedding_scale": 50.0,
                "integration_steps": 20,
                "ode_solver": "euler",
            },
        )
        serialized = config_to_dict(config)
        self.assertEqual(serialized["continual"]["replay_mode"], "flow-matching")
        self.assertEqual(serialized["flow_matching"]["integration_steps"], 20)
        self.assertEqual(config_from_dict(serialized), config)

        model_config = FlowMatchingModelConfig(
            width=config.flow_matching.width,
            integration_steps=config.flow_matching.integration_steps,
            time_embedding_scale=config.flow_matching.time_embedding_scale,
        )
        self.assertEqual(
            (
                model_config.width,
                model_config.integration_steps,
                model_config.time_embedding_scale,
            ),
            (32, 20, 50.0),
        )

    def test_invalid_flow_matching_hyperparameters_and_unknown_fields(self):
        invalid = {
            "epochs": 0,
            "batch_size": 0,
            "learning_rate": 0.0,
            "width": 5,
            "time_embedding_scale": 0.0,
            "integration_steps": 0,
            "ode_solver": "bogus",
            "rtol": 0.0,
            "atol": float("inf"),
        }
        for name, value in invalid.items():
            with self.subTest(field=name), self.assertRaisesRegex(
                ValueError, f"flow_matching.{name}"
            ):
                self.config(
                    continual={"replay_mode": "flow-matching"},
                    flow_matching={name: value},
                )

        for solver in ("euler", "midpoint", "heun2", "rk4", "dopri5"):
            configured = self.config(
                continual={"replay_mode": "flow-matching"},
                flow_matching={"ode_solver": solver, "rtol": 1e-5, "atol": 2e-5},
            )
            self.assertEqual(configured.flow_matching.ode_solver, solver)
        with self.assertRaisesRegex(ValueError, "flow_matching.typo"):
            self.config(
                continual={"replay_mode": "flow-matching"},
                flow_matching={"typo": 1},
            )

    def test_flow_section_and_replay_mode_are_strict(self):
        with self.assertRaisesRegex(ValueError, "flow_matching requires"):
            self.config(flow_matching={"epochs": 1})
        with self.assertRaisesRegex(ValueError, "continual.replay_mode"):
            self.config(continual={"replay_mode": "flow_matching"})
        fixed_ddpm = self.config(
            continual={
                "replay_mode": "diffusion",
                "replay_allocation": "fixed-total",
                "replay_budget": 10,
            }
        )
        self.assertEqual(fixed_ddpm.continual.replay_budget, 10)
        nominal_no_replay = self.config(
            continual={
                "replay_mode": "none",
                "replay_allocation": "fixed-total",
                "replay_budget": 10,
            }
        )
        self.assertEqual(nominal_no_replay.continual.replay_budget, 10)

    def test_existing_diffusion_and_no_replay_serialization_is_exactly_preserved(self):
        expected_sections = {
            "experiment",
            "environment",
            "bc",
            "diffusion",
            "continual",
            "evaluation",
            "runtime",
        }
        for path in (
            ROOT / "configs/diffcrl/diffcrl.yaml",
            ROOT / "configs/diffcrl/diffcrl_no_replay.yaml",
        ):
            with self.subTest(path=path):
                original = yaml.safe_load(path.read_text())
                resolved = config_to_dict(load_config(path))
                self.assertEqual(set(resolved), expected_sections)
                self.assertNotIn("flow_matching", resolved)
                # Existing explicit YAML values must survive resolution unchanged.
                for section, values in original.items():
                    if section == "experiment":
                        self.assertEqual(resolved[section], values)
                    else:
                        for name, value in values.items():
                            self.assertEqual(resolved[section][name], value)

    def test_flow_settings_participate_only_in_flow_scientific_identity(self):
        flow = self.config(continual={"replay_mode": "flow-matching"})
        changed = replace(
            flow,
            flow_matching=replace(flow.flow_matching, integration_steps=25),
        )
        self.assertNotEqual(_scientific_config(flow), _scientific_config(changed))
        diffusion = self.config()
        changed_unused = replace(
            diffusion,
            flow_matching=replace(diffusion.flow_matching, integration_steps=25),
        )
        self.assertEqual(
            _scientific_config(diffusion), _scientific_config(changed_unused)
        )


if __name__ == "__main__":
    unittest.main()
