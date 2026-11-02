#!/usr/bin/env python3
"""Validate GeneralPolicy transfer into the isolated Immediate SAC policy."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from src.continual.bc_policy import GeneralPolicy
from src.envs import make_env
from src.sac.policy_transfer import (
    deterministic_equivalence,
    transfer_general_policy_to_sac,
)
from src.sac.transferable_actor import TransferableSACPolicy


def collect_observations(env, count: int, seed: int) -> np.ndarray:
    """Collect valid environment states without training a policy."""
    observations = []
    observation, _ = env.reset(seed=seed)
    generator = np.random.default_rng(seed)
    while len(observations) < count:
        observations.append(np.asarray(observation, dtype=np.float32))
        action = generator.uniform(env.action_space.low, env.action_space.high)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    return np.stack(observations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="hammer-v3")
    parser.add_argument("--observations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-std-init", type=float, default=-3.0)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Optional short SAC optimization run; zero performs transfer only.",
    )
    args = parser.parse_args()
    if args.observations < 1000:
        parser.error("--observations must be at least 1000.")
    if args.smoke_steps < 0:
        parser.error("--smoke-steps must be nonnegative.")

    torch.set_num_threads(1)
    env = make_env("metaworld-v3", args.task, args.seed)
    try:
        observations = collect_observations(env, args.observations, args.seed)
        general_policy = GeneralPolicy.load(
            args.checkpoint, expected_use_layer_norm=True
        )
        model = SAC(
            TransferableSACPolicy,
            env,
            learning_rate=3e-4,
            buffer_size=max(1000, args.smoke_steps + 1),
            learning_starts=5,
            batch_size=4,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs={
                "net_arch": [256, 256, 256, 256],
                "log_std_init": args.log_std_init,
            },
            seed=args.seed,
            device="cpu",
            verbose=0,
        )
        transfer = transfer_general_policy_to_sac(model, general_policy)
        equivalence = deterministic_equivalence(
            general_policy, model, observations
        )
        if equivalence.maximum_absolute_error >= 1e-5:
            raise RuntimeError("Deterministic transfer equivalence failed.")
        result = {
            "checkpoint": str(args.checkpoint),
            "task": args.task,
            "transfer": transfer.parameter_mapping,
            "equivalence": equivalence.to_dict(),
            "log_std_initialization": {
                "weight": "zeros",
                "bias": args.log_std_init,
            },
            "smoke": None,
        }
        if args.smoke_steps:
            actor_before = {
                name: value.detach().clone()
                for name, value in model.policy.actor.named_parameters()
            }
            critic_before = {
                name: value.detach().clone()
                for name, value in model.policy.critic.named_parameters()
            }
            model.learn(args.smoke_steps)
            actor_loss = model.logger.name_to_value.get("train/actor_loss")
            critic_loss = model.logger.name_to_value.get("train/critic_loss")
            losses_finite = bool(
                actor_loss is not None
                and critic_loss is not None
                and np.isfinite(actor_loss)
                and np.isfinite(critic_loss)
            )
            actor_gradients_finite = all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.policy.actor.parameters()
            )
            actor_updated = any(
                not torch.equal(actor_before[name], value)
                for name, value in model.policy.actor.named_parameters()
            )
            critic_updated = any(
                not torch.equal(critic_before[name], value)
                for name, value in model.policy.critic.named_parameters()
            )
            finite = all(
                torch.isfinite(value).all()
                for module in (model.policy.actor, model.policy.critic)
                for value in module.state_dict().values()
            )
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "immediate_sac.zip"
                model.save(path)
                reloaded = SAC.load(path, env=env, device="cpu")
                before, _ = model.predict(observations[:32], deterministic=True)
                after, _ = reloaded.predict(observations[:32], deterministic=True)
                reload_error = float(np.max(np.abs(before - after)))
            result["smoke"] = {
                "steps": args.smoke_steps,
                "actor_updated": actor_updated,
                "critic_updated": critic_updated,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "losses_finite": losses_finite,
                "actor_gradients_finite": bool(actor_gradients_finite),
                "finite_parameters": bool(finite),
                "reload_maximum_absolute_error": reload_error,
            }
            if (
                not actor_updated
                or not critic_updated
                or not losses_finite
                or not actor_gradients_finite
                or not finite
            ):
                raise RuntimeError("SAC optimization smoke test failed.")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
