"""Public Gymnasium construction for the isolated MetaWorld KUKA-v3 task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from . import ACTION_SHAPE, GOAL_SLICE, OBSERVATION_SHAPE, EpisodeSuccessWrapper

KUKA_V3_HORIZON = 200
KUKA_V3_REGISTRATION_ID = "Meta-World-KUKA/MT1"
KUKA_V3_TASKS = (
    "kuka-reach-v3",
    "kuka-push-v3",
    "kuka-push-cube-v3",
    "kuka-handle-press-side-v3",
    "kuka-button-press-v3",
    "kuka-button-press-topdown-v3",
    "kuka-peg-unplug-side-v3",
    "kuka-hammer-v3",
    "kuka-drawer-close-v3",
    "kuka-faucet-open-v3",
    "kuka-faucet-close-v3",
    "kuka-window-open-v3",
    "kuka-window-close-v3",
    "kuka-door-open-v3",
    "kuka-pick-place-v3",
)


def kuka_v3_tasks(task: str, seed: int, cube_push_curriculum_level: int = 3):
    """Recreate the exact ordered task bank used by the registered KUKA MT1."""
    if task not in KUKA_V3_TASKS:
        raise ValueError(f"KUKA-v3 backend does not support {task!r}.")
    import metaworld_kuka  # noqa: F401
    from metaworld import _make_tasks
    from metaworld_kuka.registration import KUKA_ENVIRONMENTS

    if cube_push_curriculum_level not in (1, 2, 3):
        raise ValueError("cube_push_curriculum_level must be 1, 2, or 3")
    if task != "kuka-push-cube-v3" and cube_push_curriculum_level != 3:
        raise ValueError("Cube Push curriculum is valid only for kuka-push-cube-v3")
    return _make_tasks(
        KUKA_ENVIRONMENTS,
        {task: {"args": [], "kwargs": {"task_id": 0}}},
        {"partially_observable": False},
        seed=seed,
    )


def make_kuka_v3_env(
    task: str,
    seed: int,
    render_mode: str | None = None,
    *,
    reward_function_version: str = "v2",
    hammer_reward_variant: str = "original",
    hammer_nail_progress_weight: float = 0.0,
    peg_unplug_reward_variant: str = "original",
    peg_unplug_progress_weight: float = 0.0,
    peg_unplug_progress_gamma: float = 0.99,
    cube_push_curriculum_level: int = 3,
    cube_push_initialization: str = "default",
    cube_y_min: float = 0.490,
    near_contact_gap_m: float = 0.005,
) -> gym.Env:
    """Create a KUKA-v3 task exclusively through its public registered interface."""
    if task not in KUKA_V3_TASKS:
        raise ValueError(f"KUKA-v3 backend supports {KUKA_V3_TASKS!r}, got {task!r}.")
    if reward_function_version not in (
        "v1", "v2", "v3", "v3_1", "v3_2_retention",
        "v3_3_contact_push_progress",
        "v4_simple_approach_progress",
        "v5_contact_start_progress",
        "v6_rear_contact_push_progress",
        "v1_initial_to_contact",
        "v1_1_initial_to_contact_aperture",
        "sawyer_push_v2_adapted",
    ):
        raise ValueError(
            "reward_function_version must be v1, v2, v3, v3_1, or "
            "v3_2_retention, v3_3_contact_push_progress, or "
            "v6_rear_contact_push_progress, or sawyer_push_v2_adapted."
        )
    if (
        reward_function_version in ("v3", "v3_1", "v3_2_retention")
        and task not in ("kuka-push-v3", "kuka-push-cube-v3")
    ):
        raise ValueError(
            f"reward_function_version={reward_function_version!r} is specific to "
            "kuka-push-v3 or kuka-push-cube-v3."
        )
    if (
        reward_function_version == "v3_3_contact_push_progress"
        and task != "kuka-push-cube-v3"
    ):
        raise ValueError(
            "reward_function_version='v3_3_contact_push_progress' is specific "
            "to kuka-push-cube-v3."
        )
    if (
        reward_function_version == "v4_simple_approach_progress"
        and task != "kuka-push-cube-v3"
    ):
        raise ValueError(
            "reward_function_version='v4_simple_approach_progress' is specific "
            "to kuka-push-cube-v3."
        )
    if reward_function_version == "v5_contact_start_progress" and task != "kuka-push-cube-v3":
        raise ValueError("reward_function_version='v5_contact_start_progress' is specific to kuka-push-cube-v3.")
    if reward_function_version == "v6_rear_contact_push_progress" and task != "kuka-push-cube-v3":
        raise ValueError("reward_function_version='v6_rear_contact_push_progress' is specific to kuka-push-cube-v3.")
    if reward_function_version in (
        "v1_initial_to_contact", "v1_1_initial_to_contact_aperture"
    ) and task != "kuka-push-cube-v3":
        raise ValueError(f"reward_function_version={reward_function_version!r} is specific to kuka-push-cube-v3.")
    if reward_function_version in (
        "v1_initial_to_contact", "v1_1_initial_to_contact_aperture"
    ) and (
        cube_push_curriculum_level != 2 or cube_push_initialization != "default"
    ):
        raise ValueError(
            f"{reward_function_version} requires Cube Push curriculum level 2 with "
            "default initialization"
        )
    if cube_push_initialization != "default" and task != "kuka-push-cube-v3":
        raise ValueError("cube_push_initialization is specific to kuka-push-cube-v3")
    if cube_push_initialization in ("contact", "near_contact") and (
        cube_push_curriculum_level != 2
    ):
        raise ValueError("Cube Push contact initializations require curriculum level 2")
    if (
        reward_function_version == "sawyer_push_v2_adapted"
        and task != "kuka-push-cube-v3"
    ):
        raise ValueError(
            "reward_function_version='sawyer_push_v2_adapted' is specific "
            "to kuka-push-cube-v3."
        )
    if task != "kuka-hammer-v3":
        hammer_reward_variant = "original"
        hammer_nail_progress_weight = 0.0
    if task != "kuka-peg-unplug-side-v3":
        peg_unplug_reward_variant = "original"
        peg_unplug_progress_weight = 0.0

    import metaworld_kuka  # noqa: F401  # Performs isolated registration.

    env = gym.make(
        KUKA_V3_REGISTRATION_ID,
        env_name=task,
        seed=seed,
        use_one_hot=False,
        terminate_on_success=False,
        reward_function_version=reward_function_version,
        hammer_reward_variant=hammer_reward_variant,
        hammer_nail_progress_weight=hammer_nail_progress_weight,
        peg_unplug_reward_variant=peg_unplug_reward_variant,
        peg_unplug_progress_weight=peg_unplug_progress_weight,
        peg_unplug_progress_gamma=peg_unplug_progress_gamma,
        cube_push_curriculum_level=cube_push_curriculum_level,
        cube_push_initialization=cube_push_initialization,
        cube_y_min=cube_y_min,
        near_contact_gap_m=near_contact_gap_m,
        max_episode_steps=KUKA_V3_HORIZON,
        render_mode=render_mode,
        disable_env_checker=True,
    )
    env = EpisodeSuccessWrapper(env)
    observation, _ = env.reset(seed=seed)
    if env.observation_space.shape != OBSERVATION_SHAPE:
        env.close()
        raise ValueError(
            f"Expected observation shape {OBSERVATION_SHAPE}, "
            f"got {env.observation_space.shape}."
        )
    if env.action_space.shape != ACTION_SHAPE:
        env.close()
        raise ValueError(
            f"Expected action shape {ACTION_SHAPE}, got {env.action_space.shape}."
        )
    if observation.dtype != np.float32:
        env.close()
        raise TypeError(f"Expected float32 observations, got {observation.dtype}.")
    if not np.isfinite(observation).all():
        env.close()
        raise ValueError("KUKA-v3 reset produced a non-finite observation.")
    if not np.any(observation[GOAL_SLICE]):
        env.close()
        raise ValueError("KUKA-v3 goal is not visible in observation[36:39].")
    return env
