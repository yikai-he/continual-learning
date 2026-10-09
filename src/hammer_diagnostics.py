"""Read-only episode diagnostics for the KUKA Hammer task."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


HAMMER_TASK = "kuka-hammer-v3"
NAIL_SUCCESS_THRESHOLD = 0.09
REWARD_PROXY_OFFSET = np.array([0.16, 0.06, 0.0])
STRIKING_GEOM_LOCAL_CENTER = np.array([0.16, 0.056, 0.0])


def _collision_geom_ids(model, body_name: str) -> set[int]:
    body_id = model.body(body_name).id
    return {
        index
        for index, geom_body_id in enumerate(model.geom_bodyid)
        if int(geom_body_id) == body_id and int(model.geom_contype[index]) != 0
    }


def _striking_geom_id(model) -> int:
    candidates = _collision_geom_ids(model, "hammer")
    candidates.discard(model.geom("HammerHandleCollision").id)
    if not candidates:
        raise RuntimeError("Hammer model has no collision head geometry.")
    return min(
        candidates,
        key=lambda index: float(
            np.linalg.norm(model.geom_pos[index] - STRIKING_GEOM_LOCAL_CENTER)
        ),
    )


@dataclass(frozen=True)
class HammerEpisodeMetrics:
    final_nail_qpos: float
    max_nail_qpos: float
    progress_fraction: float
    ge_003: bool
    ge_005: bool
    ge_008: bool
    crossed_success_threshold: bool
    success: bool
    minimum_proxy_goal_distance: float
    minimum_striking_goal_distance: float
    mean_proxy_error: float
    final_proxy_position: list[float]
    final_striking_position: list[float]
    head_nail_contact: bool
    head_nail_contact_steps: int
    first_head_nail_contact_timestep: int | None

    def to_dict(self) -> dict:
        return asdict(self)


class HammerEpisodeTracker:
    """Sample MuJoCo state without changing task dynamics or observations."""

    def __init__(self, env) -> None:
        self.env = env.unwrapped
        self.striking_geom_id = _striking_geom_id(self.env.model)
        self.hammer_head_geom_ids = _collision_geom_ids(self.env.model, "hammer")
        self.hammer_head_geom_ids.discard(
            self.env.model.geom("HammerHandleCollision").id
        )
        self.nail_geom_ids = _collision_geom_ids(self.env.model, "nail_link")
        self.max_nail_qpos = -np.inf
        self.minimum_proxy_goal_distance = np.inf
        self.minimum_striking_goal_distance = np.inf
        self.proxy_errors: list[float] = []
        self.contact_steps = 0
        self.first_contact_timestep: int | None = None
        self.final_proxy = np.zeros(3)
        self.final_striking = np.zeros(3)

    def sample(self, timestep: int) -> None:
        nail = float(self.env.data.joint("NailSlideJoint").qpos[0])
        hammer = np.asarray(self.env.data.body("hammer").xpos, dtype=float)
        proxy = hammer + REWARD_PROXY_OFFSET
        striking = np.asarray(
            self.env.data.geom(self.striking_geom_id).xpos, dtype=float
        ).copy()
        goal = np.asarray(self.env._target_pos, dtype=float)
        self.max_nail_qpos = max(self.max_nail_qpos, nail)
        self.minimum_proxy_goal_distance = min(
            self.minimum_proxy_goal_distance, float(np.linalg.norm(proxy - goal))
        )
        self.minimum_striking_goal_distance = min(
            self.minimum_striking_goal_distance,
            float(np.linalg.norm(striking - goal)),
        )
        self.proxy_errors.append(float(np.linalg.norm(proxy - striking)))
        self.final_proxy = proxy.copy()
        self.final_striking = striking
        contact = any(
            (
                contact.geom1 in self.hammer_head_geom_ids
                and contact.geom2 in self.nail_geom_ids
            )
            or (
                contact.geom2 in self.hammer_head_geom_ids
                and contact.geom1 in self.nail_geom_ids
            )
            for contact in self.env.data.contact
        )
        if contact:
            self.contact_steps += 1
            if self.first_contact_timestep is None:
                self.first_contact_timestep = timestep

    def finish(self, success: bool) -> HammerEpisodeMetrics:
        final = float(self.env.data.joint("NailSlideJoint").qpos[0])
        maximum = max(self.max_nail_qpos, final)
        return HammerEpisodeMetrics(
            final_nail_qpos=final,
            max_nail_qpos=maximum,
            progress_fraction=maximum / NAIL_SUCCESS_THRESHOLD,
            ge_003=maximum >= 0.03,
            ge_005=maximum >= 0.05,
            ge_008=maximum >= 0.08,
            crossed_success_threshold=maximum > NAIL_SUCCESS_THRESHOLD,
            success=bool(success),
            minimum_proxy_goal_distance=self.minimum_proxy_goal_distance,
            minimum_striking_goal_distance=self.minimum_striking_goal_distance,
            mean_proxy_error=float(np.mean(self.proxy_errors)),
            final_proxy_position=self.final_proxy.tolist(),
            final_striking_position=self.final_striking.tolist(),
            head_nail_contact=self.contact_steps > 0,
            head_nail_contact_steps=self.contact_steps,
            first_head_nail_contact_timestep=self.first_contact_timestep,
        )


def summarize_hammer_episodes(episodes: list[dict]) -> dict[str, float]:
    """Aggregate Hammer episode dictionaries into TensorBoard-ready scalars."""
    if not episodes:
        raise ValueError("Hammer diagnostics require at least one episode.")
    values = lambda key: np.asarray([episode[key] for episode in episodes], dtype=float)
    maximum = values("max_nail_qpos")
    return {
        "hammer_mean_final_nail_qpos": float(np.mean(values("final_nail_qpos"))),
        "hammer_mean_max_nail_qpos": float(np.mean(maximum)),
        "hammer_median_max_nail_qpos": float(np.median(maximum)),
        "hammer_best_max_nail_qpos": float(np.max(maximum)),
        "hammer_mean_progress_fraction": float(np.mean(values("progress_fraction"))),
        "hammer_fraction_ge_003": float(np.mean(values("ge_003"))),
        "hammer_fraction_ge_005": float(np.mean(values("ge_005"))),
        "hammer_fraction_ge_008": float(np.mean(values("ge_008"))),
        "hammer_fraction_success_threshold": float(
            np.mean(values("crossed_success_threshold"))
        ),
        "hammer_head_nail_contact_fraction": float(np.mean(values("head_nail_contact"))),
        "hammer_mean_head_nail_contact_steps": float(
            np.mean(values("head_nail_contact_steps"))
        ),
        "hammer_mean_min_proxy_goal_distance": float(
            np.mean(values("minimum_proxy_goal_distance"))
        ),
        "hammer_mean_min_striking_goal_distance": float(
            np.mean(values("minimum_striking_goal_distance"))
        ),
        "hammer_mean_proxy_error": float(np.mean(values("mean_proxy_error"))),
    }
