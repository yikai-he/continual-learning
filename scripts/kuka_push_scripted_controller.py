"""Calibrated deterministic controller used by KUKA Push physics diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


class CalibratedPushController:
    """Replay the exact 20/35/145-step diagnostic push trajectory."""

    rho = 0.060
    tcp_z = 0.042
    horizon = 200
    approach_steps = 20
    contact_steps = 35
    push_steps = 145

    def __init__(self) -> None:
        self._targets: list[np.ndarray] = []
        self._step = 0
        self._gripper_action = 1.0

    def reset(self, object_position: np.ndarray, goal: np.ndarray) -> None:
        obj = np.asarray(object_position, dtype=float).copy()
        goal = np.asarray(goal, dtype=float).copy()
        direction = goal[:2] - obj[:2]
        distance = float(np.linalg.norm(direction))
        if distance <= 0.0:
            raise ValueError("object and goal planar positions must differ")
        direction /= distance
        behind = obj.copy()
        behind[:2] -= self.rho * direction
        behind[2] = self.tcp_z
        self._targets = [behind + [0.0, 0.0, 0.10]] * self.approach_steps
        self._targets += [behind.copy()] * self.contact_steps
        self._targets += [
            behind + np.r_[direction * advance, 0.0]
            for advance in np.linspace(0.0, distance + self.rho, self.push_steps)
        ]
        if len(self._targets) != self.horizon:
            raise AssertionError("scripted trajectory must contain exactly 200 targets")
        self._step = 0
        self._gripper_action = 1.0

    def reset_contact_start(
        self, object_position: np.ndarray, goal: np.ndarray, tcp: np.ndarray
    ) -> None:
        """Start the push phase from an already established open-pad contact."""
        obj = np.asarray(object_position, dtype=float)
        goal = np.asarray(goal, dtype=float)
        start = np.asarray(tcp, dtype=float).copy()
        direction = goal[:2] - obj[:2]
        distance = float(np.linalg.norm(direction))
        if distance <= 0.0:
            raise ValueError("object and goal planar positions must differ")
        direction /= distance
        self._targets = [
            start + np.r_[direction * advance, 0.0]
            for advance in np.linspace(0.0, distance + self.rho, self.horizon)
        ]
        self._step = 0
        self._gripper_action = 1.0

    @property
    def phase(self) -> str:
        if self._step < self.approach_steps:
            return "approach"
        if self._step < self.approach_steps + self.contact_steps:
            return "contact"
        return "push"

    def predict(self, observation: np.ndarray, base: Any) -> tuple[np.ndarray, None]:
        """Return the next action using the production environment's action scale."""
        del observation  # State feedback is the physical TCP, as in the diagnostics.
        if not self._targets:
            raise RuntimeError("reset() must be called before predict()")
        if self._step >= self.horizon:
            raise RuntimeError("scripted controller exhausted its 200-step trajectory")
        target = self._targets[self._step]
        action = np.r_[
            np.clip((target - np.asarray(base.tcp_center)) / base.action_scale, -1, 1),
            self._gripper_action,
        ].astype(np.float32)
        self._step += 1
        return action, None
