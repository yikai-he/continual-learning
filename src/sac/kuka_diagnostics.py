"""Read-only KUKA episode diagnostics used by standalone SAC evaluation."""

from __future__ import annotations

import numpy as np


class KukaEpisodeDiagnostics:
    """Accumulate optional Hammer, Push, and Stage-1 observations for one episode."""

    def __init__(self, task_name, env, observation, hammer_tracker_type):
        self.task_name = task_name
        self.env = env
        self.raw_env = env.unwrapped
        self.hammer_tracker = (
            hammer_tracker_type(env) if task_name == "kuka-hammer-v3" else None
        )
        if self.hammer_tracker is not None:
            self.hammer_tracker.sample(0)
        self.track_handoff = task_name == "kuka-push-cube-v3" and getattr(
            self.raw_env, "reward_function_version", None
        ) in ("v1_initial_to_contact", "v1_1_initial_to_contact_aperture")
        self.track_push = (
            task_name in ("kuka-push-v3", "kuka-push-cube-v3")
            and observation.shape == (39,)
            and not self.track_handoff
        )
        if self.track_handoff:
            self._start_handoff(observation)
        if self.track_push:
            self._start_push(observation)

    def _start_handoff(self, observation):
        self.handoff_instantaneous = False
        self.handoff_sustained = False
        self.handoff_first_instantaneous_step = None
        self.handoff_success_step = None
        self.handoff_max_counter = 0
        self.handoff_contact_steps = 0
        self.handoff_reward_approach = 0.0
        self.handoff_reward_tcp = 0.0
        self.handoff_reward_aperture = 0.0
        self.handoff_reward_bonus = 0.0
        self.handoff_final_violations = ()
        initial_cube = np.asarray(observation[4:7], dtype=float).copy()
        self.handoff_goal = np.asarray(observation[36:39], dtype=float).copy()
        target = self.raw_env._stage1_approach_target(initial_cube, self.handoff_goal)
        initial_distance = float(
            np.linalg.norm(np.asarray(observation[:3]) - target)
        )
        self.handoff_initial_distance = initial_distance
        self.handoff_minimum_distance = initial_distance
        self.handoff_final_distance = initial_distance
        self.handoff_apertures = []
        self.handoff_actions = []
        self.handoff_discounted_return = 0.0
        self.handoff_first_contact_step = None
        self.handoff_max_cube_displacement = 0.0
        self.handoff_max_cube_tilt = 0.0
        self.handoff_final_offsets = (0.0, 0.0, 0.0)
        self.handoff_violation_counts = {}
        self.handoff_final_tcp_potential = -initial_distance
        self.handoff_final_aperture_potential = 0.0

    def _start_push(self, observation):
        self.initial_object = np.asarray(observation[4:7], dtype=float).copy()
        self.goal = np.asarray(observation[36:39], dtype=float).copy()
        self.initial_distance = float(np.linalg.norm(self.initial_object - self.goal))
        self.minimum_distance = self.initial_distance
        self.maximum_displacement = 0.0
        self.contact_steps = self.stationary_contact_steps = 0
        self.clipped_contact_without_progress_steps = 0
        self.positive_progress_steps = self.negative_progress_steps = 0
        self.progress_contribution = self.baseline_contribution = 0.0
        self.pre_contact_caging_contribution = 0.0
        self.contact_baseline_contribution = 0.0
        self.retention_contribution = 0.0
        self.first_success_count = 0
        self.v3_3_contributions = {
            "reward_approach": 0.0,
            "reward_first_contact": 0.0,
            "reward_push": 0.0,
            "reward_goal_progress": 0.0,
            "reward_first_success": 0.0,
        }
        self.v4_contributions = {
            "reward_approach": 0.0,
            "reward_progress": 0.0,
            "reward_first_success": 0.0,
        }
        self.v5_contributions = {
            "reward_progress": 0.0,
            "reward_first_success": 0.0,
        }
        self.sawyer_adapted_contributions = {
            "reward_base": 0.0,
            "reward_near_object": 0.0,
            "reward_success_override_steps": 0.0,
        }
        self.ever_near_object = False
        self.had_contact = False
        self.contact_lost = False

    def observe_step(self, action, observation, reward, info, episode_length):
        if self.hammer_tracker is not None:
            self.hammer_tracker.sample(episode_length)
        if self.track_handoff:
            self._observe_handoff(action, observation, reward, info, episode_length)
        if self.track_push:
            self._observe_push(observation, info, episode_length)

    def _observe_handoff(self, action, observation, reward, info, episode_length):
        self.handoff_discounted_return += (0.99 ** (episode_length - 1)) * float(
            reward
        )
        instantaneous = bool(info.get("handoff_instantaneous", False))
        sustained = bool(info.get("handoff_sustained", False))
        self.handoff_instantaneous |= instantaneous
        self.handoff_sustained |= sustained
        if instantaneous and self.handoff_first_instantaneous_step is None:
            self.handoff_first_instantaneous_step = episode_length
        if sustained and self.handoff_success_step is None:
            self.handoff_success_step = episode_length
        self.handoff_max_counter = max(
            self.handoff_max_counter,
            int(info.get("handoff_stability_counter", 0)),
        )
        valid_contact = bool(info.get("handoff_valid_rear_contact", False))
        self.handoff_contact_steps += int(valid_contact)
        if valid_contact and self.handoff_first_contact_step is None:
            self.handoff_first_contact_step = episode_length
        self.handoff_reward_approach += float(info.get("reward_approach", 0.0))
        self.handoff_reward_tcp += float(
            info.get("reward_tcp_shaping", info.get("reward_approach", 0.0))
        )
        self.handoff_reward_aperture += float(
            info.get("reward_aperture_shaping", 0.0)
        )
        self.handoff_reward_bonus += float(info.get("reward_handoff", 0.0))
        self.handoff_final_tcp_potential = float(
            info.get("potential_tcp_current", -self.handoff_final_distance)
        )
        self.handoff_final_aperture_potential = float(
            info.get("potential_aperture_current", 0.0)
        )
        self.handoff_final_violations = tuple(
            info.get("handoff_violation_reasons", ())
        )
        for violation in self.handoff_final_violations:
            self.handoff_violation_counts[violation] = (
                self.handoff_violation_counts.get(violation, 0) + 1
            )
        self.handoff_actions.append(np.asarray(action, dtype=float).tolist())
        self.handoff_apertures.append(
            float(info.get("handoff_gripper_aperture", np.nan))
        )
        cube = np.asarray(observation[4:7], dtype=float)
        target = self.raw_env._stage1_approach_target(cube, self.handoff_goal)
        self.handoff_final_distance = float(
            np.linalg.norm(np.asarray(observation[:3]) - target)
        )
        self.handoff_minimum_distance = min(
            self.handoff_minimum_distance, self.handoff_final_distance
        )
        self.handoff_max_cube_displacement = max(
            self.handoff_max_cube_displacement,
            float(info.get("handoff_cube_displacement", 0.0)),
        )
        self.handoff_max_cube_tilt = max(
            self.handoff_max_cube_tilt,
            float(info.get("handoff_cube_tilt", 0.0)),
        )
        self.handoff_final_offsets = tuple(
            float(info.get(name, 0.0))
            for name in (
                "handoff_forward_offset",
                "handoff_lateral_offset",
                "handoff_vertical_offset",
            )
        )

    @staticmethod
    def _reward_prefix(info):
        # Preserve historical info-schema fallbacks; version alone is not reliable.
        return (
            "sawyer_push_v2_adapted"
            if "reward_object_grasped" in info
            else "reward_v5"
            if "reward_progress" in info and "reward_approach" not in info
            else "reward_v4"
            if "reward_progress" in info
            else "reward_v3_3"
            if "reward_total" in info
            else "reward_v3_2_retention"
            if "reward_v3_2_retention_total" in info
            else "reward_v3_1"
            if "reward_v3_1_total" in info
            else "reward_v3"
        )

    def _observe_push(self, observation, info, episode_length):
        prefix = self._reward_prefix(info)
        obj = np.asarray(observation[4:7], dtype=float)
        current_distance = float(np.linalg.norm(obj - self.goal))
        signed_progress = float(
            info.get(
                "signed_goal_progress"
                if prefix in ("reward_v3_3", "reward_v4", "reward_v5")
                else f"{prefix}_signed_progress",
                self.initial_distance - current_distance
                if episode_length == 1
                else self.previous_distance - current_distance,
            )
        )
        physical_contact = bool(
            info.get(
                "physical_robot_object_contact",
                self.raw_env.pad_contacting_object(
                    self.raw_env._get_id_main_object()
                )
                and float(observation[3]) > 0,
            )
        )
        contact = bool(
            physical_contact
            if prefix == "reward_v3_3"
            else info.get(f"{prefix}_contact", physical_contact)
        )
        at_workspace_boundary = bool(
            np.any(
                np.isclose(
                    np.asarray(self.raw_env.data.mocap_pos[0], dtype=float),
                    np.asarray(self.raw_env.mocap_low, dtype=float),
                    atol=1e-9,
                )
                | np.isclose(
                    np.asarray(self.raw_env.data.mocap_pos[0], dtype=float),
                    np.asarray(self.raw_env.mocap_high, dtype=float),
                    atol=1e-9,
                )
            )
        )
        if self.had_contact and not contact:
            self.contact_lost = True
        self.had_contact |= contact
        self.contact_steps += int(contact)
        self.stationary_contact_steps += int(
            contact and abs(signed_progress) <= 1e-5
        )
        self.clipped_contact_without_progress_steps += int(
            contact and at_workspace_boundary and abs(signed_progress) <= 1e-5
        )
        self.positive_progress_steps += int(contact and signed_progress > 1e-5)
        self.negative_progress_steps += int(contact and signed_progress < -1e-5)
        if prefix == "sawyer_push_v2_adapted":
            self.progress_contribution += 0.0
        elif prefix == "reward_v5":
            for component in self.v5_contributions:
                self.v5_contributions[component] += float(info[component])
            self.progress_contribution += float(info["reward_progress"])
        elif prefix == "reward_v4":
            for component in self.v4_contributions:
                self.v4_contributions[component] += float(info[component])
            self.progress_contribution += float(info["reward_progress"])
        elif prefix == "reward_v3_3":
            for component in self.v3_3_contributions:
                self.v3_3_contributions[component] += float(info[component])
            self.progress_contribution += float(info["reward_goal_progress"])
        else:
            self.progress_contribution += (
                1000.0
                * float(contact)
                * float(
                    info.get(
                        f"{prefix}_clipped_progress",
                        np.clip(signed_progress, -0.005, 0.005),
                    )
                )
            )
        caging = float(
            info.get(f"{prefix}_caging", info.get("grasp_reward", 0.0))
        )
        if prefix == "sawyer_push_v2_adapted":
            base = float(info["reward_base_contribution"])
            near = float(info["reward_near_object_contribution"])
            self.pre_contact_caging_contribution += base
            self.contact_baseline_contribution += near
            self.baseline_contribution += base + near
            self.sawyer_adapted_contributions["reward_base"] += base
            self.sawyer_adapted_contributions["reward_near_object"] += near
            self.sawyer_adapted_contributions[
                "reward_success_override_steps"
            ] += float(info["reward_success_override"])
        elif prefix == "reward_v5":
            pass
        elif prefix == "reward_v4":
            self.pre_contact_caging_contribution += float(info["reward_approach"])
            self.baseline_contribution += float(info["reward_approach"])
        elif prefix == "reward_v3_3":
            self.pre_contact_caging_contribution += float(info["reward_approach"])
            self.contact_baseline_contribution += float(
                info["reward_first_contact"] + info["reward_push"]
            )
            self.baseline_contribution += float(
                info["reward_approach"]
                + info["reward_first_contact"]
                + info["reward_push"]
            )
        elif prefix in ("reward_v3_1", "reward_v3_2_retention"):
            self.pre_contact_caging_contribution += caging * float(not contact)
            self.contact_baseline_contribution += 0.05 * float(contact)
            self.baseline_contribution += caging * float(not contact) + 0.05 * float(
                contact
            )
        else:
            self.pre_contact_caging_contribution += 0.1 * caging
            self.contact_baseline_contribution += 0.05 * float(contact)
            self.baseline_contribution += 0.1 * caging + 0.05 * float(contact)
        success_key = (
            "reward_success_override"
            if prefix == "sawyer_push_v2_adapted"
            else "reward_first_success"
            if prefix in ("reward_v5", "reward_v4", "reward_v3_3")
            else f"{prefix}_first_success"
        )
        self.first_success_count += int(bool(info.get(success_key, False)))
        self.retention_contribution += float(
            info.get(f"{prefix}_retention_reward", 0.0)
        )
        self.ever_near_object |= bool(info.get("near_object", False))
        self.minimum_distance = min(self.minimum_distance, current_distance)
        self.maximum_displacement = max(
            self.maximum_displacement,
            float(np.linalg.norm(obj - self.initial_object)),
        )
        self.previous_distance = current_distance

    def finish(
        self,
        *,
        seed,
        episode_return,
        episode_length,
        episode_success,
        success_diagnostic,
        final_info,
    ):
        behavior = self._finish_push(seed, episode_length, episode_success, success_diagnostic) if self.track_push else None
        hammer = (
            self.hammer_tracker.finish(episode_success).to_dict()
            if self.hammer_tracker is not None
            else None
        )
        handoff = (
            self._finish_handoff(seed, episode_return, episode_length, final_info)
            if self.track_handoff
            else None
        )
        return behavior, hammer, handoff

    def _finish_handoff(self, seed, episode_return, episode_length, final_info):
        actions = np.asarray(self.handoff_actions, dtype=float)
        apertures = np.asarray(self.handoff_apertures, dtype=float)
        return {
            "seed": seed,
            "instantaneous_success": self.handoff_instantaneous,
            "sustained_success": self.handoff_sustained,
            "first_instantaneous_step": self.handoff_first_instantaneous_step,
            "success_step": self.handoff_success_step,
            "maximum_stability_counter": self.handoff_max_counter,
            "valid_rear_contact_fraction": self.handoff_contact_steps / episode_length,
            "valid_rear_contact_episode": bool(self.handoff_contact_steps),
            "first_valid_rear_contact_step": self.handoff_first_contact_step,
            "initial_tcp_target_distance": self.handoff_initial_distance,
            "minimum_tcp_target_distance": self.handoff_minimum_distance,
            "final_tcp_target_distance": self.handoff_final_distance,
            "final_forward_offset": self.handoff_final_offsets[0],
            "final_lateral_offset": self.handoff_final_offsets[1],
            "final_vertical_offset": self.handoff_final_offsets[2],
            "mean_physical_aperture": float(np.mean(apertures)),
            "final_physical_aperture": float(apertures[-1]),
            "mean_aperture_error": float(np.mean(np.abs(apertures - 0.0841))),
            "final_aperture_error": float(abs(apertures[-1] - 0.0841)),
            "gripper_action_mean": float(np.mean(actions[:, 3])),
            "gripper_action_std": float(np.std(actions[:, 3])),
            "gripper_action_min": float(np.min(actions[:, 3])),
            "gripper_action_max": float(np.max(actions[:, 3])),
            "maximum_cube_displacement": self.handoff_max_cube_displacement,
            "maximum_cube_tilt": self.handoff_max_cube_tilt,
            "contact_loss_events": int(
                final_info.get("handoff_contact_loss_events", 0)
            ),
            "reward_approach_total": self.handoff_reward_approach,
            "reward_tcp_shaping_total": self.handoff_reward_tcp,
            "reward_aperture_shaping_total": self.handoff_reward_aperture,
            "final_tcp_potential": self.handoff_final_tcp_potential,
            "final_aperture_potential": self.handoff_final_aperture_potential,
            "final_total_potential": (
                self.handoff_final_tcp_potential
                + self.handoff_final_aperture_potential
            ),
            "reward_handoff_total": self.handoff_reward_bonus,
            "undiscounted_episode_return": episode_return,
            "discounted_episode_return_gamma_0_99": self.handoff_discounted_return,
            "contract_violation_counts": self.handoff_violation_counts,
            "final_violation_reasons": self.handoff_final_violations,
        }

    def _finish_push(self, seed, episode_length, episode_success, success_diagnostic):
        final_progress = self.initial_distance - self.previous_distance
        maximum_progress = self.initial_distance - self.minimum_distance
        if episode_success and self.previous_distance > 0.05:
            category = "overshoot"
        elif episode_success:
            category = "success"
        elif not self.ever_near_object:
            category = "no_meaningful_approach"
        elif not self.had_contact:
            category = "approach_no_contact"
        elif self.maximum_displacement <= 0.001:
            category = "stationary_contact"
        elif final_progress < -0.005:
            category = "wrong_direction"
        elif self.minimum_distance <= 0.075:
            category = "near_threshold_miss"
        elif maximum_progress >= 0.03:
            category = "useful_progress_insufficient"
        elif self.contact_lost:
            category = "contact_loss"
        else:
            category = "contact_low_progress"
        return {
            "seed": seed,
            "success": episode_success,
            **success_diagnostic,
            "category": category,
            "initial_goal_distance": self.initial_distance,
            "maximum_goal_progress": maximum_progress,
            "minimum_goal_distance": self.minimum_distance,
            "final_goal_distance": self.previous_distance,
            "maximum_puck_displacement": self.maximum_displacement,
            "contact_rate": self.contact_steps / episode_length,
            "stationary_contact_rate": self.stationary_contact_steps / episode_length,
            "workspace_clipped_contact_without_progress_rate": (
                self.clipped_contact_without_progress_steps / episode_length
            ),
            "wrong_direction_step_rate": self.negative_progress_steps / episode_length,
            "correct_direction_step_rate": self.positive_progress_steps
            / episode_length,
            "signed_progress_contribution": self.progress_contribution,
            "caging_contact_baseline_contribution": self.baseline_contribution,
            "pre_contact_caging_contribution": self.pre_contact_caging_contribution,
            "contact_baseline_contribution": self.contact_baseline_contribution,
            "success_bonus_occurrences": self.first_success_count,
            "retention_reward_contribution": self.retention_contribution,
            **{
                f"{name}_contribution": value
                for name, value in self.v3_3_contributions.items()
            },
            **{
                f"v4_{name}_contribution": value
                for name, value in self.v4_contributions.items()
            },
            **{
                f"v5_{name}_contribution": value
                for name, value in self.v5_contributions.items()
            },
            **{
                f"sawyer_adapted_{name}_contribution": value
                for name, value in self.sawyer_adapted_contributions.items()
            },
        }
