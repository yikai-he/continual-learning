"""Focused tests for read-only trajectory inspection."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.continual.trajectory_inspection import (
    action_oob_by_timestep,
    action_overshoot_statistics,
    action_saturation_statistics,
    aggregate_diagnostics,
    compare_datasets,
    diagnose_trajectory,
    distribution_statistics,
    export_csv,
    export_txt,
    load_trajectories,
    state_reference_diagnostics,
)


def trajectory(length=4):
    current = np.arange(length * 18, dtype=np.float32).reshape(length, 18) / 10
    previous = np.concatenate((current[:1], current[:-1]), axis=0)
    goal = np.repeat(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), length, axis=0)
    action = np.zeros((length, 4), dtype=np.float32)
    return np.concatenate((current, previous, goal, action), axis=1)


class TrajectoryInspectionTests(unittest.TestCase):
    def test_perfect_and_broken_previous_state_alignment(self):
        values = trajectory()
        self.assertEqual(diagnose_trajectory(values)["previous_alignment"]["max"], 0)
        broken = values.copy()
        broken[2, 18] += 2
        result = diagnose_trajectory(broken, tolerance=1e-6)["previous_alignment"]
        self.assertEqual(result["above_tolerance"], 1)
        self.assertEqual(result["max_timestep"], 2)
        self.assertAlmostEqual(result["max"], 2)

    def test_constant_and_drifting_goal(self):
        values = trajectory()
        self.assertEqual(diagnose_trajectory(values)["goal_drift"]["max"], 0)
        drifting = values.copy()
        drifting[3, 36] += 0.5
        result = diagnose_trajectory(drifting)["goal_drift"]
        self.assertEqual(result["above_tolerance"], 1)
        self.assertAlmostEqual(result["max"], 0.5)

    def test_action_oob_detection(self):
        values = trajectory()
        values[1, 39:] = [-1.1, -1, 1, 1.25]
        result = diagnose_trajectory(values)
        self.assertEqual(result["action_oob_count"], 2)
        self.assertEqual(result["action_oob_timesteps"], [1])
        self.assertAlmostEqual(result["action_oob_fraction"], 2 / 16)
        self.assertAlmostEqual(result["action_oob_timestep_fraction"], 1 / 4)
        self.assertAlmostEqual(result["action_max_overshoot"], 0.25)

    def test_nan_and_inf_detection(self):
        values = trajectory()
        values[0, 0] = np.nan
        values[1, 1] = np.inf
        result = diagnose_trajectory(values)
        self.assertEqual(result["nan_count"], 1)
        self.assertEqual(result["inf_count"], 1)

    def test_exports_do_not_modify_source(self):
        values = trajectory()
        original = values.copy()
        with tempfile.TemporaryDirectory() as tmp:
            export_csv(values, Path(tmp) / "trajectory.csv")
            export_txt(values, Path(tmp) / "trajectory.txt")
        np.testing.assert_array_equal(values, original)

    def test_loads_real_npz_and_packed_npy(self):
        values = trajectory()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            np.save(directory / "packed.npy", values[None], allow_pickle=False)
            np.savez_compressed(
                directory / "real.npz",
                episode_0_observations=values[:, :39],
                episode_0_actions=values[:, 39:],
                episode_0_rewards=np.zeros(len(values)),
                episode_0_length=len(values),
            )
            packed = load_trajectories(directory / "packed.npy")
            real = load_trajectories(directory / "real.npz")
        np.testing.assert_array_equal(packed.trajectories[0], values)
        np.testing.assert_array_equal(real.trajectories[0], values)
        self.assertIn("rewards", real.metadata[0])


    def test_percentiles_pool_all_valid_state_deltas(self):
        values = trajectory(5)
        stats = distribution_statistics(np.arange(1, 101, dtype=float))
        self.assertAlmostEqual(stats["median"], 50.5)
        self.assertAlmostEqual(stats["p90"], 90.1)
        dataset = aggregate_diagnostics((values, values[:2]))
        self.assertEqual(dataset["state_delta"]["count"], 5)
        self.assertEqual(dataset["trajectory_count"], 2)

    def test_reference_tails_and_state_dimension_statistics(self):
        real = trajectory(5)
        generated = real.copy()
        generated[2:, 0] += 100
        result = compare_datasets((real,), (generated,))
        self.assertGreater(result["generated_tail_exceedance"][">real_p99"]["fraction"], 0)
        self.assertEqual(len(result["real"]["state_dimensions"]), 18)
        self.assertGreater(result["state_support_exceedance"]["state_00"]["above_real_p99"], 0)
        self.assertIn("p01", result["generated"]["state_dimensions"]["state_00"])

    def test_action_saturation_and_overshoot_statistics(self):
        values = trajectory()
        values[:, 39:] = [[0.91, 0, 0, 0], [0.96, 0, 0, 0],
                           [1.02, 0, 0, 0], [-1.5, 0, 0, 0]]
        saturation = action_saturation_statistics(values[:, 39:])
        self.assertAlmostEqual(saturation["component_fraction"][">0.90"], 4 / 16)
        self.assertAlmostEqual(saturation["timestep_fraction"][">1.00"], 2 / 4)
        overshoot = action_overshoot_statistics((values,), top_k=1)
        self.assertEqual(overshoot["count"], 2)
        self.assertAlmostEqual(overshoot["median"], 0.26)
        self.assertAlmostEqual(overshoot["max"], 0.5)
        self.assertEqual(overshoot["top_components"][0]["timestep"], 3)

    def test_comparison_is_json_serializable_and_does_not_mutate(self):
        real, generated = trajectory(), trajectory()
        generated[1, 39] = 1.1
        real_before, generated_before = real.copy(), generated.copy()
        result = compare_datasets((real,), (generated,))
        json.dumps(result, allow_nan=False)
        np.testing.assert_array_equal(real, real_before)
        np.testing.assert_array_equal(generated, generated_before)
        self.assertEqual(len(result["generated"]["action_dimensions"]), 4)
        self.assertGreater(result["action_support_exceedance"]["action_00"]["generated_outside_real_p01_p99_fraction"], 0)


    def test_constant_state_tiny_deviation_uses_absolute_diagnostic(self):
        real = np.zeros((6, 18), dtype=np.float64)
        generated = real.copy()
        generated[:, 8] = [-1e-8, 0, 1e-8, 2e-8, -2e-8, 3e-8]
        before = generated.copy()
        constant, support = state_reference_diagnostics(real, generated)
        item = constant["state_08"]
        self.assertTrue(item["constant_reference"])
        self.assertNotIn("state_08", support)
        self.assertAlmostEqual(item["generated_abs_deviation"]["max"], 3e-8)
        self.assertEqual(item["fraction_above_tolerance"], 0)
        np.testing.assert_array_equal(generated, before)

    def test_constant_state_true_deviation_is_reported(self):
        real = np.zeros((5, 18), dtype=np.float64)
        generated = real.copy()
        generated[:, 4] = 1e-3
        constant, _ = state_reference_diagnostics(real, generated)
        item = constant["state_04"]
        self.assertAlmostEqual(item["generated_abs_deviation"]["max"], 1e-3)
        self.assertGreater(item["fraction_above_tolerance"], 0)

    def test_nonconstant_state_keeps_percentile_support(self):
        real = np.zeros((10, 18), dtype=np.float64)
        generated = real.copy()
        real[:, 3] = np.arange(10)
        generated[:, 3] = np.arange(10) + 1
        constant, support = state_reference_diagnostics(real, generated)
        self.assertNotIn("state_03", constant)
        self.assertFalse(support["state_03"]["constant_reference"])
        self.assertIn("below_real_p01", support["state_03"])

    def test_oob_timestep_clustering_and_no_mutation(self):
        trajectories = tuple(trajectory(9) for _ in range(5))
        for item in trajectories[:4]:
            item[3, 39] = 1.1
        trajectories[0][7, 40] = -1.2
        originals = tuple(item.copy() for item in trajectories)
        result = action_oob_by_timestep(trajectories, top_k=2)
        self.assertEqual(result["top_trajectory_fraction_timesteps"][0]["timestep"], 3)
        self.assertEqual(result["per_timestep"][3]["trajectory_count"], 4)
        self.assertEqual(result["per_timestep"][7]["trajectory_count"], 1)
        for item, original in zip(trajectories, originals):
            np.testing.assert_array_equal(item, original)

    def test_oob_by_timestep_all_zero(self):
        result = action_oob_by_timestep((trajectory(4), trajectory(4)))
        self.assertTrue(all(item["trajectory_fraction"] == 0 for item in result["per_timestep"]))
        self.assertTrue(all(item["component_fraction"] == 0 for item in result["per_timestep"]))


if __name__ == "__main__":
    unittest.main()
