# KUKA Cube Push Goal-Retention Counterfactual

Checkpoint: `/home/yikai/Documents/Code/Final/continual-learning/runs/reward_ablations/kuka-push-cube-v3/sawyer_push_v2_adapted/seed_0_200k/best_success_model/best_success_model.zip`

```json
{
  "checkpoint": "/home/yikai/Documents/Code/Final/continual-learning/runs/reward_ablations/kuka-push-cube-v3/sawyer_push_v2_adapted/seed_0_200k/best_success_model/best_success_model.zip",
  "task_set_seed": 10000,
  "reset_seed_base": 10000,
  "target_radius": 0.05,
  "velocity_frame": "MuJoCo world frame; TCP velocity is finite-difference over one control step",
  "stop_gripper": "holds the actuator command present at first success",
  "retraction": {
    "xy_distance": 0.08,
    "z_clearance": 0.06,
    "direction": "world-frame XY from cube toward TCP at first success",
    "gripper_action": -1.0
  },
  "validation": {
    "pre_success_actions_identical": true,
    "pre_success_states_identical": true,
    "never_success_trajectories_identical": true,
    "first_success_steps_identical": true,
    "stop_has_zero_translation": true,
    "baseline_ever_success_18_percent": true,
    "baseline_final_success_4_percent": true,
    "success_threshold_unchanged": true
  },
  "modes": {
    "original": {
      "episodes": 50,
      "ever_success_count": 9,
      "ever_success_rate": 0.18,
      "final_success_count": 2,
      "final_success_rate": 0.04,
      "goal_retention_rate": 0.2222222222222222,
      "entered_and_exited_count": 7,
      "continuous_retention_count": 2,
      "mean_final_distance": 0.17215318933175258,
      "mean_post_success_displacement": 0.12814051404246074,
      "mean_successful_steps": 42.77777777777778,
      "mean_post_success_contact_steps": 1.4444444444444444,
      "mean_cube_speed_at_first_success": 0.7758582344068125,
      "workspace_clipping_events": 7764,
      "successful_contact_releases": 0,
      "retraction_completions": 0
    },
    "stop": {
      "episodes": 50,
      "ever_success_count": 9,
      "ever_success_rate": 0.18,
      "final_success_count": 3,
      "final_success_rate": 0.06,
      "goal_retention_rate": 0.3333333333333333,
      "entered_and_exited_count": 6,
      "continuous_retention_count": 3,
      "mean_final_distance": 0.16666994987991782,
      "mean_post_success_displacement": 0.0969798740863676,
      "mean_successful_steps": 52.77777777777778,
      "mean_post_success_contact_steps": 0.0,
      "mean_cube_speed_at_first_success": 0.7758582344068125,
      "workspace_clipping_events": 6784,
      "successful_contact_releases": 0,
      "retraction_completions": 0
    },
    "retract": {
      "episodes": 50,
      "ever_success_count": 9,
      "ever_success_rate": 0.18,
      "final_success_count": 3,
      "final_success_rate": 0.06,
      "goal_retention_rate": 0.3333333333333333,
      "entered_and_exited_count": 6,
      "continuous_retention_count": 3,
      "mean_final_distance": 0.16666994987991782,
      "mean_post_success_displacement": 0.0969798740863676,
      "mean_successful_steps": 52.77777777777778,
      "mean_post_success_contact_steps": 0.0,
      "mean_cube_speed_at_first_success": 0.7758582344068125,
      "workspace_clipping_events": 6784,
      "successful_contact_releases": 0,
      "retraction_completions": 9
    },
    "terminate": {
      "episodes": 50,
      "ever_success_count": 9,
      "ever_success_rate": 0.18,
      "final_success_count": 9,
      "final_success_rate": 0.18,
      "goal_retention_rate": 1.0,
      "entered_and_exited_count": 0,
      "continuous_retention_count": 0,
      "mean_final_distance": 0.1624086905095318,
      "mean_post_success_displacement": 0.0,
      "mean_successful_steps": 1.0,
      "mean_post_success_contact_steps": 0.0,
      "mean_cube_speed_at_first_success": 0.7758582344068125,
      "workspace_clipping_events": 6784,
      "successful_contact_releases": 0,
      "retraction_completions": 0
    }
  }
}
```
