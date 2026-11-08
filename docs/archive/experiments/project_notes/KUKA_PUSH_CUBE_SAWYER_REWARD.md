---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# KUKA Cube Push: Sawyer Push v2 reward adaptation

## Scope and source

This is a source-based adaptation named `sawyer_push_v2_adapted`, enabled only
for `kuka-push-cube-v3`. It does not change physics, success, termination,
observations, actions, control, or task sampling.

The authoritative local source is
`metaworld-kuka-v3/metaworld/envs/sawyer_push_v3.py`, class
`SawyerPushEnvV3`, `compute_reward()`'s `reward_function_version == "v2"`
branch and `evaluate_state()`. The checkout reports MetaWorld 3.1.1 at commit
`2ed78a08c7bf0ce0fd7aac4e44fd0ee3f9dd76ea`.

An older class/file literally named `SawyerPushEnvV2` and the v1 source are not
present in this shallow checkout, its Git history, or the installed environment.
MetaWorld 3.1.1 exposes the task as EnvV3 while selecting the updated/original
reward through `reward_function_version`. Consequently this adaptation is an
exact reproduction of the locally available Sawyer Push **v2 reward branch**,
not a claim about an unavailable historical class implementation or the entire
Sawyer task.

## Original Sawyer reward

Let `g` be `_gripper_caging_reward`, `p` the in-place tolerance, `d_tcp` the
3D TCP/object-center distance, `o` the normalized gripper opening, and `d_goal`
the 3D object/goal-center distance. The source executes assignments:

```
reward = 2 * g
if d_tcp < 0.020 and o > 0:
    reward += 1 + reward + 5 * p
if d_goal < 0.050:
    reward = 10
```

Thus the middle branch is `1 + 4g + 5p`, while success overrides rather than
adds. Success is evaluated every timestep and is not latched. The in-place term
is `tolerance(d_goal, bounds=(0, 0.05), margin=initial_goal_distance,
sigmoid="long_tail")`.

The source caging call uses:

| Parameter | Sawyer value |
|---|---:|
| `object_reach_radius` | 0.010 m |
| `obj_radius` | 0.015 m |
| `pad_success_thresh` | 0.050 m |
| `xz_thresh` | 0.005 m |
| `high_density` | true |

Caging uses current left/right pad Y distances, their distances to the initial
object position, TCP/object XZ distance, current gripper action, `obj_init_pos`,
and `init_tcp`. Those simulator/episode values are hidden state beyond the
observation passed to `compute_reward()`.

`evaluate_state()` reports success at `d_goal <= 0.05`, `near_object` at
`d_tcp <= 0.03`, grasp/contact diagnostics, raw caging, in-place, and total
reward. The reward branch itself uses strict `<` comparisons.

## KUKA Cube adaptation

The adapted equation changes the robot-specific proximity gate and maps the
success override to the Cube task's existing inclusive success indicator:

```
reward = 2 * g
if d_tcp < 0.050 and o > 0:
    reward += 1 + reward + 5 * p
if d_goal <= 0.050:
    reward = 10
```

| Item | Sawyer | KUKA Cube | Classification |
|---|---:|---:|---|
| Equation and weights | as above | same | A: exact reproduction |
| TCP/object branch | 0.020 m | 0.050 m | B: robot geometry adaptation |
| `object_reach_radius` | 0.010 m | 0.010 m | A |
| `obj_radius` | 0.015 m | 0.015 m | A |
| `pad_success_thresh` | 0.050 m | 0.050 m | A |
| `xz_thresh` | 0.005 m | 0.005 m | A |
| high density | true | true | A |
| in-place function | long-tail, initial distance | same | A |
| success comparison | `< 0.050 m` | existing `<= 0.050 m` | D: task contract |
| TCP | Sawyer fingertip midpoint | Robotiq inner-pad-site midpoint | B |
| pad bodies | Sawyer pads | mapped Robotiq left/right pads | B |

The Sawyer puck XML has 20 mm lateral radius; the cube has 20 mm half-width.
Therefore the source's 15 mm caging radius is retained—there is no object-size
adaptation. This parameter is a caging bound, not a literal declaration of the
whole object's radius.

Three successful fixed-task scripted pushes (Tasks 4, 19, and 35) measured 286
post-step physical-contact samples. TCP/cube-center contact distances ranged
from 29.1 to 44.7 mm; 95th percentiles were 35.2--43.0 mm. Noncontact samples
ranged from 33.0 to 223.2 mm, so proximity and contact overlap. The original
20 mm branch is unreachable during those pushes. A 50 mm gate is the smallest
round centimeter threshold covering all observed contact and matches the
existing source `pad_success_thresh`; it remains a proximity condition, not a
claim of physical contact.

## Diagnostics and branch semantics

The info dictionary exposes:

- raw: `reward_object_grasped`, `reward_in_place`, `reward_near_object`
- weighted/conditional: `reward_base_contribution`,
  `reward_near_object_contribution`, `reward_pre_success_override`
- override/final: `reward_success_override`, `reward_total`
- geometry: physical, named-pad, follower contact, object-goal and TCP-object
  distances

`reward_total == reward_pre_success_override` outside success. During success,
`reward_total == 10` regardless of the pre-override value. These are assignment
branches, not five independently additive components.

## Offline comparison

This exploratory comparison re-scored deterministic rollouts on fixed Tasks 4,
19, and 35. Available policies were v3.1 at 60k/100k/180k/200k, stopped v3.3
at 20k/40k, and the unchanged scripted controller. Values are mean episode
returns over three tasks; reward scales differ and must not be compared as
policy-quality scores.

| trajectory source | success | v3.1 | v3.3 | Sawyer adapted | near-branch rate |
|---|---:|---:|---:|---:|---:|
| v3.1 60k | 2/3 | 133.37 | 159.37 | 958.26 | 49.8% |
| v3.1 100k | 0/3 | 14.14 | 21.01 | 28.68 | 1.8% |
| v3.1 180k | 0/3 | 5.43 | 0.53 | 10.86 | 0% |
| v3.1 200k | 0/3 | 6.15 | 0.36 | 12.31 | 0% |
| v3.3 20k | 0/3 | 2.00 | 0.61 | 4.00 | 0% |
| v3.3 40k | 0/3 | 3.55 | 0.50 | 7.10 | 0% |
| scripted | 3/3 | 217.57 | 282.38 | 735.87 | 63.0% |

Mean Sawyer-adapted discounted returns (`gamma=0.99`) were respectively 335.82,
14.00, 4.97, 5.12, 1.94, 3.46, and 188.84. Sawyer-adapted mean rewards earned
on steps with <=10 micrometers XY motion were 194.65, 16.11, 10.33, 12.31,
4.00, 7.10, and 39.55 per episode. This positive stationary reward is expected
from the source caging/in-place structure, not classified as an implementation
bug. Per-step adapted rewards ranged from 0 to 10 in this sample.

The scripted trajectories included tipping up to 45 degrees and all succeeded;
the reward has no orientation term. Object motion without named-pad contact is
not directly rewarded as motion, but can improve the in-place raw value and
subsequent near-branch reward. Moving away reverses that in-place improvement;
there is no explicit signed-progress term.

Offline rankings do not show that SAC will learn a better policy. The subset is
deliberately representative, not a full 50-task estimate.

## Experiment and reproducibility

Configuration:
`configs/sac/kuka_push_cube_v3_sawyer_push_v2_adapted_200k.yaml`.
It matches the completed v3.1 optimization settings and uses the same 50-task
fixed bank, task-set seed 10000, reset seeds 10000--10049, 200-step horizon,
20k evaluation/checkpoint interval, replay-buffer checkpoints, best-return and
best-success selection, final model, TensorBoard, and early stopping.

Output is isolated at:
`runs/reward_ablations/kuka-push-cube-v3/sawyer_push_v2_adapted/seed_0_200k/`.
The preflight found no existing directory.

Training command (not executed):

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
conda run --no-capture-output -n metaworld \
  python -u train_sac.py \
  --config configs/sac/kuka_push_cube_v3_sawyer_push_v2_adapted_200k.yaml
```

TensorBoard command (not executed):

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
conda run --no-capture-output -n metaworld \
  tensorboard --logdir runs/reward_ablations/kuka-push-cube-v3/sawyer_push_v2_adapted/seed_0_200k/tensorboard
```

Known limitations are the unavailable historical EnvV2/v1 files, post-control-
step rather than substep contact measurements, only three offline tasks, and
the deliberate robot-specific 50 mm proximity adaptation.
