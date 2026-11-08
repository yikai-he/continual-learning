---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# KUKA Cube Push SAC Baseline

## Research objective

This single-seed exploratory baseline tests whether Stable-Baselines3 SAC can learn to move the 40 mm KUKA Cube Push object into the existing 50 mm-radius goal region using reward `v3_1`. Cube rotation, rolling, face changes, and tipping are allowed. They are diagnostics, not failures; success is determined only by the environment's existing object-to-goal criterion.

## Environment

- Backend/task: `kuka-v3` / `kuka-push-cube-v3`
- Production cube: 40 x 40 x 40 mm, 0.75 kg, original inertia and collision geometry
- Observation/action: `(39,)` / `(4,)`
- Horizon: 200 environment steps
- Fixed wrist and production Cartesian-delta/mocap controller
- Reward: inherited Push `v3_1`, unchanged

The inherited caging helper still uses `obj_radius=0.015 m`, while the cube half-width is 0.020 m. This first baseline intentionally retains that parameter for Cylinder comparability. Preflight checks found finite rewards and components, but this geometric approximation may alter pre-contact shaping and should be treated as a known limitation rather than silently recalibrated.

## SAC configuration

| Setting | Value |
|---|---:|
| Training seed | 0 |
| Total steps | 200,000 |
| Policy | SB3 `MlpPolicy` |
| Network | `[256, 256, 256, 256]` |
| Learning rate | 3e-4 |
| Batch size | 128 |
| Replay capacity | 1,000,000 |
| Learning starts | 10,000 |
| Gamma | 0.99 |
| Tau | 0.005 |
| Train frequency | 1 |
| Gradient steps | 1 |
| Entropy coefficient | `auto` |
| Device | `auto` |

The primary reference is the completed Cylinder run at `runs/reward_ablations/kuka-push-v3/reward_v3_1_lr3e4/seed_0_200k/`. Its resolved configuration confirms learning rate 3e-4 and checkpoints every 20k. The older Cylinder directory without the `lr3e4` suffix used 1e-3 and 50k checkpoints and is not the matched optimization reference.

## Fixed-task evaluation

- Mode: `fixed-tasks`
- Episodes/tasks: 50
- Task-set seed: 10000
- Ordered reset seeds: 10000 through 10049
- Frequency: every 20,000 training steps, plus the existing initial evaluation
- Deterministic policy inference
- Primary success metric: any-step environment success
- Existing secondary outputs: final-step and sustained success diagnostics, first-success step, return mean/std, minimum/final goal distance, maximum progress/displacement, named-pad physical contact rates, direction-of-progress rates, and clipped-contact-without-progress rate

The current evaluator does not record cube tilt/rotation or follower-link contact. Those are unavailable training-evaluation diagnostics and are not added here because tipping does not alter success semantics.

## Checkpoints and stopping

- Periodic model and matching replay-buffer sidecars every 20,000 global steps
- `best_success_model/best_success_model.zip`: highest fixed-task success; mean return breaks ties
- `best_model/best_model.zip`: highest mean return
- `final_model.zip`: the policy when training exits, including early stopping
- Early stopping: enabled after at least 100,000 steps when success is at least 95% for three consecutive evaluations

The trainer always executes final-model saving after `model.learn()` returns normally, including when the callback stops training early. It then writes checkpoint manifests, summaries, and the completion marker.

## Expected output

`runs/reward_ablations/kuka-push-cube-v3/reward_v3_1/seed_0_200k/` will contain the trainer's existing artifacts: `config.yaml`, `config.json`, `run_manifest.json`, lifecycle marker, TensorBoard events, evaluation NPZ/JSON histories, periodic models and replay buffers, best-success and best-return models, `final_model.zip`, checkpoint manifests, `evaluation_history.csv`, `run_summary.json`, and `EXPERIMENT_SUMMARY.md`.

The directory did not exist during preflight. The trainer refuses to overwrite it; if it appears before launch, it creates a timestamped sibling instead.

## Manual launch

Do not run until the configuration and clean output target have been reviewed.

```bash
cd /home/yikai/Documents/Code/Final/continual-learning

conda run --no-capture-output -n metaworld \
  python -u train_sac.py \
  --config configs/sac/kuka_push_cube_v3_reward_v3_1_200k.yaml
```

TensorBoard after the run directory has been created:

```bash
cd /home/yikai/Documents/Code/Final/continual-learning

conda run --no-capture-output -n metaworld \
  tensorboard \
  --logdir runs/reward_ablations/kuka-push-cube-v3/reward_v3_1/seed_0_200k/tensorboard
```

## Limitations

- One seed cannot establish learning reliability or statistical superiority.
- The fixed 50-task bank measures deterministic performance on one specified task set.
- The inherited caging radius is Cylinder-derived and not a perfect cube geometry model.
- Cube tipping is permitted and is not represented in the trainer's evaluation archive.
- Workspace/contact diagnostics are sampled at environment-step resolution, not every MuJoCo substep.
