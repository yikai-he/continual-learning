---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# KUKA Cube Push environment and feasibility study

## Scope

`kuka-push-cube-v3` is an independent task for testing a 40 mm cube under the established KUKA Push controller and reward semantics. The original `kuka-push-v3` Python implementation and XML were not changed. No SAC configuration was created and no training was run.

## Verified implementation facts

The new class subclasses `KukaPushEnvV3` and changes only `model_name`. Consequently it inherits the same Cartesian delta controller, fixed wrist orientation, workspace, 39D observation, 4D action, 200-step horizon, XY sampling ranges, 0.15 m object-goal separation, 5 cm success radius, reset procedure, collision query, and reward code.

The cube XML uses one aligned visual/collision geom:

- MuJoCo type: `box`
- half-extents: `(0.02, 0.02, 0.02)` m, giving verified 40 x 40 x 40 mm dimensions
- body/geom center at z=0.02 m
- mass: 0.75 kg
- diagonal inertia: `(0.0002, 0.0002, 0.0002)` kg m2
- friction: `(1, 0.1, 0.002)`; `condim=4`
- `solimp=(.99,.99,.01)` and `solref=(.01,1)`, matching the cylinder

For a uniform solid cube, each principal moment is `m a^2/6 = .75*.04^2/6 = .0002 kg m2`. Shape is therefore not the only changed physical quantity: contact geometry and moment of inertia differ. The cylinder asset explicitly uses `.000880012 kg m2` on all three axes, rather than the analytical inertia of a uniform 30 mm-radius, 30 mm-tall cylinder.

After settling, the cube center was z=0.019968 m, XY drift was below 10 micrometers over the stability test, and its local vertical axis remained aligned to better than 0.1%. The observation reports the body center and quaternion in the same slots as Cylinder Push: current object position `4:7`, orientation `7:11`, previous object position `22:25`, and goal `36:39`.

## Reward compatibility

The task loads and steps under `v2`, `v3`, `v3_1`, and `v3_2_retention`. Physical contact, signed/clipped progress, success latches, and diagnostic logging are inherited unchanged. The recommended first comparison remains v3.1, as requested.

One known mismatch was intentionally not retuned: `_gripper_caging_reward` still receives `obj_radius=.015`, inherited from Cylinder Push, while the cube's axis-aligned half-width is 0.020 m and its projected width changes with yaw. Thus caging is a shared reward-semantic control, not a geometrically exact cube enclosure measure. Changing it would alter reward semantics and requires a separate approved ablation.

## Measured physical validation

The smoke controller established actual `objGeom`/Robotiq-pad collisions and displaced the cube more than 20 mm. The cube was never teleported during a push. Registration, reset, observation, and success tests use real MuJoCo stepping.

The isolated comparison artifacts are in `/tmp/kuka_push_cube_validation/`. The experiment used nine matched task vectors (indices 0, 6, 12, 18, 24, 30, 36, 42, 48) and eight pose settings per shape:

- `rho` 40, 50, and 60 mm at z=42 mm
- z 40, 45, and 50 mm at `rho=60 mm`
- lateral offsets -15 and +15 mm at `(rho,z)=(60,42) mm`

The same 20-step approach, 35-step contact phase, 145-step push, action scaling, horizon, task vectors, success radius, and controller were used for each shape. An effective trial was defined before aggregation as physical pad contact, at least 20 mm net goal-directed displacement, and mean moving-contact alignment at least 0.8.

| Metric (72 trials/shape) | Cylinder | Cube |
|---|---:|---:|
| Physical-contact probability | 94.4% | 100.0% |
| Effective-push fraction | 87.5% | 84.7% |
| Any-step success | 68.1% | 48.6% |
| Final success | 59.7% | 47.2% |
| Sustained success | 68.1% | 47.2% |
| Trial tipping frequency | 1.4% | 87.5% |
| Mean final rotation change | 8.2 deg | 105.7 deg |
| Mean motion alignment | .970 | .919 |
| Mean force alignment | .943 | .883 |
| Mean TCP tracking error | 9.25 mm | 11.64 mm |
| Workspace-clipped step rate | 16.6% | 50.4% |

At the prior calibrated `(rho,z)=(60,42) mm` pose, cylinder contact/effective/any/final/sustained rates were all 100% on this subset. Cube contact and effective pushing were 100%, but any/final/sustained success were 66.7%; mean maximum cube tilt was 79.7 degrees.

Soft floor contact penetration reported by MuJoCo averaged 0.63 mm for the cylinder and 2.73 mm for the cube, with maxima 5.32 and 4.57 mm respectively. No Robotiq pad-floor collision occurred. These signed contact distances are solver penetration, not evidence that the XML started below the table.

## Threshold sensitivity

The qualitative result depends somewhat on alignment threshold but not displacement threshold in the tested range:

| Effective threshold | Cylinder | Cube |
|---|---:|---:|
| displacement >=10/20/30 mm, alignment >=.7 | 87.5% | 100.0% |
| displacement >=10/20/30 mm, alignment >=.8 | 87.5% | 84.7% |
| displacement >=10/20/30 mm, alignment >=.9 | 80.6% | 73.6% |

The cube always acquired contact and passes a permissive alignment definition, but its direction quality, goal success, and stability were worse. This limited experiment does not support the hypothesis that the requested cube is easier to control. It also does not prove SAC cannot learn it.

The sweep was intentionally not extended to all 50 tasks because systematic tipping violates the prerequisite that the controller behave correctly. A larger task count would estimate the same problematic regime more precisely without resolving it.

## Registration and training integration

The task is registered through `Meta-World-KUKA/MT1`, appears in the KUKA adapter's supported task list, supports deterministic fixed-task serialization, and is accepted as an individual SAC task. It was not added to any continual-learning sequence. Push diagnostics in `train_sac.py` recognize both push task names. Reward identity will be recorded through the existing configuration/manifest machinery when a future configuration is explicitly created.

## Interactive viewer

The existing GLFW viewer discovers the task through `KUKA_V3_TASKS`; no viewer-specific code change was required. Because no cube policy exists, the following command uses the completed cylinder v3.2 policy only as a deterministic action source for visual/physics inspection. It is not a cube-policy evaluation:

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
MUJOCO_GL=glfw conda run --no-capture-output -n metaworld \
  python -u scripts/render_kuka_push_policy.py \
  --model runs/reward_ablations/kuka-push-v3/reward_v3_2_retention/seed_0_200k/final_model.zip \
  --env-name kuka-push-cube-v3 \
  --reward-version v3_1 \
  --task-set-seed 10000 --task-index 19 --seeds 10019 \
  --control-hz 20 --window-size 1280 900 --no-loop
```

## Recommendation and remaining uncertainty

Do not start a cube SAC experiment yet. First determine why the cube rolls onto an edge/side so frequently. The physically derived cube inertia is substantially lower than the existing cylinder's explicit inertia, so this experiment changes both shape and rotational dynamics. A useful next controlled study would compare the physical-inertia cube against an inertia-matched cube, or adjust contact height/controller behavior, but either changes the experimental definition and should be approved separately.

Even if scripted tolerance improves, broader scripted success alone would not guarantee easier SAC learning. Conversely, the inherited caging-radius mismatch may make learning harder even when physical pushing is feasible. These are distinct hypotheses and should be isolated one at a time.
