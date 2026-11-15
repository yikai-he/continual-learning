---
status: current
last_verified: 2026-11-15
applies_to: continual-learning direct-43d migration and verification
---

# Direct 43D Generative Replay Migration

This document records the acceptance evidence for the current generator
contract. Current architecture details remain canonical in
[Project Architecture](ARCHITECTURE.md); the removed design is preserved in
[Legacy 22D Representation](archive/LEGACY_22D_REPRESENTATION.md).

## Current contract

```text
Expert trajectories
        ↓
Physical 43D trajectories (39D observation + 4D action)
        ↓
43D per-feature normalization
        ↓
DDPM or Flow Matching
        ↓
Direct 43D generation
        ↓
43D denormalization
        ↓
Generated historical replay
        ↓
Behavioral Cloning (39D observation → 4D action)
        ↓
Persistent GeneralPolicy
        ↓
Continual evaluation
```

Every generated timestep contains current state `0:18`, previous state `18:36`,
goal `36:39`, and action `39:43`. Both generators consume and produce all 43
features. An integer task ID and independently normalized 3D reference goal
remain external conditions; the three goal coordinates are nevertheless also
part of the generated output.

`diffusion_data.py` fits population z-score statistics over batch and time on
the stage training split, with the existing `1e-3` scale floor and no feature
clipping. Raw action mode leaves the last four coordinates unchanged.
Pre-`tanh` mode applies its transform only to those coordinates. Optional
generated-action projection likewise changes only actions.

There is no reduced trajectory representation, trajectory decoder,
previous-state shift, goal restoration, or consistency correction. BC and
evaluation retain the physical 39D-observation/4D-action interface.

## Generators and checkpoints

### DDPM

`TrajectoryDiffusion` predicts normalized clean 43D trajectories (`x_0`). At a
uniformly sampled discrete timestep, training adds Gaussian noise according to
the stored squared-cosine schedule and minimizes mean squared x0-prediction
error. Sampling begins with standard normal `(N,T,43)` noise and applies every
ancestral posterior step in reverse order. The temporal network uses position,
discrete time, task, and projected normalized-goal conditions.

New checkpoints use kind `trajectory_diffusion_43_v3`. Loading validates the
43-feature model configuration, schedule buffers, model state, task/action and
backbone contracts, and 43D dynamic plus 3D goal normalization statistics.
Known v2 direct-43D checkpoints remain loadable as the original Temporal CNN.

### Flow Matching

`TrajectoryFlowMatching` samples independent standard-normal source data and a
continuous time in `[0,1)`. Its linear conditional path is
`x_t=(1-t)x_0+t x_1`; the target velocity is `x_1-x_0`, trained with mean squared
velocity error. Sampling integrates the learned velocity from zero to one with
`torchdiffeq.odeint`.

Supported solvers are fixed-step `euler`, `midpoint`, `heun2`, and `rk4`, plus
adaptive `dopri5`. `integration_steps` controls fixed-step size; `rtol` and
`atol` control integration tolerances. The temporal network receives position,
continuous time, task, and projected normalized-goal conditions.

New checkpoints use kind `trajectory_flow_matching_43_v3`. It stores the 43D
model configuration and normalizer, 3D goal normalizer, state, backbone
identity, and replay provenance. Known v2 direct-43D checkpoints remain
loadable as the original Temporal CNN.

Neither loader accepts legacy reduced-representation checkpoints. There is no
automatic conversion. Historical configurations, checkpoints, and experiment
reports remain evidence for the software that produced them, but legacy
generator configurations are incompatible with the current strict schema.

The action representation and generated-action projection are shared by DDPM
and Flow Matching. Their fields remain under `diffusion:` for historical YAML,
checkpoint, and report compatibility.

## Consistency and interpretation

Direct 43D generation does not guarantee:

- previous state equals the preceding generated current state;
- the generated goal is constant across an episode;
- generated goals agree with the external conditioning goal; or
- states and actions form a physically or dynamically feasible rollout.

Replay providers require the physical shape and finite values, then report
goal deviation/variation, previous-state alignment/reset error, action-bound
violations, and NaN/Inf counts. They do not repair these values.

Structural consistency describes relationships among stored coordinates.
Physical or dynamical consistency asks whether the sequence could arise from
the environment. Downstream performance asks whether replay helps the BC policy
learn and retain tasks. These are separate measurements; success in one does
not establish either of the others.

Real expert and generated trajectories share the stage acceptance contract:
the physical shape and all values must be finite. Previous-state alignment and
within-episode goal consistency remain descriptive diagnostics and do not reject
either source.

## Completed verification

The complete automated suite was run with ROS/system plugin autoload disabled:

```bash
conda run --no-capture-output -n metaworld \
  env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/ -q
```

Result: `396 passed, 2 skipped, 78 subtests passed`.

Two collision-safe, two-task end-to-end smoke runs were also completed:

```bash
conda run --no-capture-output -n metaworld \
  python -u train_diffcrl.py \
  --config configs/diffcrl/diffcrl_smoke.yaml \
  --run-name direct43d_e2e_20261111

conda run --no-capture-output -n metaworld \
  python -u train_diffcrl.py \
  --config configs/diffcrl/flow_matching_smoke.yaml \
  --run-name direct43d_e2e_20261111
```

Both used `reach-v3 → hammer-v3`, four expert trajectories per task, two
generator epochs, two BC epochs, two evaluation episodes per learned task, ten
sampling/integration steps, and CPU execution. Both completed stages zero and
one, produced finite historical replay of shape `(4,200,43)`, valid generator
and GeneralPolicy checkpoints, evaluation reports, hash-validated completion
markers, and `RUN_COMPLETE`.

The retained outputs are under
`runs/diffcrl_smoke/direct43d_e2e_20261111/` and
`runs/flow_matching_smoke/direct43d_e2e_20261111/`. Each stage report records
the generated data path and structural diagnostics; completion markers bind
policy, generator, report, evaluation, and boundary-state artifacts by hash.

| Diagnostic | DDPM | Flow Matching |
|---|---:|---:|
| Previous-state alignment mean error | 0.05659 | 0.43545 |
| Generated goal/reference mean error | 0.06603 | 0.10825 |
| Within-trajectory goal variation | 0.01071 | 0.09289 |

The nonzero diagnostics demonstrate that the end-to-end replay path preserved
generated previous-state and goal coordinates rather than imposing the old
structural decoder. These smoke runs establish pipeline execution and artifact
integrity only. Their tiny budgets and success rates do not establish generator
quality, algorithm superiority, or scientific performance.
