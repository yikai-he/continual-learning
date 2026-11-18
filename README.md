---
status: current
last_verified: 2026-11-17
applies_to: continual-learning
---

# Continual Learning for Robotic Manipulation

Research code for the bachelor’s thesis **Continual Learning for Robotic
Manipulation towards Real-World Deployment**. The repository studies whether
trajectory-level generative replay can reduce forgetting as a manipulation
policy learns a sequence of simulated tasks.

## SAC training

`train_sac.py` is the preferred entry point for new single-task SAC experts.
`best_model.zip` remains the compatible inference path, while new runs retain
synchronized Best and Final model/replay snapshots for same-task continuation.
Selection maximizes `(success_rate, mean_return)` lexicographically; exact ties
keep the earlier Best. Actor-only warm-start and historical model-state-only
continuation remain supported. Replay buffers stay in training runs and are not
registered as experts. See [SAC training](docs/SAC_TRAINING.md) for details.

Optimization uses the `runtime.seed` bank, validation defaults to bank `10000`,
and post-selection final tests default to banks `20000` and `20001`. Serialized
task identities are checked for pairwise disjointness before training.
The SAC entropy mode and network are fixed implementation choices. New formal
runs also reproduce the Legacy schedule: random actions through timestep
10,000, optimization from timestep 1,000, 50 updates every 50 steps, and the
Gaussian entropy target corresponding to output standard deviation `0.089`.
New experts use the Legacy `continual_diff_pytorch`
four-layer 256-unit actor/critic core: the first linear layer is followed by
LayerNorm and Tanh, and the remaining layers use LeakyReLU (`0.01`). Stable-
Baselines3 still owns squashed-Gaussian sampling, twin/target critics, entropy
tuning, replay, and optimization; the actor evaluates the tanh correction with
the Legacy stable pre-tanh formula to retain finite saturation gradients.
Existing standard-SB3 checkpoints retain their original serialized policy
class; historical `LegacySACPolicy` weights remain loadable with the corrected
distribution.

```bash
python train_sac.py --config configs/sac/sac.yaml --seed 0 --device cpu
```

Three Sawyer Early Stopping runs are prepared as isolated configurations:
Push Back receives only compatible Faucet Close actor tensors into otherwise
fresh SAC state; Stick Pull and Push restore same-task SB3 model state but reset
and refill replay because no replay-buffer artifact exists. Each validation uses
50 deterministic episodes: one ordered pass over all 50 unique MetaWorld MT1
configurations from bank 10000. Checks run every 40,000 steps and require three
consecutive 50/50-success results; 49/50 resets the target counter. Banks 20000
and 20001 remain reserved for post-selection testing. Plateau stopping is
disabled because SAC may improve after long validation plateaus. The minimum
new-step requirements for target-success stopping are 120k, 400k, and 200k for
Stick Pull, Push, and Push Back; maximum budgets remain 500k, 1M, and 1M.

The general fixed-task evaluator still supports episode counts above the bank
size by deterministic cycling, recording repeated indices, hashes, reset seeds,
and pass counts. These three experiments intentionally use one pass so no task
configuration repeats within a validation check.

```bash
# Validate configs, manifests, hashes, and output isolation; do not train:
PYTHON_BIN=python ./tools/experiments/train_sawyer_early_stop_three.sh --preflight-only

# Run Stick Pull, Push, then Push Back sequentially:
PYTHON_BIN=python ./tools/experiments/train_sawyer_early_stop_three.sh --run
```

Both modes refuse any existing destination under
`runs/sac_target_success_only_20261117/`; historical Plateau-enabled runs under
`runs/sac_early_stop_20261117/` remain separate. Normal completion is either
three sustained perfect validations after the task's minimum new steps or the
maximum additional-step budget. Inspect `results.json` → `early_stopping` for
the recorded reason (`target-success` or `maximum-training-steps`); do not infer
it from TensorBoard. Reaching perfect validation is not guaranteed.

The individual configs are
`sawyer_push_back_from_faucet_close_early_stop.yaml`,
`sawyer_stick_pull_continuation_early_stop.yaml`, and
`sawyer_push_continuation_early_stop.yaml` under `configs/sac/`.

The main pipeline, DiffCRL, is inspired by DISTR (Diffusion-based Trajectory
Replay), but it is **not an exact reproduction**. DISTR trains an Immediate
Policy during continual reinforcement learning. This implementation first
trains independent single-task SAC experts, collects their trajectories, and
continually distils those trajectories into one persistent behavior-cloning
policy. Diffusion or Flow Matching models provide old-task replay.

The recommended pipeline collects every requested expert episode with the
deterministic action from each frozen SAC expert, combines current data with No
Replay, Expert Recollection, DDPM, or Flow Matching replay, trains the persistent
`GeneralPolicy` with physical-action MSE, and evaluates it deterministically on
fixed task banks. Expert Recollection performs fresh rollouts from frozen
historical experts; it is distinct from the retired stored real-trajectory mode.
The canonical `configs/diffcrl/diffcrl.yaml` makes the key choices explicit:

```yaml
continual:
  expert_action_mode: deterministic
  collection_mode: all
bc:
  loss: deterministic_mse
evaluation:
  mode: fixed-tasks
```

Squashed-Gaussian NLL, stochastic expert collection, success-preferred
collection, and stochastic NLL execution remain supported for reproducing the
corresponding controlled ablations; they are not the recommended defaults.

## Documentation

| Document | Purpose | Status | Last verified |
|---|---|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Current pipeline, module, data, evaluation, and artifact contracts | current | 2026-11-11 |
| [Direct 43D migration](docs/DIRECT_43D_MIGRATION.md) | Generator migration contract and completed acceptance evidence | current | 2026-11-11 |
| [Development status](docs/DEVELOPMENT_STATUS.md) | Implemented and validated scope, limitations, and remaining work | current | 2026-11-11 |
| [SAC training](docs/SAC_TRAINING.md) | Expert training, evaluation, checkpoints, and smoke validation | current | 2026-11-07 |
| [Workflow index](docs/WORKFLOW_INDEX.md) | Supported entry points and configuration families | current | 2026-11-11 |
| [Flow replay architecture](docs/FLOW_REPLAY_ARCHITECTURE.md) | Flow Matching replay and allocation contract | current | 2026-11-11 |
| [Runs inventory](docs/RUNS_INVENTORY.md) | Retained canonical artifact families | current | 2026-11-07 |
| [Research code standards](docs/RESEARCH_CODE_STANDARDS.md) | Local quality priorities and proportional acceptance checks | current | 2026-11-07 |
| [Documentation archive](docs/archive/README.md) | Historical reports, provenance, and archive conventions | current index | 2026-11-11 |

Historical documents describe their original project state and are not current
operating instructions.

### Documentation maintenance

Update an existing canonical document before creating another Markdown file.
Maintained documents must carry `status` and an actually verified
`last_verified` date. Mark historical experiments explicitly, update relevant
documentation with behavior changes, and never silently rewrite archived
experiment records or refresh verification dates without checking their claims.

## Implemented scope

### Environments

- **MetaWorld v3** through `Meta-World/MT1`.
- **KUKA v3** through the workspace-local `Meta-World-KUKA/MT1` fork.

All adapters expose a 39-dimensional observation, a 4-dimensional continuous
action, and episode-level success metadata. Fixed-task evaluation is supported
for both backends.

Hammer and Peg Unplug use fixed reward definitions in their KUKA-v3 task
implementations. Their former reward-variant and progress-weight configuration
fields are intentionally not part of this repository's environment API.

### Methods

- **Single-task SAC experts**: one independent Stable-Baselines3 SAC policy per
  task.
- **No Replay**: persistent `GeneralPolicy` trained only on current expert data.
- **Expert Recollection**: fresh deterministic rollouts from frozen historical
  SAC experts in their task environments. This is a privileged reference with
  historical expert and environment access, not memory-based replay.
- **Diffusion Replay**: current expert data plus task- and goal-conditioned DDPM
  trajectory replay.
- **Flow Matching Replay**: current expert data plus trajectories produced by a
  task- and goal-conditioned conditional flow model.

Both generators model the complete 43D physical trajectory directly (39D
observation plus 4D action) while retaining separate task and goal conditions.
Generated previous-state and goal coordinates are not reconstructed or replaced.

The four active DiffCRL replay modes are `none`, `expert-recollection`,
`diffusion`, and `flow-matching`. Historical YAML using `real` records the
retired stored-trajectory protocol and is intentionally rejected by the active
loader rather than reinterpreted. The ContinualManipulation Forward Transfer metric requires
explicitly aligned learning and reference curves; it is not inferred from the endpoint evaluation
matrix.

## Repository structure

```text
continual-learning/
├── train_sac.py                 # single-task SAC experts
├── train_diffcrl.py             # expert-driven continual BC and replay
├── run_replay_comparison.py     # no/recollection/DDPM comparison runner
├── manage_diffcrl_sync.py       # validated stage backup/restore
├── configs/                     # active, experimental, and archived YAML
├── src/
│   ├── continual/               # BC, replay, generators, evaluation, artifacts
│   └── support/                 # manifests, hashing, persistence, sync
├── scripts/                     # collection and evaluation entry points
├── tests/                       # unit, regression, and integration tests
├── docs/                        # architecture, protocols, audits, status
└── runs/                        # generated and retained experiment artifacts
```

The root entry points are retained for compatibility. Specialized commands
live under `scripts/`; historical configurations remain under `configs/archive/`
or `configs/sac/archive/` and should not be treated as current defaults.

## Installation

The tested direct dependencies are pinned in `requirements.txt`. From the
repository root:

```bash
conda create -n metaworld python=3.10
conda activate metaworld
python -m pip install -r requirements.txt
```

PyTorch is installed transitively by Stable-Baselines3. Select a CUDA-matched
PyTorch build separately when GPU execution is required.

KUKA experiments require `metaworld-kuka-v3/` as a sibling checkout:

```text
Thesis/
├── continual-learning/
└── metaworld-kuka-v3/
```

Install the fork into the same environment:

```bash
python -m pip install -e ../metaworld-kuka-v3 --no-deps --no-build-isolation
```

The fork provides both `metaworld` and `metaworld_kuka`; merely placing it next
to this repository does not make it importable.

## Quick start

Use the smoke configurations to verify a workflow without launching a formal
experiment:

```bash
python train_sac.py --config configs/sac/sac_smoke.yaml --seed 0
python train_diffcrl.py --config configs/diffcrl/diffcrl_smoke.yaml \
  --seed 0 --run-name smoke-seed0
```

These commands still create environments and train models. They are pipeline
checks, not scientific experiments and not evidence of method performance.

### Train experts and DiffCRL

DiffCRL requires a compatible SAC expert for every task. Train an expert using
a task-specific copy of `configs/sac/sac.yaml`:

```bash
python -u train_sac.py --config path/to/expert.yaml --seed 0
```

Set `continual.experts` in the DiffCRL YAML to the qualified expert checkpoints,
then run:

```bash
python -u train_diffcrl.py \
  --config configs/diffcrl/diffcrl.yaml \
  --seed 0 \
  --run-name diffusion-seed0
```

Named curricula may use `continual.sequence`. Sawyer / MetaWorld-v3 supports
`cm1`, `cm3`, `cm5`, and `cm10`; CM10 follows the original Continual World CW10
task ordering with MetaWorld-v3 identifiers. KUKA / MetaWorld-KUKA-v3 supports
the custom `kuka_cm4`, `kuka_cm5`, and `kuka_cm10` curricula. The former KUKA
CM5 ordering remains available as `kuka_cm5_legacy`; the project also retains
the backward-compatible `kuka_cw10_inspired_5` alias. The complete definitions
and protocol caveats are documented in the
[benchmark README](packages/continual-manipulation/README.md#named-task-sequences).

A configuration must not set both `continual.sequence` and `continual.tasks`.
Resolved run configurations store the explicit ordered task list, so historical
snapshots and explicit configurations keep their original scientific identity
when an alias definition changes. Expert paths remain explicit for every
selected DiffCRL task. `cm20` is not exposed because the current unique-task
identity and evaluation contracts cannot safely represent a repeated curriculum.

Use `configs/diffcrl/diffcrl_no_replay.yaml` for the no-replay ablation. Flow
Matching, Expert Recollection, KUKA, fixed-budget, and pilot configurations are
kept in the same directory and are labelled by filename and comments. Historical
YAML using `real` is provenance only and is rejected by current validation.
Inspect a YAML before running it; not every configuration is a maintained formal
protocol.

Flow Matching sampling uses `torchdiffeq.odeint`. Configure `ode_solver` as
`euler`, `midpoint`, `heun2`, `rk4`, or `dopri5`; `integration_steps` determines
the fixed step size for the first four methods, while `rtol` and `atol` control
adaptive `dopri5`. Replay reports record the velocity network's actual function
evaluation count rather than inferring NFE from the configured step count.

### Collection and evaluation

```bash
python scripts/collection/collect_expert_trajectories.py \
  --config path/to/trajectory-collection.yaml

python scripts/evaluation/evaluate_sac.py \
  --task reach-v3 \
  --model runs/path/to/model.zip \
  --episodes 50

python scripts/evaluation/evaluate_continual.py --help
```

Evaluation records success, return, episode length, reset seeds, and task-bank
provenance. The continual matrix stores success rate `R[k,j]` for task `j`
after stage `k`, from which Average Performance and end-of-task forgetting are
computed.

## Configuration

`src/config/` provides separate top-level schemas for standalone SAC and
DiffCRL. Each YAML exposes only the sections meaningful to
that experiment; unknown keys and incompatible sections are rejected. CLI flags
can override seed, device, run naming, and narrowly scoped entry-point options.
Resolved configs serialize the complete selected schema, including defaults.

Maintained starting points are:

- `configs/sac/sac.yaml`
- `configs/diffcrl/diffcrl.yaml`
- `configs/diffcrl/diffcrl_no_replay.yaml`

The corresponding smoke YAML files are developer presets. Configurations under
an `archive/` directory are retained for historical reproducibility.

## Outputs and reproducibility

Runs normally write beneath `runtime.output`. New run directories are
collision-protected; existing results are not overwritten. DiffCRL writes
resolved configuration and provenance, policy/generator checkpoints, stage
datasets and reports, evaluation-matrix snapshots, and a hash-validated
`stage_<index>_COMPLETE.json` marker. `RUN_COMPLETE` marks a fully completed
run.

DiffCRL runs always start in a fresh, collision-protected directory. Stage
markers authenticate completed checkpoints, reports, and evaluation matrices
for artifact synchronization and offline analysis; they are not restart points.
See
[Architecture](docs/ARCHITECTURE.md#8-configuration-and-reproducibility) for
the full contract and [Runs inventory](docs/RUNS_INVENTORY.md) for retained
artifact policy.

For repeatable experiments:

1. record the YAML, seed, repository revision, and environment versions;
2. use expert manifests and fixed task banks where available;
3. use a fresh run name for every seed and method;
4. compare methods with matched task order, budgets, and evaluation protocol;
5. do not interpret smoke or incomplete runs as thesis results.

Exact numerical equivalence is not guaranteed across CPU/GPU backends or GPU
models even when saved RNG state is restored.

## Testing

Run lightweight and integration tests with the intended environment:

```bash
conda run --no-capture-output -n metaworld \
  env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/ -q
```

Disabling plugin autoload avoids unrelated ROS/system pytest plugins. Tests that
need the local KUKA fork, CUDA, or retained large artifacts
may skip or require additional resources. No test result substitutes for a
multi-seed scientific evaluation.

The maintained two-task integration presets are
`configs/diffcrl/diffcrl_smoke.yaml` and
`configs/diffcrl/flow_matching_smoke.yaml`. Always provide a fresh
`--run-name`; see [Direct 43D migration](docs/DIRECT_43D_MIGRATION.md#completed-verification)
for verified commands, outputs, and interpretation limits.

## Status and limitations

- The repository targets simulation; no real-robot deployment path is present.
- DiffCRL is expert-driven continual imitation/distillation, not the original
  DISTR online Immediate Policy algorithm.
- CUDA device-native DDPM replay sampling is implemented and
  CPU-equivalence-tested, but CUDA correctness/performance is not yet validated
  in this repository.
- Flow Matching workflows remain experimental; documented results must be read
  with their protocol and limitations.

See [Development status](docs/DEVELOPMENT_STATUS.md) for the complete
implemented/validated/experimental classification.

## References and acknowledgements

- Feng Chen et al., [“Stable Continual Reinforcement Learning via
  Diffusion-based Trajectory Replay”](https://arxiv.org/abs/2411.10809), 2024.
- [MetaWorld](https://github.com/Farama-Foundation/Metaworld) provides the
  manipulation benchmark and task interfaces.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) provides the
  SAC implementation.

This thesis code builds on those projects while retaining its own environment
adapters, trajectory contracts, continual orchestration, replay models, and
evaluation/provenance machinery.
