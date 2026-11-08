---
status: current
last_verified: 2026-11-07
applies_to: continual-learning
---

# Continual Learning for Robotic Manipulation

Research code for the bachelor’s thesis **Continual Learning for Robotic
Manipulation towards Real-World Deployment**. The repository studies whether
trajectory-level generative replay can reduce forgetting as a manipulation
policy learns a sequence of simulated tasks.

## SAC training

`train_sac.py` is the preferred entry point for new single-task SAC experts. New
runs default to low-disk checkpointing: `best_model.zip` is the validation
checkpoint maximizing `(success_rate, mean_return)` lexicographically, and
`latest_model.zip` is the most recently persisted policy. Exact ties retain the
existing best checkpoint.

Low-disk mode does not save replay buffers and therefore does not support exact
training continuation. Set `sac.checkpoint_mode: full` when replay-backed resume
or the historical periodic checkpoint layout is required. Historical
`best_model.zip` files may be return-best and must not be reinterpreted under the
new naming convention.

Optimization uses the `runtime.seed` bank, validation defaults to bank `10000`,
and post-selection final tests default to banks `20000` and `20001`. Serialized
task identities are checked for pairwise disjointness before training.
The SAC entropy mode and four-layer 256-unit network are fixed implementation
choices and therefore are not YAML options.

```bash
python train_sac.py --config configs/sac/sac.yaml --seed 0 --device cpu
```

The main pipeline, DiffCRL, is inspired by DISTR (Diffusion-based Trajectory
Replay), but it is **not an exact reproduction**. DISTR trains an Immediate
Policy during continual reinforcement learning. This implementation first
trains independent single-task SAC experts, collects their trajectories, and
continually distils those trajectories into one persistent behavior-cloning
policy. Diffusion or Flow Matching models provide old-task replay.

## Documentation

| Document | Purpose | Status | Last verified |
|---|---|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Current pipeline, module, data, evaluation, and resume contracts | current | 2026-11-07 |
| [Development status](docs/DEVELOPMENT_STATUS.md) | Implemented and validated scope, limitations, and remaining work | current | 2026-11-07 |
| [SAC training](docs/SAC_TRAINING.md) | Expert training, evaluation, checkpoints, and smoke validation | current | 2026-11-07 |
| [Workflow index](docs/WORKFLOW_INDEX.md) | Supported entry points and configuration families | current | 2026-11-07 |
| [Flow replay architecture](docs/FLOW_REPLAY_ARCHITECTURE.md) | Flow Matching replay and allocation contract | current | 2026-11-07 |
| [Forward transfer](docs/FORWARD_TRANSFER_EXPERIMENT.md) | Supported GeneralPolicy-to-SAC transfer experiment | current | 2026-11-07 |
| [Runs inventory](docs/RUNS_INVENTORY.md) | Retained canonical artifact families | current | 2026-11-07 |
| [Research code standards](docs/RESEARCH_CODE_STANDARDS.md) | Local quality priorities and proportional acceptance checks | current | 2026-11-07 |
| [Documentation archive](docs/archive/README.md) | Historical reports, provenance, and archive conventions | current index | 2026-11-07 |

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
- **Sequential SAC**: one SAC learner carried across MetaWorld tasks, with its
  replay buffer reset at task boundaries.
- **No Replay**: persistent `GeneralPolicy` trained only on current expert data.
- **Real Replay**: current expert data plus sampled stored trajectories from
  previous tasks.
- **Diffusion Replay**: current expert data plus task- and goal-conditioned DDPM
  trajectory replay.
- **Flow Matching Replay**: current expert data plus trajectories produced by a
  task- and goal-conditioned conditional flow model.

The four DiffCRL replay modes are `none`, `real`, `diffusion`, and
`flow-matching`. Forward-transfer utilities exist for aligned learning curves;
forward transfer is not inferred from the endpoint evaluation matrix.

## Repository structure

```text
continual-learning/
├── train_sac.py                 # single-task SAC experts
├── train_diffcrl.py             # expert-driven continual BC and replay
├── train_continual_sac.py       # sequential SAC baseline
├── run_replay_comparison.py     # no/real/DDPM comparison runner
├── manage_diffcrl_sync.py       # validated stage backup/restore
├── configs/                     # active, experimental, and archived YAML
├── src/
│   ├── env_adapters/            # MetaWorld/KUKA integration
│   ├── continual/               # BC, replay, generators, evaluation, resume
│   ├── sac/                     # SAC checkpoint and transfer utilities
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
python train_continual_sac.py \
  --config configs/continual_sac/continual_sac_smoke.yaml \
  --run-name smoke-seed0
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

Use `configs/diffcrl/diffcrl_no_replay.yaml` for the no-replay ablation. Flow
Matching, Real Replay, KUKA, fixed-budget, and pilot configurations are kept in
the same directory and are labelled by filename and comments. Inspect a YAML
before running it; not every configuration is a maintained formal protocol.

Flow Matching sampling uses `torchdiffeq.odeint`. Configure `ode_solver` as
`euler`, `midpoint`, `heun2`, `rk4`, or `dopri5`; `integration_steps` determines
the fixed step size for the first four methods, while `rtol` and `atol` control
adaptive `dopri5`. Replay reports record the velocity network's actual function
evaluation count rather than inferring NFE from the configured step count.

### Sequential SAC

```bash
python -u train_continual_sac.py \
  --config configs/continual_sac/continual_sac.yaml \
  --run-name sequential-sac-seed0
```

The current sequential SAC implementation constructs MetaWorld environments
directly. KUKA DiffCRL is implemented, but KUKA sequential SAC is not.

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

`src/config.py` loads strict YAML sections for `environment`, `sac`, `bc`,
`diffusion`, `flow_matching`, `continual`, `evaluation`, and `runtime`. Unknown
keys and incompatible experiment sections are rejected. CLI flags can override
seed, device, output/run naming, and entry-point-specific resume options.

Maintained starting points are:

- `configs/sac/sac.yaml`
- `configs/diffcrl/diffcrl.yaml`
- `configs/diffcrl/diffcrl_no_replay.yaml`
- `configs/continual_sac/continual_sac.yaml`

The corresponding smoke YAML files are developer presets. Configurations under
an `archive/` directory are retained for historical reproducibility.

## Outputs, resume, and reproducibility

Runs normally write beneath `runtime.output`. New run directories are
collision-protected; existing results are not overwritten. DiffCRL writes
resolved configuration and provenance, policy/generator checkpoints, stage
datasets and reports, evaluation-matrix snapshots, and a hash-validated
`stage_<index>_COMPLETE.json` marker. `RUN_COMPLETE` marks a fully completed
run.

Resume a marker-based DiffCRL run with its original scientific configuration:

```bash
python -u train_diffcrl.py \
  --config path/to/original.yaml \
  --resume-from /path/to/existing/run
```

Completed artifacts and expert hashes are validated before stages are skipped.
Interrupted-stage outputs are not overwritten. Periodic SAC resume requires
both the model checkpoint and its replay-buffer sidecar. See
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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Disabling plugin autoload avoids unrelated ROS/system pytest plugins. Tests that
need the local KUKA fork, CUDA, or retained large artifacts
may skip or require additional resources. No test result substitutes for a
multi-seed scientific evaluation.

## Status and limitations

- The repository targets simulation; no real-robot deployment path is present.
- DiffCRL is expert-driven continual imitation/distillation, not the original
  DISTR online Immediate Policy algorithm.
- KUKA sequential SAC is not implemented.
- CUDA device-native DDPM replay sampling is implemented and
  CPU-equivalence-tested, but CUDA correctness/performance is not yet validated
  in this repository.
- Flow Matching and forward-transfer workflows remain experimental; documented
  results must be read with their protocol and limitations.

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
