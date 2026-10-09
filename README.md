# Diffusion Replay for Continual RL

This repository contains the continual-learning and DiffCRL experiments for
modern MetaWorld v3. Default Sawyer tasks use the `metaworld` Conda
environment.

## Environment setup

Run from the repository root:

```bash
conda activate metaworld
python -m pip install -r requirements.txt
```

The expected package version is `metaworld==3.1.1`.

## Testing

Run the full test suite from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Plugin autoload is disabled because unrelated system or ROS pytest plugins can
interfere with the test run. Optional KUKA or hardware-specific integration
tests may be skipped when their local resources are unavailable.

## KUKA experiments

KUKA experiments require a separate custom MetaWorld fork, `metaworld-kuka`.

Clone or obtain that repository separately, then install it in the dedicated
`metaworld-kuka` Conda environment:

```bash
conda activate metaworld-kuka
cd metaworld-kuka
python -m pip install -e .
```

### KUKA Hammer working reward

The selected working configuration for `kuka-hammer-v3` uses the v2 reward
with normalized nail-progress shaping:

```yaml
environment:
  backend: kuka-v3
  reward_function_version: v2
  hammer_reward_variant: nail_progress
  hammer_nail_progress_weight: 4.0
```

Development screening found that the original Sawyer-derived dense reward
could produce high return without reliable task success because it did not
directly reward nail displacement. Weight 4 was chosen as the project default
after the screened seed reached stable high success; it is a practical choice,
not a claim that the weight is globally optimal. This adapts the KUKA Hammer
reward only. Success remains `NailSlideJoint.qpos[0] > 0.09`, and the original
reward remains available with `hammer_reward_variant: original` and weight
`0.0` for ablation and reproducibility.

The formal fresh-expert preset is `configs/sac/kuka_hammer_v3_1m.yaml`. Its 1M
steps are a safety ceiling and success-based early stopping may finish the run
earlier. Do not resume an original-reward Hammer checkpoint or replay buffer
under this preset. For downstream trajectory collection, use the qualified
`best_success_model/best_success_model.zip` checkpoint rather than selecting
the Hammer expert primarily by shaped return.

## Experiment types

- **Single-task SAC expert training** trains an independent expert checkpoint
  for one task with `train_sac.py`; DiffCRL uses these experts to collect data.
- **Sequential SAC** is a continual-learning baseline that trains one SAC
  learner across tasks without pretrained experts, behavioral cloning, or
  diffusion replay.
- **DiffCRL with diffusion replay** trains the shared behavior-cloning policy
  from current expert trajectories and generated trajectories for prior tasks.
- **DiffCRL without replay** is the corresponding ablation: it retains expert
  supervision and behavioral cloning but disables prior-task diffusion replay.

## Maintained configurations

- `configs/sac/sac.yaml`
- `configs/diffcrl/diffcrl.yaml`
- `configs/diffcrl/diffcrl_no_replay.yaml`
- `configs/continual_sac/continual_sac.yaml`

The corresponding `*_smoke.yaml` files are small developer/testing presets.

## Single-task experts and DiffCRL

First, train the SAC experts required by the selected DiffCRL configuration.
For each task, copy `configs/sac/sac.yaml`, update `continual.tasks[0]`, and run:

```bash
python -u train_sac.py --config path/to/expert-config.yaml --seed 0
```

Periodic SAC checkpoints include a matching replay-buffer sidecar. To resume
one into a collision-safe new run directory, keep the desired total timestep
target in the YAML's `sac.steps` field and run:

```bash
python -u train_sac.py \
  --config path/to/expert-config.yaml \
  --seed 0 \
  --resume-from runs/reward_v2/hammer-v3/seed_0/checkpoints/sac_500000_steps.zip
```

For example, with `sac.steps: 1000000`, a 500,000-step checkpoint trains only
the remaining approximately 500,000 steps. Model-only legacy checkpoints are
rejected because they cannot restore SAC's replay buffer faithfully.

Development and smoke-test runs may optionally stop after stable evaluation
success by adding this block under `sac`. It is disabled by default and should
remain disabled for fixed-budget formal experiments:

```yaml
sac:
  steps: 1000000  # Maximum total global timestep budget.
  early_stopping:
    enabled: true
    min_steps: 100000
    success_threshold: 0.95
    patience: 3
```

The criterion uses globally aligned evaluation results and global timesteps,
including after resume. Three consecutive qualifying evaluations are required
in this example; any failed evaluation resets the count.

Then update the expert checkpoint paths under `continual.experts` in
`configs/diffcrl/diffcrl.yaml` and start the continual-learning run:

```bash
python -u train_diffcrl.py \
  --config configs/diffcrl/diffcrl.yaml \
  --seed 0 \
  --run-name example-seed0
```

Results and checkpoints are written under `runs/`. Use a new `--run-name` for
each run because existing run directories are not overwritten.

The canonical trained KUKA v3 experts are stored under
`runs/experts/kuka-v3/<task>/expert_model.zip`. Model provenance, reward
configuration, hashes, and retained evaluation evidence live beside each
checkpoint. Hammer's canonical expert uses reward v2 with `nail_progress`
weight `4.0`.

These are standalone KUKA-v3 experts and must not currently be substituted for
the Sawyer MetaWorld-v3 experts referenced by the DiffCRL configurations. KUKA
continual-learning support is not yet implemented.

SAC checkpoint evaluation for MetaWorld uses an explicit ordered MT1 bank. The
`evaluation.task_set_seed` generates the fixed task configurations; the rollout
seed (`runtime.seed + evaluation.seed_offset` for SAC) controls the separate
per-episode reset-seed sequence. Every checkpoint evaluation uses the same bank
order, and its hashes are stored in `evaluations/evaluations.npz`.

New SAC runs write checkpoint-specific `*.manifest.json` sidecars containing
the task identity, checkpoint hash, environment contract, training identity,
and fixed-bank qualification metrics. DiffCRL validates these manifests. Older
checkpoints without a sidecar remain usable through a warned legacy path, but
their task/reward identity cannot be proven and they should be requalified
before a new formal experiment.

Before a fixed-task formal run, the training and evaluation MT1 task hashes are
checked for disjointness. Formal workflows also write `run_manifest.json` with
repository, environment, config, expert, and task-bank provenance. `RUNNING`
marks an incomplete run and is atomically replaced by `RUN_COMPLETE` only after
all final outputs succeed.

SAC evaluation archives produced before the fixed-bank callback change contain
reset seeds but not fixed task identities. They must **not** be described as
fixed-task evaluations.

For a fast pipeline check, use `configs/diffcrl/diffcrl_smoke.yaml`.

Run the no-replay ablation with:

```bash
python -u train_diffcrl.py \
  --config configs/diffcrl/diffcrl_no_replay.yaml \
  --seed 0 \
  --run-name no-replay-seed0
```

## Sequential SAC baseline

Sequential SAC maintains one SAC learner across the task sequence. The learner
state persists across task boundaries, while task-local replay is reset at each
boundary. It does not load pretrained experts or use behavioral cloning or
diffusion replay.

Smoke run:

```bash
python -u train_continual_sac.py \
  --config configs/continual_sac/continual_sac_smoke.yaml
```

Formal five-task run:

```bash
python -u train_continual_sac.py \
  --config configs/continual_sac/continual_sac.yaml
```

These configurations write to `runs/continual_sac_smoke/` and
`runs/formal/continual_sac_5task/`, respectively. Existing run directories are
not overwritten; use `--run-name NAME` to place a run in a distinct named
subdirectory.

## Utilities

Evaluate a trained SAC policy:

```bash
python scripts/evaluation/evaluate_sac.py \
  --task reach-v3 \
  --model runs/path/to/model.zip \
  --episodes 50
```

Render a saved SAC or DiffCRL policy to MP4 (requires FFmpeg):

```bash
python scripts/visualization/render_policy_video.py \
  --task reach-v3 \
  --model path/to/checkpoint \
  --policy-type sac \
  --output videos/reach.mp4
```

Use `--policy-type diffcrl` for a saved `GeneralPolicy` checkpoint. Run any
entry point with `--help` to see its remaining options.
