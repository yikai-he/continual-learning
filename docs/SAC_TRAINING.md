---
status: current
last_verified: 2026-11-15
applies_to: continual-learning
---

# Canonical SAC Training

Use `train_sac.py` for new single-task MetaWorld-v3 and KUKA-v3 SAC runs.

## Training schedule

Formal configurations reproduce the effective Legacy `continual_diff_pytorch`
schedule while retaining Stable-Baselines3 SAC: uniform actions through
zero-based environment timestep 10,000, optimization beginning at timestep
1,000, and 50 gradient updates every 50 environment steps. The Legacy Gaussian
entropy target is derived from `target_output_std: 0.089` (approximately
`-4.00072` for four actions); automatic entropy tuning remains enabled. Formal
validation runs every 40,000 steps. Smoke configurations deliberately retain
short warmups and one-step updates so they remain inexpensive infrastructure
checks.

The actor uses SB3's squashed diagonal Gaussian, but evaluates the tanh
Jacobian from the cached pre-tanh action with the algebraically stable Legacy
formula. This avoids the zero-gradient branch that can occur when float32
`tanh` rounds a large latent action to exactly `-1` or `1`. Sampling,
deterministic actions, distribution parameters, losses, and checkpoint tensor
keys are otherwise unchanged. New expert manifests identify this implementation
as `legacy-continual-diff-pytorch-4x256-stable-tanh-v2`; older expert weights
remain loadable through the same `LegacySACPolicy` class.

## Checkpoint semantics

New standalone SAC runs retain synchronized Best and Final training snapshots:

```text
best_model.zip              # selected inference checkpoint (compatible path)
best_replay_buffer.pkl      # replay state at the same validation step
best_checkpoint.json        # hashes, step, task, spaces, and buffer metadata
final_model.zip             # actual policy at normal termination
final_replay_buffer.pkl     # replay state at the final step
final_checkpoint.json       # synchronized Final metadata
final_model.zip.manifest.json # identity sidecar for same-task continuation
```

Success rate is primary. Mean return breaks equal-success ties, and exact ties
keep the earlier checkpoint. Metrics must be finite. Model and replay files are
serialized and hashed before metadata is published; a failed Best replacement
restores the previous complete trio. Final is saved for normal maximum-budget
and Early Stopping termination and is not promoted automatically to the Expert
Registry. No periodic recovery snapshots are created.

## Scratch, actor initialization, and continuation

The three initialization modes are deliberately separate:

- Scratch training creates a new SAC model and all training state. This remains
  the default.
- Actor Warm-Start copies only compatible actor weights; critics, targets,
  optimizers, entropy state, counters, and replay are fresh.
- Same-task continuation loads the serialized SAC model into a new
  collision-safe run.
  Actor, critics, target critics, optimizer states, entropy state, and timestep
  counters stored by SB3 are restored. Warm-start and resume are mutually
  exclusive.

```yaml
actor_init: {mode: scratch, checkpoint: null}
resume:
  enabled: true
  checkpoint: experts/sawyer/peg-unplug-side-v3/seed_0/best_model.zip
  replay_buffer: null
sac:
  steps: 200000  # hard maximum of additional environment steps
  early_stopping:
    enabled: true
    min_steps: 20000
    success_threshold: 1.0
    patience: 3
    plateau_patience: null
    min_delta: 0.01
```

SB3 model archives do not contain replay buffers. With `replay_buffer: null`,
the run is labeled `model-state-only`, starts with an empty buffer, and uses the
explicit refill described below. New matched continuation points use either
the `best_*` model/replay/metadata trio or the corresponding `final_*` trio.
Hashes, step, task, backend, reward, spaces, capacity, stored transition count,
and source configuration are verified before optimization. A Best model cannot
be paired with a Final buffer. Neither mode is exact process-level resume: RNG
streams, environment state, callback state, and TensorBoard state are not
restored.

For a matched continuation, point both paths at the same completed training run:

```yaml
resume:
  enabled: true
  checkpoint: runs/source/best_model.zip       # or final_model.zip
  replay_buffer: runs/source/best_replay_buffer.pkl
  refill_steps: 0
```

The adjacent `best_checkpoint.json` (or `final_checkpoint.json`) is mandatory.
Source checkpoint steps remain the TensorBoard/global-step origin; refill and
new training interactions count toward the additional `sac.steps` budget, while
evaluation interactions do not.

The serialized source hyperparameters remain effective during continuation and
are recorded in the result provenance; the new `sac.steps` is the additional
budget rather than a request to rebuild the optimizer. SB3 archives do not
preserve a live random-number-generator stream, so continuation reseeds the
model and environment from `runtime.seed` and records that RNG state as reset.

For model-state-only continuation, `resume.refill_steps` collects transitions
with the restored SAC policy's ordinary stochastic action sampling before
optimization resumes. Actor, critics, optimizers, and entropy state are not
updated during refill. The interactions advance the global counter, consume the
additional-step budget, and count toward early-stopping `min_steps`. This avoids
the Legacy 50-update boundary repeatedly sampling a nearly empty buffer. Refill
must be zero when an explicit replay buffer is restored.

At termination, `final_policy_validation` evaluates the actual in-memory Final
policy on the validation bank unless that exact timestep was already evaluated.
This is separate from Best validation and independent final-bank testing of
Best. Validation and final testing may use different episode counts through
`evaluation.episodes` and optional `evaluation.final_episodes`; null preserves
the historical behavior of using the validation count for final banks.

Resume requires a hash-valid manifest matching task, backend, reward version,
horizon, and observation/action contracts. Unsupported custom reward variants
are rejected. The source policy is evaluated before the first update and
participates in success-first, return-second selection. `min_steps` counts
additional steps. `patience` counts consecutive held-out evaluations meeting
`success_threshold`; optional `plateau_patience` counts evaluations without a
success-rate improvement greater than `min_delta` and is disabled by default.
`results.json` records source, final-policy, and independent final-test metrics,
restored/reset state, additional/total steps, and the stopping reason.

The three prepared Sawyer target-success-only experiments explicitly set
`plateau_patience: null`. They run Stick Pull, Push, then Push Back, prioritizing
the stronger existing same-task experts before the less certain Faucet Close →
Push Back actor transfer. Their fresh destinations are under
`runs/sac_target_success_only_20261117/`; the launcher refuses collisions and
does not alter historical `runs/sac_early_stop_20261117/` artifacts. Stick Pull
and Push restore compatible model and optimizer state, refill an empty replay
buffer for 10,000 interactions, and count those interactions against the new
budget. Push Back transfers compatible actor tensors only and starts fresh
critics, targets, optimizers, entropy state, replay, and counters. Normal
termination is either sustained target success after the configured minimum or
the maximum additional-step budget; long zero-success periods do not stop a
run. The best validation checkpoint remains success-first, mean-return-second,
and is selected before independent testing on banks 20000 and 20001.

## Evaluation separation

- Optimization: `runtime.seed`
- Validation/selection: `evaluation.task_set_seed` (default `10000`)
- Final test: `evaluation.final_task_set_seeds` (default `[20000, 20001]`)

Serialized configuration identities are checked pairwise for overlap. Final
tests run only after selection and cannot change checkpoint ranking.

## Manual engineering smoke test

The smoke configuration exercises environment creation, training, four periodic
validation events, atomic best-expert replacement, independent final testing,
result serialization, and synchronized Best/Final replay persistence. It is an
infrastructure check, not a performance benchmark, and does not require the
policy to solve the task.

From the repository root, with the `metaworld` Conda environment active, run:

```bash
python train_sac.py \
  --config configs/sac/smoke_low_disk.yaml \
  --run-name manual_YYYYMMDD_HHMMSS
```

Replace the timestamp placeholder so the run has a new path under
`runs/smoke/sac_low_disk/`. The trainer also refuses to overwrite an existing
run directory. A successful run contains synchronized Best and Final
model/replay trios, the Best expert manifest, `config.yaml`,
`run_manifest.json`, `results.json`, and evaluation records. Replay buffers
remain training artifacts and are never copied into `experts/`. A
one-million-entry buffer is typically several hundred MB, so Best and Final are
the only retained snapshots. CSV, Markdown, and duplicate JSON summaries are not
written automatically; `evaluations/evaluations.npz` is the validation-history
authority.

Stop safely with `Ctrl-C`. An interrupted run is intentionally not marked
complete; use a fresh run name for the next attempt. This workflow does not
provide crash recovery or exact process-level resume.

## Sawyer Push Back and Push scratch runs (2M maximum)

Two independent seed-0 scratch experiments are defined in
`configs/sac/sawyer_scratch_2m/`. Push Back runs first because it blocks CM5;
Push follows as a separate CM10 candidate. Both use MetaWorld v3 reward v2,
`LegacySACPolicy`, a maximum of 2,000,000 training interactions, and a fresh
actor, critics, optimizers, entropy state, timestep counter, and replay buffer.
They do not load the historical Push continuation or Faucet Close actor.

Validation is deterministic on all 50 fixed MT1 configurations once every
40,000 training interactions. Training stops normally after three consecutive
50/50 validations or at the maximum budget; `plateau_patience: null` means long
periods without improvement cannot stop it. The initial step-0 evaluation is
eligible for Best selection but not Early Stopping. Selected-Best final tests
use distinct task banks 20000 and 20001 for 100 total episodes and never affect
selection.

Each run writes synchronized `best_*` and `final_*` model, replay-buffer, and
checkpoint-metadata trios under
`runs/sac_sawyer_scratch_2m/<task>/seed_0/`, plus the resolved configuration,
actor initialization, run manifest, evaluation history, results, TensorBoard
events, and console log. Historical experts and transfer/continuation runs stay
separate and are never promoted automatically.

```bash
# Validate both tasks without creating run directories.
./tools/experiments/train_sawyer_scratch_2m.sh --preflight-only

# Run either task independently.
python -u train_sac.py --config configs/sac/sawyer_scratch_2m/push_back_v3.yaml
python -u train_sac.py --config configs/sac/sawyer_scratch_2m/push_v3.yaml

# Run Push Back then Push, stopping on the first unexpected failure.
./tools/experiments/train_sawyer_scratch_2m.sh --run
```

## Five Sawyer experts on three machines

The first five tasks in the original Continual World CW10 order are launched as
independent single-task experts. Machine 1 runs `hammer-v3` then `push-wall-v3`;
Machine 2 runs `faucet-close-v3` then `push-back-v3`; Machine 3 runs
`stick-pull-v3`. Activate the same Python environment on each machine, clone the
same repository revision, and run the assigned command:

```bash
# On each machine (or activate the equivalent environment with venv/mamba):
conda activate metaworld
cd /path/to/continual-learning

# Machine 1
PYTHON_BIN=python ./scripts/experiments/train_cw5_machine1.sh

# Machine 2
PYTHON_BIN=python ./scripts/experiments/train_cw5_machine2.sh

# Machine 3
PYTHON_BIN=python ./scripts/experiments/train_cw5_machine3.sh
```

`PYTHON_BIN` may be omitted when the activated environment's `python` is the
correct interpreter. Runs are written below
`runs/cw5_experts/legacy_sac/reward_v2/<task>/seed_0/`; console logs are written
below `logs/cw5_experts/legacy_sac/`. Monitor a task with, for example,
`tail -f logs/cw5_experts/legacy_sac/hammer-v3_seed0.log`. The selected expert is
`best_model.zip` inside the task run. After collecting the three machines'
outputs, verify that every task directory contains `RUN_COMPLETE`,
`best_model.zip`, and `best_model.zip.manifest.json`. Identical seeds preserve the
protocol but do not guarantee bitwise-identical results across hardware.

## Six stable-tanh CW10 experts

The second CW10 task and tasks 6--10 are prepared as six independent experts in
this exact order:

```text
push-wall-v3
handle-press-side-v3
push-v3
shelf-place-v3
window-close-v3
peg-unplug-side-v3
```

This ordering follows the Legacy `continualworld/tasks.py` CW10 definition after
updating its `-v2` environment names to MetaWorld v3. Every YAML below
`configs/sac/cw10/` creates a fresh model and replay buffer with seed 0, one
million environment steps, reward v2, and the fixed-task evaluation protocol
described above. No weights or optimizer state pass between tasks.

Run all six sequentially in a persistent tmux session:

```bash
conda activate metaworld
cd /path/to/continual-learning
tmux new -s cw10-experts
PYTHON_BIN=python ./scripts/experiments/train_cw10_six_sequential.sh
```

Detach with `Ctrl-b d` and reattach with `tmux attach -t cw10-experts`. Results
are written to
`runs/cw10_experts/stable_tanh_v2/reward_v2/<task>/seed_0/`; task logs are
written to `logs/cw10_experts/stable_tanh_v2/<task>_seed0.log`. The launcher
preflights every configuration and refuses any existing target run or log
before starting the first task. It prints timestamped task boundaries and one
unbuffered progress line at each 40,000-step evaluation, including success rate
and mean return. Monitor the durable output with, for example:

```bash
tail -f logs/cw10_experts/stable_tanh_v2/push-wall-v3_seed0.log
```

The earlier Rich progress bar was connected to a pipe by the launcher's `tee`,
so it could not reliably render as an interactive terminal display. SAC now
uses the evaluation progress lines instead; they render consistently both in
tmux and redirected logs. If a task fails or is interrupted, the launcher
prints `FAILED`, preserves Python's exit status, and stops without retrying or
starting the next task.
