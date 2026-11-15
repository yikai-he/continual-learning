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

## Checkpoint semantics

Standalone SAC retains one selected expert:

```text
best_model.zip    # highest validation (success_rate, mean_return)
```

Success rate is primary. Mean return breaks equal-success ties, and exact ties
keep the earlier checkpoint. Metrics must be finite. The file is serialized to
a temporary path and atomically replaced; replay buffers are not saved.

No periodic models, replay buffers, final model, or resume checkpoint are
written. Interrupted runs must be restarted in a fresh collision-safe directory.

## Evaluation separation

- Optimization: `runtime.seed`
- Validation/selection: `evaluation.task_set_seed` (default `10000`)
- Final test: `evaluation.final_task_set_seeds` (default `[20000, 20001]`)

Serialized configuration identities are checked pairwise for overlap. Final
tests run only after selection and cannot change checkpoint ranking.

## Manual engineering smoke test

The smoke configuration exercises environment creation, training, four periodic
validation events, atomic best-expert replacement, independent final testing,
and result serialization. It is an infrastructure check, not a performance
benchmark, and does not require the policy to solve the task.

From the repository root, with the `metaworld` Conda environment active, run:

```bash
python train_sac.py \
  --config configs/sac/smoke_low_disk.yaml \
  --run-name manual_YYYYMMDD_HHMMSS
```

Replace the timestamp placeholder so the run has a new path under
`runs/smoke/sac_low_disk/`. The trainer also refuses to overwrite an existing
run directory. A successful run contains `best_model.zip`, its expert manifest,
`config.yaml`, `run_manifest.json`, `results.json`, and evaluation records.
`results.json` contains the early-stopping outcome, selected-expert hash, and
post-selection final evaluations. No replay-buffer file or periodic checkpoint
directory is expected. CSV, Markdown, and duplicate JSON summaries are not
written automatically; `evaluations/evaluations.npz` is the validation-history
authority.

Stop safely with `Ctrl-C`. An interrupted run is intentionally not marked
complete; use a fresh run name for the next attempt because low-disk checkpoints
are inference-only and cannot provide exact training resumption.

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
