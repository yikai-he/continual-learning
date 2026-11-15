---
status: planned
last_verified: 2026-11-10
---

# KUKA Flow Matching sampling-step ablation protocol

This is an execution protocol, not a scientific results report. No diagnostic
samples were generated while preparing it.

The experiment re-samples historical replay from the completed five-seed KUKA
Flow Matching runs. It compares `fm_structured_22d` and `fm_full_43d` at 10,
25, 50, 100, 200, and 400 fixed Euler steps. For every training seed, method,
stage, and historical task, all step-count cells reuse the same checkpoint,
task condition, reconstructed goal, selected training configurations, and
sampling seed. Consequently `TrajectoryFlowMatching.sample` creates the same
seeded Gaussian initial state; only the Euler step size/count changes.

## Checkpoint provenance

Replay at continual stage N is generated before the stage-N generator update.
The diagnostic therefore reads each generated source in that stage's original
`report.json`, requires all sources to agree, verifies the recorded SHA-256,
and requires the path to be the immutable checkpoint after stage N-1. It also
loads the checkpoint and compares its model-state hash with
`flow_matching_before_state_sha256`. The report's top-level checkpoint after
stage N is deliberately not used.

Stage 1 samples Hammer from the post-Hammer checkpoint. Stage 2 samples Hammer
and Handle from the post-Handle checkpoint. Stage 3 samples Hammer, Handle,
and Button from the post-Button checkpoint. Original fixed-total replay counts
(20; 10+10; 7+7+6) are retained.

## Execution

Run from the repository root in the `metaworld` Conda environment. A complete
dry run performs checkpoint/marker/config/hash validation and prints all 60
commands, but does not sample:

```bash
bash scripts/experiments/run_kuka_fm_sampling_steps.sh --dry-run
```

Execute Seed 0 for both representations:

```bash
bash scripts/experiments/run_kuka_fm_sampling_steps.sh --seeds 0
```

Execute Seeds 1–4:

```bash
bash scripts/experiments/run_kuka_fm_sampling_steps.sh --seeds 1,2,3,4
```

Analyze completed diagnostic JSON without writing a report:

```bash
conda run -n metaworld python scripts/evaluation/analyze_kuka_fm_sampling_steps.py
```

To save an analysis after all experiments have run, choose a new path (the
script uses exclusive creation and will not overwrite it):

```bash
conda run -n metaworld python scripts/evaluation/analyze_kuka_fm_sampling_steps.py \
  --output docs/experiments/kuka_fm_sampling_steps_results.json
```

The launcher accepts comma-separated `--seeds`, `--methods`, and `--steps`.
Existing completed cells are skipped, so subsets can be resumed without
overwriting. Results are stored as
`runs/diagnostics/fm_sampling_steps/<method>_seed<seed>/steps_<steps>.json`.
Each file records immutable checkpoint identities, conditioning/configuration
identities, sampling seed, NFE, sampling duration, trajectory hash, and the
existing goal, previous-state, action-bound, shape, NaN, and Inf diagnostics.

No generator/BC/SAC training, environment rollout, expert collection, or
modification of historical experiment artifacts is part of this protocol.
