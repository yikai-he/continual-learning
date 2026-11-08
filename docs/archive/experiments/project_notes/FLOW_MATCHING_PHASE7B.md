---
status: archived
original_date: unknown
archived_on: 2026-10-30
last_verified: unknown
---

> Historical document. Describes an earlier project state and may not match the current implementation. Do not use its commands or configuration examples without checking the relevant historical code revision.

# Phase 7B: Flow Matching generation-quality ablation

This is a preparation-only experiment grid. Each job has an independent output
directory and can be run without launching any other job. The 50-epoch jobs load
the immutable Phase 7A checkpoint; the 100- and 200-epoch jobs train a new model.

## Jobs

- `integration_steps_10`, `integration_steps_20`, `integration_steps_50`,
  `integration_steps_100`
- `training_epochs_50`, `training_epochs_100`, `training_epochs_200`

Run generation and offline diagnostics first:

```bash
scripts/flow_matching/launch_flow_matching_ablation.sh integration_steps_10 generate
```

Run downstream BC and real-environment evaluation only after generation finishes:

```bash
scripts/flow_matching/launch_flow_matching_ablation.sh integration_steps_10 bc
```

`all` runs both stages. Repeating the same command validates completion markers
and skips completed work. A killed attempt is retained in an append-only
`*_attempt_*` directory; rerunning creates a new attempt and never overwrites it.

For another Python installation or MetaWorld checkout:

```bash
FLOW_ABLATION_PYTHON=/path/to/python \
METAWORLD_ROOT=/path/to/metaworld \
scripts/flow_matching/launch_flow_matching_ablation.sh training_epochs_100 all
```

Monitor a redirected run with `tail -f LOGFILE`. Completion can also be checked
with:

```bash
find runs/flow_matching_ablation/button_press_v3_seed0 -name '*.complete.json' -print
```

Interactive runs show epoch/step counters, elapsed time, ETA, live train and
validation losses, and the current best epoch. Set `FLOW_ABLATION_PROGRESS=0`
to disable progress bars while retaining stage-level start/completion messages:

```bash
FLOW_ABLATION_PROGRESS=0 \
scripts/flow_matching/launch_flow_matching_ablation.sh training_epochs_100 all
```

Only directories containing `experiment.complete.json` are aggregated:

```bash
python scripts/flow_matching/aggregate_flow_matching_ablation.py \
  --root runs/flow_matching_ablation/button_press_v3_seed0 \
  --output runs/flow_matching_ablation/button_press_v3_seed0/aggregate.csv
```

The aggregator verifies hashes for the artifacts named by each final marker and
refuses to replace an existing aggregate.
