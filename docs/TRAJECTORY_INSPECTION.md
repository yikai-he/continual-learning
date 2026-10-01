# Trajectory inspection

`scripts/diagnostics/inspect_trajectory.py` is a read-only utility for the
physical trajectory artifacts produced by DiffCRL. It loads files, computes
descriptive diagnostics, prints or exports them, and optionally plots them. It
does not normalize, clip, rewrite, or validate data through the training path.

Supported inputs are:

- `real_trajectories.npz`, containing collected `Trajectory` fields per episode;
- `data_task_<index>.npy`, containing packed `(N,T,43)` physical trajectories.

The latter can hold generated replay for an old task or real data for the
current task. The array alone does not record provenance; consult that stage's
report/source metadata to distinguish them.

The 43 columns use repository constants: current state `[0:18]`, previous state
`[18:36]`, goal `[36:39]`, and action `[39:43]`. The repository does not define
stable meanings for individual dimensions within the 18D state, so the utility
uses indexed names and does not infer semantic labels.

## Examples

Inspect and export one collected real episode:

```bash
python scripts/diagnostics/inspect_trajectory.py \
  --input runs/<run>/<stage>/real_trajectories.npz \
  --traj-index 3 \
  --output-txt trajectory_003.txt \
  --output-csv trajectory_003.csv
```

Inspect one generated trajectory from a prior task's stage group:

```bash
python scripts/diagnostics/inspect_trajectory.py \
  --input runs/<run>/<stage>/data_task_0.npy \
  --traj-index 3
```

Rank suspicious trajectories in a file:

```bash
python scripts/diagnostics/inspect_trajectory.py \
  --input runs/<run>/<stage>/data_task_0.npy \
  --scan-all --top-k 5
```

Compare aggregate real and generated statistics:

```bash
python scripts/diagnostics/inspect_trajectory.py \
  --compare-real runs/.../stage_0_button-press-v3/real_trajectories.npz \
  --compare-generated runs/.../stage_1_faucet-close-v3/data_task_0.npy \
  --output-json button_real_vs_generated.json
```

Add `--plot` for action/state traces or comparison histograms when matplotlib
is installed. Use `--state-dims 0 4 10` to choose plotted state indices.
Plots and comparisons are descriptive diagnostics only and do not establish
physical validity.


## Dataset comparison output

Comparison pools every finite state-step delta across all trajectories and
reports mean, standard deviation, median, p90, p95, p99, maximum, and sample
count. Generated delta tails are measured against the provided real reference's
p90, p95, and p99. Action output includes component- and timestep-level
saturation at `0.90`, `0.95`, `0.99`, and `1.00`, plus generated overshoot
percentiles and the worst individual components.

Indexed `state_00` through `state_17` and `action_00` through `action_03`
summaries include distribution percentiles. State support comparisons report
generated fractions below the real p01 and above the real p99; action support
reports the fraction outside the corresponding real p01–p99 interval.

The two inputs are described only as the **provided real reference** and
**provided generated dataset**. Filenames are not treated as provenance.
`data_task_<index>.npy` may contain either generated old-task replay or real
current-task data; consult stage report/source metadata to establish which.

Previous-state alignment and goal drift are primarily structural pipeline
checks because structured replay reconstructs them deterministically.
State-delta, action-distribution, tail, and support comparisons are descriptive
replay-quality diagnostics. None of these metrics alone demonstrates simulator
or physical validity.


## Near-constant states and temporal OOB concentration

When a real state dimension has standard deviation below `1e-6`, comparison
uses the real median as a constant reference. It reports generated absolute-deviation
percentiles and the fraction exceeding an absolute tolerance of
`1e-6`, rather than presenting p01/p99 support exceedance as the primary
metric. Tiny deviations such as `1e-8` around a real constant zero are normally
numerical deviations and are not by themselves evidence of meaningful state-
distribution failure. Non-constant dimensions continue to use real p01/p99
support comparisons.

The comparison JSON also contains `generated_oob_by_timestep`. For every
available timestep it reports action-component OOB counts/fractions and the
number/fraction of trajectories with any OOB component. Its ranked lists and
optional unsmoothed plot can reveal whether violations cluster around specific
trajectory phases. Variable-length inputs use only trajectories that contain
the reported timestep.

Both constant-reference deviation and OOB-by-timestep results are descriptive.
Neither establishes physical or simulator validity.
