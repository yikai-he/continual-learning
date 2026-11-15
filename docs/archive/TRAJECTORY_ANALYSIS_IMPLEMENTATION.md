# Archived trajectory-analysis implementation

This document records the generated-trajectory quality analysis that existed at Git revision
`f4f36c762c31de95cc0c0c4f18af7811fb3251a3` before its removal. It describes the source as
implemented, not a new benchmark protocol. Historical results and documents remain valid records
of that implementation.

## Scope and purpose

The analysis described whether generated replay looked structurally similar to physical MetaWorld
trajectories. It was diagnostic only: it did not score continual-learning performance and did not
change DDPM, Flow Matching, BC, replay allocation, evaluation, or checkpoint selection. The main
features were action-bound summaries, goal consistency, previous-state alignment, generated versus
expert action disagreement, simple distribution summaries, and representation/sampling-step reports.

The training algorithms used physical tensors shaped `(N, T, 43)`: 39 observation coordinates
followed by four action coordinates. Observation slices were current state `0:18`, previous state
`18:36`, and goal `36:39`.

## Online training diagnostics

### Generated action projection diagnostics

Source: `src/continual/diffusion_data.py`, `generated_action_diagnostics`, called by
`src/continual/replay_common.py`, `prepare_generated_replay`.

Inputs were the decoded physical tensor before projection, the tensor after optional action
projection, and the projection name. The function first required equal shapes and exact equality of
all observation coordinates. For action tensors `a_before` and `a_after`, it stored:

- pre/post out-of-bounds component fraction: `mean(abs(a) > 1)`;
- pre-projection out-of-bounds timestep fraction: `mean(any(abs(a_before) > 1, axis=action))`;
- clipped component fraction: `mean(a_before != a_after)`;
- fractions exactly at one and above `0.99` after projection;
- the last two fractions separately for each action coordinate.

The values were merged into each generated replay source record. Action decoding and projection
themselves were separate required operations and were not diagnostic: raw actions were unchanged,
pre-tanh actions were decoded with `tanh`, and the `clip` projection clamped physical actions to
`[-1, 1]`.

For pre-tanh replay, `prepare_generated_replay` also calculated a BC latent cap
`atanh(1 - BC_CLAMP_EPSILON)` and stored the fraction of sampled latent action components whose
absolute value exceeded that cap as `generated_latent_abs_gt_bc_cap_fraction`.

### Structural trajectory diagnostics

Source: `src/continual/diffusion_data.py`, `generated_trajectory_diagnostics`, called once per
historical task by `src/continual/replay_common.py`, `generate_replay_groups`.

Inputs were generated physical trajectories and one reference goal per trajectory. The result
contained `invalid_shape` and, for floating tensors, exact NaN and Inf counts. When shapes were
valid, it calculated:

- goal error at each timestep: Euclidean norm `||goal[t] - reference_goal||_2`;
- mean and maximum goal error;
- within-trajectory goal variation: mean `||goal[t] - goal[0]||_2`;
- fraction of goal errors greater than a default tolerance of `1e-3`;
- previous-state alignment for `t > 0`: `||previous[t] - current[t-1]||_2`;
- mean, maximum, and fraction above tolerance for that alignment;
- timestep-zero reset error: mean `||previous[0] - current[0]||_2`;
- action bound violation `max(abs(action) - 1, 0)`, reported as component fraction,
  timestep fraction, and maximum violation.

This dictionary was stored as `source["structural_diagnostics"]`. Reference goals were also stored
in source provenance. Those reference goals were not merely diagnostic: later stage preparation used
them as the generator's goal-conditioning targets and therefore they remain part of training.

### Real and BC action summaries

Sources: `src/continual/diffcrl.py` and `src/continual/stage_dataset.py`.

Expert collection stored `real_diffusion_encode_clip_fraction`, the fraction of physical expert
action components outside `[-1 + action_clamp_epsilon, 1 - action_clamp_epsilon]`. Stage preparation
stored `bc_target_clip_fraction`: actions were affinely scaled using the policy action bounds and the
same kind of near-boundary fraction was calculated with `BC_CLAMP_EPSILON`.

These were descriptive. The actual pre-tanh transform, BC inverse-tanh handling, physical action
bounds, and rejection of nonfinite or invalid-shape data were separate algorithm/safety behavior.

### Stage-report fields

`src/continual/reporting.py`, `build_diffcrl_stage_report`, copied the real encode fraction and full
source provenance into `stage_*/report.json`. It also emitted diagnostic placeholders
`nan_count: 0`, `inf_count: 0`, `generated_success: null`, and `action_targets_modified`, the latter
being true when any source's `clipped_component_fraction` was positive. Historical JSON files keep
their original meaning; new reports no longer emit these trajectory-quality fields.

## Validation that was adjacent but not analysis

The following behavior was not removed:

- trajectory shape and finite-value checks;
- full-horizon collection and packing;
- physical expert-action bound checks;
- action decoding and configured projection;
- generated replay shape checks and pre-tanh physical-bound checks;
- exact task-bank and goal reconstruction;
- reference goals required for generator conditioning;
- real-trajectory goal constancy and previous/current-state contract validation;
- train/validation split leakage checks and trajectory hashes.

These checks protect training semantics or data integrity rather than produce quality summaries.

## Offline analysis tools

The removed dedicated tools were:

- `scripts/evaluation/analyze_ddpm_representation.py`: regenerated DDPM replay and wrote structural,
  action, and coordinate-distribution summaries. Its distribution helper reported mean, standard
  deviation, minimum, maximum, and selected quantiles over flattened coordinates.
- `scripts/evaluation/sample_kuka_fm_sampling_steps.py` and
  `analyze_kuka_fm_sampling_steps.py`: sampled Flow Matching at multiple integration-step counts,
  timed generation, and summarized the structural metrics above.
- `scripts/evaluation/analyze_kuka_expert_disagreement.py`: evaluated frozen SAC experts on saved
  generated observations. For generated and expert actions it calculated per-sample squared L2 error
  (sum over four coordinates), then mean, median, population standard deviation, 95th percentile,
  maximum, and per-coordinate MSE. Variants corrected saved goals, previous-state coordinates, both,
  or clipped generated actions before comparison. It also reported action means, variances, bound
  fractions, and maximum violation.
- `scripts/evaluation/analyze_kuka_generator_200plus200.py`: recomputed structural diagnostics from
  saved generated arrays and optionally incorporated expert-disagreement results.
- `scripts/evaluation/analyze_kuka_representation_four_stage.py` and
  `analyze_kuka_representation_multiseed.py`: combined continual-learning outcomes with stored or
  recomputed trajectory consistency summaries for representation ablations.
- `scripts/evaluation/validate_stochastic_bc_cross_task_replay.py`: retrained diagnostic policies
  against saved generated replay while reporting action saturation and stored structural summaries.
- `scripts/evaluation/validate_fixed_replay_bc_seeds.py`: extended that diagnostic retraining path
  across fixed saved replay seeds and imported its trajectory-quality summary helpers.
- `scripts/experiments/run_kuka_fm_sampling_steps.sh`: orchestrated the removed Flow Matching
  sampling-step diagnostic sampler.

The retained `analyze_kuka_flow_matching_ablation.py` also exposed stored structural summaries in
its result records; that field was removed while its performance, timing, pairing, and continual-
learning metric comparisons remain available.

Associated exclusive tests covered the expert-disagreement formulas and Flow Matching sampling-step
diagnostic aggregation. General DDPM/FM numerical, replay, evaluation, and continual-learning tests
were independent and remain.

No production nearest-neighbor trajectory comparison was found in this revision. Mentions of
nearest interpolation in the temporal U-Net were neural-network operations, not trajectory analysis.

## Configuration and dependencies

No dedicated production configuration switch enabled these online diagnostics: they ran
unconditionally whenever generated replay was prepared. `generated_action_projection` and
`action_clamp_epsilon` appeared in their outputs but are algorithm/checkpoint settings and remain.
Offline scripts had their own input/output CLI arguments and read existing run artifacts. Historical
representation-ablation YAML files are retained as experiment records.

Diagnostic calculation used NumPy and PyTorch already required by training. Expert-disagreement
analysis additionally loaded Stable-Baselines3 SAC experts and the saved replay arrays.

## Limitations

- Structural consistency did not demonstrate environment-dynamics validity or rollout success.
- Expert actions on generated, off-distribution observations were defined but could be meaningless.
- Action-bound statistics did not measure whether replay improved continual learning.
- Goal and previous-state summaries depended on the chosen reference goals and tolerance.
- Most summaries pooled all trajectories and timesteps, hiding temporal and multimodal structure.
- Diagnostic computations increased CPU/GPU tensor work and enlarged per-stage JSON provenance.

## Reconstruction

Use the Git revision above to recover the exact removed implementation. The primary recovery points
are `generated_action_diagnostics` and `generated_trajectory_diagnostics` in
`src/continual/diffusion_data.py`, their invocation in `src/continual/replay_common.py`, and the seven
offline scripts listed above. Historical reports under `docs/experiments/` and existing run artifacts
show the serialized field names and representative outputs. Reintroducing the analysis should remain
read-only with respect to generated tensors so sampling order and training RNG state are unaffected.
