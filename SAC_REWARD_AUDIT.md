# SAC Reward System Audit

Date: 2026-10-31
Scope: read-only inspection of the continual-learning repository and the local
`metaworld-kuka-v3` environment package. No training was launched and no source,
configuration, test, checkpoint, dataset, or result file was changed.

## Executive summary

The scalar reward consumed by SAC is computed by the selected environment, not by
`train_sac.py`. The training script forwards configuration into `make_env()`, sums
the rewards returned by `env.step()`, and selects/logs checkpoints from episode
return and success. Its large reward-specific section is observational evaluation
diagnostics: it reads reward components already emitted in `info` and reconstructs
behavior summaries. It does not replace the environment reward.

There are three real sources of complexity:

1. KUKA Cube Push contains many scientifically distinct, stateful reward
   experiments. They are active configurations, are tested, and must be kept for
   reproducibility.
2. The same reward-option validation is repeated in `src/config.py`,
   `src/env_adapters/kuka_v3.py`, and `metaworld_kuka/registration.py`.
3. `FixedSeedEvalCallback._evaluate()` contains a large schema-detection and
   aggregation implementation for historical KUKA reward diagnostics. This code is
   logging-only but makes the SAC trainer appear to implement rewards.

No reward implementation is currently safe to delete solely on the evidence in
this audit. The safest first refactor is to move evaluation-only KUKA diagnostic
aggregation out of `train_sac.py`, with characterization tests for every emitted
artifact and TensorBoard key.

Focused verification of the existing reward/configuration/checkpoint contracts:
`153 passed, 1 skipped, 60 subtests passed`. The skip is an existing optional
environment case; no failure was observed.

## 1. Current architecture and configuration flow

```text
YAML environment fields
  -> src.config.config_from_dict / validate_config
  -> ExperimentConfig.environment (EnvironmentConfig)
  -> train_sac.main
  -> src.envs.make_env (train and evaluation environments)
  -> backend adapter
       MetaWorld: make_metaworld_v3_env
       KUKA v2:   make_kuka_v2_env (reward option ignored by dispatcher)
       KUKA v3:   make_kuka_v3_env
  -> gym.make / KUKA registration
  -> task environment compute_reward/evaluate_state
  -> env.step returns scalar reward plus info components
  -> SB3 SAC consumes scalar reward
  -> FixedSeedEvalCallback sums returns and observes info diagnostics
  -> evaluations.npz, JSON diagnostics, TensorBoard, checkpoints, manifests
```

`train_sac.py` constructs both training and evaluation environments with the same
reward settings. Its `config.json` records the generic version and both task
variants/weights. The output hierarchy includes `reward_function_version` for
unnamed runs. Expert manifests record `reward_function_version` and, for Hammer,
the Hammer variant and weight. They do not currently record the Peg Unplug
variant/weight/gamma or all Cube Push distribution fields.

The separate `EnvironmentSettings` dataclass in `src/envs.py` mirrors
`EnvironmentConfig` and is used by other construction paths. Its `kwargs()` method
provides the full environment option bundle. `train_sac.py` currently passes the
same fields individually rather than using that object.

## 2. Reward-related inventory

### Configuration and dispatch

| Item | Responsibility and scope | Computes reward? | Callers / necessity | Assessment |
|---|---|---:|---|---|
| `EnvironmentConfig` in `src/config.py` | Typed identity and parameters for all backends | No | All parsed experiments; required | **KEEP** |
| `reward_function_version` | Selects MetaWorld v1/v2 or KUKA reward family | Indirectly | Active SAC, continual, replay and expert identity paths | **KEEP** |
| `hammer_reward_variant`, `hammer_nail_progress_weight` | Selects original or progress-augmented KUKA Hammer reward | Indirectly | Active Hammer SAC and KUKA DiffCRL configs | **KEEP** |
| `peg_unplug_reward_variant`, `peg_unplug_progress_weight`, `peg_unplug_progress_gamma` | Selects potential-based KUKA Peg Unplug shaping | Indirectly | Active baseline, ablation and resume configs | **KEEP** |
| Cube curriculum/initialization/distribution fields | Constrain which experimental Cube Push rewards are meaningful | No, but alter initial-state distribution | Active Cube Push experiments | **KEEP**; do not label them reward fields |
| `validate_config()` reward branches | Cross-field, task, backend, experiment and weight validation | No | Every YAML load | **KEEP**, then **SIMPLIFY** duplicated predicates |
| `EnvironmentSettings` | Immutable environment-construction option bundle | No | Continual/environment factory paths | **CONSOLIDATE** with call sites, not necessarily with parser type |
| `make_env()` | Backend dispatch and option forwarding | No | SAC, collection, continual evaluation and replay | **KEEP** |
| `make_metaworld_env()` | Backward-compatible MetaWorld alias | No | Tests and existing callers | **DEPRECATE** only after caller/history audit; not removable now |
| `make_metaworld_v3_env()` | Validates v1/v2 and forwards version to registered MT1 | No | MetaWorld training/evaluation | **KEEP** |
| `make_kuka_v2_env()` / adapter | Preserves the legacy environment's returned reward | No new logic | KUKA-v2 local/legacy configs | **KEEP** for reproducibility |
| `make_kuka_v3_env()` | Validates/scopes options and forwards them to registration | No | All KUKA-v3 paths | **KEEP**, **SIMPLIFY** duplicated validation |
| `make_kuka_mt1()` in the environment package | Registration boundary; validates again and installs task-specific options | No | Public KUKA Gym entry point | **KEEP**; it must defend non-repository callers |
| `EpisodeSuccessWrapper` | Latches any-step success into terminal `is_success` | No | Every backend | **KEEP**; success semantics, not reward computation |

### Training, evaluation, metadata and diagnostics

| Item | What it does | Computes reward? | Scope / callers | Assessment |
|---|---|---:|---|---|
| `TrainingSuccessCallback` | Rolling terminal any-step success for TensorBoard | No | SAC training callback | **KEEP** |
| `FixedSeedEvalCallback._evaluate()` | Deterministic rollouts; sums scalar returns; observes success and task-specific `info` | No | SAC evaluation | Core loop **KEEP**; KUKA diagnostic branches **MOVE/CONSOLIDATE** |
| `summarize_success_trace()` | Any/final/sustained success retention statistics | No | Evaluation and tests | **KEEP**, possibly move to shared evaluation diagnostics |
| `HammerEpisodeTracker` / `HammerEpisodeMetrics` | Read-only MuJoCo geometry, nail progress and contact diagnostics | No | Hammer evaluation only | **KEEP**; already appropriately separate |
| `summarize_hammer_episodes()` | Aggregates Hammer diagnostics for TensorBoard | No | SAC evaluation | **KEEP** |
| Push/handoff accumulators inside `_evaluate()` | Aggregate environment-emitted reward components and physical behavior | No | KUKA Push/Cube Push evaluation | **MOVE** to KUKA evaluation diagnostics |
| `_record_evaluation()` | Mean/std return, success rate, TensorBoard, best-return and best-success saves | No | SAC evaluation | **KEEP** |
| `export_run_summaries()` | Exports evaluation history, best-return and best-success summaries | No | End of SAC run and tests | **KEEP**; shares ranking rules that should be centralized |
| `qualification_record()` | Stores success, return, std and evaluation identity | No | Expert manifests | **KEEP** |
| Expert manifest reward fields/validation | Prevents use of experts trained under incompatible reward identity | No | DiffCRL/expert loading | **KEEP**; extend identity coverage before simplifying |
| Collector/schema reward handling | Validates finite scalar rewards, stores arrays and checks episode-return sum | No | Expert trajectory collection | **KEEP** |

## 3. Native MetaWorld rewards

MetaWorld v3 accepts only `v1` and `v2` through this repository's adapter.
`train_sac.py` does not contain MetaWorld reward formulas. It forwards the selected
version to `gym.make("Meta-World/MT1", ...)`; each MetaWorld task's
`compute_reward()` selects the native implementation. The adapter then adds only
observation compatibility and episode-success latching.

- `v2` is the repository default and the version used by current standard SAC,
  continual-learning, DDPM and Flow Matching configurations.
- `v1` remains explicitly tested by `tests/test_reward_version.py` and remains a
  supported scientific option even though no active top-level SAC configuration
  found by this audit selects it.
- Reward normalization wrappers exist in the external MetaWorld package but are
  not enabled by this repository's `make_metaworld_v3_env()` construction path.

Classification: both versions **KEEP**. Removing v1 would break the public
configuration contract, tests and historical reproducibility.

## 4. KUKA rewards

### KUKA v2

KUKA v2 is a legacy backend. `KukaV2GymnasiumAdapter.step()` converts API types but
returns the raw environment's scalar reward unchanged. `make_env()` does not pass
`reward_function_version` into the v2 constructor. Thus a YAML value such as `v2`
is metadata/configuration compatibility, not a selectable KUKA-v2 reward formula.

This is mildly confusing but not safely removable: local KUKA-v2 configs, tests,
trajectory collection and historical runs still exist. Classification: **KEEP**
with documentation; consider backend-specific configuration projection later.

### Standard KUKA v3 tasks

All 15 KUKA-v3 task classes own their `compute_reward()` implementation. Reach,
Push, Cube Push, Handle Press Side, Button Press, Button Press Topdown, Peg Unplug,
Hammer, Drawer Close, Faucet Open/Close, Window Open/Close, Door Open and Pick
Place are registered. Most use the KUKA task's v2 formula; several retain a v1
branch, while stricter classes reject anything except v2. This difference is
task-specific environment behavior, not training-script duplication.

Every standard `compute_reward()` is required by its environment. The repeated
version checks could eventually share constants/error formatting inside the KUKA
package, but merging task formulas would be inappropriate.

### KUKA Push and Cube Push variants

| Variant | Task/config usage | Actual behavior | Status |
|---|---|---|---|
| `v2` | Standard KUKA tasks; active baselines | Native dense task reward | **KEEP** |
| `v3` | Push/Cube Push; archived reward probe | Caging/contact plus clipped signed progress and one-time success | **KEEP** for archived reproducibility |
| `v3_1` | Active Push and Cube Push configs | Contact-gated progress; caging when not in contact; one-time success | **KEEP** |
| `v3_2_retention` | Active Push config | v3.1 plus post-success retention reward/penalty | **KEEP** |
| `v3_3_contact_push_progress` | Active Cube Push curriculum configs | Approach potential, first-contact bonus, contact-attributed displacement, goal progress and first success | **KEEP** |
| `v4_simple_approach_progress` | Active Cube Push config | Approach potential, signed progress, one success bonus; zero after success | **KEEP** |
| `v5_contact_start_progress` | Active contact-start Cube Push config | Signed progress and first-success bonus; zero after success | **KEEP** |
| `v6_rear_contact_push_progress` | Four active contact/near-contact Cube Push configs | Rear-contact bonus, attributable forward push, signed progress, first success | **KEEP** |
| `v1_initial_to_contact` | Active Stage-1 Cube Push config | TCP approach potential plus first sustained-handoff bonus | **KEEP** |
| `v1_1_initial_to_contact_aperture` | Active Stage-1 Cube Push config | v1 Stage-1 behavior plus aperture potential | **KEEP** |
| `sawyer_push_v2_adapted` | Active baseline/resume Cube Push configs | Sawyer-v2-style caging/in-place reward adapted to Robotiq geometry | **KEEP** |

These variants have distinct RNG-neutral state, latches and `info` schemas. They
must not be collapsed merely because some share progress or success terms.

### Task-specific additive variants

- Hammer `original`: the native v2 grasp/orientation/in-place formula.
- Hammer `nail_progress`: adds
  `hammer_nail_progress_weight * clip(nail_qpos / 0.09, 0, 1)` before the existing
  success cap. It is used by the active 1M expert, development config and six KUKA
  DiffCRL configs; archived controlled ablations also cover original and weights 2/4.
- Peg Unplug `original`: native v2 formula.
- Peg Unplug `peg_unplug_progress_v1`: potential-based extraction progress using
  the configured weight and gamma. Active 200k and resume-to-600k configurations
  require it.

All four selections **KEEP**. Their parameters change actual training reward, not
only logging.

## 5. Logging and checkpoint selection versus reward computation

The responsibilities are behaviorally separated, but not visually separated:

- SAC receives only `reward` from `env.step()`.
- Episode return is the undiscounted sum used for evaluation reporting. One
  Stage-1 diagnostic additionally computes a fixed gamma-0.99 discounted return;
  this is logging-only and is not the SAC objective.
- `best_model/best_model.zip` is selected by strict improvement in mean return.
- `best_success_model/best_success_model.zip` is selected lexicographically by
  `(success_rate, mean_return)`.
- Early stopping is success-based and independent of reward version.
- TensorBoard records `eval/mean_reward`, `eval/std_reward`, `eval/success_rate`,
  rollout success and optional KUKA behavior/reward-component summaries.
- `evaluations.npz` stores per-episode returns, lengths, successes and return std.
  Separate JSON files store behavior, success, Hammer and handoff diagnostics.
- `export_run_summaries()` recomputes the same best-return and lexicographic
  best-success indices for CSV/JSON/Markdown export.

The return/success ranking duplicated between callback recording, final manifest
qualification and summary export is legitimate output behavior but should use two
small shared pure index helpers to prevent drift.

## 6. Active and historical evidence

Static parsing found these non-archived SAC selections:

- 23 configurations: v2 with original task variants.
- 2: v2 Hammer nail-progress.
- 2: v2 Peg Unplug progress.
- 3: v3.1.
- 1 each: v3.2 retention, v4, v5, Stage-1 v1 and Stage-1 v1.1.
- 4 each: v3.3 and v6.
- 2: Sawyer-v2 adapted.

Outside `configs/sac`, six active KUKA DiffCRL configurations select Hammer
nail-progress and the standard continual/replay configurations use v2. Archived
configs still directly reference v3, v3.1, v2 Hammer original and nail-progress.
Tests deliberately load several archived ablations to enforce controlled
comparisons. Therefore “archived” does not mean removable.

## 7. Redundancy and complexity findings

### Genuine redundancy

1. The allowed reward-version tuple and task-scoping rules are repeated in
   `src/config.py`, `src/env_adapters/kuka_v3.py`, and KUKA registration.
2. `EnvironmentConfig` and `EnvironmentSettings` contain the same environment
   fields, while `train_sac.py` manually forwards them twice.
3. Best-return and best-success index logic appears in the callback, final expert
   manifest preparation and summary exporter.
4. Cube Push diagnostic schema inference in `_evaluate()` uses presence of keys
   such as `reward_progress`, `reward_total` and `reward_object_grasped`. The
   environment already exposes `reward_function_version`; inference is redundant
   and fragile.
5. Push diagnostic accumulation contains repeated branches for each historical
   reward component schema.

### Legitimate differences

- Task-specific `compute_reward()` methods are not duplication: they encode
  different objects, success conditions and geometry.
- Cube Push v3-v6 and Stage-1 variants are scientifically different objectives.
- Success metrics and reward returns are intentionally separate; neither should
  replace the other.
- Validation at the external KUKA registration boundary is necessary even if the
  repository validates earlier, because the package has independent callers.

### Hard-coded knowledge in `train_sac.py`

The evaluator knows KUKA task names, observation slices, private environment
methods/state, reward component key names, reward versions and a literal diagnostic
discount of 0.99. This does not alter reward computation, but it couples the SAC
runner tightly to KUKA implementation details. In particular, the diagnostic 0.99
should be labeled explicitly or sourced from the configured SAC gamma if it is
intended to represent the training discount.

### Metadata gaps

`config.json` is complete enough to reproduce current runs, but compact summaries
display only `reward_function_version`. Expert manifests encode Hammer variants
but not Peg Unplug variants or all Cube Push curriculum/initialization/distribution
identity. Removing any configuration field before extending/checking these
contracts would make historical checkpoint identity less reliable.

## 8. Removal-safety decisions

| Candidate | Production/config/test/checkpoint evidence | Decision |
|---|---|---|
| MetaWorld v1 | Public adapter/continual paths and `test_reward_version`; historical semantics | **KEEP** |
| KUKA Push v3 | Archived config is loaded by current regression tests | **DEPRECATE** for new work if desired, but do not remove |
| All other Cube Push variants | Non-archived configs and dedicated tests | **KEEP** |
| Hammer original/progress | Active and archived configs, DiffCRL and manifest validation | **KEEP** |
| Peg Unplug original/progress | Active baseline/ablation/resume and tests | **KEEP** |
| KUKA-v2 reward compatibility | Production backend and local historical configs/tests | **KEEP** |
| `make_metaworld_env` alias | Compatibility callers/tests exist | **DEPRECATE**, not remove |
| Adapter-side KUKA validation | Protects repository callers but overlaps config/registration | **SIMPLIFY**, not remove wholesale |
| Registration-side validation | Public package boundary | **KEEP** |
| Reward-component diagnostics in `_evaluate()` | Produces JSON/TensorBoard artifacts; tests cover portions | **MOVE/CONSOLIDATE**, not remove yet |
| Schema inference by reward keys | Version is already available from environment | **SIMPLIFY** after exact artifact tests |
| Duplicate ranking calculations | Same intended semantics in three locations | **CONSOLIDATE** into pure helpers |
| Reward statistics/export code | Current user-visible artifacts and checkpoint qualification | **KEEP** |

There are no evidence-supported immediate **REMOVE** candidates. Safe removal can
be reconsidered only after a checkpoint/config inventory establishes that a
deprecated variant has no required historical reproduction path.

## 9. Recommended architecture

Keep the design small:

1. Environment classes continue to own scalar reward computation and component
   emission in `info`.
2. Configuration owns typed fields and experiment-level cross-field validation.
3. Backend adapters forward options and perform only boundary validation that
   cannot be delegated safely.
4. External KUKA registration remains independently defensive.
5. SAC training remains reward-agnostic: consume scalar rewards, calculate returns,
   and select checkpoints.
6. Move KUKA-specific observational aggregation to a small diagnostics module with
   a lifecycle such as `start_episode`, `observe_step`, `finish_episode`. This is
   composition, not a reward framework or trainer hierarchy.
7. Select diagnostic parsing explicitly from `(task_name,
   reward_function_version)` rather than guessing from `info` keys. Unknown schemas
   should either produce generic metrics or fail clearly in strict diagnostic mode.
8. Use shared pure functions for best-return and best-success indices.
9. Define a single repository-side KUKA reward capability table for config and
   adapter validation, while retaining equivalent validation in the external
   package boundary.
10. Extend checkpoint identity metadata before removing or hiding reward-related
    configuration fields.

## 10. Risks and required tests

Any refactor must preserve:

- exact per-step scalar rewards and `info` components for fixed state/action traces;
- reset behavior of success/contact latches and potential-shaping history;
- environment construction acceptance/rejection for every version/task/backend;
- train/evaluation environment option equality;
- fixed-task rollout seeds and task hashes;
- return arrays, return std, success arrays and all diagnostic JSON schemas;
- TensorBoard tag names and values;
- strict best-return and lexicographic best-success tie-breaking;
- early-stopping behavior;
- old expert manifest loading and reward identity checks;
- resume configs, especially Peg Unplug and Cube Push resume experiments.

Recommended characterization coverage before changes:

1. Parameterize every reward version over valid and invalid tasks.
2. Record exact fixed transition traces for each KUKA variant, including reset and
   post-success behavior.
3. Snapshot `_evaluate()` artifacts and TensorBoard records for generic,
   Hammer, Push and both Stage-1 handoff families.
4. Test reward-schema selection explicitly rather than only testing individual
   helper summaries.
5. Round-trip old and new expert manifests, including legacy manifests without
   newly introduced optional identity fields.
6. Compare uninterrupted and resumed runs' environment identity metadata without
   launching full training.

## Prioritized refactoring plan

1. **Characterize evaluation outputs first.** Add small deterministic tests for
   every diagnostic family and ranking/tie behavior.
2. **Extract logging-only KUKA diagnostics from `FixedSeedEvalCallback`.** Preserve
   every output key and value; do not touch reward formulas.
3. **Centralize return/success checkpoint ranking.** Reuse pure helpers in callback,
   manifest qualification and exporters.
4. **Replace reward-key inference with explicit version dispatch.** Retain a tested
   compatibility fallback for historical/custom environments if required.
5. **Reduce repository-side validation duplication.** Introduce constants/pure
   predicates shared by `src/config.py` and the KUKA adapter; keep external
   registration validation.
6. **Adopt `EnvironmentSettings` consistently at construction call sites.** Do this
   only after tests prove identical kwargs for train, evaluation and continual
   paths.
7. **Complete reward identity metadata.** Add backward-compatible optional Peg and
   Cube fields before considering deprecation/removal.
8. **Review deprecated variants with artifact evidence.** Only then decide whether
   v3 or compatibility aliases can move to a historical support module. No reward
   variant should be deleted in the initial cleanup.
