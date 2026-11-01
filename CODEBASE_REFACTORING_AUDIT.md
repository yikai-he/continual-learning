# Continual Learning Codebase Refactoring Audit

## Scope and method

This is a read-only source audit performed on 2026-10-31. The only repository
change made by the audit is this report. No source, configuration, checkpoint,
dataset, result, or run artifact was modified.

The requested `IMPLEMENTATION_ARCHITECTURE.md` is not present in this checkout,
so the implementation, imports, CLI entry points, tests, README, and current
configuration tree were treated as authoritative. Static inspection covered all
Python files under `src/`, `scripts/`, and `tests/`, plus the top-level Python
entry points. Import searches, call-site searches, documentation references,
configuration hashes, and checkpoint serialization code were also inspected.

### Safety baseline

- The worktree was already heavily modified. It contains modified, renamed,
  deleted, and untracked work across configuration, source, scripts, tests, and
  documentation. Those changes were preserved.
- `runs/`, checkpoints, datasets, evaluation results, and historical artifacts
  were not touched.
- The documented full validation command is:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`.
- Focused suites can be run with the same prefix followed by explicit test paths.
- Additional safe checks are `python -m compileall -q src scripts *.py`,
  configuration-loading tests, CLI `--help` smoke tests, and `git diff --check`.
- DiffCRL, GeneralPolicy, DDPM, and Flow checkpoints store model/optimizer state
  dictionaries plus metadata rather than pickled project class instances.
  Internal module moves therefore should not alter their tensor format, but load
  APIs, metadata validation, and resume tests must still remain stable. SB3 ZIP
  checkpoints must continue to be loaded through Stable-Baselines3.

## 1. Current architecture overview

The public flow is:

```text
YAML -> src.config -> top-level CLI
                       |
                       +-> train_sac.py -> src.envs/adapters -> SB3 SAC
                       +-> train_continual_sac.py -> sequential_sac -> evaluation
                       +-> train_diffcrl.py -> DiffCRLTrainer
                                                |-> expert collection
                                                |-> grouped stage dataset
                                                |-> DDPM / Flow / real replay
                                                |-> persistent GeneralPolicy BC
                                                |-> fixed-task evaluation
                                                `-> reports, checkpoints, resume markers
```

`src/envs.py` is the environment facade. `src/env_adapters/` isolates backend
differences. `src/continual/` contains the scientific pipeline and models.
`src/support/` contains persistence, provenance, synchronization, and stage-data
helpers. Top-level files are training entry points; `scripts/` contains offline
collection, evaluation, visualization, diagnostics, and experimental studies.

The scientific boundaries are mostly sound. The main complexity hotspots are
`train_sac.py` (1,475 lines), `src/continual/diffcrl.py` (1,288 lines),
`scripts/render_kuka_push_policy.py` (1,100 lines), and `src/config.py` (932
lines). The strongest genuine duplication is between DDPM and Flow model fitting,
checkpoint validation, and replay generation.

## 2. Production source inventory

“Active” means reachable from a CLI, another production module, an offline
script, or tests. “Overlap” identifies related behavior, not an automatic merge.

### Environment and adapters

| File | Responsibility and important API | Dependents / status / overlap |
|---|---|---|
| `src/envs.py` | Public backend facade: `EnvironmentSettings`, `make_env`, `make_metaworld_env`, horizon/base-env/target/configuration helpers. | Widely used by all trainers, collection, evaluation, diagnostics, and tests; **active**. Delegates rather than duplicates adapters; keep as the stable facade. |
| `src/env_adapters/__init__.py` | Shared `EpisodeSuccessWrapper` and adapter package boundary. | Imported through package-relative paths by adapters; **active**. Tiny but meaningful package contract. |
| `src/env_adapters/metaworld_v3.py` | MetaWorld-v3 factory and 39D observation compatibility via `MetaWorldObservationWrapper`. | Used by `src.envs`; **active**, backend-specific, no improper overlap. |
| `src/env_adapters/kuka_v2.py` | Legacy KUKA-v2 Gymnasium adapter and factory. | Used by `src.envs` and integration tests; **active compatibility backend**. Do not remove while KUKA-v2 configs/tests are retained. |
| `src/env_adapters/kuka_v3.py` | KUKA-v3 task discovery and configured factory. | Used by `src.envs`, task banks, renderer, diagnostics, and tests; **active**. Some diagnostic scripts repeat low-level MuJoCo discovery, but that is analysis-specific. |
| `src/hammer_diagnostics.py` | `HammerEpisodeTracker`, metrics, and summary for KUKA Hammer evaluation. | Used by `train_sac.py`, checkpoint comparison, and tests; **active** and appropriately task-specific. |

### Configuration and experiment management

| File | Responsibility and important API | Dependents / status / overlap |
|---|---|---|
| `src/config.py` | All typed experiment dataclasses; strict YAML merge/load/validation; device, CLI override, output, and resolved-config helpers. | Used by every trainer and many tests/scripts; **active**. Large, but centralizing schema and cross-field validation avoids divergent behavior. Internal organization can improve without splitting the public schema prematurely. |
| `src/support/run_manifest.py` | Creates canonical provenance manifests and completion records. | Used by SAC, sequential SAC, DiffCRL, and tests; **active**. Slight overlap with resume markers, but run lifecycle and stage lifecycle differ. |
| `src/support/expert_manifest.py` | Expert identity, hash/sidecar validation, qualification record, and manifest writing. | Used by SAC, DiffCRL, evaluator, expert-space validation, and tests; **active**. Keep separate from run manifests. |
| `src/continual/resume.py` | Scientific-config comparison, RNG state, immutable stage-boundary markers, and resume loading. | Used by DiffCRL, stage sync, offline Flow analysis, and tests; **active**. Some hash/path mechanics overlap with support persistence. |
| `src/support/stage_sync.py` | Verified copy, validation, status, and restore for completed stage backups. | Used by `train_diffcrl.py`, `manage_diffcrl_sync.py`, and tests; **active operational boundary**. |
| `src/support/io.py` | Exclusive/atomic JSON and Torch persistence: `write_json`, `atomic_write_json`, `atomic_torch_save`. | Used throughout model/checkpoint code; **active**. |
| `src/support/reproducibility.py` | `file_hash` and deterministic model `state_hash`. | Widely used; **active**. Small but coherent and stable. |

### Trajectory collection and data processing

| File | Responsibility and important API | Dependents / status / overlap |
|---|---|---|
| `src/continual/schema.py` | `Trajectory` dataclass and strict trajectory validation. | Used by collector, packing, and tests; **active foundational contract**. |
| `src/continual/policy_adapter.py` | Common `Policy` protocol and `SB3SACPolicy`. | Used by collection, evaluation scripts, sequential SAC, and tests; **active**, prevents SB3 leakage into generic collection. |
| `src/continual/collector.py` | Single/multiple episode collection through the policy protocol. | Used by evaluation and expert collection; **active**. |
| `src/continual/expert_collection.py` | Standalone expert-data config, successful rollout collection, validation, and dataset export. | Used by collection CLI and tests; **active**. It intentionally builds on collector/packing. |
| `src/continual/expert_space.py` | Validates expert observation/action spaces and authenticated legacy KUKA repair. | Called by DiffCRL via relative import and directly tested; **active compatibility logic**. |
| `src/continual/diffusion_data.py` | Physical `(obs, action)` packing; action transforms; goal-conditioned 22D encode/decode; stage-group validation. | Used by models, replay, expert collection, inspection, dataset support, and tests; **active core data contract**. |
| `src/support/dataset.py` | Deterministic whole-episode split, BC flattening, group validation/preparation/concatenation. | Used by DiffCRL, Flow sanity, and tests; **active**. It is continual-stage-specific despite residing in `support`. |
| `src/continual/task_bank.py` | Deterministic backend task banks, identities/hashes, disjointness, training configuration sampling, and replay goal reconstruction. | Used by trainers/evaluation/replay and tests; **active reproducibility boundary**. |
| `src/continual/task_sequence.py` | `TaskSpec`, `TaskSequence`, and lazy `TaskSwitcher`. | Used by orchestration/evaluation/scripts/tests; **active**. |

### Models, BC, and replay

| File | Responsibility and important API | Dependents / status / overlap |
|---|---|---|
| `src/continual/bc_policy.py` | Persistent `GeneralPolicy`; checkpoint save/load; `fit_bc` with best-validation model and optimizer restoration. | Used by DiffCRL, comparison runner, studies, evaluators, and tests; **active**. Keep model/training together unless checkpoint API remains unchanged. |
| `src/continual/trajectory_diffusion.py` | `FeatureNormalizer`, temporal denoiser, goal/task-conditioned `TrajectoryDiffusion`, save/load/sample/loss, `fit_diffusion`. | Used by DiffCRL/replay/studies/tests; **active**. Shares training/checkpoint patterns with Flow. |
| `src/continual/trajectory_flow_matching.py` | Flow configuration, path/embedding/velocity network, `TrajectoryFlowMatching`, save/load/sample/loss, `fit_flow_matching`. | Used by DiffCRL/replay/studies/tests; **active**. Correctly separate algorithm, but duplicates mechanics from DDPM. |
| `src/continual/diffcrl_replay.py` | DDPM `ReplayRequest`/`ReplayResult` and previous-task replay generation. | Called by DiffCRL via relative import; directly tested; **active**. Strong structural overlap with Flow replay. |
| `src/continual/flow_matching_replay.py` | Flow request/result validation, checkpoint metadata validation, conditioned generation, decode, diagnostics, and provenance. | Called by DiffCRL via relative import; directly tested; **active**. Strong structural overlap with DDPM replay. |
| `src/continual/real_replay.py` | Deterministic training-only historical sampling with replacement and provenance. | Called by DiffCRL and tested; **active**. Semantics differ enough from generated replay to remain separate. |
| `src/continual/replay_allocation.py` | Exact deterministic uniform allocation and legacy/fixed-total policy resolution. | Called by DiffCRL and used by experiment tests; **active**. Small but a meaningful scientific policy boundary. |
| `src/continual/diagnostics.py` | `generated_action_diagnostics` for replay action bounds. | Used by both replay providers through relative imports; **active**, but only 39 lines and tightly coupled to replay/data. Candidate for consolidation. |
| `src/continual/generation_quality.py` | Offline expert-disagreement and state-delta trajectory metrics. | Used only by Flow analysis/validation scripts; **active offline analysis**, not part of training. |

### Continual orchestration, evaluation, and reporting

| File | Responsibility and important API | Dependents / status / overlap |
|---|---|---|
| `src/continual/diffcrl.py` | Stage dataclasses/report construction and `DiffCRLTrainer`: experts, banks, replay, collection, splitting, generator/BC fitting, evaluation, checkpointing, resume. | Main DiffCRL engine used by CLIs, comparison runner, and many tests; **active**. At 1,288 lines it combines too many lifecycle concerns. |
| `src/continual/sequential_sac.py` | Persistent no-replay sequential SAC baseline, finite transition callback, switching/evaluation/checkpointing. | Used by `train_continual_sac.py` and tests; **active distinct algorithm**. |
| `src/continual/evaluation.py` | `EvaluationMatrix`, fixed task creation, active-task validation, task/stage rollout evaluation, persistent matrix updates. | Used by all trainers and evaluation/diagnostic scripts; **active central protocol**. |
| `src/continual/metrics.py` | Average performance, forgetting, and forward transfer with missing-value safety. | Used by evaluation/reporting; **active scientific metrics**. |
| `src/continual/reporting.py` | Formats/prints evaluation, stage, and final reports from recorded results. | Used by evaluation and sequential SAC; **active presentation layer**. Keep separate from metric computation. |
| `src/continual/trajectory_inspection.py` | Read-only loading, statistics, diagnostics, CSV/text export, scans, and comparisons. | Used by inspection and Flow study scripts/tests; **active offline tooling**, large but outside training. |
| `src/continual/__init__.py` | Package marker and architecture statement. | Python package boundary; **active**, intentionally minimal. |
| `src/support/__init__.py` | Support package marker. | Python package boundary; **active**, intentionally minimal. |

## 3. Top-level entry points

| File | Responsibility / important API | Dependents / status / overlap |
|---|---|---|
| `train_sac.py` | Full standalone SAC workflow: summaries, success/evaluation/global callbacks, resume, provenance, CLI, training. | Documented, called by shell scripts, heavily tested; **active**. Excessively large and mixes CLI, callbacks, resume, and exports. |
| `train_diffcrl.py` | Parses DiffCRL config, creates/resumes trainer, optionally synchronizes stages. | Documented and used by launch scripts/tests; **active thin CLI**, good boundary. |
| `train_continual_sac.py` | Parses config and calls sequential SAC `run`. | Documented/tested; **active thin CLI**. |
| `run_replay_comparison.py` | Builds matched no/real/DDPM arms, enforces hashes/splits/allocations/shared Stage-0 state, writes comparison summary. | Used by current replay pilots and tests; **active experiment CLI**. It belongs under an experiments directory once its interface is stabilized. |
| `manage_diffcrl_sync.py` | CLI for stage backup sync/status/validate/restore. | Documented and backed by stage-sync tests; **active operational CLI**. |

## 4. Script inventory

### Collection, evaluation, and visualization

| File | Main responsibility and API | Status / overlap |
|---|---|---|
| `scripts/collection/collect_expert_trajectories.py` | Thin CLI around `collect_expert_dataset`. | **Active**; no problematic duplication. |
| `scripts/evaluation/evaluate_continual.py` | Loads GeneralPolicy/SB3 policy and performs deterministic stage or cross-expert audit. | **Active**; overlaps `evaluate_sac.py` in policy/path loading only. |
| `scripts/evaluation/evaluate_sac.py` | Resolves and evaluates a single SAC checkpoint, validates expert identity. | **Active** and documented. Keep separate CLI, share loader helpers only if interfaces remain stable. |
| `scripts/evaluation/compare_kuka_hammer_checkpoints.py` | Fixed-bank KUKA Hammer checkpoint comparison and table/CSV output. | **Active specialized analysis**. |
| `scripts/evaluation/evaluate_kuka_cube_push.py` | Fixed-distribution KUKA Cube Push evaluation. | **Active specialized evaluation**. |
| `scripts/visualization/render_policy_video.py` | Generic deterministic rollout-to-MP4 renderer and writer. | **Active**, tested. Some checkpoint resolution overlaps evaluators. |
| `scripts/render_kuka_push_policy.py` | Interactive KUKA replay/scripted control, camera/HUD/video, task-specific diagnostics. | **Active**, extensively tested, but 1,100 lines and a strong split candidate. |
| `scripts/render_kuka_10_tasks.py` | Batch launcher for ten KUKA renders. | **Active convenience CLI**; subprocess wrapper is reasonable. |
| `scripts/kuka_push_scripted_controller.py` | Reusable calibrated `CalibratedPushController`. | **Active**, used by renderer and geometry diagnostics. |

### Diagnostics

| File | Main responsibility and important API | Status / overlap |
|---|---|---|
| `scripts/diagnostics/inspect_trajectory.py` | CLI/report/plot wrapper over `trajectory_inspection`. | **Active**; correctly keeps plotting out of core. |
| `scripts/diagnostics/diagnose_kuka_pick_place.py` | Saved-policy failure-stage diagnosis and CSV/Markdown reports. | **Active standalone diagnostic**. |
| `scripts/diagnostics/diagnose_kuka_pick_place_reaching.py` | Scripted reaching probe, correlations, and report. | **Active standalone diagnostic**. |
| `scripts/diagnostics/analyze_kuka_near_contact_v6_checkpoints.py` | Read-only fixed-bank physical checkpoint diagnosis. | **Active but specialized**; repeats hashing/git/environment helpers found in other diagnostics. |
| `scripts/diagnostics/sweep_kuka_cube_near_contact_gap.py` | Physical gap feasibility sweep. | **Active specialized diagnostic**; repeats hash/git and MuJoCo contact helpers. |
| `scripts/diagnostics/validate_kuka_cube_near_contact_common_distribution.py` | Validates the fixed-50 common distribution. | **Active experiment validator**. |
| `scripts/diagnostics/validate_kuka_stage2_robustness.py` | Controlled contact-perturbation Stage-2 evaluator. | **Active specialized diagnostic**, intentionally self-contained but repeats atomic JSON/hash/MuJoCo helpers. |
| `scripts/diagnostics/analyze_kuka_push_goal_retention.py` | Post-success counterfactual rollout analysis. | **Active**, directly tested. |
| `scripts/diagnostics/analyze_kuka_push_object_geometry.py` | Geometry/contact variants with scripted control. | **Active**, directly tested. |

### Flow Matching studies

| File | Main responsibility and important API | Status / overlap |
|---|---|---|
| `scripts/experiments/run_flow_matching_sanity.py` | Matched single-task real/DDPM/Flow BC sanity study. | **Active**, imported by later studies and tested through config tests. Contains reusable private helpers now shared by import. |
| `scripts/flow_matching/analyze_continual_pilot.py` | Append-only offline diagnostics for completed Flow stages. | **Active**, imports sanity helpers. |
| `scripts/flow_matching/run_harder_task_validation.py` | Resumable matched harder-task validation jobs. | **Currently broken/incomplete**: imports missing `scripts.flow_matching.run_flow_matching_ablation`. Do not classify as dead; restore dependency or retire workflow with evidence. |
| `scripts/flow_matching/aggregate_flow_matching_ablation.py` | Hash-validates and aggregates Phase 7B outputs. | **Active aggregator**, but its producer is missing. |
| `scripts/flow_matching/aggregate_harder_task_validation.py` | Hash-validates and aggregates harder-task outputs. | **Active aggregator**. |

Shell launchers are not Python inventory items, but one safety issue matters:
`scripts/flow_matching/launch_flow_matching_ablation.sh` invokes missing
`run_flow_matching_ablation.py`; `launch_harder_task_validation.sh` reaches a
Python file that imports the same missing module. These workflows cannot run in
the current checkout.

## 5. Test inventory

Each test module is active under pytest discovery unless explicitly skipped by
its own environment guards. The table gives its principal contract and important
fixtures/classes; listing every individual `test_*` function would repeat the
source without improving the audit.

### Core continual learning and models

| Test file | Principal coverage / important fixtures |
|---|---|
| `tests/test_bc_policy.py` | GeneralPolicy bounds, BC objectives, optimizer/checkpoint behavior; `BCPolicyTests`. |
| `tests/test_collector.py` | Episode collection, success/shape/provenance and DiffCRL collection paths. |
| `tests/test_diffusion_data.py` | Physical packing, structured encode/decode, action transforms/projection. |
| `tests/test_trajectory_diffusion.py` | DDPM checkpoint round trip and seeded sampling; `TrajectoryDiffusionTests`. |
| `tests/test_trajectory_flow_matching.py` | Flow path/loss/embedding/network/sampling/checkpoint contracts. |
| `tests/test_diffcrl_replay.py` | DDPM replay request, metadata, generation, determinism, decoding, action policy. |
| `tests/test_flow_matching_replay.py` | Flow replay alignment, conditioning, normalization, checkpoint validation, BC compatibility. |
| `tests/test_real_replay.py` | Training-only IDs, replacement/provenance, deterministic sampling, matched allocation, no-replay bypass. |
| `tests/test_replay_allocation.py` | Exact uniform/fixed-total allocation and legacy policy. |
| `tests/test_stage_dataset.py` | Whole-episode split, training-only history, concatenation, hashes/provenance; `StageDatasetTests`. |
| `tests/test_generator_interchangeability.py` | Shared downstream contracts between DDPM and Flow. |
| `tests/test_diffcrl_progress.py` | Progress flag forwarding in BC/generator paths. |
| `tests/test_diffcrl_stage_report.py` | Stage-report fields, hashes, normalization, generator metadata. |
| `tests/test_diffcrl_resume.py` | Boundary markers, exact restoration, config/hash rejection, resume continuation. |
| `tests/test_diffcrl_flow_resume.py` | Flow-specific checkpoint and resume restoration. |
| `tests/test_diffcrl_flow_matching.py` | Flow integration into DiffCRL stage fitting/checkpoint selection. |
| `tests/test_replay_comparison.py` | Matched isolated no/real/DDPM comparison arms and inherited budgets. |
| `tests/test_evaluation_contract.py` | Evaluation matrix, fixed tasks, success/return recording and protocol invariants. |
| `tests/test_reporting.py` | Console formatting from immutable recorded results; `ReportingTests`. |
| `tests/test_trajectory_inspection.py` | Offline trajectory loaders/statistics/export/comparison; `TrajectoryInspectionTests`. |

### Configuration, provenance, persistence, and entry points

| Test file | Principal coverage / important fixtures |
|---|---|
| `tests/test_config.py` | Strict schema, merge, experiment field restrictions, replay/Flow validation. |
| `tests/test_training_config.py` | All training entry points consume typed YAML/runtime overrides; `TrainingConfigTests`. |
| `tests/test_run_manifest.py` | Required provenance and completion markers. |
| `tests/test_expert_manifest.py` | Expert sidecar identity, hashes, qualification, legacy behavior. |
| `tests/test_expert_space.py` | Space/backend compatibility and authenticated repair. |
| `tests/test_stage_sync.py` | Verified retry, marker ordering, corruption detection, restore, RNG safety. |
| `tests/test_sac_export.py` | SAC JSON/CSV summaries; `SacExportTests`. |
| `tests/test_sac_resume.py` | Faithful standalone SAC resume; `SACResumeTests`. |
| `tests/test_sac_success_checkpoint.py` | Success-primary fixed-seed checkpoint selection; `FixedSeedEvalCallbackTests`. |
| `tests/test_task_bank_safety.py` | Bank reproducibility and overlap rejection. |
| `tests/test_expert_collection.py` | Standalone collection config, filtering, serialization, trajectory contracts. |
| `tests/test_reward_version.py` | Reward selection reaches environment factory; `RewardVersionTests`. |

### Backend, SAC, rendering, and diagnostic coverage

| Test file | Principal coverage / important fixtures |
|---|---|
| `tests/test_kuka_v2_integration.py` | Legacy adapter API, seeding, horizon/goal/success, factory. |
| `tests/test_kuka_v3_sac.py` | KUKA-v3 SAC configs, task/reward scoping, factory contracts. |
| `tests/test_kuka_diffcrl_phase1.py` | KUKA DiffCRL configs, task identity/banks, expert manifests, packed trajectories. |
| `tests/test_kuka_fixed_evaluation.py` | Deterministic KUKA task banks, parity/disjointness, fixed evaluation. |
| `tests/test_hammer_diagnostics.py` | Hammer tracker and fixed-bank comparison behavior. |
| `tests/test_render_policy_video.py` | Generic offline video utility; fake policy/env/writer fixtures. |
| `tests/test_render_kuka_push_policy.py` | Large renderer CLI, checkpoint/task/reward selection, cleanup, video, diagnostics. |
| `tests/test_kuka_push_scripted_controller.py` | Calibrated controller and renderer integration. |
| `tests/test_kuka_push_goal_retention_diagnostic.py` | Retraction/controller/aggregate semantics. |
| `tests/test_kuka_push_object_geometry_diagnostic.py` | Mass/inertia/controller/configuration contracts. |
| `tests/test_kuka_push_retention_ranking.py` | Offline post-success reward ranking. |
| `tests/test_kuka_push_cube_integration.py` | Cube training adapter and renderer discovery. |

### Experiment/configuration preflight modules

| Test file | Principal coverage |
|---|---|
| `tests/test_flow_matching_sanity.py` | Full/debug sanity config isolation and unknown fields. |
| `tests/test_flow_matching_5task_pilot.py` | Pilot parity, settings, allocations, expert assets. |
| `tests/test_flow_matching_hammer_first.py` | Hammer-first task/order/output/allocation contracts. |
| `tests/test_flow_matching_hammer_first_100real_100pseudo.py` | Budget-only variant and combined counts. |
| `tests/test_harder_task_validation.py` | Harder-task full/debug configs and required assets. |
| `tests/test_kuka_push_cube_common_distribution_config.py` | Common-distribution config/output isolation. |
| `tests/test_kuka_push_cube_contact_start_config.py` | Contact-start config contract. |
| `tests/test_kuka_push_cube_sac_config.py` | Initial Cube Push SAC baseline. |
| `tests/test_kuka_push_cube_sawyer_reward_config.py` | Sawyer-adapted reward scoping and resume. |
| `tests/test_kuka_push_cube_stage1_config.py` | Stage-1 config and environment rejection. |
| `tests/test_kuka_push_cube_stage1_v1_1_config.py` | Stage-1 v1.1 isolation/scoping. |
| `tests/test_kuka_push_cube_v3_3_sac_config.py` | v3.3 training contract/scoping. |
| `tests/test_kuka_push_cube_v4_sac_config.py` | v4 training contract/scoping. |
| `tests/test_kuka_push_cube_v6_sac_config.py` | v6 contact-start contract. |
| `tests/__init__.py` | Test package marker used by cross-test fixture imports. Keep. |

## 6. Configuration inventory

There are 123 YAML files: 67 under `configs/sac`, 22 under
`configs/diffcrl`, 19 under `configs/local`, 11 under `configs/archive`, two
continual-SAC configs, and one each for the Flow sanity and harder-task studies.
No two YAML files are byte-identical by SHA-256.

The large number is mostly scientific provenance rather than code duplication:

- `configs/archive/` and `configs/sac/archive/` explicitly retain historical
  controls, probes, and ablations.
- `configs/local/` contains KUKA-v2 integration/sweep/trajectory work.
- Active SAC files encode distinct task, reward, curriculum, geometry, resume,
  or optimizer contracts and are referenced by tests.
- DiffCRL includes DDPM, Flow, KUKA, smoke, no-replay, fixed-budget, and replay
  comparison protocols.

Do not delete or fold these files merely because fields repeat. Safe cleanup is
organizational: document status and use directories/naming consistently. YAML
inheritance would change the current self-contained provenance model and add a
resolver; it is not recommended for this cleanup.

## 7. Dependency relationships

The highest-impact dependency hubs are:

- `src.config`: every training entry point, orchestration, manifests, resume,
  studies, and configuration tests.
- `src.envs`: all environment-facing code and most diagnostics.
- `src.continual.diffcrl`: DiffCRL CLI/comparison plus stage, replay, resume,
  and reporting tests.
- `src.continual.evaluation`: SAC, sequential SAC, DiffCRL, evaluators,
  renderers, diagnostics, and reporting.
- `src.continual.diffusion_data`: both generators, replay providers, collection,
  inspection, and grouped datasets.
- `src.support.io` and `reproducibility`: checkpoint/resume/provenance backbone.

Meaningful lower-level boundaries are `schema -> collector -> evaluation`,
`diffusion_data -> generator model -> replay provider`, and
`task_bank/task_sequence -> collection/evaluation/replay`. Refactoring should
preserve these dependency directions and avoid making models depend on the
trainer.

## 8. Duplicate and overlapping functionality

### Confirmed duplication worth consolidating

1. **DDPM and Flow replay providers** (`diffcrl_replay.py`,
   `flow_matching_replay.py`) repeat request checks, checkpoint state/hash and
   metadata validation, per-task seeded generation loops, goal reconstruction,
   decode/action handling, physical-group validation, diagnostics, and source
   provenance assembly. The algorithm-specific operations are checkpoint load
   and sampling. A private shared replay helper can own the common mechanics
   while retaining both public request/result/functions. **Risk: Medium.**

2. **Generator fitting/checkpoint mechanics** (`trajectory_diffusion.py`,
   `trajectory_flow_matching.py`) repeat finite 3D tensor validation, Adam setup,
   deterministic shuffling, gradient clipping, per-epoch validation, best-state
   copying/restoration, and state-dict/normalizer/metadata checkpoint envelopes.
   Losses and sampling must remain separate. Extracting only tested private
   training/checkpoint utilities would reduce drift. **Risk: Medium.**

3. **Checkpoint/path loading in evaluation scripts** (`evaluate_sac.py`,
   `evaluate_continual.py`, `render_policy_video.py`) partially repeats model
   path resolution and SB3/GeneralPolicy adaptation. A small script-level helper
   is safe if CLI behavior and expert-manifest checks remain explicit.
   **Risk: Low-Medium.**

4. **Standalone diagnostic plumbing** repeats SHA-256, git revision, exclusive
   output directory, CSV/JSON serialization, and some KUKA MuJoCo contact/tilt
   helpers. Consolidate only stable, exact operations; task-specific physical
   definitions must remain local. **Risk: Low for hashing/atomic output; Medium
   for physical helpers.**

### Similarity that should not be merged

- DDPM and Flow model/network/loss/sampling implementations are different
  algorithms and should remain separate.
- Real replay and generated replay have different data/provenance semantics.
- Run manifests, expert manifests, and stage resume markers establish different
  identities and lifecycles.
- Evaluation metrics and report formatting are cleanly separated.
- Environment adapters should remain backend-specific behind `src.envs`.
- SAC, sequential SAC, and DiffCRL are different training algorithms.

## 9. Potentially obsolete, incomplete, or generated code

### Demonstrably removable

- **REMOVE generated `__pycache__/` directories and `*.pyc` files from working
  copies/artifact bundles.** Evidence: they are interpreter-generated caches,
  ignored by source discovery, not imported as source, and are recreated
  automatically. This does not mean deleting source files.

No production Python source file can currently be demonstrated dead with enough
confidence to recommend deletion. Every `src` module is reached through a
production relative/absolute import, a CLI, or focused tests. Static name search
also found no top-level function/class whose name occurs only at its definition.

### Incomplete or questionable, requiring a decision rather than deletion

- `scripts/flow_matching/run_flow_matching_ablation.py` is missing, while
  `launch_flow_matching_ablation.sh` invokes it and
  `run_harder_task_validation.py` imports helpers from it. This is concrete
  breakage. Determine whether the missing file exists in another branch/run
  archive. Restore it if results/configs depend on it; otherwise retire the
  launch/import chain only after documenting that evidence.
- `IMPLEMENTATION_ARCHITECTURE.md` is referenced by the audit request but absent.
  Either restore it or designate this report plus `src/continual/README.md` as
  the maintained architecture documentation.
- KUKA-v2 is a legacy backend, but it still has configs, factory routing, and
  integration tests. It is not dead and should not be removed without an
  explicit support-policy decision.
- Offline Flow quality and inspection modules are not used by primary training,
  but current study scripts import them. They are analysis code, not dead code.

## 10. Unnecessary complexity and fragmentation

- `DiffCRLTrainer` owns validation, environment/expert setup, model creation,
  resume, collection, stage data, two generator branches, BC, checkpointing,
  evaluation, reporting, and lifecycle control. Its many typed stage records are
  useful; the monolithic implementation is the problem.
- `train_sac.py` combines reusable callbacks and resume/export logic with CLI and
  orchestration. Tests import its internals, turning a script into a de facto API.
- `render_kuka_push_policy.py` combines argument validation, checkpoint/task
  selection, viewer state, video writing, task debug text, scripted control, and
  the render loop.
- `src/config.py` is large because strict validation is comprehensive. Splitting
  it into many experiment-specific modules would increase indirection; clearer
  internal validation sections/table-driven rules are preferable first.
- `diagnostics.py` is unnecessarily fragmented: its sole function is shared only
  by generated replay providers and uses replay data constants.
- `src/support/dataset.py` is not generic support; it encodes DiffCRL stage split
  semantics. Its location obscures ownership.
- Several Flow scripts import underscore-prefixed helpers from another executable
  script. Shared study utilities should have a non-CLI module with public names.

## 11. Proposed simplified structure

This is a target shape, not a recommendation for a one-shot move:

```text
src/
  config.py                         # stable public config API
  envs.py                           # stable environment facade
  env_adapters/                     # unchanged backend boundaries
  sac/
    training.py                     # extracted standalone SAC orchestration
    callbacks.py                    # success/eval/global callbacks
    resume.py                       # standalone SAC resume helpers
    export.py                       # JSON/CSV summaries
  continual/
    schema.py
    collector.py
    task_sequence.py
    task_bank.py
    evaluation.py
    metrics.py
    reporting.py
    bc_policy.py
    diffusion_data.py
    stage_dataset.py                # relocated support/dataset.py
    replay_allocation.py
    real_replay.py
    generators/
      diffusion.py                  # current trajectory_diffusion.py
      flow_matching.py              # current trajectory_flow_matching.py
      replay_common.py              # private common generation/provenance loop
      diffusion_replay.py
      flow_matching_replay.py
    diffcrl.py                      # lifecycle coordinator
    diffcrl_stage.py                # stage preparation/training/checkpoint helpers
    resume.py
    sequential_sac.py
  support/                          # generic IO/provenance only
    io.py
    reproducibility.py
    expert_manifest.py
    run_manifest.py
    stage_sync.py

scripts/
  collection/
  evaluation/
  visualization/
  diagnostics/
  experiments/
    replay_comparison.py
    flow_matching/
      common.py                     # shared study helpers, not a CLI
      ...
```

Moving model modules into `generators/` is optional and should happen only after
common mechanics are extracted. Avoid combining relocation with behavior
changes.

## 12. Recommended actions and risk assessment

| Type | Current files | Proposed change | Public interface | Risk |
|---|---|---|---|---|
| **KEEP** | Environment adapters + `envs.py` | Preserve backend facade and separate adapters. | None. | Low |
| **KEEP** | `schema.py`, `collector.py`, `policy_adapter.py` | Preserve clean trajectory/policy boundaries. | None. | Low |
| **KEEP** | `metrics.py`, `reporting.py`, `evaluation.py` | Preserve computation/presentation/protocol separation. | None. | Low |
| **KEEP** | `real_replay.py`, `replay_allocation.py` | Preserve distinct semantics and scientific policy. | None. | Low |
| **SIMPLIFY** | Flow study scripts | Restore missing ablation module or remove broken references after provenance review; move shared underscore helpers to `scripts/experiments/flow_matching/common.py`. | CLI unchanged; internal imports change. | Low-Medium |
| **MERGE** | `diagnostics.py` + replay-common code | Move generated-action diagnostic into a shared replay helper or `diffusion_data.py`; preserve function behavior. | Compatibility re-export advisable. | Low |
| **RELOCATE** | `src/support/dataset.py` | Move to `src/continual/stage_dataset.py`, initially leave a compatibility re-export. | Old import retained temporarily. | Low-Medium |
| **MERGE** | DDPM/Flow replay providers | Extract private common request/provenance/generation loop; keep public request/result and entry functions. | No intended public change. | Medium |
| **SIMPLIFY** | DDPM/Flow trainers | Share finite validation, deterministic epoch loop, best-state, and checkpoint-envelope helpers only. | Checkpoint schemas and model APIs unchanged. | Medium |
| **SIMPLIFY** | `src/continual/diffcrl.py` | Extract stage preparation/training/checkpoint helper functions; keep `DiffCRLTrainer` facade and order of operations. | No intended public change. | Medium-High |
| **RELOCATE** | `run_replay_comparison.py` | Move under `scripts/experiments/` after current pilots stabilize; leave top-level wrapper. | Existing command remains valid. | Low |
| **SIMPLIFY** | `train_sac.py` | Extract callbacks, resume, and export into `src/sac/`; keep top-level imports/re-exports during migration. | Existing CLI/imports retained initially. | High |
| **SIMPLIFY** | KUKA renderer | Split CLI/task resolution, viewer/video, diagnostics, and loop into script helper modules. | Preserve CLI exactly. | Medium |
| **RELOCATE** | `generation_quality.py`, `trajectory_inspection.py` | Optionally place under `src/analysis/` only if analysis is declared a supported package. | Requires compatibility imports. | Medium; low benefit now |
| **REMOVE** | `__pycache__`, `*.pyc` | Remove generated caches only. | None. | Low |
| **KEEP** | All experiment YAML and `runs/` artifacts | Retain for provenance; improve inventory documentation instead of deduplication. | None. | Low |

## 13. Suggested implementation order

1. **Repair/document the Flow workflow gap.** Locate the missing ablation runner,
   decide whether it is supported, and add an import/CLI smoke test. This is a
   correctness repair, not refactoring.
2. **Add characterization tests before structural work.** Snapshot replay source
   metadata, stage reports, resolved configs, checkpoint keys, seeded generated
   tensors, optimizer restoration, and resume markers.
3. **Consolidate trivial replay diagnostics.** This is the safest source cleanup.
4. **Extract a private common generated-replay helper.** Keep DDPM and Flow
   public APIs and algorithm calls unchanged; compare seeded outputs byte-for-byte.
5. **Extract generator training/checkpoint mechanics.** Do not share losses,
   architectures, or samplers.
6. **Relocate stage dataset ownership** with a compatibility module and update
   imports in a separate commit.
7. **Reduce `DiffCRLTrainer` size** by moving pure stage preparation/report and
   checkpoint helpers, one concern per change.
8. **Split `train_sac.py`** only after its callback/resume/export tests explicitly
   characterize all imported internals and ZIP/replay-buffer behavior.
9. **Split the KUKA renderer** independently from scientific training changes.
10. **Revisit legacy support/config organization** only after an explicit policy
    decision; do not infer obsolescence from age or location.

## 14. Tests required after refactoring

At every step:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q src scripts *.py
git diff --check
```

Additional gates by area:

- **Replay consolidation:** `test_diffcrl_replay.py`,
  `test_flow_matching_replay.py`, `test_real_replay.py`,
  `test_replay_allocation.py`, `test_generator_interchangeability.py`, and
  seeded before/after tensor/provenance snapshots.
- **Generator fitting/checkpoints:** `test_trajectory_diffusion.py`,
  `test_trajectory_flow_matching.py`, `test_diffcrl_flow_matching.py`, both
  resume suites, and exact checkpoint-key/metadata comparisons.
- **Stage data/orchestration:** `test_stage_dataset.py`, collector/evaluation,
  stage-report, progress, KUKA DiffCRL, task-bank, and replay-comparison tests.
- **SAC extraction:** training-config, export, resume, success checkpoint,
  reward, KUKA-v3 SAC, run-manifest, and a short smoke run that compares resolved
  config and checkpoint metadata.
- **Environment changes:** MetaWorld/KUKA adapter tests plus fixed-bank hashes and
  reward-version forwarding.
- **Script moves:** CLI `--help`, import smoke tests, existing renderer/diagnostic
  tests, and documentation command checks.
- **Checkpoint compatibility:** load representative GeneralPolicy, DDPM, Flow,
  DiffCRL resume, and SB3 checkpoints without rewriting them.
- **Scientific equivalence:** for fixed seeds, compare split IDs, replay
  allocations, source hashes, generated tensor hashes, normalization statistics,
  best epochs, evaluation task hashes, and evaluation matrices.

## Prioritized changes

1. Resolve the missing Flow ablation runner/import chain.
2. Consolidate the tiny generated-action diagnostic into shared replay code.
3. Extract common DDPM/Flow replay plumbing while preserving public APIs.
4. Extract only the common generator training/checkpoint mechanics.
5. Move stage-dataset code into the continual package with a compatibility shim.
6. Decompose `DiffCRLTrainer` into pure stage helpers without changing ordering.
7. Extract standalone SAC callbacks/resume/export behind the current CLI/API.
8. Split the KUKA renderer by UI, task resolution, diagnostics, and loop.
9. Improve config/workflow inventory documentation; retain all scientific YAML
   and artifacts unless a separate evidence-backed archival decision is made.
