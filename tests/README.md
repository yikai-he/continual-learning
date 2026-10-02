# Test suite

Run the full suite from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Disabling plugin autoload prevents unrelated system or ROS pytest plugins from
interfering with collection.

| Test file | Purpose | Type / notes |
| --- | --- | --- |
| `test_bc_policy.py` | Validates behavior-cloning preprocessing, losses, bounds, and checkpoint compatibility. | Unit test |
| `test_collector.py` | Checks repeated trajectory collection, seed ordering, callbacks, and environment cleanup. | Unit test |
| `test_config.py` | Validates configuration defaults, schemas, YAML loading, overrides, and persistence. | Config/schema test |
| `test_diffcrl_progress.py` | Checks DiffCRL progress controls and console-output behavior. | Unit test |
| `test_diffcrl_replay.py` | Validates deterministic previous-task replay generation and checkpoint safeguards. | Replay/diffusion test |
| `test_diffcrl_stage_report.py` | Checks stage-report construction, persistence, integrity, and state transitions. | Evaluation/reporting test |
| `test_diffusion_data.py` | Validates trajectory encoding, decoding, and action transformations. | Replay/diffusion unit test |
| `test_evaluation_contract.py` | Checks evaluation-matrix persistence and protocol consistency across stages. | Evaluation test |
| `test_expert_collection.py` | Validates KUKA expert-collection filtering, configuration, datasets, and ledgers. | KUKA test; real-expert case is optional |
| `test_kuka_v2_integration.py` | Checks KUKA task exposure, adapter behavior, seeding, and environment construction. | KUKA integration; legacy fork cases may skip |
| `test_render_policy_video.py` | Validates offline policy-video rollout, frame handling, and CLI checks. | Utility unit test |
| `test_reporting.py` | Checks continual-learning metric and console-report formatting without mutation. | Evaluation/reporting test |
| `test_reward_version.py` | Verifies reward-version selection and propagation through training and evaluation. | Environment contract test |
| `test_sac_success_checkpoint.py` | Checks success-first SAC checkpoint selection and return-based tie-breaking. | Training unit test |
| `test_stage_dataset.py` | Validates deterministic DiffCRL stage-dataset assembly and provenance. | Replay/data test |
| `test_training_config.py` | Verifies training entry-point configuration wiring, overrides, and persistence. | Training/config wiring; CUDA case may skip |
| `test_trajectory_diffusion.py` | Checks diffusion checkpoint metadata and deterministic seeded sampling. | Replay/diffusion test |
| `test_trajectory_inspection.py` | Validates read-only trajectory diagnostics, loading, comparison, and exports. | Diagnostics unit test |

Optional KUKA, CUDA, or other resource-dependent integration tests may be
skipped when the required local resources are unavailable.
