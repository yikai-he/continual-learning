# KUKA-v3 SAC Expert Audit and Task-Sequence Recommendation

Audit date: 2026-11-01. Scope is exclusively KUKA-v3. No MetaWorld/Sawyer
checkpoint was evaluated or used. Existing checkpoints, configurations,
manifests, evaluations, and the earlier MetaWorld audit were not modified.

## Executive recommendation

- **Five-task benchmark: GO**, using the five STRONG experts recommended below.
- **Ten-task benchmark: CONDITIONAL GO.** All ten experts are usable and nine
  are STRONG, but Window Close is only GOOD (90%), failed demonstrations are
  retained, and Peg-Unplug's configured reward variant is not forwarded by the
  current GeneralPolicy stage-evaluation path.
- **Expert availability: complete.** Every requested task has one hash-verified
  KUKA-v3 `policy.zip` and an expert manifest.

## Uniform audit evaluation

Each expert was evaluated with deterministic SAC actions on the ordered
KUKA-v3 fixed task bank: reward v2, task-set seed 10000, reset/evaluation seed
10000, 50 episodes, 50 distinct configurations, horizon 200, and established
any-step episodic success. Task-specific authenticated v2 variants were
preserved: Hammer used `nail_progress` weight 4; Peg-Unplug used
`peg_unplug_progress_v1` weight 2 and gamma 0.99. All episodes had length 200.

| Task | Checkpoint step | Success | Mean return | Return SD | Numerical/environment failures | Status |
|---|---:|---:|---:|---:|---:|---|
| kuka-reach-v3 | 520,000 | 48/50 (96%) | 1825.159 | 86.926 | 0 | STRONG |
| kuka-hammer-v3 | 100,000 | 50/50 (100%) | 1510.453 | 263.805 | 0 | STRONG |
| kuka-handle-press-side-v3 | 600,000 | 48/50 (96%) | 898.820 | 230.725 | 0 | STRONG |
| kuka-button-press-v3 | 200,000 | 50/50 (100%) | 1462.717 | 76.863 | 0 | STRONG |
| kuka-faucet-open-v3 | 120,000 | 50/50 (100%) | 1853.337 | 87.243 | 0 | STRONG |
| kuka-drawer-close-v3 | 80,000 | 50/50 (100%) | 1725.501 | 120.566 | 0 | STRONG |
| kuka-window-close-v3 | 100,000 | 45/50 (90%) | 1429.256 | 424.954 | 0 | GOOD |
| kuka-door-open-v3 | 160,000 | 49/50 (98%) | 1602.702 | 172.535 | 0 | STRONG |
| kuka-button-press-topdown-v3 | 60,000 | 50/50 (100%) | 1306.420 | 187.947 | 0 | STRONG |
| kuka-peg-unplug-side-v3 | 240,000 | 50/50 (100%) | 1272.198 | 229.491 | 0 | STRONG |

Thresholds are STRONG ≥95%, GOOD 90–<95%, PARTIAL 70–<90%, and WEAK <70%.
Performance status is separate from provenance confidence.

The individual JSON records contain the ordered 50 task hashes, preventing
silent configuration repetition. The aggregate registry contains full hashes,
sizes, selection roles, and documented run metadata:

- `runs/audits/sac_experts_kuka_v3_20261101/expert_registry.json`
- `runs/audits/sac_experts_kuka_v3_20261101/<task>.json`

## Inventory and provenance

All ten `policy.zip` files loaded under Stable-Baselines3 2.9.0. Each sidecar
manifest authenticated task name, `kuka-v3`, SAC, reward v2, seed 0, `(39,)`
observations, `(4,)` actions, horizon 200, and the current checkpoint SHA-256.
The audit recomputed every hash. Hammer's manifest additionally authenticates
its reward variant. Peg's manifest records its progress variant, although the
current validator does not enforce Peg-specific fields.

The checkpoint selections are identifiable from checkpoint `num_timesteps`
and the saved summaries:

- Best-success before the final training state: Reach, Hammer, Drawer Close,
  Button Press Topdown, and Peg Unplug.
- Best-success equal to the final state: Handle Press Side, Button Press,
  Faucet Open, Window Close, and Door Open.

The registry gives exact values. `config.json` and historical summaries
document seed, planned/final steps, reward settings, evaluation histories, and
environment metadata. They are stronger provenance than filenames alone, but
the manifests do not hash or reference the training configuration.

### Environment provenance limitation

Eight older configurations document environment commit
`c4af9a6f5f45cfd74c118b8c32b3be4253dada2f` on branch `kuka-v3`.
Button Press Topdown and Peg Unplug document main commit
`2ed78a08c7bf0ce0fd7aac4e44fd0ee3f9dd76ea`. The checkout used by this audit
has the latter HEAD plus pre-existing uncommitted KUKA task/adapter changes.
Therefore the new scores verify compatibility with the **current working KUKA
pipeline**, but cannot reproduce the exact historical training environment
byte-for-byte. The checkpoint and manifest hashes themselves are verified.

## Architecture and semantic compatibility

Every checkpoint stores the same SAC actor:

`39 → 256 ReLU → 256 ReLU → 256 ReLU → 256 ReLU → {mu(4), log_std(4)} → squashed Gaussian`

`use_sde=false`; deterministic evaluation uses the squashed mean action.
Checkpoint actions and current KUKA environment actions are float32 Box
`[-1,1]^4`.

The 39D observation is not accepted by shape alone. The current KUKA adapter
and KUKA environment source establish the semantics:

- indices 0–17: current KUKA physical state;
- indices 18–35: previous KUKA physical state;
- indices 36–38: visible Cartesian goal;
- each 18D physical state contains TCP xyz, normalized physical Robotiq 2F-85
  aperture, and padded object position/quaternion slots.

The 4D action means clipped Cartesian XYZ mocap displacement plus Robotiq
effort, where -1 opens and +1 closes. This is specifically KUKA semantics, not
Sawyer semantics.

Eight older checkpoints required the narrowly scoped authenticated repair in
`src/continual/expert_space.py`: their stored observation-space goal bounds are
zero while the current environment exposes the goal bounds. Only metadata is
reconciled after manifest/hash validation; observations are not transformed.
Topdown and Peg did not require this repair.

## DiffCRL path verification

Confirmed compatible components:

- `src/env_adapters/kuka_v3.py` constructs `Meta-World-KUKA/MT1`, checks 39D
  float32 finite observations, visible goals, 4D actions, and the task list.
- `src/continual/diffcrl.py` requires non-legacy manifests for KUKA-v3,
  validates expert spaces, selects KUKA training-bank configurations, and
  collects deterministic full-horizon trajectories.
- `src/continual/collector.py` checks finite 39D/4D values and uses the common
  any-step success definition.
- DDPM packing uses 39 observation + 4 action features, stores task labels and
  goals, and reconstructs KUKA task-bank goals for replay.
- GeneralPolicy accepts raw 39D observations and emits bounded physical 4D
  actions. Both `[256,256]` and `[256,256,256,256]` are supported with stored
  checkpoint architecture metadata.
- Fixed-total replay allocation and one-/two-trajectory split behavior passed
  the focused test suite.

Confirmed caveats:

1. `scripts/evaluation/evaluate_sac.py` has a stale CLI restriction and does
   not expose `kuka-v3`, despite the underlying fixed-bank evaluator supporting
   it. The audit therefore uses the included audit-only driver, built from the
   same adapter, collector, task-bank, manifest, and policy components.
2. GeneralPolicy stage evaluation forwards Hammer settings but `TaskSwitcher`
   omits Peg-Unplug settings. Peg success remains comparable, but reported
   continual-evaluation returns use the original Peg reward instead of the
   expert's progress variant. Expert collection itself correctly uses the full
   `EnvironmentSettings` and this audit evaluated Peg with its recorded variant.
3. The standalone `trajectory_collection` configuration loader supports
   MetaWorld-v3 and KUKA-v2 only. This does not block DiffCRL, whose internal
   KUKA-v3 collection path is implemented and tested, but that standalone CLI
   is not KUKA-v3 compatible.
4. Expert collection retains unsuccessful episodes. Window Close's measured
   90% rate implies an illustrative `1 - 0.9^15 = 79.4%` chance that a 15-shot
   set contains at least one failure if episodes were independent. The fixed
   training bank is different from this evaluation bank, so this is a risk
   estimate, not a prediction.

## Recommended five-task sequence

`kuka-hammer-v3 → kuka-button-press-topdown-v3 → kuka-door-open-v3 → kuka-faucet-open-v3 → kuka-drawer-close-v3`

This five-task sequence is **GO**. All five are STRONG. It begins with tool use
so Hammer retention is measured across four later stages, separates top-down
pressing from later articulated manipulation, and covers tool interaction,
precision pressing, hinged rotation, faucet rotation, and sliding closure.
Drawer Close is comparatively simple and is placed last so a complex task is
not consumed as the unmeasured final retention anchor. Transfer/interference
relationships are structural hypotheses, not established empirical results.

## Recommended ten-task sequence

`kuka-hammer-v3 → kuka-window-close-v3 → kuka-button-press-v3 → kuka-peg-unplug-side-v3 → kuka-faucet-open-v3 → kuka-handle-press-side-v3 → kuka-drawer-close-v3 → kuka-button-press-topdown-v3 → kuka-door-open-v3 → kuka-reach-v3`

This is a **CONDITIONAL GO**:

- Hammer, Peg Unplug, and the lower-performing Window Close occur early enough
  to measure long-term retention.
- related press tasks are separated (stages 3, 6, and 8);
- articulated tasks are interleaved rather than grouped;
- simple Reach is last, avoiding placement of a complex task where its
  post-learning retention cannot be observed;
- all experts are compatible and none is WEAK, but Window Close is only GOOD,
  and the Peg evaluation-variant caveat should be resolved or explicitly
  accepted before interpreting return-based continual metrics.

## Compatibility with fixed 15+15 replay

The sequences are mechanically compatible with 15 current-task expert
trajectories, DDPM, the GeneralPolicy depth comparison, and fixed-total uniform
replay. This paragraph describes a historical research comparison; active code
now supports only the four-hidden-layer architecture.
The configuration must explicitly select `replay_allocation: fixed-total` and
`replay_budget: 15`; the existing formal KUKA five-task configuration instead
uses legacy generator-default allocation and 200 current trajectories.

For a ten-task sequence, the 15 replay trajectories are allocated over prior
tasks in sequence order as follows:

| Current stage | Prior-task replay counts |
|---:|---|
| 2 | 15 |
| 3 | 8, 7 |
| 4 | 5, 5, 5 |
| 5 | 4, 4, 4, 3 |
| 6 | 3, 3, 3, 3, 3 |
| 7 | 3, 3, 3, 2, 2, 2 |
| 8 | 3, 2, 2, 2, 2, 2, 2 |
| 9 | 2, 2, 2, 2, 2, 2, 2, 1 |
| 10 | 2, 2, 2, 2, 2, 1, 1, 1, 1 |

At stage 10, four historical tasks receive one replay trajectory and five
receive two. A one-trajectory group is training-only; two trajectories split
one train/one validation. The 15 current trajectories keep global validation
nonempty. Generator self-replay remains valid, but late-stage old-task coverage
is extremely sparse, remainder allocation favors earlier tasks, and retention
variance will be high. A ten-task 15+15 run should be described as a
fixed-budget stress test.

## Reproduction commands

The project SAC CLI does not currently expose KUKA-v3, so use the audit-only
driver saved with the results. It writes outputs exclusively (`open("x")`) and
therefore refuses to overwrite prior records.

```bash
cd continual-learning
AUDIT_DIR=runs/audits/sac_experts_kuka_v3_repro_seed10000
mkdir -p "$AUDIT_DIR"
for TASK in reach-v3 hammer-v3 handle-press-side-v3 button-press-v3 faucet-open-v3 drawer-close-v3 window-close-v3 door-open-v3 button-press-topdown-v3 peg-unplug-side-v3; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=../metaworld-kuka-v3 \
  conda run -n metaworld python \
    runs/audits/sac_experts_kuka_v3_20261101/audit_evaluate_kuka_sac.py \
    --repo . --task "kuka-$TASK" \
    --model "runs/experts/kuka-v3/$TASK/policy.zip" \
    --episodes 50 --seed 10000 --task-set-seed 10000 \
    --output "$AUDIT_DIR/$TASK.json"
done
```

## Verification performed

- Ten checkpoints loaded; actual actor modules and spaces inspected.
- Ten manifests and checkpoint hashes validated.
- 500 total fixed-configuration episodes completed with no numerical or
  environment failures.
- Each result contains 50 distinct ordered task hashes.
- Registry/result hashes cross-checked.
- Focused compatibility suite: **85 passed, 3 subtests passed**.

No SAC retraining, DiffCRL training, or trajectory collection experiment was
started.
