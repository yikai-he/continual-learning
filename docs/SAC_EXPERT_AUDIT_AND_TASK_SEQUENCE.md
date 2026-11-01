# SAC Expert Audit and Task-Sequence Recommendation

Audit date: 2026-11-01. This audit was read-only with respect to source code,
configurations, checkpoints, and prior experiment artifacts. New evaluation
records were written only under
`runs/audits/sac_experts_metaworld_v3_20261101/`.

## Executive conclusion

**NO-GO for the requested ten-task MetaWorld-v3 benchmark.** Six of the ten
requested tasks have authentic, loadable MetaWorld-v3 checkpoint candidates;
four have only KUKA-v3 checkpoints and must not be mixed into this benchmark.
Of the six evaluated MetaWorld policies, four are STRONG and two are PARTIAL.

**GO for a revised five-task benchmark** if the explicitly distinct
`faucet-close-v3` supplemental expert is accepted:

`button-press-v3 → faucet-close-v3 → hammer-v3 → handle-press-side-v3 → window-close-v3`

All five scored 50/50 under the common protocol. This does not authenticate
their undocumented historical training provenance, and it does not establish
transfer relationships empirically.

## Evidence and interpretation rules

- **Verified** means directly read from the checkpoint, filesystem, current
  environment, or new evaluation output.
- **Documented** means stated in a legacy `config.json`, but not authenticated
  by an expert manifest.
- **Inferred** means a defensible interpretation, not historical proof.
- Performance category and provenance are deliberately separate. A STRONG
  score does not prove how the checkpoint was trained.

The machine-readable inventory is
`runs/audits/sac_experts_metaworld_v3_20261101/expert_registry.json`. Individual
evaluation JSON files retain the ordered hashes of all fixed task
configurations, preventing silent configuration repetition.

## Inventory and qualification

All six requested MetaWorld checkpoints and the supplemental faucet-close
checkpoint loaded with Stable-Baselines3 2.9.0. Their checkpoint actors report
four 256-unit hidden layers, ReLU activations, diagonal squashed-Gaussian SAC
output, and `use_sde=false`. The spaces are float32 `(39,)` observations and
`(4,)` actions bounded by `[-1, 1]^4`.

| Requested task | Checkpoint | Size (bytes) | Checkpoint steps | Success | Return mean ± SD | Backend | Status |
|---|---|---:|---:|---:|---:|---|---|
| reach-v3 | `runs/experts/metaworld-v3/reach-v3/policy.zip` | 9,266,656 | 420,000 | 44/50 (88%) | 1795.73 ± 166.78 | verified MetaWorld evaluation | PARTIAL |
| hammer-v3 | `runs/experts/metaworld-v3/hammer-v3/policy.zip` | 9,266,669 | 980,000 | 50/50 (100%) | 1624.83 ± 182.01 | verified MetaWorld evaluation | STRONG |
| handle-press-side-v3 | `runs/experts/metaworld-v3/handle-press-side-v3/policy.zip` | 9,266,679 | 920,000 | 50/50 (100%) | 1792.61 ± 39.91 | verified MetaWorld evaluation | STRONG |
| button-press-v3 | `runs/experts/metaworld-v3/button-press-v3/policy.zip` | 9,266,673 | 340,000 | 50/50 (100%) | 1532.96 ± 88.30 | verified MetaWorld evaluation | STRONG |
| faucet-open-v3 | **missing** | — | — | not evaluated | — | KUKA candidate only | INVALID |
| drawer-close-v3 | **missing** | — | — | not evaluated | — | KUKA candidate only | INVALID |
| window-close-v3 | `runs/experts/metaworld-v3/window-close-v3/policy.zip` | 9,266,659 | 1,000,000 | 50/50 (100%) | 1588.95 ± 32.21 | verified MetaWorld evaluation | STRONG |
| door-open-v3 | **missing** | — | — | not evaluated | — | KUKA candidate only | INVALID |
| button-press-topdown-v3 | **missing** | — | — | not evaluated | — | KUKA candidate only | INVALID |
| peg-unplug-side-v3 | `runs/experts/metaworld-v3/peg-unplug-side-v3/policy.zip` | 9,266,666 | 300,000 | 44/50 (88%) | 984.76 ± 449.13 | verified MetaWorld evaluation | PARTIAL |

Supplemental evidence, not an automatic substitution:

| Task | Checkpoint | Success | Return mean ± SD | Status |
|---|---|---:|---:|---|
| faucet-close-v3 | `runs/experts/metaworld-v3/faucet-close-v3/policy.zip` | 50/50 (100%) | 1766.18 ± 12.98 | STRONG |

Exact sizes, SHA-256 values, KUKA candidate details, and full-precision metrics
are in the registry. The legacy MetaWorld `config.json` files document seed 0,
one million planned steps, reward v2, `MlpPolicy`, and `[256,256,256,256]`.
There are no MetaWorld expert manifests to authenticate those claims. The
checkpoint `num_timesteps` values above are directly verified. Because most
are below the documented training horizon, `policy.zip` was evidently curated
before that horizon, but whether it was named best, final, or intermediate in
the original run is **unknown**. Existing historical registry results used
only 10 deterministic episodes and were not used for the new qualification.

The four same-named KUKA checkpoints load and have matching tensor dimensions,
but their manifests/configurations identify `kuka-v3`. KUKA and Sawyer
MetaWorld observations/actions have different robot semantics; shape equality
does not make them interchangeable.

## Compatibility with the current DiffCRL pipeline

The current shared interface is defined in `src/env_adapters/__init__.py`:
18 current-state values, 18 previous-state values, and a 3D goal, for 39 total;
the action has four bounded components. `src/env_adapters/metaworld_v3.py`
constructs fixed-horizon (200-step), non-terminating-on-success MetaWorld MT1
v3 environments with reward v2. All six evaluated requested checkpoints passed
the current space checks and task-specific rollouts.

For MetaWorld, the 18D state follows MetaWorld's physical observation layout:
end-effector position, gripper opening, and object-state slots (positions and
quaternions); it is not semantically interchangeable with the KUKA adapter.
No observation normalization wrapper or VecNormalize state was found in these
checkpoint evaluation paths. SAC internally applies its standard bounded-action
mapping; the environment and stored demonstrations use `[-1,1]^4`.

All ten exact requested identifiers exist in the installed MetaWorld v3 task
set. Identifier validity does not imply that a compatible expert exists.

## Evaluation methodology

The audit used `scripts/evaluation/evaluate_sac.py`, deterministic SAC actions,
reward v2, evaluation/task-set seed 10000, and the established any-step
episodic success definition. Success does not end the episode, so every rollout
had length 200 (mean 200, SD 0). There were no environment or numerical
failures.

The request preferred 100 distinct configurations. Installed MetaWorld MT1
provides exactly 50 training tasks for each identifier; the evaluator correctly
rejects requests above that bank size rather than repeating tasks. Therefore,
the largest comparable bank—50 episodes and 50 unique configurations—was used.

Reproduction from the repository root (choose a new directory to retain the
evaluator's exclusive-output guarantee):

```bash
cd continual-learning
AUDIT_DIR=runs/audits/sac_experts_metaworld_v3_repro_seed10000
mkdir -p "$AUDIT_DIR"
for TASK in reach-v3 hammer-v3 handle-press-side-v3 button-press-v3 window-close-v3 peg-unplug-side-v3 faucet-close-v3; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 scripts/evaluation/evaluate_sac.py \
    --model "runs/experts/metaworld-v3/$TASK/policy.zip" \
    --task "$TASK" --env-backend metaworld-v3 --reward-function-version v2 \
    --evaluation-mode fixed-tasks --episodes 50 \
    --task-set-seed 10000 --seed 10000 \
    --output "$AUDIT_DIR/$TASK.json"
done
```

Use the project environment that supplies MetaWorld, Torch, and SB3 2.9.0 if
system `python3` is not that environment. The attempted 100-episode form is
expected to fail because it requests more unique configurations than MT1
provides; this is a safety property, not an evaluation failure.

## Demonstration reliability

`src/continual/collector.py` retains every collected episode and records
any-step success; it does not filter failed expert rollouts. Consequently:

- the four requested STRONG experts are suitable for small BC demonstration
  collection under the measured distribution;
- reach and peg-unplug are not reliable enough for an unfiltered 15-trajectory
  demonstration set. At an assumed independent 88% rate, the chance of at
  least one failure in 15 is about 85.3%; independence is only an illustrative
  calculation, not a guarantee;
- missing MetaWorld experts cannot be replaced with KUKA experts.

Even 50/50 is an estimate, not proof of perfect performance under every task
configuration or seed.

## Sequence review

### Proposed five tasks

`button-press → drawer-close → faucet-open → peg-unplug-side → hammer` is a
**NO-GO**. It has useful diversity (pressing, articulated objects, extraction,
and tool use), but drawer-close and faucet-open lack MetaWorld experts, and peg
is PARTIAL. It would contaminate a 15-shot BC comparison with backend mismatch
or failed demonstrations.

The recommended qualified sequence is:

`button-press-v3 → faucet-close-v3 → hammer-v3 → handle-press-side-v3 → window-close-v3`

It spans button contact, articulated rotation, tool use, side pressing, and a
sliding articulated object. Placing related contact/control skills apart also
creates opportunities to observe forgetting and later retention. These are
hypotheses about task structure; no positive or negative transfer claim is
empirically established here. Faucet-close is explicitly a new benchmark
choice, not a relabeling or substitute checkpoint for faucet-open.

### Proposed ten tasks

The proposed order is diverse and long enough to measure retention, but is a
**NO-GO with current artifacts**: four experts are missing/incompatible and two
more are PARTIAL. There are only seven compatible MetaWorld checkpoint
candidates even after adding distinct faucet-close, and only five are STRONG.
Thus no scientifically qualified ten-task sequence can be formed from current
experts without duplicating tasks or lowering the qualification bar.

The requested ten-task order can remain a **conditional future target**:

`reach → button-press → drawer-close → faucet-open → hammer → handle-press-side → door-open → button-press-topdown → peg-unplug-side → window-close`

Before using it, obtain and uniformly qualify authentic MetaWorld-v3 experts
for the four missing tasks and improve or explicitly accept the reach/peg
failure rates. The order mixes elementary positioning, related press tasks,
articulated manipulation, tool use, and extraction, which makes forgetting and
retention measurable; expected transfer remains unverified.

## Fixed 15+15 replay assessment

The current fixed-total uniform allocator gives remainders to earlier task IDs.
With 15 generated trajectories total, the allocations over old tasks are:

| Stage (current task index) | Prior-task allocation in sequence order |
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

The allocation remains mechanically valid for DDPM and either two- or
four-layer GeneralPolicy. `src/continual/stage_dataset.py` keeps a one-episode
group entirely in training; groups of two split one train/one validation.
The 15 current real trajectories still provide a validation set, so the global
split is nonempty. Generator self-replay and goal reconstruction remain valid.

Scientifically, late-stage retention estimates are weak: by stage 10 four old
tasks receive only one replay trajectory and five receive two. Early tasks also
receive deterministic remainder preference. This creates severe sampling
variance and dilution across tasks; correctness of the code path is not
evidence that 15 total replay trajectories are adequate for long-term
retention.

## Risks, limitations, and next experiments

1. Train or recover authentic MetaWorld-v3 experts for faucet-open,
   drawer-close, door-open, and button-press-topdown; add immutable manifests
   at creation time rather than retroactively asserting provenance.
2. Re-qualify those checkpoints with this exact 50-task-bank protocol.
3. Improve or explicitly exclude reach and peg-unplug before using unfiltered
   15-trajectory BC data.
4. Run the revised five-task 15+15 benchmark first. Treat a ten-task fixed-15
   replay study as a replay-budget stress test, not a well-powered retention
   benchmark.
5. If a ten-task benchmark becomes possible, report per-task replay counts and
   repeat across seeds; the audit does not recommend changing the existing
   algorithm in place.

No checkpoints, historical results, manifests, or formal configurations were
deleted, moved, renamed, or overwritten during this audit.
