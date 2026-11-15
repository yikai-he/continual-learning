# Archived SAC policy-transfer implementation

This document records the GeneralPolicy-to-SAC policy-transfer subsystem as it existed at Git
revision `f4f36c762c31de95cc0c0c4f18af7811fb3251a3`. The subsystem was removed from the active
thesis workflow because online SAC policy transfer is outside the current scope. It was separate
from both single-task SAC expert training and the general Forward Transfer metric in the standalone
ContinualManipulation benchmark.

## Objective

The experiment tested whether initializing a fresh SAC actor from a DiffCRL `GeneralPolicy`
improved learning on the immediately following task. It paired two conditions with the same target
task, training seed, SAC settings, evaluation grid, and source-checkpoint provenance:

- `random`: use the freshly initialized specialized SAC actor unchanged;
- `transferred`: copy the compatible GeneralPolicy actor pathway before SAC training.

Critics, target critics, entropy state, replay buffer, and optimizer states were always fresh. The
experiment therefore measured actor initialization, not continuation of an SAC learner.

## Transferable actor

`src/sac/transferable_actor.py` defined `TransferableActor`, an SB3 squashed-Gaussian `Actor`, and
`TransferableSACPolicy`, an `SACPolicy` that changed only actor construction. The critic remained
the normal SB3 SAC critic.

The actor required 39-dimensional observations, four continuous actions through the environment
space, no gSDE, and exactly four hidden layers of width 256. Its policy trunk was:

1. `Linear(39, 256)`;
2. `LayerNorm(256)`;
3. ReLU for the standard activation, or Tanh for the Legacy-inspired leaky-ReLU design;
4. three further `Linear(256, 256)` layers, each followed by ReLU or LeakyReLU(0.01).

SB3's ordinary linear `mu` and `log_std` heads were retained. On fresh construction the log-standard-
deviation head weights were zero and its bias was `log_std_init`, giving an initially state-independent
exploration scale. The default experiment value was `-3.0` and validation constrained it to SB3's
`[LOG_STD_MIN, LOG_STD_MAX]` range.

## Compatibility and parameter copying

`src/sac/policy_transfer.py` accepted only a task-agnostic, LayerNorm-enabled, squashed-Gaussian-NLL
GeneralPolicy whose hidden sizes, activation, observation shape, action shape, and physical action
bounds matched the specialized SAC actor. Deterministic-MSE policies, task-conditioned policies,
width mismatches, activation mismatches, and ordinary SB3 policies were rejected.

The implementation derived the mapping by pairing corresponding `Linear` and `LayerNorm` modules,
then copied state dictionaries under `torch.no_grad()` with strict loading. For the supported ReLU
four-layer layout the mapping was:

| GeneralPolicy source | SAC target |
|---|---|
| `shared_network.0` | `actor.latent_pi.0` |
| `shared_network.1` | `actor.latent_pi.1` |
| `shared_network.3` | `actor.latent_pi.3` |
| `shared_network.5` | `actor.latent_pi.5` |
| `shared_network.7` | `actor.latent_pi.7` |
| `mean_head` | `actor.mu` |
| `log_std_head` | `actor.log_std` |

Thus both stochastic action heads were inherited along with all trunk linear and LayerNorm
parameters. No non-actor SAC state was copied.

## Action-equivalence verification

`deterministic_equivalence` required a finite, nonempty `(N, 39)` float32 observation array. It
compared bounded physical actions from deterministic GeneralPolicy inference with
`SAC.predict(..., deterministic=True)`, then reported mean absolute error, maximum absolute error,
and the number of observations having any coordinate error greater than or equal to the tolerance.
The diagnostic script used a tolerance of `1e-5`, collected at least 1,000 valid environment
observations with uniformly sampled actions, and failed if the maximum error reached that threshold.
An optional smoke phase verified finite losses and gradients, actor and critic updates, and save/load
prediction equivalence after short SAC optimization.

## Training and comparison protocol

`scripts/run_forward_transfer.py` used a fixed MetaWorld-v3 target-task bank and reused
`train_sac.FixedSeedEvalCallback`. It evaluated at timestep zero and every configured evaluation
interval, saved a checkpoint and replay buffer at each matching interval, and supported resuming an
incomplete run from the latest paired model/replay checkpoint. Completed runs stored configuration,
source identity, task-bank verification, actor/critic hashes, evaluation arrays, final model and replay
hashes, and a summary before writing `RUN_COMPLETE`.

The source GeneralPolicy checkpoint had to prove from its metadata that it was trained through
exactly the declared task history and immediately preceded the target task. The supplied experiment
used history `[button-press-v3, faucet-close-v3]` and target `hammer-v3`.

Per-condition summaries recorded trapezoidal success AUC normalized by the training-step span,
final success, final return, and the first measured success-threshold crossing. The aggregation
script paired random and transferred conditions by seed and reported transferred-minus-random AUC
and final-success differences, threshold steps, means, and sample standard deviations. These were
experiment-specific comparisons, not the ContinualManipulation normalized Forward Transfer metric.

## Configuration

The original configurations were `configs/forward_transfer/hammer_4layer_random.yaml` and
`hammer_4layer_transferred.yaml`. Their principal values were:

- environment: MetaWorld-v3, reward version v2;
- SAC: 100,000 steps, learning rate `3e-4`, buffer 1,000,000, learning starts 10,000,
  batch size 256, gamma 0.99, tau 0.005, train frequency 1, gradient steps 1;
- actor log-standard-deviation initialization: `-3.0`;
- checkpoint and evaluation frequency: 10,000 steps;
- evaluation: 50 episodes, reset seed 10,000, task-bank seed 10,000, success threshold 0.8;
- runtime: CPU, seed 0, output under `runs/forward_transfer_4layer`.

Strict validation required positive finite settings, a training budget divisible by evaluation
frequency, checkpoint frequency equal to evaluation frequency, learning starts below total steps,
and a seed in the uint32 range.

## Original source paths

- `src/sac/forward_transfer.py`
- `src/sac/policy_transfer.py`
- `src/sac/transferable_actor.py`
- `scripts/run_forward_transfer.py`
- `scripts/aggregate_forward_transfer.py`
- `scripts/test_general_policy_sac_transfer.py`
- `configs/forward_transfer/hammer_4layer_random.yaml`
- `configs/forward_transfer/hammer_4layer_transferred.yaml`
- `tests/test_forward_transfer.py`
- `tests/test_policy_transfer.py`
- `docs/FORWARD_TRANSFER_EXPERIMENT.md`

## Known limitations

- The experiment supported MetaWorld-v3 only and one fixed 39D/4D interface.
- Transfer required an unconditioned four-by-256 GeneralPolicy and could not use one-hot task
  conditioning or the supported 512-wide GeneralPolicy variant.
- The specialized actor was not the standard actor used by independent SAC expert training, so the
  random condition controlled for its custom LayerNorm architecture rather than ordinary SB3 SAC.
- Only actor initialization was transferred; no critic, entropy, replay, or optimizer knowledge was
  reused.
- Temporal validity depended on complete and correct checkpoint metadata.
- The experiment-specific AUC difference was not the Continual World-style normalized Forward
  Transfer metric retained in `continual_manipulation.metrics.forward_transfer`.
- Exact equivalence and subsequent optimization remained subject to floating-point and device
  behavior.
