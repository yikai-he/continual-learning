---
status: archived
last_verified: 2026-11-11
applies_to: continual-learning legacy generative replay
---

# Legacy 22D Trajectory Representation

This document records the reduced trajectory representation that preceded
direct 43D generation. It describes the implementation removed in November
2026 and is a historical reference, not an active implementation guide.

## Motivation and design

The generator treated parts of the physical observation as deterministic
structure. Instead of learning duplicated temporal state and an episode-
constant goal, it modeled only the current state and action. This reduced the
per-timestep learning problem from 43 to 22 features and guaranteed two physical
observation equalities after decoding. Task identity and the episode goal were
supplied as separate conditioning inputs.

The omitted data was the 18D previous state and the 3D goal. The previous state
was reconstructed from current states. The normalized 3D goal and integer task
ID conditioned the DDPM denoiser or Flow Matching velocity network; the same
physical goal was later copied into every decoded timestep.

## Original layouts

The physical trajectory was `(N,T,43)`:

| Indices | Width | Meaning |
|---|---:|---|
| `0:18` | 18 | current environment state |
| `18:36` | 18 | previous environment state |
| `36:39` | 3 | goal |
| `39:43` | 4 | action |

The generated trajectory was `(N,T,22)`:

| Indices | Width | Meaning |
|---|---:|---|
| `0:18` | 18 | current environment state |
| `18:22` | 4 | action |

The external goal condition was one float32 `(N,3)` vector per episode. It was
normalized independently with a `FeatureNormalizer` fitted on training-split
goals. Task conditioning was an optional `(N,)` integer label embedded by the
temporal network.

## Encoding

`src/continual/diffusion_data.py::encode_trajectory` accepted observations of
shape `(N,T,39)` and actions of shape `(N,T,4)`. It required matching batch,
time, dtype, and device, finite floating tensors, and exact physical structure:

```python
previous[:, 0] == current[:, 0]
previous[:, 1:] == current[:, :-1]
goals == goals[:, :1].expand_as(goals)
```

It returned `torch.cat((current, actions), dim=-1)` and a clone of
`goals[:, 0]`. Real stage groups were also checked for previous-state alignment,
constant goals, shape, and finiteness before training. The pre-`tanh` action
mode applied `atanh(clamp(action, -1 + epsilon, 1 - epsilon))` only to indices
`18:22`; raw mode copied the data unchanged.

`prepare_generator_data` encoded training and validation splits, fitted the
22-feature normalizer only on the training split, and fitted the separate
3-feature goal normalizer on training goals. Population standard deviation and
a `1e-3` scale floor were used, without feature clipping.

## Decoding

`decode_trajectory(dynamic, goals)` rebuilt `(N,T,43)` data. Given generated
current states `current`, it formed:

```python
previous = torch.cat((current[:, :1], current[:, :-1]), dim=1)
physical = torch.cat((
    current,
    previous,
    goals[:, None].expand(-1, T, -1),
    dynamic[..., 18:22],
), dim=-1)
```

Thus timestep zero used its own current state as its previous state, later
timesteps used the preceding generated current state, and the conditioning goal
overwrote all goal coordinates. Actions were preserved except for the configured
action-coordinate conversion and optional `[-1,1]` projection. No other state
projection or dynamics correction was performed.

## Implementation details

The principal implementation points were:

- `diffusion_data.py`: `DIFFUSION_FEATURES = 22`, `DIFFUSION_ACTION_SLICE`,
  `encode_trajectory`, `decode_trajectory`, action transforms, normalization
  preparation, and physical-stage validation.
- `replay_common.py::decode_generated_replay`: pre-`tanh` action decoding,
  22D-to-43D decoding, optional action projection, and action diagnostics.
- `diffcrl_replay.py` and `flow_matching_replay.py`: sampled normalized 22D
  trajectories, denormalized them, reconstructed 43D replay, and asserted exact
  previous-state and goal alignment.
- `trajectory_diffusion.py`: `TemporalDenoiser` predicted normalized clean 22D
  trajectories with discrete time, task, position, and projected goal inputs.
- `trajectory_flow_matching.py`: `TemporalVelocityModel` predicted a normalized
  22D velocity field with continuous time, task, position, and goal inputs.
- `diffcrl.py`: encoded each physical stage split before generator fitting and
  kept the BC dataset at the physical `(N,T,43)` interface.

Both temporal models retained their six residual dilated Conv1d blocks. DDPM
kept its cosine-derived schedule and x0 objective; Flow Matching kept its linear
conditional path and ODE integration. Only representation handling differed.

The checkpoint identifiers were `trajectory_diffusion_22_v1` and
`trajectory_flow_matching_22_v1`. Their serialized model configuration used
`features=22`; dynamic normalizer mean and scale had shape `(22,)`, while goal
normalizer mean and scale had shape `(3,)`. Checkpoints also stored model state,
task order, training-bank provenance, selected training configurations, action
space, clamp epsilon, and generated-action projection metadata.

## Trade-offs

Advantages were a smaller output space, no need to learn duplicated state or a
constant per-episode signal, and exact decoded previous-state and goal
structure. The reduced data also lowered first/last network-layer parameter
counts.

The design assumed exact alignment in every real trajectory and a constant goal
within each episode. More importantly, reconstruction made structural
diagnostics trivially perfect and prevented the generator from learning or
expressing errors and correlations in previous-state and goal coordinates. The
external goal was both a condition and a decoder instruction rather than a
generated feature.

Direct 43D generation instead models current state, previous state, goal, and
action jointly. It retains task and goal conditioning but does not enforce
alignment or goal constancy. Finiteness, shape, action bounds, and read-only
structural diagnostics remain meaningful; learned inconsistency is not repaired.

## Migration

The migration removed the 43D-to-22D encoder, 22D-to-43D decoder, shift-based
previous-state reconstruction, goal restoration, 22D constants, representation
selection settings, and compatibility loading. Both generators now normalize,
learn, sample, and denormalize exactly 43 physical features. BC, evaluation,
expert collection, and environment observation/action contracts remain 39D/4D.

New checkpoints use `trajectory_diffusion_43_v2` and
`trajectory_flow_matching_43_v2`. Old 22D checkpoints and intermediate 40D/43D
ablation checkpoint schemas are incompatible and fail explicitly; there is no
automatic conversion. Historical checkpoints, datasets, run records, and
experiment reports remain valid evidence for the implementation that produced
them, but cannot be resumed by the direct-43D implementation.
