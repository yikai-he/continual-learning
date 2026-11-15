---
status: current
last_verified: 2026-11-09
applies_to: continual-learning
---

# KUKA trajectory representation analysis

## 1. Experiment setup

This report analyzes three completed single-task DDPM runs for
`kuka-hammer-v3`, seed 0, using 100 expert trajectories and 100 generated
trajectories per representation. The controlled variable is the generated
trajectory representation: `structured_22d`, `goal_conditioned_40d`, or
`full_43d`. Each run used 100 diffusion epochs, 100 reverse-diffusion steps,
width 64, batch size 8, learning rate 0.001, raw actions, no generated-action
projection, integer task conditioning, and normalized 3D reference-goal
conditioning.

Authoritative sources are the [22D diagnostics](../../runs/representation_ablation/single_hammer_ddpm/structured_22d_seed0/generated_diagnostics_100/diagnostics.json),
[40D diagnostics](../../runs/representation_ablation/single_hammer_ddpm/goal_conditioned_40d_seed0/generated_diagnostics_100/diagnostics.json),
[43D diagnostics](../../runs/representation_ablation/single_hammer_ddpm/full_43d_seed0/generated_diagnostics_100/diagnostics.json), and their corresponding
[22D stage report](../../runs/representation_ablation/single_hammer_ddpm/structured_22d_seed0/stage_0_kuka-hammer-v3/report.json),
[40D stage report](../../runs/representation_ablation/single_hammer_ddpm/goal_conditioned_40d_seed0/stage_0_kuka-hammer-v3/report.json), and
[43D stage report](../../runs/representation_ablation/single_hammer_ddpm/full_43d_seed0/stage_0_kuka-hammer-v3/report.json).

## 2. Comparability audit

The comparison is matched on task, seed, data, split, conditioning, training
budget, and sampling budget:

- All runs used the same SAC expert SHA-256:
  `1900fe4dd8d3150f7d484f7f8886f4e65616bf11fa681d5a29ae6b7a06696c8c`.
- All packed source files `data_task_0.npy` have SHA-256
  `ce5940cee3ada91a7c1ca8a164531683c194e2bd7cb913f5bafed66c8b0bde00`.
- All archived `real_trajectories.npz` files have SHA-256
  `9b82fb38ab443b3577e90c5c0f6d520e5601f26d80846558f30968f340fc5541`.
- All runs used collection seeds 0--99 and the same 80 training IDs and 20
  validation IDs. The canonicalized split records have identical SHA-256
  `67276ac10e579143b4548df39ebe92d2d763c06bf6976478dc0a54ef19cf919d`.
- Resolved scientific configurations differ only in representation identity
  and the corresponding model feature width (22, 40, or 43); run names and
  run directories necessarily differ.
- All samples used task index 0, reference goals reconstructed from the same
  task bank, sampling seed 101000, and horizon 200.

The dimensionality change also changes the DDPM input/output parameter shapes
and learning problem. It is therefore not a pure post-processing comparison,
and validation losses across widths are not directly comparable quality scores.
Only one training and generation seed was run, so timing and quality differences
do not establish between-seed effects.

## 3. Main comparison

Errors are Euclidean norms. Goal and previous-state means are averaged over
timesteps (previous-state alignment excludes timestep zero). Fractions are over
timesteps except action-component OOB, which is over all scalar action
components. The recorded structural tolerance is `1e-3`.

| Metric | Structured 22D | Goal-conditioned 40D | Full 43D |
|---|---:|---:|---:|
| Goal deviation, mean per timestep | 0 | 0 | 3.44065e-5 |
| Goal deviation, maximum timestep | 0 | 0 | 1.23903e-4 |
| Goal variation, mean per timestep | 0 | 0 | 7.79069e-5 |
| Goal tolerance violations | 0% | 0% | 0% |
| Previous-state alignment, mean per timestep | 0 | 0.0150481 | 0.0160189 |
| Previous-state alignment, maximum timestep | 0 | 0.0933536 | 0.123051 |
| Previous-state tolerance violations | 0% | 100% | 100% |
| Timestep-zero reset error, mean per trajectory | 0 | 0.0166805 | 0.0220090 |
| OOB action components | 14.6275% | 9.7875% | 8.05625% |
| Timesteps with any OOB action | 42.31% | 31.27% | 23.53% |
| Maximum action-bound violation | 0.341963 | 0.604113 | 0.601517 |
| Generator training duration | 61.2223 s | 65.4043 s | 66.2148 s |
| Sampling duration, 100 trajectories | 7.91611 s | 8.05549 s | 9.14877 s |
| Selected epoch | 85 | 94 | 93 |
| Selected validation loss | 0.0541951 | 0.0345872 | 0.0385349 |
| Training / validation trajectories | 80 / 20 | 80 / 20 | 80 / 20 |
| Generated trajectories | 100 | 100 | 100 |

## 4. Structural consistency

The 22D decoder enforces exact goal repetition, previous-state shifting, and
timestep-zero reset; all corresponding measured errors are exactly zero. The
40D decoder enforces the reference goal but preserves generated previous states,
producing mean alignment error 0.0150 and a 100% violation rate at tolerance
`1e-3`. Full 43D preserves both generated previous states and generated goals.
It produces similar previous-state error (0.0160) and measurable but small goal
drift (mean 3.44e-5, maximum 1.24e-4); no goal timestep exceeds `1e-3`.

For scale, the real current-state marginal standard deviations have joint L2
norm 0.3047, while the real mean adjacent-current-state displacement is 0.01133.
The 40D and 43D mean previous-state errors are about 4.9% and 5.3% of the former,
but 1.33 and 1.41 times the latter. Thus the errors are small relative to the
overall state distribution but material relative to a typical one-step state
change. This comparison measures representation consistency, not environment
dynamics validity.

## 5. Action validity

All arms generated raw actions and used no projection. Reduced generated
dimensionality did **not** improve the observed OOB fractions: 22D had the
largest component and timestep OOB rates. Full 43D had the lowest rates, although
its worst scalar violation (0.602 beyond the legal bound) was larger than the
22D maximum (0.342) and comparable to 40D (0.604). Consequently, OOB frequency
and worst-case severity rank the representations differently.

Action marginals also differ. Relative to real actions, the L2 differences in
the four-coordinate mean vectors were 0.307 (22D), 0.144 (40D), and 0.0827
(43D); L2 differences in standard-deviation vectors were 0.157, 0.0591, and
0.0795. These descriptive distances favor 40D/43D for these marginals, but do
not test temporal or state-action consistency.

## 6. Generated distributions

Each cell below is `mean ± population std; [5th, 50th, 95th percentile];
minimum..maximum`, computed over 20,000 timesteps. Coordinates follow the
implemented 18D current-state and 4D action ordering. Real statistics use the
same 100 source trajectories as all three arms.

### Current-state coordinates

| Coord. | Real expert | Structured 22D | Goal-conditioned 40D | Full 43D |
|---:|---|---|---|---|
| 0 | 0.2053 ± 0.0836; [0.000265, 0.2460, 0.2502]; -0.0827..0.2511 | 0.2058 ± 0.0888; [-0.0138, 0.2456, 0.2551]; -0.0905..0.2687 | 0.2082 ± 0.0799; [0.0158, 0.2435, 0.2569]; -0.0811..0.2647 | 0.2060 ± 0.0844; [-0.00142, 0.2442, 0.2538]; -0.1151..0.2689 |
| 1 | 0.6431 ± 0.0938; [0.4198, 0.6890, 0.7032]; 0.4189..0.7071 | 0.6464 ± 0.0950; [0.4197, 0.6961, 0.7064]; 0.3956..0.7254 | 0.6456 ± 0.0900; [0.4317, 0.6852, 0.7073]; 0.3839..0.7252 | 0.6431 ± 0.0959; [0.4166, 0.6899, 0.7065]; 0.3802..0.7185 |
| 2 | 0.0540 ± 0.0342; [0.0347, 0.0451, 0.1433]; 0.0308..0.2133 | 0.0524 ± 0.0343; [0.0361, 0.0432, 0.1443]; 0.0263..0.2225 | 0.0521 ± 0.0332; [0.0369, 0.0422, 0.1415]; 0.0308..0.2274 | 0.0542 ± 0.0335; [0.0344, 0.0459, 0.1443]; 0.0246..0.2267 |
| 3 | 0.5785 ± 0.198; [0.1820, 0.5975, 0.9567]; 0.0139..1.000 | 0.5282 ± 0.214; [0.2838, 0.4560, 0.9643]; 0.1859..1.079 | 0.6442 ± 0.182; [0.1551, 0.6426, 0.9285]; -0.1053..1.017 | 0.6017 ± 0.208; [0.1603, 0.6206, 0.9761]; 0.000662..1.106 |
| 4 | 0.1397 ± 0.0824; [-0.0376, 0.1575, 0.2457]; -0.0991..0.2560 | 0.1672 ± 0.0914; [-0.0297, 0.2030, 0.2510]; -0.1152..0.2628 | 0.1358 ± 0.0804; [-0.0144, 0.1497, 0.2362]; -0.0998..0.2485 | 0.1146 ± 0.0826; [-0.0511, 0.1372, 0.2317]; -0.1133..0.2679 |
| 5 | 0.6485 ± 0.0922; [0.4519, 0.6847, 0.7235]; 0.4209..0.7284 | 0.6470 ± 0.0924; [0.4344, 0.6867, 0.7258]; 0.4128..0.7362 | 0.6396 ± 0.0862; [0.4670, 0.6760, 0.7198]; 0.3929..0.7381 | 0.6276 ± 0.0997; [0.4332, 0.6761, 0.7188]; 0.3893..0.7339 |
| 6 | 0.0443 ± 0.0226; [0.0275, 0.0284, 0.0789]; 0.0234..0.1713 | 0.0646 ± 0.0250; [0.0284, 0.0698, 0.0973]; 0.0249..0.1878 | 0.0453 ± 0.0265; [0.0276, 0.0301, 0.0918]; 0.0222..0.1914 | 0.0433 ± 0.0273; [0.0253, 0.0291, 0.1053]; 0.0208..0.1795 |
| 7 | 0.9849 ± 0.0243; [0.9432, 0.9908, 0.9996]; 0.8024..1.000 | 0.9827 ± 0.0183; [0.9446, 0.9885, 0.9997]; 0.8501..1.005 | 0.9853 ± 0.0322; [0.9123, 0.9983, 1.001]; 0.7951..1.009 | 0.9747 ± 0.0295; [0.9154, 0.9860, 1.001]; 0.8112..1.009 |
| 8 | -0.0333 ± 0.0655; [-0.1759, -0.00152, 0.0188]; -0.2584..0.1901 | -0.0506 ± 0.0772; [-0.2046, -0.0158, 0.0378]; -0.2316..0.1656 | -0.00823 ± 0.0391; [-0.0810, 0.000356, 0.0183]; -0.2006..0.1890 | -0.0208 ± 0.0649; [-0.1666, 0.00204, 0.0478]; -0.2338..0.2001 |
| 9 | -0.00107 ± 0.0552; [-0.0333, -0.0296, 0.1015]; -0.0709..0.4420 | 0.0532 ± 0.0755; [-0.0304, 0.0393, 0.1785]; -0.0642..0.4754 | 0.0161 ± 0.0731; [-0.0330, -0.0217, 0.1365]; -0.0474..0.4712 | 0.00148 ± 0.0792; [-0.0393, -0.0273, 0.1938]; -0.0704..0.4534 |
| 10 | -0.0993 ± 0.106; [-0.3145, -0.0774, 0.0103]; -0.4944..0.1270 | -0.0710 ± 0.0981; [-0.2684, -0.0243, 0.0320]; -0.4314..0.0524 | -0.0826 ± 0.116; [-0.3900, -0.0347, 0.00668]; -0.5365..0.0387 | -0.1486 ± 0.129; [-0.3857, -0.1442, 0.0187]; -0.4897..0.0602 |
| 11 | 0.2400 ± 2.67e-5; [0.2400, 0.2400, 0.2400]; 0.2400..0.2400 | 0.2400 ± 2.10e-5; [0.2400, 0.2400, 0.2400]; 0.2399..0.2401 | 0.2400 ± 2.03e-5; [0.2400, 0.2400, 0.2400]; 0.2399..0.2401 | 0.2400 ± 2.21e-5; [0.2400, 0.2400, 0.2400]; 0.2399..0.2401 |
| 12 | 0.7184 ± 0.0411; [0.6400, 0.7409, 0.7415]; 0.6400..0.7464 | 0.7179 ± 0.0414; [0.6364, 0.7400, 0.7442]; 0.6236..0.7540 | 0.7197 ± 0.0387; [0.6405, 0.7403, 0.7437]; 0.6281..0.7561 | 0.7189 ± 0.0419; [0.6338, 0.7414, 0.7455]; 0.6217..0.7556 |
| 13 | 0.1100 ± 4.45e-6; [0.1100, 0.1100, 0.1100]; 0.1100..0.1100 | 0.1100 ± 2.29e-5; [0.1099, 0.1100, 0.1100]; 0.1099..0.1101 | 0.1100 ± 2.19e-5; [0.1100, 0.1100, 0.1100]; 0.1099..0.1101 | 0.1100 ± 1.53e-5; [0.1100, 0.1100, 0.1100]; 0.1099..0.1101 |
| 14 | 1.000 ± 0; [1.000, 1.000, 1.000]; 1.000..1.000 | 1.000 ± 1.87e-5; [1.000, 1.000, 1.000]; 0.9999..1.000 | 1.000 ± 2.58e-5; [0.9999, 1.000, 1.000]; 0.9999..1.000 | 1.000 ± 2.15e-5; [1.000, 1.000, 1.000]; 0.9999..1.000 |
| 15 | 0 ± 0; [0, 0, 0]; 0..0 | 5.60e-6 ± 2.12e-5; [-2.57e-5, 5.41e-6, 3.94e-5]; -1.13e-4..8.34e-5 | -4.11e-6 ± 1.90e-5; [-3.42e-5, -5.50e-6, 2.79e-5]; -8.11e-5..1.14e-4 | 8.12e-6 ± 2.31e-5; [-2.71e-5, 5.63e-6, 4.70e-5]; -8.92e-5..1.08e-4 |
| 16 | 0 ± 0; [0, 0, 0]; 0..0 | 9.27e-6 ± 1.85e-5; [-1.94e-5, 7.35e-6, 4.13e-5]; -5.08e-5..6.48e-5 | 4.53e-6 ± 1.75e-5; [-2.22e-5, 4.78e-6, 3.25e-5]; -6.37e-5..9.03e-5 | 5.25e-6 ± 1.60e-5; [-2.16e-5, 4.85e-6, 3.30e-5]; -5.73e-5..7.41e-5 |
| 17 | 0 ± 0; [0, 0, 0]; 0..0 | -7.70e-6 ± 2.20e-5; [-4.04e-5, -8.70e-6, 3.16e-5]; -1.00e-4..9.06e-5 | -1.07e-5 ± 1.84e-5; [-3.61e-5, -1.16e-5, 1.93e-5]; -8.33e-5..1.06e-4 | -1.79e-6 ± 1.70e-5; [-2.87e-5, -2.44e-6, 2.86e-5]; -6.83e-5..6.03e-5 |

The current-state mean-vector L2 differences from real were 0.0881 (22D),
0.0751 (40D), and 0.0656 (43D); standard-deviation-vector differences were
0.0319, 0.0392, and 0.0362. All representations cover broadly similar marginal
ranges, with visible coordinate-specific shifts (notably coordinates 3, 4, 6,
9, and 10). There is no marginal evidence of wholesale state collapse.

### Action coordinates

| Coord. | Real expert | Structured 22D | Goal-conditioned 40D | Full 43D |
|---:|---|---|---|---|
| 0 | 0.5945 ± 0.442; [-0.3443, 0.7766, 0.9860]; -0.9743..0.9992 | 0.7578 ± 0.327; [0.1685, 0.8349, 1.019]; -1.342..1.150 | 0.6069 ± 0.431; [-0.1790, 0.8079, 0.9958]; -1.604..1.324 | 0.5649 ± 0.426; [-0.2238, 0.6924, 1.005]; -1.527..1.228 |
| 1 | 0.4989 ± 0.508; [-0.6568, 0.7186, 0.9774]; -0.9913..0.9972 | 0.6067 ± 0.521; [-0.5814, 0.7990, 1.119]; -1.214..1.273 | 0.4957 ± 0.514; [-0.4313, 0.6423, 1.098]; -1.121..1.331 | 0.5269 ± 0.459; [-0.3239, 0.6343, 1.093]; -1.134..1.425 |
| 2 | -0.7390 ± 0.234; [-0.9849, -0.7874, -0.3312]; -0.9987..0.8064 | -0.8722 ± 0.128; [-1.033, -0.8873, -0.6791]; -1.101..-0.2247 | -0.7075 ± 0.177; [-0.9658, -0.6954, -0.4011]; -1.039..-0.2178 | -0.7071 ± 0.292; [-1.001, -0.8047, -0.0463]; -1.080..0.2143 |
| 3 | 0.1668 ± 0.577; [-0.9247, 0.1460, 0.9613]; -0.9999..0.9878 | -0.0291 ± 0.588; [-0.9327, -0.00938, 0.8205]; -1.280..1.297 | 0.3064 ± 0.589; [-0.8106, 0.2625, 1.212]; -1.168..1.514 | 0.1022 ± 0.595; [-0.9730, 0.2241, 0.9925]; -1.602..1.202 |

Marginal similarity is not proof of coherent state-action pairs, temporal
transitions, or simulator validity. The generated distributions can match these
statistics while violating dynamics.

## 7. Training and sampling efficiency

Training time rises from 61.2 s (22D) to 65.4 s (40D, +6.8%) and 66.2 s (43D,
+8.2%). Sampling 100 trajectories takes 7.92, 8.06, and 9.15 s respectively;
43D is about 15.6% slower than 22D in this single run. These wall-clock values
are machine- and load-dependent and lack repeated timing trials.

The selected epochs were 85, 94, and 93. Recorded validation losses were
0.0542, 0.0346, and 0.0385, but the objectives average over different target
widths and must not be ranked as directly comparable model-quality scores.

## 8. Interpretation

Direct measurements establish the intended ablation behavior: deterministic
22D decoding removes both measured structural errors; 40D exposes previous-state
inconsistency while retaining exact goals; 43D exposes previous-state
inconsistency and small goal drift. The independently generated previous state
is inconsistent on every measured transition under the strict `1e-3` tolerance,
and its mean error is at least as large as a typical real one-step displacement.
This is large enough to warrant testing downstream BC sensitivity.

The results do not support a hypothesis that fewer generated dimensions improve
action validity: 22D has the worst OOB frequency here. They also do not show
numerical instability or obvious marginal collapse: every output has expected
shape `(100, 200, 43)`, zero NaNs and Infs, and nondegenerate state/action
marginals. Rare extremes and OOB actions remain a quality concern in every arm.

These are single-task, single-seed generation diagnostics. They do not show
that any representation improves continual learning, retention, task success,
or physical dynamics consistency.

## 9. Limitations

- One task and one seed cannot separate representation effects from stochastic
  training variation.
- Marginal summaries discard temporal dependence, state-action coupling, and
  multimodality.
- No environment rollouts establish whether generated transitions are feasible.
- Goal drift is measurable in 43D but remains below the configured tolerance;
  its downstream importance is unknown.
- Action projection was intentionally disabled, so BC will receive OOB targets.
- Different target dimensions change model parameter shapes and optimization;
  structural enforcement is not the only changed property.
- Wall-clock timings are single observations without uncertainty estimates.

## 10. Recommendation

Proceed with the planned two-task `kuka-hammer-v3` →
`kuka-handle-press-side-v3` ablation, including DDPM 22D/40D/43D, Real Replay,
and No Replay under the already matched data, replay, BC, and evaluation
budgets. The observed previous-state errors and action-validity differences are
large enough to motivate downstream testing, while the absence of numerical
failures makes the runs technically reasonable. Interpret the two-task result
as a pilot: retain the representation diagnostics at stage one, compare BC and
retention metrics, verify identical current-task data hashes across arms, and
repeat with additional seeds before making thesis-level causal claims.
