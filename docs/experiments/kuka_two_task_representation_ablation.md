# KUKA two-task trajectory-representation ablation

## Scope and experiment setup

This report audits the completed seed-0 runs under
`runs/representation_ablation/two_task/` for the sequence
`kuka-hammer-v3` → `kuka-handle-press-side-v3`. The five methods are DDPM
with `structured_22d`, `goal_conditioned_40d`, or `full_43d`, Real Replay,
and No Replay. All values below come from saved run artifacts; no training or
sampling was performed for this analysis.

The common setup is KUKA-v3 with reward function v2, 100 expert trajectories
per task, BC for 100 epochs (batch size 256, learning rate 0.001,
post-tanh MSE), seed 0, and fixed-task evaluation on 20 episodes per cell
(evaluation/task-bank seed 10000). The replay arms use a fixed-total budget of
100 historical trajectories at stage 1 and train BC on those 100 Hammer
trajectories plus the 100 new Handle trajectories. The DDPM arms use 100
epochs, 100 denoising steps, batch size 8, learning rate 0.001, width 64, raw
actions, and no generated-action projection. Only the trajectory
representation/dimensionality differs among the three DDPM arms.

Average performance (AP) and forgetting are not redefined here. They are
calculated by the repository implementation in `src/continual/metrics.py`:
AP is the mean of the completed stage row, while forgetting is the mean of
the diagonal endpoint minus the final endpoint, including the current task's
zero term. Consequently, for two tasks a 0.15 Hammer drop produces forgetting
of 0.075, not 0.15.

## Comparability and completion audit

All five runs are valid and complete for the requested comparison.

| Run | Stage 0 marker | Stage 1 marker | `RUN_COMPLETE` | Final matrix |
| --- | ---: | ---: | ---: | --- |
| DDPM 22D | valid | valid | present | valid, required cells finite |
| DDPM 40D | valid | valid | present | valid, required cells finite |
| DDPM 43D | valid | valid | present | valid, required cells finite |
| Real Replay | valid | valid | present | valid, required cells finite |
| No Replay | valid | valid | present | valid, required cells finite |

The repository's `completed_stage_markers` validator found two contiguous
markers in every run and verified every marker-referenced artifact against its
recorded SHA-256. `EvaluationMatrix.load` accepted every final matrix. Each
matrix has measured cells `R[0,0]`, `R[1,0]`, and `R[1,1]`, with the
not-yet-learned `R[0,1]` correctly unmeasured.

The expert identities are identical across methods:

- Hammer: `runs/experts/kuka-v3/hammer-v3/policy.zip`, SHA-256
  `1900fe4dd8d3150f7d484f7f8886f4e65616bf11fa681d5a29ae6b7a06696c8c`.
- Handle: `runs/experts/kuka-v3/handle-press-side-v3/policy.zip`, SHA-256
  `090824e59d5dcfe3a3be0700a020877b97e5545b3bf9e507a02d1b132dc47049`.

Both manifests identify SAC, KUKA-v3, reward v2, training seed 0, observation
shape 39, action shape 4, and horizon 200. The collected stage-0 Hammer arrays
are byte-identical across all five methods (SHA-256 `ce5940...de00`), as are
the stage-1 Handle arrays (SHA-256 `31a027...494`). The Handle train/validation
IDs and fixed evaluation task banks are also identical. All replay arms report
`replay_budget: 100`, allocation `{kuka-hammer-v3: 100}`, and exactly 100
historical trajectories; No Replay correctly reports no replay budget or
historical source.

Thus the configurations are scientifically comparable for this ablation.
Expected differences are replay mode, DDPM representation, generated replay
content, generator cost, and the larger stage-1 BC dataset in replay arms.

## Continual evaluation results

All entries are success rates except AP and forgetting, which are derived from
the success-rate matrix using the implementation described above.

| Method | Hammer Stage 0 | Hammer Stage 1 | Handle Stage 1 | Final AP | Forgetting |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDPM 22D | 1.00 | 0.85 | 1.00 | 0.925 | 0.075 |
| DDPM 40D | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 |
| DDPM 43D | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 |
| Real Replay | 1.00 | 1.00 | 1.00 | 1.000 | 0.000 |
| No Replay | 1.00 | 0.00 | 1.00 | 0.500 | 0.500 |

With 20 evaluation episodes, the observed success-rate resolution is 0.05.
The 22D final Hammer result corresponds to 17/20 successes; the other replay
arms have 20/20, and No Replay has 0/20.

## Stage-1 historical replay diagnostics

Diagnostics operate on the 100 decoded Hammer replay trajectories, each of
length 200. Goal and previous-state errors are Euclidean norms. Action OOB is
the fraction of scalar action components outside `[-1,1]`; maximum violation
is `max(max(|a|-1, 0))`. No action projection was applied.

| DDPM representation | Goal mean deviation | Goal max deviation | Within-trajectory goal variation | Previous-state mean error | Previous-state max error | Action OOB fraction | Max action violation | Numerical validity | Sampling duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `structured_22d` | 0 | 0 | 0 | 0 | 0 | 11.3063% | 0.797593 | valid (0 NaN, 0 Inf) | 14.6594 s |
| `goal_conditioned_40d` | 0 | 0 | 0 | 0.0148745 | 0.0991345 | 7.3513% | 0.649380 | valid (0 NaN, 0 Inf) | 15.3718 s |
| `full_43d` | 2.69329e-5 | 1.05114e-4 | 6.01627e-5 | 0.0150218 | 0.0879561 | 15.2375% | 1.038789 | valid (0 NaN, 0 Inf) | 14.4568 s |

The stored 40D and 43D report diagnostics were cross-checked against the saved
arrays. The 22D report predates storage of the nested structural-diagnostics
block, so its values above were recomputed read-only from its saved
`data_task_0.npy`, using the same formulas as
`generated_trajectory_diagnostics` and the corresponding stage-0 reference
goals. Its shape is `(100, 200, 43)` and it is finite.

The representation-dependent structural pattern from the single-task Hammer
sanity experiment remains present during stage-1 replay:

- 22D reconstructs both goals and previous states exactly.
- 40D reconstructs goals exactly but preserves generated previous-state
  inconsistency.
- 43D preserves both small generated goal drift and previous-state
  inconsistency.

The action-OOB ordering does **not** remain the same. In the single-task run,
component OOB was 14.6275% (22D), 9.7875% (40D), and 8.05625% (43D). During
stage-1 historical replay it is 11.3063%, 7.3513%, and 15.2375%, respectively:
43D changes from the lowest to the highest OOB fraction. This difference may
reflect the generator's continual update and sampling context, but the saved
results alone do not identify a cause.

## Interpretation

### Measured findings

1. **Hammer retention.** DDPM 40D and DDPM 43D tie for the best generated-replay
   retention at 1.00. They also tie Real Replay. DDPM 22D retains 0.85.
2. **Structural consistency versus retention.** Lower structural inconsistency
   does not correspond to better retention in this seed: exact 22D replay is
   worse than the structurally inconsistent 40D and 43D replay by 0.15 Hammer
   success. Structural exactness is therefore not sufficient to predict this
   downstream result.
3. **Action OOB versus retention.** There is no monotonic relationship. 40D has
   the lowest OOB fraction and perfect retention, but 43D has the highest OOB
   fraction and also perfect retention; 22D lies between them and retains only
   0.85. No projection-setting confound is present because all use `none`.
4. **Versus Real Replay.** 40D and 43D match Real Replay on all measured success
   endpoints and both CL metrics. 22D trails Real Replay by 0.15 on final Hammer,
   0.075 AP, and 0.075 forgetting.
5. **Versus No Replay.** Every DDPM arm substantially improves Hammer retention:
   0.85 or 1.00 versus 0.00. Their final AP is 0.925 or 1.00 versus 0.50.
6. **Catastrophic forgetting.** Learning Handle causes complete measured Hammer
   forgetting without replay (1.00 to 0.00). The 22D arm shows partial forgetting
   (1.00 to 0.85). The 40D, 43D, and Real Replay arms show none at this endpoint.
7. **Current-task learning.** There is no success-rate evidence that replay harms
   Handle learning: every method reaches 1.00. This endpoint is at ceiling, so it
   cannot establish equivalence in learning speed, robustness, or return quality.

### Hypotheses, not established conclusions

The better 40D/43D retention could arise from richer replay features, favorable
generated distributions, BC optimization effects, or seed-specific variation;
these data do not isolate the mechanism. In particular, the results do not
support claiming that previous-state inconsistency or action OOB is beneficial.
They show only that neither diagnostic, by itself, orders downstream retention
in this run.

## Limitations

- There is one training seed and one two-task order. No variance estimate or
  statistical ranking among representations is available.
- Evaluation uses only 20 episodes per cell; success changes in increments of
  0.05, and 1.00 values are ceiling-censored.
- Replay diagnostics describe structural validity, not task feasibility or
  coverage of behaviorally important states.
- The generator is updated at stage 1, so the historical samples are not simply
  a repeat of the single-task sampling experiment.
- AP and forgetting summarize endpoint success only. They do not measure
  learning speed or within-stage transient forgetting.
- Real Replay and DDPM replay both double the stage-1 BC sample count relative
  to No Replay by design. Thus replay presence and dataset size are coupled in
  the intended protocol.

## Recommendations

Run at least four additional seeds (for example 1–4), retaining seed 0 for a
five-seed aggregate. Keep expert checkpoints, collected source trajectories,
splits, task banks, replay allocation, action projection, and sampling seeds
paired across representations. Report per-seed matrices plus mean and a
seed-level uncertainty interval rather than pooling evaluation episodes as if
they were independent training replicates.

Because 40D and 43D are tied at the success ceiling while 22D differs by only
three evaluation episodes, prioritize replication before changing the model.
For additional discrimination, preserve the canonical success metrics but also
report the already stored mean returns and, if feasible in a future experiment,
more evaluation episodes and learning curves. Continue recording the same
structural diagnostics so that correlations with retention can be evaluated
across seeds rather than inferred from three points.

## Artifact and implementation references

- Architecture/data contracts: `docs/ARCHITECTURE.md`
- Metrics: `src/continual/metrics.py` (`average_performance`, lines 22–25;
  `forgetting`, lines 28–36 at analysis time)
- Diagnostics: `src/continual/diffusion_data.py`
  (`generated_trajectory_diagnostics`, lines 384–415 at analysis time)
- Completion validation: `src/continual/resume.py`
  (`completed_stage_markers`)
- Per-run sources: each run's `config.yaml`, `run_manifest.json`,
  `stage_{0,1}_COMPLETE.json`, `evaluation_matrix.json`, stage-1 `report.json`,
  and `timing_summary.json` under `runs/representation_ablation/two_task/`
- Single-task comparison: `docs/experiments/kuka_trajectory_representation_analysis.md`
