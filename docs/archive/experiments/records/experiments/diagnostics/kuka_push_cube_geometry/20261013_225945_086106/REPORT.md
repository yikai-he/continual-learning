# KUKA Cube Push Object-Geometry Diagnostic

Generated: `20261013_225945_086106`

No training or policy update occurred. All geometry changes existed only in isolated compiled MuJoCo model instances.

## Protocol

- Reward: `sawyer_push_v2_adapted`; official success unchanged (3D center distance <= 0.05 m).
- Fixed controller: rho=0.060 m, TCP z=0.042 m.
- Geometry-aware formulas: rho=0.040+max(width,depth)/2; TCP z=height/2+0.022 m.
- Larger objects would penetrate the table under the raw z=0.020 m reset. Evaluation explicitly sets object and goal center z to half-height.
- Tipping: >=15 deg. Rolling diagnostic: >=45 deg plus >=1 rad/s angular speed.

## Inspected baseline physics

- Object collision geom: box half-size `(0.02, 0.02, 0.02)` m; full size 40 mm cubed; mass 0.75 kg; diagonal inertia `(0.0002, 0.0002, 0.0002)` kg m2.
- Object friction `(slide, torsion, roll)=(1, 0.1, 0.002)`, `condim=4`, `solimp=(0.99, 0.99, 0.01)`, `solref=(0.01, 1)`.
- Global solver: Newton, dense Jacobian, elliptic cone, 50 iterations, tolerance `1e-10`, timestep 0.0025 s.
- Object XY sampling is `x=[-0.10,0.10]`, `y=[0.484,0.55]`; goal is `x=[-0.10,0.10]`, `y=[0.65,0.75]`, with at least 0.15 m planar separation. TCP workspace is `[-0.25,0.42,0.04]` to `[0.25,0.75,0.50]` m.
- Robotiq named pad collision boxes have half-size `(0.011,0.004,0.009375)` m. Upper/lower pad friction is 0.7/0.6 with `solimp=(0.95,0.99,0.001)` and `solref=(0.004,1)`.
- The scripted controller pushes along the normalized object-to-goal XY vector after a 20-step elevated approach and 35-step contact phase; its final 145-step sweep advances through `distance + rho`.
- `sawyer_push_v2_adapted` retains the official 0.05 m 3D center-distance success rule and uses fixed caging geometry (`obj_radius=0.015`, `pad_success_thresh=0.05`, `xz_thresh=0.005`), which was not retuned for variants.

## Results

| Configuration | Ever | Final | Contact | Effective push | Tip | Roll | Final distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| cube_40__constant_mass__fixed | 52.0% | 42.0% | 100.0% | 100.0% | 86.0% | 0.0% | 0.0732 |
| cube_60__constant_mass__fixed | 40.0% | 26.0% | 98.0% | 98.0% | 98.0% | 0.0% | 0.0888 |
| cube_60__constant_mass__geometry_aware | 28.0% | 20.0% | 100.0% | 100.0% | 96.0% | 2.0% | 0.0998 |
| cube_60__constant_density__fixed | 30.0% | 18.0% | 100.0% | 98.0% | 78.0% | 0.0% | 0.1022 |
| cube_60__constant_density__geometry_aware | 22.0% | 14.0% | 100.0% | 100.0% | 92.0% | 14.0% | 0.1051 |
| cube_80__constant_mass__fixed | 30.0% | 20.0% | 96.0% | 96.0% | 90.0% | 12.0% | 0.1353 |
| cube_80__constant_mass__geometry_aware | 18.0% | 8.0% | 100.0% | 100.0% | 100.0% | 2.0% | 0.1307 |
| cube_80__constant_density__fixed | 30.0% | 16.0% | 100.0% | 98.0% | 74.0% | 0.0% | 0.1280 |
| cube_80__constant_density__geometry_aware | 12.0% | 8.0% | 100.0% | 100.0% | 100.0% | 8.0% | 0.1218 |
| box_60x60x30__constant_mass__fixed | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.2066 |
| box_60x60x30__constant_mass__geometry_aware | 0.0% | 0.0% | 100.0% | 100.0% | 10.0% | 0.0% | 0.2076 |
| box_60x60x30__constant_density__fixed | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.2076 |
| box_60x60x30__constant_density__geometry_aware | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.2082 |

## Interpretation boundary

This report separates geometry, mass strategy, and controller protocol. Constant-density variants change both mass and inertia; geometry-aware runs also change controller targets. The larger-object runs include a documented vertical placement correction and therefore are not literal raw-reset comparisons.

See `summary.json`, `configurations.csv`, `episodes.csv`, `selected_step_telemetry.csv`, `contact_analysis.json`, `object_motion_analysis.json`, and `geometry_feasibility.json` for measurements.

## Contact analysis

- Contact acquisition was already saturated by the baseline: 100% of episodes, mean first contact at step 68.0, and mean duration 85.8 steps. Larger cubes therefore had no contact-rate headroom to improve.
- The 60 mm cubes contacted in 98--100% of episodes. Fixed control reached them earlier (mean step 64.0--64.2), while geometry-aware control reached them later (step 71.3).
- The 80 mm cubes contacted in 96--100% of episodes, but constant-mass contact duration fell to 26.8 fixed / 23.2 aware steps. This indicates unstable sustained contact despite easy acquisition.
- The low box contacted in 100% of episodes for 44.2--47.6 steps, yet moved only 6--8 mm toward the goal on average. Physical contact alone was not an effective push.
- Contact loss/reacquisition and pad/follower breakdowns are in `contact_analysis.json`; actual MuJoCo contacts, not TCP proximity, define these metrics.

## Motion, success, and overshooting

- Baseline was best: 52% ever-success, 42% final-success, 80.8% retention among successful episodes, and 0.0732 m mean final distance.
- Best larger variant was the 60 mm constant-mass cube with fixed control: 40% ever, 26% final, 65% retention, and 0.0888 m mean final distance.
- Fixed 80 mm variants reached 30% ever and 16--20% final success. Their mean final distances (0.1280--0.1353 m) were substantially worse than baseline.
- The low-profile box reduced tipping to 0--10%, but every condition had 0% ever/final success. Reduced tipping did not improve this task.
- Goal-exit frequency was 10% for baseline, 12--14% for fixed 60 mm, and 16--18% for fixed 80 mm. Successful larger objects also had worse retention, consistent with overshooting/contact-control limitations rather than failed acquisition.
- Every configuration encountered workspace clipping in at least one step of every episode because the open-loop target continues beyond reachable task space. This is a controller limitation shared by all comparisons.

## Answers and recommendation

1. Larger cubes did not improve contact acquisition; baseline was already at 100% and larger variants were 96--100%.
2. Larger cubes did not improve goal-directed pushing or success. The permissive `effective_push` event remained 96--100%, but displacement, final distance, and success worsened.
3. Mass mattered, but did not reverse the conclusion. Constant-density 60 mm performance was below constant-mass; 80 mm ever-success tied under fixed control while final-success was slightly lower at constant density.
4. The low box strongly reduced tipping.
5. Reduced tipping did not improve success: the box never succeeded and barely advanced.
6. Fixed-controller goal exits rose with cube size (10% baseline, 14% at 60 mm constant mass, 16% at 80 mm constant mass); this is evidence of worse retention, though failures before goal entry also increased.
7. The 80 mm cube is not impossible to contact, but it is too large for this controller/Robotiq setup to recommend: contact remains high while duration, progress, and success degrade.
8. Geometry-aware control did not outperform fixed control for any non-baseline geometry. Its deterministic offset/height formulas generally delayed contact and reduced success.
9. Improvements were not robust across the 50-task bank; no larger geometry beat baseline aggregate success.
10. The 40 mm baseline is the best candidate among tested geometries. If a geometry follow-up is required, test the 60 mm constant-mass cube with a controller designed to decelerate and stop near the goal, not the present open-loop sweep.

Measured results support the recommendation above. A physically motivated hypothesis is that larger faces make initial contact easy but amplify sustained-contact, clearance, and open-loop overshoot problems. Whether a learned SAC policy could exploit a larger object remains unverified; no frozen-policy evaluation or training was performed.

## Reproduction

```bash
cd /home/yikai/Documents/Code/Final/continual-learning
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n metaworld \
  pytest -q tests/test_kuka_push_object_geometry_diagnostic.py
MUJOCO_GL=egl conda run --no-capture-output -n metaworld \
  python -u scripts/diagnostics/analyze_kuka_push_object_geometry.py --smoke
MUJOCO_GL=egl conda run --no-capture-output -n metaworld \
  python -u scripts/diagnostics/analyze_kuka_push_object_geometry.py
```
