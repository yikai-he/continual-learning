# KUKA Cube Push Object-Geometry Diagnostic

Generated: `20261013_225240_837260`

No training or policy update occurred. All geometry changes existed only in isolated compiled MuJoCo model instances.

## Protocol

- Reward: `sawyer_push_v2_adapted`; official success unchanged (3D center distance <= 0.05 m).
- Fixed controller: rho=0.060 m, TCP z=0.042 m.
- Geometry-aware formulas: rho=0.040+max(width,depth)/2; TCP z=height/2+0.022 m.
- Larger objects would penetrate the table under the raw z=0.020 m reset. Evaluation explicitly sets object and goal center z to half-height.
- Tipping: >=15 deg. Rolling diagnostic: >=45 deg plus >=1 rad/s angular speed.

## Results

| Configuration | Ever | Final | Contact | Effective push | Tip | Roll | Final distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| cube_40__constant_mass__fixed | 66.7% | 66.7% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0549 |
| cube_60__constant_mass__fixed | 66.7% | 33.3% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0647 |
| cube_60__constant_mass__geometry_aware | 33.3% | 33.3% | 100.0% | 100.0% | 100.0% | 33.3% | 0.0667 |
| cube_60__constant_density__fixed | 66.7% | 33.3% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0769 |
| cube_60__constant_density__geometry_aware | 33.3% | 33.3% | 100.0% | 100.0% | 100.0% | 33.3% | 0.0770 |
| cube_80__constant_mass__fixed | 33.3% | 33.3% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0779 |
| cube_80__constant_mass__geometry_aware | 66.7% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0733 |
| cube_80__constant_density__fixed | 66.7% | 66.7% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0599 |
| cube_80__constant_density__geometry_aware | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 33.3% | 0.1207 |
| box_60x60x30__constant_mass__fixed | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.2179 |
| box_60x60x30__constant_mass__geometry_aware | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.2190 |
| box_60x60x30__constant_density__fixed | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.2179 |
| box_60x60x30__constant_density__geometry_aware | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.2189 |

## Interpretation boundary

This report separates geometry, mass strategy, and controller protocol. Constant-density variants change both mass and inertia; geometry-aware runs also change controller targets. The larger-object runs include a documented vertical placement correction and therefore are not literal raw-reset comparisons.

See `summary.json`, `configurations.csv`, `episodes.csv`, `selected_step_telemetry.csv`, and `geometry_feasibility.json` for measurements.
