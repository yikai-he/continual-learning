# KUKA Cube Push Object-Geometry Diagnostic

Generated: `20261013_225416_042580`

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

See `summary.json`, `configurations.csv`, `episodes.csv`, `selected_step_telemetry.csv`, and `geometry_feasibility.json` for measurements.
