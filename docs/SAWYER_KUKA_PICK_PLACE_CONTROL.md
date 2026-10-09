# Sawyer–KUKA Pick Place Control

This control compares official MetaWorld Sawyer `pick-place-v3` against KUKA
`kuka-pick-place-v3` under the same 50k-step single-task SAC protocol.

Both experiments use seed 0, reward v2, a 200-step horizon, no termination on
success, identical SAC hyperparameters, deterministic evaluation every 5,000
steps, and 20 fixed evaluation seeds (10000–10019).

Sawyer config: `configs/sac/sawyer_pick_place_v3_short_50k.yaml`

Expected Sawyer run directory:
`runs/reward_v2/pick-place-v3/seed_0_short_50k/`

No result interpretation is made here because the Sawyer control has not been
run.
