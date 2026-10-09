#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for output in runs/kuka_diffcrl_5task runs/kuka_diffcrl_no_replay_5task; do
  if [[ -e "$output" ]]; then
    echo "[$(date --iso-8601=seconds)] Refusing to overwrite existing output: $output" >&2
    exit 1
  fi
done

echo "[$(date --iso-8601=seconds)] Starting formal KUKA DiffCRL"
python train_diffcrl.py --config configs/diffcrl/kuka_diffcrl_5task.yaml
echo "[$(date --iso-8601=seconds)] Completed formal KUKA DiffCRL"

echo "[$(date --iso-8601=seconds)] Starting formal KUKA No-Replay"
python train_diffcrl.py --config configs/diffcrl/kuka_diffcrl_no_replay_5task.yaml
echo "[$(date --iso-8601=seconds)] Completed formal KUKA No-Replay"
