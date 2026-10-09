#!/usr/bin/env bash
set -euo pipefail

group="${1:-all}"
seed="${SEED:-0}"

easy=(
  dev_kuka_button_press_v3
  dev_kuka_drawer_close_v3
  dev_kuka_window_open_v3
  dev_kuka_window_close_v3
  dev_kuka_faucet_open_v3
  dev_kuka_door_open_v3
)
remaining=(
  dev_kuka_faucet_close_v3
  dev_kuka_reach_v3
  dev_kuka_handle_press_side_v3
  dev_kuka_hammer_v3
)

case "$group" in
  easy) configs=("${easy[@]}") ;;
  remaining) configs=("${remaining[@]}") ;;
  all) configs=("${easy[@]}" "${remaining[@]}") ;;
  *)
    echo "Usage: $0 [easy|remaining|all]" >&2
    exit 2
    ;;
esac

for name in "${configs[@]}"; do
  config="configs/sac/${name}.yaml"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${name} (seed ${seed})"
  python -u train_sac.py --config "$config" --seed "$seed"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] FINISH ${name} (seed ${seed})"
done
