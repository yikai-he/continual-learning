#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

case "${1:-}" in
  --dry-run|--preflight-only) execute=false ;;
  --run) execute=true ;;
  *) echo "Usage: $0 --dry-run|--preflight-only|--run" >&2; exit 2 ;;
esac
[[ $# -eq 1 ]] || { echo "Usage: $0 --dry-run|--preflight-only|--run" >&2; exit 2; }

python_bin="${PYTHON_BIN:-python}"
tasks=(peg-unplug-side-v3 stick-pull-v3 push-v3 push-wall-v3)
configs=(peg_unplug_side.yaml stick_pull.yaml push.yaml push_wall.yaml)
budgets=(300000 500000 1000000 1000000)
output_root="$repo_root/runs/sac_continuation/sawyer"

command -v "$python_bin" >/dev/null 2>&1 || {
  echo "Python interpreter not found: $python_bin" >&2
  exit 127
}
"$python_bin" -c "import stable_baselines3, continual_manipulation" || {
  echo "Required SAC dependencies are unavailable; activate the metaworld Conda environment." >&2
  exit 1
}

for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  config="$repo_root/configs/sac/continuation/sawyer/${configs[$index]}"
  run_dir="$output_root/$task/seed_0"
  [[ -f "$config" ]] || { echo "Missing configuration: $config" >&2; exit 1; }
  [[ ! -e "$run_dir" ]] || { echo "Refusing existing run: $run_dir" >&2; exit 1; }
  "$python_bin" - "$config" "$task" "${budgets[$index]}" "$run_dir" <<'PY'
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from src.config import load_config, output_directory
from src.support.expert_manifest import validate_expert_checkpoint, load_expert_manifest

path, task, budget, expected_output = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]).resolve()
cfg = load_config(path, expected_experiment="sac")
assert cfg.continual.tasks == [task]
assert cfg.environment.backend == "metaworld-v3"
assert cfg.environment.reward_function_version == "v2"
assert cfg.runtime.seed == 0
assert cfg.sac.steps == budget
assert cfg.actor_init.mode == "scratch" and cfg.actor_init.checkpoint is None
assert cfg.resume.enabled and cfg.resume.replay_buffer is None
assert cfg.resume.refill_steps == 10_000
assert cfg.sac.early_stopping.enabled
assert cfg.sac.early_stopping.min_steps == 50_000
assert cfg.sac.early_stopping.success_threshold == 1.0
assert cfg.sac.early_stopping.patience == 3
assert cfg.sac.early_stopping.plateau_patience is None
assert cfg.evaluation.episodes == 50
assert cfg.evaluation.final_episodes == 100
assert cfg.evaluation.frequency == 20_000
assert cfg.evaluation.mode == "fixed-tasks"
assert cfg.evaluation.task_set_seed == 10_000
assert cfg.evaluation.final_task_set_seeds == [20_000]
assert output_directory(cfg) == expected_output
checkpoint = Path(cfg.resume.checkpoint).resolve()
manifest_path = checkpoint.parent / "manifest.json"
manifest = load_expert_manifest(manifest_path)
assert manifest.get("reward_configuration") is None
validate_expert_checkpoint(
    checkpoint,
    expected_task=task,
    expected_backend="metaworld-v3",
    expected_reward_function_version="v2",
    expected_observation_shape=(39,),
    expected_action_shape=(4,),
    expected_horizon=200,
    allow_legacy=False,
)
model = SAC.load(checkpoint, device="cpu", custom_objects={"buffer_size": 1})
assert type(model.policy).__module__ + "." + type(model.policy).__name__ == "src.sac_policy.LegacySACPolicy"
assert model.observation_space.shape == (39,) and model.action_space.shape == (4,)
assert np.array_equal(model.action_space.low, -np.ones(4))
assert np.array_equal(model.action_space.high, np.ones(4))
PY
  echo "order=$((index + 1)) task=$task budget=${budgets[$index]} output=$run_dir"
  printf '  %q ' "$python_bin" -u train_sac.py --config "$config"
  printf '\n'
done

if [[ "$execute" == false ]]; then
  echo "Preflight only: all four continuations are ready; no training was launched."
  exit 0
fi

run_with_console_log() {
  local run_dir="$1"
  shift
  local temporary status=0
  temporary="$(mktemp "${TMPDIR:-/tmp}/sawyer-continuation-console.XXXXXX")"
  "$@" 2>&1 | tee "$temporary" || status="${PIPESTATUS[0]}"
  if [[ -d "$run_dir" ]]; then
    [[ ! -e "$run_dir/console.log" ]] || { echo "Refusing existing console log: $run_dir/console.log" >&2; return 1; }
    mv "$temporary" "$run_dir/console.log"
  else
    echo "Run directory was not created; console retained at $temporary" >&2
  fi
  return "$status"
}

for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  config="configs/sac/continuation/sawyer/${configs[$index]}"
  run_dir="$output_root/$task/seed_0"
  echo "[$(date --iso-8601=seconds)] START task=$task"
  run_with_console_log "$run_dir" "$python_bin" -u train_sac.py --config "$config" || {
    status="$?"; echo "FAILED task=$task status=$status" >&2; exit "$status";
  }
  echo "[$(date --iso-8601=seconds)] COMPLETE task=$task"
  if ((index + 1 < ${#tasks[@]})); then
    echo "NEXT task=${tasks[$((index + 1))]}"
  else
    echo "All prepared Sawyer continuation runs completed."
  fi
done
