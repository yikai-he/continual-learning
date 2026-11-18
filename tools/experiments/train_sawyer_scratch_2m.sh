#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
seed=0
mode="${1:-}"
if [[ "$mode" != "--preflight-only" && "$mode" != "--run" ]]; then
  echo "Usage: $0 (--preflight-only|--run)" >&2
  exit 2
fi
shift
if (($#)); then
  echo "Usage: $0 (--preflight-only|--run)" >&2
  exit 2
fi

tasks=(push-back-v3 push-v3)
configs=(push_back_v3.yaml push_v3.yaml)
output_root="$repo_root/runs/sac_sawyer_scratch_2m"

run_with_console_log() {
  local run_dir="$1"
  shift
  local temporary status=0
  temporary="$(mktemp "${TMPDIR:-/tmp}/sawyer-scratch-2m-console.XXXXXX")"
  "$@" 2>&1 | tee "$temporary" || status="${PIPESTATUS[0]}"
  if [[ -d "$run_dir" ]]; then
    [[ ! -e "$run_dir/console.log" ]] || {
      echo "Refusing existing console log: $run_dir/console.log" >&2
      return 1
    }
    mv "$temporary" "$run_dir/console.log"
  else
    echo "Run directory was not created; console retained at $temporary" >&2
  fi
  return "$status"
}

command -v "$python_bin" >/dev/null 2>&1 || {
  echo "Python interpreter not found: $python_bin" >&2
  exit 127
}
[[ -f "$repo_root/train_sac.py" ]] || {
  echo "Missing training entry point: $repo_root/train_sac.py" >&2
  exit 1
}
"$python_bin" -c "import stable_baselines3, continual_manipulation" || {
  echo "Required SAC dependencies are unavailable via $python_bin" >&2
  exit 1
}

for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  config="$repo_root/configs/sac/sawyer_scratch_2m/${configs[$index]}"
  run_dir="$output_root/$task/seed_$seed"
  [[ -f "$config" ]] || { echo "Missing configuration: $config" >&2; exit 1; }
  "$python_bin" - "$config" "$task" <<'PY'
import json
import sys

from continual_manipulation.environments import horizon_for_backend, make_env
from continual_manipulation.evaluation import fixed_mt1_tasks
from src.config import load_config, output_directory

config = load_config(sys.argv[1], expected_experiment="sac")
task = sys.argv[2]
assert config.continual.tasks == [task]
assert config.environment.backend == "metaworld-v3"
assert config.environment.reward_function_version == "v2"
assert config.sac.steps == 2_000_000
assert config.actor_init.mode == "scratch" and config.actor_init.checkpoint is None
assert not config.resume.enabled and config.resume.checkpoint is None
stop = config.sac.early_stopping
assert stop.enabled and stop.min_steps == 0
assert stop.success_threshold == 1.0 and stop.patience == 3
assert stop.plateau_patience is None
assert config.evaluation.episodes == 50
assert config.evaluation.final_episodes == 50
assert config.evaluation.frequency == 40_000
assert config.evaluation.mode == "fixed-tasks"
assert config.evaluation.task_set_seed == 10_000
assert config.evaluation.final_task_set_seeds == [20_000, 20_001]
assert horizon_for_backend(config.environment.backend) == 200
assert output_directory(config).name == "seed_0"

env = make_env(
    config.environment.backend,
    task,
    config.runtime.seed,
    reward_function_version=config.environment.reward_function_version,
)
try:
    assert env.observation_space.shape == (39,)
    assert env.action_space.shape == (4,)
finally:
    env.close()
tasks, identities = fixed_mt1_tasks(task, config.evaluation.task_set_seed)
assert len(tasks) == len(identities) == 50
assert len({json.dumps(identity, sort_keys=True) for identity in identities}) == 50
PY
  [[ ! -e "$run_dir" ]] || { echo "Refusing existing run: $run_dir" >&2; exit 1; }
  echo "validated task=$task seed=$seed output=$run_dir"
done

if [[ "$mode" == "--preflight-only" ]]; then
  echo "Preflight only: no training was launched."
  exit 0
fi

for index in "${!tasks[@]}"; do
  task="${tasks[$index]}"
  config="configs/sac/sawyer_scratch_2m/${configs[$index]}"
  run_dir="$output_root/$task/seed_$seed"
  echo "[$(date --iso-8601=seconds)] START task=$task seed=$seed"
  if run_with_console_log "$run_dir" \
    "$python_bin" -u train_sac.py --config "$config"; then
    echo "[$(date --iso-8601=seconds)] COMPLETE task=$task seed=$seed"
  else
    status="$?"
    echo "[$(date --iso-8601=seconds)] FAILED task=$task seed=$seed status=$status" >&2
    exit "$status"
  fi
done
