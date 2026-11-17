#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

case "${1:-}" in
  --preflight-only|--dry-run) execute=false ;;
  --run) execute=true ;;
  *) echo "Usage: $0 --preflight-only|--dry-run|--run" >&2; exit 2 ;;
esac
[[ $# -eq 1 ]] || { echo "Usage: $0 --preflight-only|--dry-run|--run" >&2; exit 2; }

python_bin="${PYTHON_BIN:-python}"
configs=(
  configs/sac/sawyer_stick_pull_continuation_early_stop.yaml
  configs/sac/sawyer_push_continuation_early_stop.yaml
  configs/sac/sawyer_push_back_from_faucet_close_early_stop.yaml
)
tasks=(stick-pull-v3 push-v3 push-back-v3)
source_steps=(1300000 1540000 560000)
outputs=(
  runs/sac_target_success_only_20261117/stick-pull-v3/seed_0
  runs/sac_target_success_only_20261117/push-v3/seed_0
  runs/sac_target_success_only_20261117/push-back-v3/seed_0
)

command -v "$python_bin" >/dev/null 2>&1 || { echo "Python not found: $python_bin" >&2; exit 127; }
"$python_bin" -c "import stable_baselines3, continual_manipulation" || {
  echo "Activate the project's MetaWorld environment or set PYTHON_BIN." >&2
  exit 1
}

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  task="${tasks[$index]}"
  source_step="${source_steps[$index]}"
  output="${outputs[$index]}"
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }
  [[ ! -e "$output" ]] || { echo "Refusing existing run: $output" >&2; exit 1; }
  "$python_bin" - "$config" "$task" "$source_step" "$output" <<'PY'
import json
import sys
import zipfile
from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from continual_manipulation.environments import make_env
from src.config import load_config, output_directory
from src.support.expert_manifest import load_expert_manifest, validate_expert_checkpoint
from src.support.reproducibility import file_hash
from train_sac import validate_fixed_evaluation_preflight

config_path, task = sys.argv[1], sys.argv[2]
expected_source_step = int(sys.argv[3])
expected_output = Path(sys.argv[4]).resolve()
cfg = load_config(config_path, expected_experiment="sac")
assert cfg.continual.tasks == [task]
assert cfg.environment.backend == "metaworld-v3"
assert cfg.environment.reward_function_version == "v2"
assert cfg.runtime.seed == 0
assert cfg.sac.early_stopping.enabled
assert cfg.sac.early_stopping.success_threshold == 1.0
assert cfg.sac.early_stopping.patience == 3
assert cfg.sac.early_stopping.plateau_patience is None
assert cfg.evaluation.mode == "fixed-tasks"
assert cfg.evaluation.episodes == 50
assert cfg.evaluation.frequency == 40_000
assert cfg.evaluation.task_set_seed == 10_000
assert cfg.evaluation.final_task_set_seeds == [20_000, 20_001]
assert output_directory(cfg) == expected_output

eval_env = Monitor(make_env(
    cfg.environment.backend,
    task,
    cfg.evaluation.task_set_seed,
    reward_function_version=cfg.environment.reward_function_version,
))
try:
    evaluation_check = validate_fixed_evaluation_preflight(
        eval_env,
        task_name=task,
        task_set_seed=cfg.evaluation.task_set_seed,
        episodes=cfg.evaluation.episodes,
        first_reset_seed=cfg.runtime.seed + cfg.evaluation.seed_offset,
        backend=cfg.environment.backend,
    )
finally:
    eval_env.close()
assert evaluation_check["task_bank_size"] == 50
assert evaluation_check["task_bank_repetitions"] == 1
assert evaluation_check["task_bank_indices"] == tuple(range(50))
assert evaluation_check["same_pair_reset_reproducible"] is True
print(
    "evaluation_preflight "
    f"task={task} bank_size={evaluation_check['task_bank_size']} "
    f"episodes={cfg.evaluation.episodes} "
    f"passes={evaluation_check['task_bank_repetitions']} "
    "explicit_selection=true "
    f"second_pass_seed_changed_observation="
    f"{evaluation_check['second_pass_reset_seed_changed_observation']}"
)

source_name = cfg.resume.checkpoint if cfg.resume.enabled else cfg.actor_init.checkpoint
source = Path(source_name).resolve()
manifest_path = source.parent / "manifest.json"
manifest = load_expert_manifest(manifest_path)
assert source.is_file()
assert file_hash(source) == manifest["checkpoint_sha256"]
with zipfile.ZipFile(source) as archive:
    source_data = json.loads(archive.read("data"))
assert source_data["num_timesteps"] == expected_source_step
assert manifest["backend"] == "metaworld-v3"
assert manifest["reward_function_version"] == "v2"
assert manifest["observation_shape"] == [39]
assert manifest["action_shape"] == [4]
assert manifest["horizon"] == 200
if cfg.resume.enabled:
    validate_expert_checkpoint(
        source,
        expected_task=task,
        expected_backend="metaworld-v3",
        expected_reward_function_version="v2",
        expected_observation_shape=(39,),
        expected_action_shape=(4,),
        expected_horizon=200,
        manifest_path=manifest_path,
        allow_legacy=False,
    )
else:
    assert cfg.actor_init.mode == "sac_actor"
    assert manifest["task_name"] == "faucet-close-v3"
PY
  echo "READY order=$((index + 1)) task=$task config=$config output=$output"
done

if [[ "$execute" == false ]]; then
  echo "Preflight complete; no training was launched."
  exit 0
fi

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  task="${tasks[$index]}"
  output="${outputs[$index]}"
  temporary="$(mktemp "${TMPDIR:-/tmp}/sawyer-target-success-${index}.XXXXXX")"
  status=0
  echo "[$(date --iso-8601=seconds)] START task=$task"
  "$python_bin" -u train_sac.py --config "$config" 2>&1 | tee "$temporary" || status="${PIPESTATUS[0]}"
  if [[ -d "$output" && ! -e "$output/console.log" ]]; then
    mv "$temporary" "$output/console.log"
  else
    echo "Console retained at $temporary" >&2
  fi
  [[ "$status" -eq 0 ]] || { echo "FAILED task=$task status=$status" >&2; exit "$status"; }
  echo "[$(date --iso-8601=seconds)] COMPLETE task=$task"
done
