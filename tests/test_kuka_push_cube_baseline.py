from dataclasses import replace
from pathlib import Path
import pytest
from src.config import config_from_dict, load_config
from src.env_adapters.kuka_v3 import kuka_v3_tasks

PILOT=Path("configs/sac/kuka_push_cube_v3_sawyer_v2_pilot_200k.yaml")
FULL=Path("configs/sac/kuka_push_cube_v3_sawyer_v2_1m.yaml")

def test_clean_cube_configs_differ_only_in_budget_and_output():
    pilot=load_config(PILOT,expected_experiment="sac")
    full=load_config(FULL,expected_experiment="sac")
    assert pilot.environment == full.environment
    assert pilot.continual == full.continual
    assert pilot.evaluation == full.evaluation
    assert pilot.runtime.seed == full.runtime.seed == 0
    assert pilot.sac == replace(full.sac,steps=200_000)
    assert not pilot.sac.early_stopping.enabled

def test_fixed_task_bank_is_deterministic():
    first=kuka_v3_tasks("kuka-push-cube-v3",10000)
    second=kuka_v3_tasks("kuka-push-cube-v3",10000)
    assert [task.data for task in first] == [task.data for task in second]
    assert len(first)==50

@pytest.mark.parametrize("field,value", [
    ("cube_push_curriculum_level",2), ("cube_push_initialization","contact"),
    ("near_contact_gap_m",.005),
])
def test_obsolete_environment_options_are_rejected(field,value):
    with pytest.raises(ValueError,match="not used"):
        config_from_dict({"experiment":"sac","environment":{"backend":"kuka-v3",field:value},
                          "continual":{"tasks":["kuka-push-cube-v3"]}},expected_experiment="sac")

def test_obsolete_reward_is_rejected():
    with pytest.raises(ValueError,match="reward_function_version"):
        config_from_dict({"experiment":"sac","environment":{"backend":"kuka-v3","reward_function_version":"v3_1"},
                          "continual":{"tasks":["kuka-push-cube-v3"]}},expected_experiment="sac")
