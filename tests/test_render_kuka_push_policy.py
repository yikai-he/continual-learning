import importlib.util
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location("render_kuka_push_policy",Path("scripts/render_kuka_push_policy.py"))
replay=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(replay)

def test_clean_push_defaults():
    args=replay.parse_args([])
    assert args.env_name=="kuka-push-v3" and args.reward_version=="v2"

def test_cube_v2_is_supported():
    args=replay.parse_args(["--env-name","kuka-push-cube-v3","--reward-version","v2"])
    assert args.env_name=="kuka-push-cube-v3"

@pytest.mark.parametrize("version",["v3_1","v3_3_contact_push_progress","sawyer_push_v2_adapted"])
def test_obsolete_cube_rewards_are_rejected(version):
    with pytest.raises(SystemExit): replay.parse_args(["--env-name","kuka-push-cube-v3","--reward-version",version])
