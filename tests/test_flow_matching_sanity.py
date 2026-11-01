import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/experiments/run_flow_matching_sanity.py"
SPEC = importlib.util.spec_from_file_location("flow_sanity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONFIG = Path(__file__).parents[1] / "configs/flow_matching_sanity/button_press_v3.yaml"


def test_full_and_debug_configuration_are_isolated():
    full = MODULE._load_config(CONFIG, False)
    debug = MODULE._load_config(CONFIG, True)
    assert full["flow"]["generated_trajectories"] == 200
    assert debug["flow"]["generated_trajectories"] == 8
    assert debug["output_dir"].endswith("_debug")
    assert full["output_dir"] != debug["output_dir"]


def test_unknown_configuration_field_is_rejected(tmp_path):
    text = CONFIG.read_text() + "unknown: true\n"
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="Unknown"):
        MODULE._load_config(path, False)
