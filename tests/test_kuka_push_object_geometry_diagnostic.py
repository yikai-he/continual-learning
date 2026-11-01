import importlib.util
from pathlib import Path
import sys

import numpy as np


PATH = Path(__file__).parents[1] / "scripts/diagnostics/analyze_kuka_push_object_geometry.py"
SPEC = importlib.util.spec_from_file_location("geometry_diagnostic", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_box_mass_and_inertia_are_exact():
    assert np.isclose(MODULE.variant_mass("cube_40", "constant_density"), 0.75)
    assert np.isclose(MODULE.variant_mass("cube_60", "constant_density"), 2.53125)
    assert np.isclose(MODULE.variant_mass("cube_80", "constant_density"), 6.0)
    assert np.isclose(MODULE.variant_mass("box_60x60x30", "constant_density"), 1.265625)
    assert np.allclose(MODULE.box_inertia(0.75, (0.04, 0.04, 0.04)), (2e-4,) * 3)


def test_geometry_aware_controller_preserves_baseline():
    assert np.allclose(MODULE.controller_parameters("cube_40", "geometry_aware"), (0.060, 0.042))
    assert np.allclose(MODULE.controller_parameters("cube_80", "geometry_aware"), (0.080, 0.062))


def test_configuration_deduplication_and_contract():
    rows = MODULE.configurations()
    assert len(rows) == 13
    assert sum(r.geometry == "cube_40" for r in rows) == 1
    assert all(r.mass_kg > 0 and all(i > 0 for i in r.inertia_kg_m2) for r in rows)
