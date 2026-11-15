"""Regression contracts for the internal ContinualManipulation package."""

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from continual_manipulation import sequences
from continual_manipulation.environments import make_env
from continual_manipulation.task_banks import bank_hashes


ROOT = Path(__file__).parents[1]
BENCHMARK = ROOT / "packages/continual-manipulation/src/continual_manipulation"


def test_benchmark_import_does_not_load_algorithm_frameworks() -> None:
    code = """
import sys
import continual_manipulation
forbidden = ('torch', 'stable_baselines3', 'src.sac', 'src.continual')
loaded = sorted(name for name in sys.modules if name.startswith(forbidden))
if loaded:
    raise SystemExit(','.join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_benchmark_source_has_no_algorithm_imports() -> None:
    forbidden = (
        "stable_baselines3", "torch", "src.sac", "src.continual",
        "bc_policy", "diffcrl", "trajectory_diffusion",
        "trajectory_flow_matching", "replay_allocation",
    )
    imports = "\n".join(path.read_text() for path in BENCHMARK.rglob("*.py"))
    assert not any(name in imports for name in forbidden)


def test_named_sequence_identity_is_unchanged() -> None:
    assert sequences.resolve_named_sequence("cm10") is sequences.CM10
    sequence = sequences.TaskSequence.from_names(sequences.CM10)
    assert tuple(sequence.task_names) == sequences.CM10
    assert tuple(task.task_id for task in sequence.tasks) == sequences.CM10


def test_task_bank_hashes_remain_ordered_sha256_identities() -> None:
    tasks = [SimpleNamespace(data=b"first"), SimpleNamespace(data=b"second")]
    assert bank_hashes(tasks) == [
        hashlib.sha256(task.data).hexdigest() for task in tasks
    ]
