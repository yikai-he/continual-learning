import hashlib
from dataclasses import dataclass

import pytest

from src.continual.task_bank import verify_disjoint_task_banks


@dataclass
class FakeTask:
    data: bytes


def provider(_task_name, seed):
    return [FakeTask(f"{seed}:{index}".encode()) for index in range(3)]


def test_disjoint_banks_pass_and_are_reproducible():
    first = verify_disjoint_task_banks("reach-v3", 0, 10000, task_provider=provider)
    second = verify_disjoint_task_banks("reach-v3", 0, 10000, task_provider=provider)
    assert first == second
    assert first["disjoint"]


def test_overlap_identifies_task_and_hash():
    shared = FakeTask(b"shared")

    def overlapping(_task_name, seed):
        return [shared, FakeTask(str(seed).encode())]

    digest = hashlib.sha256(b"shared").hexdigest()
    with pytest.raises(ValueError) as error:
        verify_disjoint_task_banks(
            "hammer-v3", 0, 10000, task_provider=overlapping
        )
    assert "hammer-v3" in str(error.value)
    assert digest in str(error.value)
