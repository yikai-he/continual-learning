"""Task metadata and lazy, explicit environment switching; observations stay 39D."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

CM10 = (
    "hammer-v3",
    "push-wall-v3",
    "faucet-close-v3",
    "push-back-v3",
    "stick-pull-v3",
    "handle-press-side-v3",
    "push-v3",
    "shelf-place-v3",
    "window-close-v3",
    "peg-unplug-side-v3",
)
CM1 = CM10[:1]
CM3 = CM10[:3]
CM5 = CM10[:5]

# KUKA-v3 tasks at official Continual World CW10 positions 1, 3, 6, 9, and 10.
# This preserves relative order but is not the official CW10 benchmark.
KUKA_CW10_INSPIRED_5 = (
    "kuka-hammer-v3",
    "kuka-faucet-close-v3",
    "kuka-handle-press-side-v3",
    "kuka-window-close-v3",
    "kuka-peg-unplug-side-v3",
)

KUKA_CM4 = (
    "kuka-drawer-close-v3",
    "kuka-window-close-v3",
    "kuka-door-open-v3",
    "kuka-peg-unplug-side-v3",
)
KUKA_CM5_LEGACY = KUKA_CM4 + ("kuka-reach-v3",)

KUKA_CM10 = (
    "kuka-reach-v3",
    "kuka-push-cube-v3",
    "kuka-hammer-v3",
    "kuka-handle-press-side-v3",
    "kuka-button-press-topdown-v3",
    "kuka-faucet-open-v3",
    "kuka-drawer-close-v3",
    "kuka-window-close-v3",
    "kuka-door-open-v3",
    "kuka-peg-unplug-side-v3",
)
KUKA_CM5 = KUKA_CM10[:5]

# Named curricula are immutable aliases for exact task orders, not backend-specific
# templates. Backends validate these unchanged names before a run is created.
NAMED_SEQUENCES = MappingProxyType(
    {
        "cm1": CM1,
        "cm3": CM3,
        "cm5": CM5,
        "cm10": CM10,
        "kuka_cm4": KUKA_CM4,
        "kuka_cm5": KUKA_CM5,
        "kuka_cm10": KUKA_CM10,
        "kuka_cm5_legacy": KUKA_CM5_LEGACY,
        "kuka_cw10_inspired_5": KUKA_CW10_INSPIRED_5,
    }
)


def resolve_named_sequence(name: str) -> tuple[str, ...]:
    """Return an exact named curriculum or reject an unknown name."""
    if not isinstance(name, str):
        raise ValueError("continual.sequence must be a string.")
    try:
        return NAMED_SEQUENCES[name]
    except KeyError as error:
        supported = ", ".join(NAMED_SEQUENCES)
        raise ValueError(
            f"Unknown continual.sequence {name!r}; supported sequences: {supported}."
        ) from error


@dataclass(frozen=True)
class TaskSpec:
    """Map a stored task ID to a backend-validated environment name."""

    task_id: str
    task_name: str

    def __post_init__(self):
        if (
            not isinstance(self.task_id, str)
            or not self.task_id.strip()
            or not isinstance(self.task_name, str)
            or not self.task_name.strip()
        ):
            raise ValueError("Require nonempty task ID and task name strings.")


@dataclass(frozen=True)
class TaskSequence:
    """Ordered curriculum whose indices also label diffusion task conditions."""

    tasks: tuple[TaskSpec, ...]

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if not self.tasks or len({t.task_id for t in self.tasks}) != len(self.tasks):
            raise ValueError("Sequence must be nonempty with unique task IDs.")

    @classmethod
    def from_names(cls, names):
        return cls(tuple(TaskSpec(n, n) for n in names))

    @property
    def num_tasks(self):
        return len(self.tasks)

    @property
    def task_names(self):
        return [t.task_name for t in self.tasks]

    def task(self, index):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.num_tasks
        ):
            raise IndexError("Task index outside sequence.")
        return self.tasks[index]


# Legacy three-task pilot default; configured runs supply their own curriculum.
PILOT_SEQUENCE = TaskSequence.from_names(CM3)
