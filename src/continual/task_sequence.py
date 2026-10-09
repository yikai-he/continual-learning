"""Task metadata and lazy, explicit environment switching; observations stay 39D."""

from __future__ import annotations

from dataclasses import dataclass

from src.envs import EnvironmentSettings, make_env


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
PILOT_SEQUENCE = TaskSequence.from_names(["reach-v3", "push-v3", "hammer-v3"])


class TaskSwitcher:
    """Own at most one environment. Switching closes it and creates a fresh one.

    The caller explicitly resets the returned environment. Do not use this owner
    to close an environment that is still attached to a training algorithm.
    """

    def __init__(
        self,
        sequence=PILOT_SEQUENCE,
        *,
        factory=make_env,
        backend="metaworld-v3",
        reward_function_version="v2",
        hammer_reward_variant="original",
        hammer_nail_progress_weight=0.0,
        render_mode=None,
    ):
        self.sequence = sequence
        self.factory = factory
        self.factory_uses_backend = factory is make_env
        self.settings = EnvironmentSettings(
            backend=backend,
            reward_function_version=reward_function_version,
            hammer_reward_variant=hammer_reward_variant,
            hammer_nail_progress_weight=hammer_nail_progress_weight,
        )
        self.backend = backend
        self.env = None
        self.index = None
        self.reward_function_version = reward_function_version
        self.render_mode = render_mode

    def switch(self, index, *, seed):
        spec = self.sequence.task(index)
        self.close()
        args = (
            (self.backend, spec.task_name, seed)
            if self.factory_uses_backend
            else (spec.task_name, seed)
        )
        kwargs = (
            self.settings.kwargs()
            if self.factory_uses_backend
            else {"reward_function_version": self.reward_function_version}
        )
        if self.render_mode is not None:
            kwargs["render_mode"] = self.render_mode
        self.env = self.factory(*args, **kwargs)
        self.index = index
        return self.env

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env = None
        self.index = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
