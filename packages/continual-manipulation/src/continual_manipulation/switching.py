"""Lazy, explicit environment ownership and task switching."""

from .environments import EnvironmentSettings, make_env
from .sequences import PILOT_SEQUENCE


class TaskSwitcher:
    """Own at most one environment, closing it before each task switch."""

    def __init__(self, sequence=PILOT_SEQUENCE, *, factory=make_env,
                 backend="metaworld-v3", reward_function_version="v2",
                 render_mode=None):
        self.sequence = sequence
        self.factory = factory
        self.factory_uses_backend = factory is make_env
        self.settings = EnvironmentSettings(
            backend=backend, reward_function_version=reward_function_version
        )
        self.backend = backend
        self.env = None
        self.index = None
        self.reward_function_version = reward_function_version
        self.render_mode = render_mode

    def switch(self, index, *, seed):
        spec = self.sequence.task(index)
        self.close()
        args = ((self.backend, spec.task_name, seed) if self.factory_uses_backend
                else (spec.task_name, seed))
        kwargs = (self.settings.kwargs() if self.factory_uses_backend else
                  {"reward_function_version": self.reward_function_version})
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
