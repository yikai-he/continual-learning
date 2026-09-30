"""One persistent SAC, sequential tasks, no cross-task replay. No expert loading.

Each task gets a fixed step budget. Boundary warm-up uses random actions and
withholds updates; initial learning_starts is configured separately. Global timesteps persist.
ReplayBuffer.reset logically discards all prior-task transitions; old storage is
not sampled. An unfinished episode at a budget boundary is discarded with it.
"""

from src.config import ExperimentConfig, parse_config
from src.continual.sequential_sac import run


def parse_args(argv=None) -> ExperimentConfig:
    return parse_config("continual_sac", __doc__, argv, config_required=True)


if __name__ == "__main__":
    run(parse_args())
