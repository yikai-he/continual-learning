"""Run pretrained-expert DiffCRL from YAML settings and runtime overrides.

Physical trajectories and BC use 39D observations plus 4D actions. When replay
is enabled, diffusion uses a separate 22D dynamics encoding and 3D goal condition.
"""

from src.config import parse_config
from src.continual.diffcrl import DiffCRLTrainer


def main(argv=None):
    """Run every configured DiffCRL stage and print the final matrix path."""
    config = parse_config("diffcrl", __doc__, argv)
    trainer = DiffCRLTrainer(config)
    trainer.run()
    print(trainer.output / "evaluation_matrix.json")


if __name__ == "__main__":
    main()
