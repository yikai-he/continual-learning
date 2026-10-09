"""Run pretrained-expert DiffCRL from YAML settings and runtime overrides.

Physical trajectories and BC use 39D observations plus 4D actions. When replay
is enabled, diffusion uses a separate 22D dynamics encoding and 3D goal condition.
"""

from src.config import parse_config
from src.continual.diffcrl import DiffCRLTrainer
from src.support.stage_sync import sync_completed_stage


def main(argv=None):
    """Run every configured DiffCRL stage and print the final matrix path."""
    def configure(parser):
        parser.add_argument(
            "--resume-from",
            type=str,
            default=None,
            help="Existing DiffCRL run directory to resume at its latest complete stage.",
        )
        parser.add_argument(
            "--stage-sync-to",
            type=str,
            default=None,
            help="Backup directory synchronized and verified after every completed stage.",
        )

    config, args = parse_config(
        "diffcrl", __doc__, argv, configure_parser=configure, return_args=True
    )
    callback = None
    if args.stage_sync_to:
        destination = args.stage_sync_to

        def callback(run_dir, stage, task_names):
            sync_completed_stage(run_dir, destination, task_names, stage)

    trainer = DiffCRLTrainer(
        config, resume_from=args.resume_from, stage_complete_callback=callback
    )
    trainer.run()
    print(trainer.output / "evaluation_matrix.json")


if __name__ == "__main__":
    main()
