from os import path
from typing import Any

from hyperencoder.datamodels.training import TrainingConfig
from hyperencoder.datamodels.experiment_logging import ExperimentLoggingConfig


def get_strategy_from_config(cfg: TrainingConfig):
    match cfg.strategy:
        case "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy

            strategy = DeepSpeedStrategy(
                **cfg.strategy_kwargs # pyright: ignore[reportAny]
            )
        case _:
            strategy = "ddp_find_unused_parameters_true" if cfg.num_gpus > 1 else "auto"

    return strategy



def get_checkpoint_dir(cfg: ExperimentLoggingConfig, experiment_logger: Any):
    match cfg.type:
        case "wandb":
            return path.join(
                cfg.config["save_dir"],
                experiment_logger.experiment.project,
                experiment_logger.experiment.id,
                "checkpoints",
            )
        case _:
            return None
        # Initialize the wandb or comet logger first to get the experiment ID
    # logger.watch(None)  # Watch can be updated later when the model is created

        # if args.save_dir and isinstance(logger.experiment.id, str):
        #     checkpoint_dir = path.join(
        #         args.save_dir,
        #         logger.experiment.project,
        #         logger.experiment.id,
        #         "checkpoints",
        #     )
        #     log_dir = path.join(
        #         args.save_dir,
        #         logger.experiment.project,
        #         logger.experiment.id,
        #         "logs",
        #     )
    #     else:
    #         checkpoint_dir = None
    #         log_dir = args.save_dir
    # else:
    #     logger = None
    #     checkpoint_dir = args.save_dir if args.save_dir else None
    #     log_dir = args.save_dir
