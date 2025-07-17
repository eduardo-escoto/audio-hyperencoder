import logging
from typing import Any

from pytorch_lightning.loggers import WandbLogger

from hyperencoder.datamodels.experiment_logging import ExperimentLoggingConfig


def create_experiment_logger_from_config(cfg: ExperimentLoggingConfig):
    logger = logging.getLogger(__name__)
    logger.info(f"Creating experiment logger from config: {cfg}")
    match cfg.type:
        case "wandb":
            return create_wandb_from_config(cfg.config)
        case _:
            raise ValueError(f"Unsupported logger type: {cfg.type}")

def create_wandb_from_config(cfg: dict[str, Any]) -> WandbLogger:
    return WandbLogger(**cfg)