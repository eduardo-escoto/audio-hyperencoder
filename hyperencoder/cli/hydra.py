"""
Main CLI entrypoint for hyperencoder.

This module provides the main Hydra-based CLI that dispatches to various tasks
like training and pre-encoding. It uses the new task-specific configuration
architecture with experiments.
"""

import logging
from os import path
from warnings import filterwarnings

import hydra
from omegaconf import OmegaConf, DictConfig

from hyperencoder.datamodels.tasks import TrainingTaskConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="config")  # pyright: ignore[reportAny]
def main(cfg: DictConfig) -> None:
    """Main entrypoint with task dispatch."""
    logger = logging.getLogger(__name__)
    logger.info(f"Current working directory: {path.abspath(path.curdir)}")
    # Filtering Warnings
    filterwarnings(
        "ignore", category=FutureWarning, module="vector_quantize_pytorch"
    )
    filterwarnings("ignore", category=FutureWarning, module="torch")
    filterwarnings(
        "ignore", 
        category=UserWarning, 
        module="torchmetrics.utilities.imports",
        # message=".*pkg_resources is deprecated.*"
    )

    o_cfg = OmegaConf.to_container(cfg, resolve=True)

    logger.info("🚀 Starting hyperencoder!")
    match cfg.task: # pyright: ignore[reportAny]
        case "train":
            from hyperencoder.core.training import train
            # Should get a TrainingTaskModel from the config
            t_cfg = TrainingTaskConfig.model_validate(o_cfg)

            train(t_cfg)
        case "pre_encode":
            # Should get a PreEncodingTaskModel from the config
            from hyperencoder.core.pre_encoding import pre_encode
            pre_encode(cfg)
        case _:  # pyright: ignore[reportAny]
            logger.info("Available tasks: train, pre_encode")

if __name__ == "__main__":
    main()
