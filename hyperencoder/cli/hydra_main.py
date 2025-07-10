"""
Main CLI entrypoint for hyperencoder.

This module provides the main Hydra-based CLI that dispatches to various tasks
like training and pre-encoding. It uses the new task-specific configuration
architecture with experiments.
"""

import logging

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main entrypoint with task dispatch."""
    logger = logging.getLogger(__name__)

    # Beautiful colored logs thanks to hydra_colorlog
    logger.info(f"🚀 Starting {cfg.task} task")

    if cfg.task == "train":
        from hyperencoder.cli.ml_tasks.train import train_task

        train_task(cfg)
    elif cfg.task == "pre_encode":
        from hyperencoder.cli.ml_tasks.pre_encode import pre_encode_task

        pre_encode_task(cfg)
    else:
        logger.error(f"❌ Unknown task: {cfg.task}")
        raise ValueError(f"Unknown task: {cfg.task}")


if __name__ == "__main__":
    main()
