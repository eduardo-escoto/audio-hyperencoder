"""Main CLI entrypoint for Audio Hyperencoder with task dispatch.

This module provides a single Hydra entrypoint that dispatches to different tasks
based on configuration. It supports all Hydra features including multirun, 
tab completion, and working directory management.
"""

import hydra
import logging
from omegaconf import DictConfig
from pathlib import Path

# Get the bundled configs path
config_path = str(Path(__file__).parent.parent / "configs")

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entrypoint with colorful logging and task dispatch.
    
    Args:
        cfg: Hydra configuration object containing task and other settings
        
    Raises:
        ValueError: If an unknown task is specified
    """
    logger = logging.getLogger(__name__)
    
    # Validate task
    valid_tasks = ["train", "pre_encode"]
    if cfg.task not in valid_tasks:
        error_msg = f"Unknown task: {cfg.task}. Available tasks: {', '.join(valid_tasks)}"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # Beautiful colored logs thanks to hydra_colorlog
    logger.info(f"🚀 Starting Audio Hyperencoder - Task: {cfg.task}")
    logger.info(f"📋 Configuration: {cfg.get('_target_', 'Default')}")
    
    try:
        if cfg.task == "train":
            logger.info("🎯 Launching training task...")
            from hyperencoder.cli.tasks.train import train_task
            train_task(cfg)
        elif cfg.task == "pre_encode":
            logger.info("🔄 Launching pre-encoding task...")
            from hyperencoder.cli.tasks.pre_encode import pre_encode_task
            pre_encode_task(cfg)
            
    except Exception as e:
        logger.error(f"💥 Task {cfg.task} failed: {str(e)}")
        raise
    
    logger.info(f"✅ Task {cfg.task} completed successfully!")

if __name__ == "__main__":
    main() 