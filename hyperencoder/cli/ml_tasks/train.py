"""
Training task for hyperencoder using Hydra configuration.

This module provides the main training entry point for hyperencoder models,
configured via Hydra and using PyTorch Lightning for training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from hyperencoder.training.hyperencoder import HyperEncoderLightningModule
from hyperencoder.data.utils import create_datamodule_from_config


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function configured with Hydra.
    
    Args:
        cfg: Hydra configuration object containing all training parameters
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting hyperencoder training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed for reproducibility
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")
    
    # Initialize wandb if configured
    wandb_logger = None
    if cfg.get("logging", {}).get("use_wandb", False):
        wandb_logger = setup_wandb_logging(cfg)
    
    # Create data module
    logger.info("📊 Creating data module")
    data_config = cfg.get("data", {})
    datamodule = create_datamodule_from_config(data_config)
    
    # Create model
    logger.info("🧠 Creating model")
    model_config = cfg.get("model", {})
    training_config = cfg.get("training", {})
    demo_config = cfg.get("demo", {})
    
    model = HyperEncoderLightningModule(
        model_config=model_config,
        training_config=training_config,
        demo_config=demo_config,
    )
    
    # Set up callbacks
    callbacks = setup_callbacks(cfg)
    
    # Create trainer
    logger.info("🏋️ Creating trainer")
    trainer_config = cfg.get("trainer", {})
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=trainer_config.get("max_epochs", 100),
        accelerator=trainer_config.get("accelerator", "auto"),
        devices=trainer_config.get("devices", "auto"),
        strategy=trainer_config.get("strategy", "auto"),
        precision=trainer_config.get("precision", "32"),
        gradient_clip_val=trainer_config.get("gradient_clip_val", 0.0),
        gradient_clip_algorithm=trainer_config.get("gradient_clip_algorithm", "norm"),
        accumulate_grad_batches=trainer_config.get("accumulate_grad_batches", 1),
        val_check_interval=trainer_config.get("val_check_interval", 1.0),
        check_val_every_n_epoch=trainer_config.get("check_val_every_n_epoch", 1),
        enable_checkpointing=trainer_config.get("enable_checkpointing", True),
        enable_progress_bar=trainer_config.get("enable_progress_bar", True),
        enable_model_summary=trainer_config.get("enable_model_summary", True),
    )
    
    # Start training
    logger.info("🎯 Starting training")
    try:
        trainer.fit(model, datamodule)
        logger.info("✅ Training completed successfully")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        # Clean up wandb
        if wandb_logger:
            wandb.finish()


def setup_wandb_logging(cfg: DictConfig) -> WandbLogger:
    """Set up wandb logging with configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Configured WandbLogger instance
    """
    logging_config = cfg.get("logging", {})
    
    # Extract wandb configuration
    project_name = logging_config.get("project_name", "hyperencoder")
    experiment_name = logging_config.get("experiment_name", "default")
    tags = logging_config.get("tags", [])
    
    # Create wandb logger
    wandb_logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    logging.getLogger(__name__).info(f"Initialized wandb: {project_name}/{experiment_name}")
    return wandb_logger


def setup_callbacks(cfg: DictConfig) -> list[Any]:
    """Set up training callbacks.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_config = cfg.get("checkpointing", {})
    if checkpoint_config.get("enabled", True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_config.get("dirpath", "checkpoints"),
            filename=checkpoint_config.get("filename", "hyperencoder-{epoch:02d}-{val_loss:.2f}"),
            monitor=checkpoint_config.get("monitor", "val_loss"),
            mode=checkpoint_config.get("mode", "min"),
            save_top_k=checkpoint_config.get("save_top_k", 3),
            save_last=checkpoint_config.get("save_last", True),
            every_n_epochs=checkpoint_config.get("every_n_epochs", 1),
            verbose=checkpoint_config.get("verbose", True),
        )
        callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping_config = cfg.get("early_stopping", {})
    if early_stopping_config.get("enabled", False):
        early_stopping_callback = EarlyStopping(
            monitor=early_stopping_config.get("monitor", "val_loss"),
            mode=early_stopping_config.get("mode", "min"),
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.0),
            verbose=early_stopping_config.get("verbose", True),
        )
        callbacks.append(early_stopping_callback)
    
    return callbacks


if __name__ == "__main__":
    main()
