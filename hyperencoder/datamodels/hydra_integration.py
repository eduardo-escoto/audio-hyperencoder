"""
Hydra integration utilities for hyperencoder datamodels.

This module provides utilities to load and validate configurations
from Hydra without needing complex bridge methods.
"""

import logging
from typing import Any, cast
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from .training import TrainingConfig
from .model_config import ModelConfig
from .data_config import DataConfig
from .pre_encode_config import PreEncodeConfig


def create_training_config_from_hydra(cfg: DictConfig) -> TrainingConfig:
    """
    Create a TrainingConfig from Hydra configuration.
    
    Args:
        cfg: The Hydra configuration containing training parameters
        
    Returns:
        A validated TrainingConfig instance
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Extract training config from the main config
        training_cfg = cfg.get("training", {})
        
        # Also extract top-level settings that belong in training config
        top_level_settings = {
            "project": cfg.get("project_name", "audio-hyperencoder"),
            "name": cfg.get("experiment_name", "default_experiment"),
            "seed": cfg.get("seed", 42),
        }
        
        # Convert OmegaConf to dict and merge with top-level settings
        if isinstance(training_cfg, DictConfig):
            training_dict = OmegaConf.to_container(training_cfg, resolve=True)
            if isinstance(training_dict, dict):
                # Merge dictionaries properly
                merged_dict = {**training_dict, **top_level_settings}
                logger.debug(f"Creating TrainingConfig with: {merged_dict}")
                return TrainingConfig(**cast(dict[str, Any], merged_dict))
        
        # Fallback to just top-level settings
        logger.debug(f"Creating TrainingConfig with defaults and: {top_level_settings}")
        return TrainingConfig(**top_level_settings)
        
    except Exception as e:
        logger.error(f"Failed to create TrainingConfig from Hydra config: {e}")
        logger.info("Using default TrainingConfig")
        return TrainingConfig()


def create_model_config_from_hydra(cfg: DictConfig) -> ModelConfig:
    """
    Create a ModelConfig from Hydra configuration.
    
    Args:
        cfg: The Hydra configuration containing model parameters
        
    Returns:
        A validated ModelConfig instance
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Extract model config from the main config
        model_cfg = cfg.get("model", {})
        
        if isinstance(model_cfg, DictConfig):
            model_dict = OmegaConf.to_container(model_cfg, resolve=True)
            if isinstance(model_dict, dict):
                logger.debug(f"Creating ModelConfig with: {model_dict}")
                return ModelConfig(**cast(dict[str, Any], model_dict))
        
        # Fallback to default
        logger.debug("Creating ModelConfig with defaults")
        return ModelConfig()
        
    except Exception as e:
        logger.error(f"Failed to create ModelConfig from Hydra config: {e}")
        logger.info("Using default ModelConfig")
        return ModelConfig()


def create_data_config_from_hydra(cfg: DictConfig) -> DataConfig:
    """
    Create a DataConfig from Hydra configuration.
    
    Args:
        cfg: The Hydra configuration containing data parameters
        
    Returns:
        A validated DataConfig instance
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Extract data config from the main config
        data_cfg = cfg.get("data", {})
        
        if isinstance(data_cfg, DictConfig):
            data_dict = OmegaConf.to_container(data_cfg, resolve=True)
            if isinstance(data_dict, dict):
                logger.debug(f"Creating DataConfig with: {data_dict}")
                return DataConfig(**cast(dict[str, Any], data_dict))
        
        # Fallback to default
        logger.debug("Creating DataConfig with defaults")
        return DataConfig()
        
    except Exception as e:
        logger.error(f"Failed to create DataConfig from Hydra config: {e}")
        logger.info("Using default DataConfig")
        return DataConfig()


def create_pre_encode_config_from_hydra(cfg: DictConfig) -> PreEncodeConfig:
    """
    Create a PreEncodeConfig from Hydra configuration.
    
    Args:
        cfg: The Hydra configuration containing pre-encode parameters
        
    Returns:
        A validated PreEncodeConfig instance
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Extract pre_encode config from the main config
        pre_encode_cfg = cfg.get("pre_encode", {})
        
        if isinstance(pre_encode_cfg, DictConfig):
            pre_encode_dict = OmegaConf.to_container(pre_encode_cfg, resolve=True)
            if isinstance(pre_encode_dict, dict):
                logger.debug(f"Creating PreEncodeConfig with: {pre_encode_dict}")
                return PreEncodeConfig(**cast(dict[str, Any], pre_encode_dict))
        
        # Fallback to default
        logger.debug("Creating PreEncodeConfig with defaults")
        return PreEncodeConfig()
        
    except Exception as e:
        logger.error(f"Failed to create PreEncodeConfig from Hydra config: {e}")
        logger.info("Using default PreEncodeConfig")
        return PreEncodeConfig()


def setup_hydra_logging() -> logging.Logger:
    """
    Set up logging using Hydra's configuration.
    
    Returns:
        The root logger configured by Hydra
    """
    # Hydra automatically configures logging, so we just get the logger
    logger = logging.getLogger(__name__)
    logger.debug("Using Hydra-managed logging configuration")
    return logger


def get_experiment_output_dir(cfg: DictConfig, logger_name: str | None = None) -> Path:
    """
    Get the output directory for the experiment in the format:
    project/experiment_name/experiment_id
    
    Args:
        cfg: The Hydra configuration
        logger_name: Optional logger name (wandb/comet) to extract experiment ID
        
    Returns:
        Path to the experiment output directory
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Extract basic info
        project_name = cfg.get("project_name", "audio-hyperencoder")
        experiment_name = cfg.get("experiment_name", "default_experiment")
        
        # Get base save directory
        base_save_dir = Path(cfg.get("training", {}).get("save_dir", "outputs"))
        
        # Create nested structure: project/experiment_name/experiment_id
        if logger_name == "wandb":
            # For wandb, we'll use the run ID once available
            experiment_dir = base_save_dir / project_name / experiment_name
        else:
            # For other loggers or no logger, use timestamp or default
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = base_save_dir / project_name / experiment_name / timestamp
        
        logger.debug(f"Experiment output directory: {experiment_dir}")
        return experiment_dir
        
    except Exception as e:
        logger.error(f"Failed to determine experiment output directory: {e}")
        return Path("outputs") / "default"


def load_training_config(cfg: DictConfig) -> TrainingConfig:
    """
    Load and validate a training configuration from Hydra config.
    
    Args:
        cfg: The Hydra configuration containing training parameters

    Returns:
        A validated TrainingConfig instance
        
    Deprecated: Use create_training_config_from_hydra instead
    """
    logger = logging.getLogger(__name__)
    logger.warning("load_training_config is deprecated, use create_training_config_from_hydra instead")
    return create_training_config_from_hydra(cfg)


def validate_and_resolve_paths(config: TrainingConfig) -> TrainingConfig:
    """
    Validate and resolve all paths in the configuration.

    Args:
        config: The training configuration to validate

    Returns:
        The validated configuration with resolved paths
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Since we're using Pydantic, validation happens automatically
        # We just need to resolve relative paths if needed

        if config.save_dir and not config.save_dir.is_absolute():
            config.save_dir = Path.cwd() / config.save_dir
            logger.debug(f"Resolved save_dir to: {config.save_dir}")

        if config.model_config_path and not config.model_config_path.is_absolute():
            config.model_config_path = Path.cwd() / config.model_config_path
            logger.debug(f"Resolved model_config_path to: {config.model_config_path}")

        return config
        
    except Exception as e:
        logger.error(f"Failed to validate and resolve paths: {e}")
        raise


def print_config_summary(config: TrainingConfig) -> None:
    """
    Print a summary of the training configuration.

    Args:
        config: The training configuration to summarize
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🔧 Training Configuration Summary:")
    logger.info(f"  📝 Project: {config.project}")
    logger.info(f"  🎯 Experiment: {config.name}")
    logger.info(f"  📊 Batch Size: {config.batch_size}")
    logger.info(f"  👥 Workers: {config.num_workers}")
    logger.info(f"  🎲 Seed: {config.seed}")
    logger.info(f"  📊 Strategy: {config.strategy}")
    logger.info(f"  💾 Save Dir: {config.save_dir}")
    logger.info(f"  ⚡ Precision: {config.precision}")
    logger.info(f"  🔄 Checkpoint Every: {config.checkpoint_every} epochs")
    logger.info(f"  🔄 Persistent Workers: {config.persistent_workers}")
