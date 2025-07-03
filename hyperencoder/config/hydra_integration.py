"""
Hydra integration for hyperencoder configuration system.

This module provides utilities to convert between Hydra/OmegaConf DictConfig
objects and our Pydantic configuration models.
"""

from typing import Any, TypeVar, Type, Optional, cast
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from .base import BaseConfig
from .training import TrainingConfig


T = TypeVar('T', bound=BaseConfig)


def dictconfig_to_pydantic(cfg: DictConfig, model_class: Type[T]) -> T:
    """
    Convert a Hydra/OmegaConf DictConfig to a Pydantic model.
    
    Args:
        cfg: The OmegaConf DictConfig to convert
        model_class: The Pydantic model class to instantiate
        
    Returns:
        An instance of the specified Pydantic model
        
    Raises:
        ValidationError: If the configuration is invalid
    """
    # Convert OmegaConf to plain dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Handle None values and convert to proper types
    if cfg_dict is None or not isinstance(cfg_dict, dict):
        cfg_dict = {}
    
    # Create and validate the Pydantic model
    try:
        return model_class(**cast(dict[str, Any], cfg_dict))
    except ValidationError as e:
        print(f"❌ Configuration validation failed for {model_class.__name__}:")
        print(f"   {e}")
        raise


def pydantic_to_dictconfig(model: BaseConfig) -> DictConfig:
    """
    Convert a Pydantic model to a Hydra/OmegaConf DictConfig.
    
    Args:
        model: The Pydantic model to convert
        
    Returns:
        A DictConfig representation of the model
    """
    # Convert to dict first
    model_dict = model.model_dump()
    
    # Convert Path objects to strings for OmegaConf compatibility
    def path_to_str(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: path_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [path_to_str(item) for item in obj]
        return obj
    
    cleaned_dict = path_to_str(model_dict)
    
    # Create OmegaConf DictConfig
    return cast(DictConfig, OmegaConf.create(cleaned_dict))


def load_training_config(cfg: DictConfig) -> TrainingConfig:
    """
    Load and validate a training configuration from Hydra config.
    
    Args:
        cfg: The Hydra configuration containing training parameters
        
    Returns:
        A validated TrainingConfig instance
    """
    # Extract the training configuration
    training_cfg = cfg.get('training', {})
    if not isinstance(training_cfg, DictConfig):
        training_cfg = OmegaConf.create(training_cfg)
    
    # Convert to Pydantic model
    return dictconfig_to_pydantic(cast(DictConfig, training_cfg), TrainingConfig)


def validate_and_resolve_paths(config: TrainingConfig, base_path: Optional[Path] = None) -> TrainingConfig:
    """
    Validate and resolve relative paths in the configuration.
    
    Args:
        config: The training configuration to process
        base_path: Base path for resolving relative paths (defaults to current directory)
        
    Returns:
        A new TrainingConfig with resolved paths
    """
    if base_path is None:
        base_path = Path.cwd()
    
    # Get the current config as a dict, but bypass validation to get raw values
    config_dict = config.model_dump()
    
    # Path fields that need resolution
    path_fields = ['model_config_path', 'dataset_config', 'val_dataset_config', 
                   'save_dir', 'ckpt_path', 'pretrained_ckpt_path', 'pretransform_ckpt_path']
    
    for field in path_fields:
        if config_dict.get(field) is not None:
            current_path = config_dict[field]
            
            # If we have a base_path different from current directory,
            # and the path looks like it was originally relative,
            # re-resolve it against the new base_path
            if isinstance(current_path, Path):
                # Check if this path was likely resolved from a relative path
                current_cwd = Path.cwd()
                if (base_path != current_cwd and 
                    current_path.is_absolute() and 
                    str(current_path).startswith(str(current_cwd))):
                    
                    # Try to extract the relative part and re-resolve
                    try:
                        relative_part = current_path.relative_to(current_cwd)
                        config_dict[field] = base_path / relative_part
                    except ValueError:
                        # If we can't make it relative, keep the original
                        pass
    
    # Create new config with resolved paths, bypassing validator to avoid double resolution
    new_config = TrainingConfig.model_validate(config_dict)
    return new_config


def print_config_summary(config: TrainingConfig) -> None:
    """
    Print a nice summary of the configuration.
    
    Args:
        config: The training configuration to summarize
    """
    print("🔧 Training Configuration Summary")
    print("=" * 50)
    print(f"📋 Run Name: {config.name}")
    print(f"📊 Project: {config.project}")
    print(f"🎯 Batch Size: {config.batch_size}")
    print(f"👥 Workers: {config.num_workers}")
    print(f"🎲 Seed: {config.seed}")
    print(f"💾 Strategy: {config.strategy}")
    print(f"🔢 Precision: {config.precision}")
    print(f"📁 Save Dir: {config.save_dir}")
    print(f"⚙️  Model Config: {config.model_config_path}")
    print(f"📈 Logger: {config.logger}")
    print("=" * 50) 