"""
Hyperencoder Configuration System

This module contains Pydantic configuration models for the hyperencoder project.
It replaces the previous prefigure-based configuration system with a modern
Hydra + OmegaConf + Pydantic stack.
"""

from .base import BaseConfig
from .training import TrainingConfig
from .hydra_integration import (
    dictconfig_to_pydantic,
    pydantic_to_dictconfig,
    load_training_config,
    validate_and_resolve_paths,
    print_config_summary,
)

__all__ = [
    "BaseConfig",
    "TrainingConfig", 
    "dictconfig_to_pydantic",
    "pydantic_to_dictconfig",
    "load_training_config",
    "validate_and_resolve_paths",
    "print_config_summary",
] 