"""
Hyperencoder Configuration System

This module contains Pydantic configuration models for the hyperencoder project.
It replaces the previous prefigure-based configuration system with a modern
Hydra + OmegaConf + Pydantic stack.
"""

from .base import BaseConfig
from .training import TrainingConfig
from .pre_encode_config import PreEncodeConfig
from .data_config import DataConfig, DatasetEntry, CropConfig
from .model_config import (
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    BottleneckConfig,
    OptimizerConfig,
    SchedulerConfig,
    OptimizerSchedulerConfig,
    DemoConfig,
)
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
    "PreEncodeConfig",
    "DataConfig",
    "DatasetEntry",
    "CropConfig",
    "ModelConfig",
    "EncoderConfig",
    "DecoderConfig",
    "BottleneckConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "OptimizerSchedulerConfig",
    "DemoConfig",
    "dictconfig_to_pydantic",
    "pydantic_to_dictconfig",
    "load_training_config",
    "validate_and_resolve_paths",
    "print_config_summary",
] 