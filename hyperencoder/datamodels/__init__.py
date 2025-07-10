"""
Hyperencoder Configuration System

This module contains Pydantic configuration models for the hyperencoder project.
It uses Hydra's structured configs for direct instantiation without conversion bridges.
"""

from .base import BaseConfig
from .training import TrainingConfig
from .data_config import CropConfig, DataConfig, DatasetEntry, MidiMetadataConfig
from .midi_metadata import MidiMetadata
from .hydra_config import (
    MainConfig,
    TaskConfig,
    HydraConfig,
    HydraJobConfig,
    HydraRunConfig,
    TrainTaskConfig,
    ExperimentConfig,
    HydraSweepConfig,
    TrainingMainConfig,
    TrainingTaskConfig,
    PreEncodeTaskConfig,
)
from .model_config import (
    DemoConfig,
    ModelConfig,
    DecoderConfig,
    EncoderConfig,
    OptimizerConfig,
    SchedulerConfig,
    BottleneckConfig,
    OptimizerSchedulerConfig,
)
from .auxiliary_heads import AuxiliaryHeadConfig
from .hydra_integration import (
    load_training_config,
    print_config_summary,
    validate_and_resolve_paths,
)
from .pre_encode_config import PreEncodeConfig

__all__ = [
    "BaseConfig",
    "TrainingConfig",
    "PreEncodeConfig",
    "DataConfig",
    "DatasetEntry",
    "CropConfig",
    "MidiMetadataConfig",
    "MidiMetadata",
    "ModelConfig",
    "EncoderConfig",
    "DecoderConfig",
    "BottleneckConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "OptimizerSchedulerConfig",
    "DemoConfig",
    "AuxiliaryHeadConfig",
    "HydraConfig",
    "HydraRunConfig",
    "HydraJobConfig",
    "HydraSweepConfig",
    "ExperimentConfig",
    "TaskConfig",
    "TrainingTaskConfig",
    "TrainTaskConfig",
    "PreEncodeTaskConfig",
    "MainConfig",
    "TrainingMainConfig",
    "load_training_config",
    "validate_and_resolve_paths",
    "print_config_summary",
]
