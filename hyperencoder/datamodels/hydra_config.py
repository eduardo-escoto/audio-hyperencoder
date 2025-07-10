"""
Hydra and task configuration models for hyperencoder.

This module defines Pydantic models for Hydra configuration and task
configuration, ensuring these are also part of the single source of truth.
"""

from typing import Union

from pydantic import Field

from .base import BaseConfig

# Type for Hydra defaults - can be a string (like "_self_") or a dict (like {"data": "hyperencoder"})
HydraDefault = Union[str, dict[str, str]]


class HydraRunConfig(BaseConfig):
    """Configuration for Hydra run settings."""

    dir: str = Field(
        default="outputs/${project_name}/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}",
        description="Output directory for Hydra runs",
    )


class HydraJobConfig(BaseConfig):
    """Configuration for Hydra job settings."""

    name: str = Field(
        default="hyperencoder_${hydra:runtime.choices.model}",
        description="Job name template",
    )


class HydraSweepConfig(BaseConfig):
    """Configuration for Hydra sweep settings."""

    dir: str = Field(
        default="multirun/${project_name}/${now:%Y-%m-%d_%H-%M-%S}",
        description="Output directory for multirun sweeps",
    )

    subdir: str = Field(
        default="${hydra:job.num}", description="Subdirectory template for sweep runs"
    )


class HydraJobLoggingConfig(BaseConfig):
    """Configuration for Hydra job logging."""
    
    formatters: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "simple": {
                "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
            }
        },
        description="Logging formatters configuration",
    )
    
    handlers: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": "${hydra:runtime.output_dir}/hyperencoder.log"
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "colorlog": {
                "class": "colorlog.StreamHandler",
                "formatter": "colorlog",
                "stream": "ext://sys.stdout"
            }
        },
        description="Logging handlers configuration",
    )
    
    root: dict[str, str | list[str]] = Field(
        default_factory=lambda: {
            "level": "INFO",
            "handlers": ["colorlog", "file"]
        },
        description="Root logger configuration",
    )
    
    disable_existing_loggers: bool = Field(
        default=False,
        description="Whether to disable existing loggers",
    )


class HydraConfig(BaseConfig):
    """Complete Hydra configuration."""
    
    run: HydraRunConfig = Field(
        default_factory=HydraRunConfig,
        description="Configuration for Hydra run settings",
    )
    
    job: HydraJobConfig = Field(
        default_factory=HydraJobConfig,
        description="Configuration for Hydra job settings",
    )
    
    sweep: HydraSweepConfig = Field(
        default_factory=HydraSweepConfig,
        description="Configuration for Hydra sweep settings",
    )
    
    job_logging: HydraJobLoggingConfig = Field(
        default_factory=HydraJobLoggingConfig,
        description="Configuration for Hydra job logging",
    )
    
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging",
    )


class TrainTaskConfig(BaseConfig):
    """Configuration for train.yaml - complete training task configuration."""

    defaults: list[HydraDefault] = Field(
        default=[
            {"/data": "default"},
            {"/model": "default"},
            {"/training": "default"},
            {"/hydra": "default"},
            "_self_",
        ],
        description="Complete training task configuration composition",
    )

    experiment_name: str = Field(
        default="baseline_experiment", description="Name of the experiment"
    )

    description: str = Field(
        default="Basic hyperencoder experiment with default settings",
        description="Description of the task",
    )

    project_name: str = Field(
        default="audio-hyperencoder", description="Name of the project"
    )

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    task: str = Field(default="train", description="Task to execute")


class PreEncodeTaskConfig(BaseConfig):
    """Configuration for pre_encode.yaml - complete pre-encoding task configuration."""

    defaults: list[HydraDefault] = Field(
        default=[
            {"/data": "default"},
            {"/model": "default"},
            {"/pre_encode": "default"},
            {"/hydra": "default"},
            "_self_",
        ],
        description="Complete pre-encoding task configuration composition",
    )

    run_name: str = Field(default="baseline_run", description="Name of the run")

    description: str = Field(
        default="run of baseline pre encoder over dataset",
        description="Description of the task",
    )

    project_name: str = Field(
        default="audio-hyperencoder", description="Name of the project"
    )

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    task: str = Field(default="pre_encode", description="Task to execute")


# Legacy configs for backward compatibility - these will be removed eventually
class ExperimentConfig(BaseConfig):
    """DEPRECATED: Legacy experiment configuration. Use TrainTaskConfig instead."""

    defaults: list[HydraDefault] = Field(
        default=[
            {"/data": "default"},
            {"/model": "default"},
            {"/training": "default"},
            {"/hydra": "default"},
            "_self_",
        ],
        description="DEPRECATED: Legacy experiment configuration composition",
    )

    name: str = Field(
        default="hyperencoder_basic_experiment", description="Name of the experiment"
    )

    description: str = Field(
        default="Basic hyperencoder experiment with default settings",
        description="Description of the experiment",
    )

    tags: list[str] = Field(
        default=["basic", "hyperencoder", "baseline"],
        description="Tags for organizing experiments",
    )

    project_name: str = Field(
        default="audio-hyperencoder", description="Name of the project"
    )

    experiment_name: str = Field(
        default="baseline_experiment", description="Name of the experiment run"
    )

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")


class TaskConfig(BaseConfig):
    """DEPRECATED: Legacy task configuration. Use TrainTaskConfig or PreEncodeTaskConfig instead."""

    defaults: list[HydraDefault] = Field(
        default=[{"experiment": "default"}, "_self_"],
        description="DEPRECATED: Legacy task configuration composition",
    )

    task: str = Field(
        default="train", description="Task to execute (train, pre_encode, etc.)"
    )


class TrainingTaskConfig(TaskConfig):
    """DEPRECATED: Legacy training task configuration. Use TrainTaskConfig instead."""

    task: str = Field(default="train", description="Training task")


class MainConfig(BaseConfig):
    """DEPRECATED: Legacy main configuration. Use TrainTaskConfig instead."""

    defaults: list[HydraDefault] = Field(
        default=[
            {"/data": "default"},
            {"/model": "default"},
            {"/training": "default"},
            {"/experiment": "default"},
            "_self_",
        ],
        description="DEPRECATED: Legacy configuration composition",
    )

    project_name: str = Field(
        default="audio-hyperencoder", description="Name of the project"
    )

    experiment_name: str = Field(
        default="baseline_experiment", description="Name of the experiment run"
    )

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    task: str = Field(
        default="train", description="Task to execute (train, pre_encode, etc.)"
    )


class TrainingMainConfig(BaseConfig):
    """DEPRECATED: Legacy training main configuration. Use TrainTaskConfig instead."""

    defaults: list[HydraDefault] = Field(
        default=["config", "_self_"],
        description="DEPRECATED: Legacy training configuration composition",
    )
