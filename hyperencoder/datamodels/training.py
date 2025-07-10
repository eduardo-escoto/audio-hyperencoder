"""
Training configuration for the hyperencoder project.

This module defines the Pydantic model for training parameters,
replacing the previous INI-based configuration.
"""

from typing import Literal
from pathlib import Path

from pydantic import Field, field_validator

from .base import BaseConfig


class TrainingConfig(BaseConfig):
    """
    Configuration for training hyperencoder models.

    This replaces the hyperencoder/defaults/train_defaults.ini file
    with a modern, type-safe, validated configuration system.
    """

    # Run identification
    name: str = Field(
        default="vqvae_hyperencoder",
        description="Name of the training run for logging and checkpointing",
        examples=["vqvae_rotation_trick_hyperencoder", "basic_autoencoder_run"],
    )

    project: str = Field(
        default="hyperencoder",
        description="Project name for experiment tracking (e.g., WandB project)",
    )

    # Training hyperparameters
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for training",
        examples=[16, 32, 64, 128],
    )

    num_workers: int = Field(
        default=8, ge=0, le=32, description="Number of CPU workers for the DataLoader"
    )

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")

    accum_batches: int = Field(
        default=1, ge=1, description="Number of batches for gradient accumulation"
    )

    # Hardware configuration
    num_nodes: int = Field(
        default=1,
        ge=1,
        description="Number of compute nodes to use for distributed training",
    )

    devices: str = Field(
        default="auto",
        description="Device specification for PyTorch Lightning",
        examples=["auto", "gpu", "cpu", "1", "2", "[0,1]"],
    )

    strategy: Literal["auto", "ddp", "ddp_find_unused_parameters_true", "fsdp"] = Field(
        default="auto", description="Multi-GPU strategy for PyTorch Lightning"
    )

    precision: Literal["16-mixed", "bf16-mixed", "32-true", "64-true"] = Field(
        default="16-mixed", description="Precision to use for training"
    )

    persistent_workers: bool = Field(
        default=False,
        description="Whether to keep DataLoader workers persistent between epochs",
    )

    # Training loop control
    max_epochs: int = Field(
        default=10000000,
        ge=1,
        description="Maximum number of training epochs",
    )

    log_every_n_steps: int = Field(
        default=1,
        ge=1,
        description="Log metrics every N training steps",
    )

    # Checkpointing and validation
    checkpoint_every: int = Field(
        default=1, ge=1, description="Number of epochs between checkpoints"
    )

    val_every: int = Field(
        default=-1,
        ge=-1,
        description="Number of steps between validation runs (-1 to disable)",
    )

    save_top_k: int = Field(
        default=20, ge=-1, description="Save top K model checkpoints (-1 for all)"
    )

    recover: bool = Field(
        default=False, description="Attempt to resume training from latest checkpoint"
    )

    # Optimization
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-8,
        le=1.0,
        description="Learning rate for training",
    )

    gradient_clip_val: float = Field(
        default=0.0,
        ge=0.0,
        description="Gradient clipping value (0.0 disables clipping)",
    )

    # Paths (now using Path objects instead of hard-coded strings)
    model_config_path: Path | None = Field(
        default=None,
        description="Path to model configuration file",
        examples=["configs/models/hyperencoder_vqvae.json"],
    )

    dataset_config: Path | None = Field(
        default=None,
        description="Path to training dataset configuration file",
        examples=["configs/data/hyperencoder.json"],
    )

    val_dataset_config: Path | None = Field(
        default=None, description="Path to validation dataset configuration file"
    )

    save_dir: Path | None = Field(
        default=None,
        description="Directory to save checkpoints and model files",
        examples=["data/experiments/my_run"],
    )

    # Resume/checkpoint paths
    ckpt_path: Path | None = Field(
        default=None, description="Trainer checkpoint file to restart training from"
    )

    pretrained_ckpt_path: Path | None = Field(
        default=None, description="Model checkpoint file to start new training run from"
    )

    pretransform_ckpt_path: Path | None = Field(
        default=None, description="Checkpoint path for the pretransform model if needed"
    )

    # Training metadata
    run_id: str | None = Field(
        default=None, description="Unique identifier for this training run"
    )

    ckpt_name: str | None = Field(
        default=None, description="Specific checkpoint name to resume from"
    )

    # Logging
    logger: Literal["wandb", "tensorboard", "csv"] = Field(
        default="wandb", description="Logger type to use for experiment tracking"
    )

    # Legacy options
    remove_pretransform_weight_norm: bool = Field(
        default=False, description="Remove weight norm from the pretransform model"
    )

    @field_validator(
        "save_dir",
        "model_config_path",
        "dataset_config",
        "val_dataset_config",
        mode="before",
    )
    @classmethod
    def resolve_paths(cls, v):
        """Convert string paths to Path objects and resolve them."""
        if v is None or v == "":
            return None
        return Path(v).resolve() if not isinstance(v, Path) else v.resolve()
