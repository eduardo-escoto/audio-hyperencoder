"""
Data configuration models for hyperencoder.

This module defines the Pydantic models for dataset configurations,
replacing the previous JSON-based data configs.
"""

from typing import Literal
from pathlib import Path

from pydantic import Field, field_validator

from .base import BaseConfig


class CropConfig(BaseConfig):
    """Configuration for data cropping/augmentation."""

    random_crop: bool = Field(
        default=True, description="Whether to apply random cropping to the data"
    )

    crop_ratio: int = Field(
        default=8, ge=1, le=32, description="Ratio for cropping the data"
    )

    original_crop_length: int = Field(
        default=47, ge=1, description="Original length for cropping operations"
    )


class MidiMetadataConfig(BaseConfig):
    """Configuration for MIDI metadata injection."""
    
    enabled: bool = Field(
        default=False,
        description="Whether to enable MIDI metadata injection"
    )
    
    midi_dir: str | None = Field(
        default=None,
        description="Directory containing MIDI files"
    )
    
    default_midi_extension: str = Field(
        default=".mid",
        description="Default MIDI file extension to try",
        examples=[".mid", ".midi"]
    )
    
    alternative_extensions: list[str] = Field(
        default_factory=lambda: [".mid", ".midi", ".MID", ".MIDI"],
        description="Alternative MIDI file extensions to try if default fails"
    )
    
    fail_on_missing_midi: bool = Field(
        default=True,
        description="Whether to fail if MIDI file is not found"
    )
    
    # Advanced metadata extraction options
    extract_harmony: bool = Field(
        default=False,
        description="Whether to extract harmonic analysis (chord progressions, key centers)"
    )
    
    extract_rhythm: bool = Field(
        default=False,
        description="Whether to extract rhythmic patterns"
    )
    
    # Caching options
    enable_caching: bool = Field(
        default=True,
        description="Whether to cache extracted MIDI metadata"
    )
    
    cache_dir: str | None = Field(
        default=None,
        description="Directory to store cached MIDI metadata (defaults to system temp)"
    )
    
    cache_max_age_days: int = Field(
        default=30,
        ge=1,
        description="Maximum age of cached metadata in days"
    )

    @field_validator("midi_dir", mode="before")
    @classmethod
    def resolve_midi_dir(cls, v):
        """Convert string paths to resolved paths."""
        if v is None or v == "":
            return None
        # Don't resolve paths with OmegaConf interpolations
        if isinstance(v, str) and "${" in v:
            return v
        return str(Path(v).resolve()) if not isinstance(v, Path) else str(v.resolve())


class DatasetEntry(BaseConfig):
    """Configuration for a single dataset entry."""

    path: str = Field(
        description="Path to the dataset directory or file",
        examples=[
            "/path/to/dataset",
            "./data/pre_encoded",
            "${oc.env:DATA_ROOT}/pre_encoded_babyslakh",
        ],
    )

    name: str | None = Field(default=None, description="Optional name for the dataset")

    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for this dataset when combining multiple datasets",
    )


class DataConfig(BaseConfig):
    """
    Configuration for hyperencoder data loading and processing.

    This replaces the previous JSON-based data configurations with
    a modern, type-safe, validated configuration system.
    """

    # Core configuration
    target_: str = Field(
        default="hyperencoder.data.latent.LatentsForHyperEncoderDataset",
        alias="_target_",
        description="Target class for instantiating the dataset",
        examples=[
            "hyperencoder.data.latent.LatentsForHyperEncoderDataset",
            "hyperencoder.data.audio.AudioDataset",
        ],
    )

    # Dataset type and loading strategy
    dataset_type: Literal[
        "latents_for_hyperencoder", "audio_dataset", "pre_encoded_dataset"
    ] = Field(default="latents_for_hyperencoder", description="Type of dataset to load")

    split_type: Literal["auto", "manual", "none"] = Field(
        default="auto", description="How to split the data into train/val/test sets"
    )

    loading_strategy: Literal["lazy", "eager", "memory_mapped"] = Field(
        default="lazy", description="Strategy for loading data into memory"
    )

    # Data loading parameters
    batch_size: int = Field(
        default=32, ge=1, le=512, description="Number of samples per batch"
    )

    num_workers: int = Field(
        default=8, ge=0, le=32, description="Number of CPU workers for the DataLoader"
    )

    random_seed: int = Field(
        default=42, ge=0, description="Random seed for reproducibility in data loading"
    )

    persistent_workers: bool = Field(
        default=False,
        description="Whether to keep DataLoader workers persistent between epochs",
    )

    # Data splits
    train_split_pct: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Percentage of data to use for training",
    )

    val_split_pct: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Percentage of data to use for validation",
    )

    test_split_pct: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Percentage of data to use for testing"
    )

    # Dataset paths
    datasets: list[DatasetEntry] = Field(
        default_factory=list, description="List of dataset configurations to load"
    )

    # Single dataset path (alternative to datasets list)
    path: str | None = Field(
        default=None, description="Single dataset path (alternative to datasets list)"
    )

    # Crop configuration
    crop_config: CropConfig | None = Field(
        default=None, description="Configuration for data cropping/augmentation"
    )

    # MIDI metadata configuration
    midi_metadata: MidiMetadataConfig | None = Field(
        default=None, description="Configuration for MIDI metadata injection"
    )

    # Data processing options
    normalize: bool = Field(default=True, description="Whether to normalize the data")

    shuffle: bool = Field(
        default=True, description="Whether to shuffle the data during training"
    )

    # Caching options
    cache_dir: str | None = Field(
        default=None, description="Directory for caching processed data"
    )

    use_cache: bool = Field(
        default=False, description="Whether to use cached data if available"
    )

    # Advanced options
    max_files: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of files to load (for debugging)",
    )

    file_extension: str = Field(
        default=".pt",
        description="File extension for dataset files",
        examples=[".pt", ".npy", ".wav", ".mp3"],
    )

    recursive: bool = Field(
        default=True,
        description="Whether to search for files recursively in directories",
    )

    # Model validation (runs after all fields are set)
    def model_post_init(self, __context):
        """Post-initialization validation."""
        # Check that splits sum to 1.0
        total = self.train_split_pct + self.val_split_pct + self.test_split_pct
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test splits must sum to 1.0, got {total}")

    @field_validator("datasets")
    @classmethod
    def validate_datasets_or_path(cls, v, info):
        """Ensure either datasets list or path is provided."""
        if not v and not info.data.get("path"):
            raise ValueError("Either 'datasets' list or 'path' must be provided")
        return v

    @field_validator("path", "cache_dir", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        """Convert string paths to resolved paths."""
        if v is None or v == "":
            return None
        # Don't resolve paths with OmegaConf interpolations
        if isinstance(v, str) and "${" in v:
            return v
        return str(Path(v).resolve()) if not isinstance(v, Path) else str(v.resolve())
