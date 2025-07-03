"""Pre-encoding configuration models for hyperencoder."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from pathlib import Path
from .base import BaseConfig


class PreEncodeConfig(BaseConfig):
    """Configuration for pre-encoding audio files to latents.
    
    This configuration controls the pre-encoding process that converts audio files
    to latent representations using stable-audio-tools pretrained models.
    """
    
    # Model configuration
    model_name: str = Field(
        default="stabilityai/stable-audio-open-1.0",
        description="HuggingFace model name for the pretrained audio encoder",
        examples=["stabilityai/stable-audio-open-1.0"]
    )
    
    # Authentication
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace token for accessing gated models",
        examples=["hf_xxxxxxxxxx"]
    )
    
    # Input/Output paths
    input_dir: str = Field(
        description="Directory containing audio files to encode",
        examples=["/path/to/audio/files", "./data/audio"]
    )
    
    output_dir: str = Field(
        description="Directory to save encoded latent files",
        examples=["/path/to/output", "./data/encoded"]
    )
    
    # File processing configuration
    file_pattern: str = Field(
        default="*.wav",
        description="File pattern to match audio files",
        examples=["*.wav", "*.mp3", "*.flac"]
    )
    
    batch_pattern: str = Field(
        default=r"Track\d*",
        description="Regex pattern to group files into batches",
        examples=[r"Track\d*", r"Song\d*", r".*"]
    )
    
    # Hardware configuration
    n_devices: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of GPU devices to use for encoding",
        examples=[1, 2, 4]
    )
    
    batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Batch size for processing audio files",
        examples=[1, 2, 4]
    )
    
    # Advanced options
    create_output_dir: bool = Field(
        default=True,
        description="Whether to create output directory if it doesn't exist"
    )
    
    log_failures: bool = Field(
        default=True,
        description="Whether to log failed files to failures.log"
    )
    
    use_path_file: bool = Field(
        default=False,
        description="Use a file containing paths instead of directory scanning"
    )
    
    path_file: Optional[str] = Field(
        default=None,
        description="Path to file containing list of audio files to process",
        examples=["./file_list.txt"]
    )
    
    # Model configuration
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "model_name": "stabilityai/stable-audio-open-1.0",
                    "input_dir": "/path/to/audio/files",
                    "output_dir": "/path/to/output",
                    "n_devices": 1,
                    "batch_size": 1,
                    "file_pattern": "*.wav",
                    "batch_pattern": "Track\\d*"
                }
            ]
        }
    ) 