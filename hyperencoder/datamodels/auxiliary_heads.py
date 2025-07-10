"""
Auxiliary head configuration models for hyperencoder.

This module defines the Pydantic models for auxiliary head configurations,
which enable multi-task learning on MIDI metadata to improve semantic
representation learning.
"""

from typing import Literal
from pydantic import BaseModel, Field


class AuxiliaryHeadConfig(BaseModel):
    """Configuration for a single auxiliary head.
    
    This defines the architecture, target, and loss configuration for an
    auxiliary prediction head that predicts MIDI metadata from latent space.
    
    Examples:
        >>> # Tempo regression head
        >>> tempo_config = AuxiliaryHeadConfig(
        ...     name="tempo_predictor",
        ...     target_key="tempo_bpm",
        ...     head_type="regression",
        ...     loss_type="mse",
        ...     loss_weight=0.1,
        ...     target_min=60.0,
        ...     target_max=200.0
        ... )
        
        >>> # Time signature classification head
        >>> time_sig_config = AuxiliaryHeadConfig(
        ...     name="time_signature_predictor",
        ...     target_key="time_signature_numerator",
        ...     head_type="classification",
        ...     loss_type="ce",
        ...     loss_weight=0.1,
        ...     num_classes=16
        ... )
    """
    
    name: str = Field(description="Name of the auxiliary head")
    target_key: str = Field(description="Key in MIDI metadata dict to predict")
    
    head_type: Literal["regression", "classification", "multi_label"] = Field(
        description="Type of prediction head"
    )
    
    # Architecture configuration
    hidden_dims: list[int] = Field(
        default_factory=lambda: [512, 256], 
        description="Hidden layer dimensions"
    )
    dropout_rate: float = Field(
        default=0.1, 
        ge=0.0, 
        le=1.0,
        description="Dropout rate for regularization"
    )
    activation: str = Field(
        default="relu", 
        description="Activation function name"
    )
    
    # Loss configuration
    loss_type: str = Field(description="Loss function type")
    loss_weight: float = Field(
        default=1.0, 
        gt=0.0, 
        description="Weight for this loss in multi-loss training"
    )
    
    # Classification specific
    num_classes: int | None = Field(
        default=None, 
        description="Number of classes for classification heads"
    )
    
    # Target preprocessing
    target_min: float | None = Field(
        default=None, 
        description="Min value for target normalization"
    )
    target_max: float | None = Field(
        default=None, 
        description="Max value for target normalization"
    )
    log_transform: bool = Field(
        default=False, 
        description="Apply log transform to target values"
    ) 