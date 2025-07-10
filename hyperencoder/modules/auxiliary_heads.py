"""
Auxiliary prediction heads for hyperencoder multi-task learning.

This module provides configurable auxiliary heads that can predict MIDI metadata
from latent representations, enabling multi-task learning to improve semantic
representation learning.
"""

import logging
from typing import Any

import torch
from torch import nn, Tensor

from ..datamodels.auxiliary_heads import AuxiliaryHeadConfig


class AuxiliaryHead(nn.Module):
    """A single auxiliary prediction head.
    
    This module takes latent representations as input and predicts a specific
    MIDI metadata target. It handles different prediction types (regression,
    classification, multi-label) and includes target preprocessing.
    
    Args:
        config: Configuration specifying architecture and target details
        input_dim: Dimension of input latent representations
        
    Examples:
        >>> config = AuxiliaryHeadConfig(
        ...     name="tempo_predictor",
        ...     target_key="tempo_bpm", 
        ...     head_type="regression",
        ...     loss_type="mse"
        ... )
        >>> head = AuxiliaryHead(config, input_dim=128)
        >>> 
        >>> # Forward pass
        >>> latents = torch.randn(32, 128, 64)  # (batch, channels, time)
        >>> prediction = head(latents)  # (32, 1) for regression
    """
    
    def __init__(self, config: AuxiliaryHeadConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Build the network
        self.network = self._build_network()
        
        self.logger.info(
            f"Created auxiliary head '{config.name}' -> {config.target_key} "
            f"({config.head_type}, input_dim={input_dim})"
        )
    
    def _validate_config(self) -> None:
        """Validate the head configuration."""
        if self.config.head_type in ["classification", "multi_label"]:
            if self.config.num_classes is None:
                raise ValueError(
                    f"num_classes must be specified for {self.config.head_type} heads"
                )
            if self.config.num_classes <= 0:
                raise ValueError("num_classes must be positive")
        
        if self.config.target_min is not None and self.config.target_max is not None:
            if self.config.target_min >= self.config.target_max:
                raise ValueError("target_min must be less than target_max")
    
    def _build_network(self) -> nn.Module:
        """Build the prediction network."""
        layers = []
        prev_dim = self.input_dim
        
        # Hidden layers
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(self.config.activation),
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        output_dim = self._get_output_dim()
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _get_output_dim(self) -> int:
        """Get the output dimension based on head type."""
        if self.config.head_type == "regression":
            return 1
        elif self.config.head_type in ["classification", "multi_label"]:
            if self.config.num_classes is None:
                raise ValueError(f"num_classes must be specified for {self.config.head_type}")
            return self.config.num_classes
        else:
            raise ValueError(f"Unknown head type: {self.config.head_type}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "swish": nn.SiLU(),
            "elu": nn.ELU(),
        }
        
        if activation not in activations:
            self.logger.warning(
                f"Unknown activation '{activation}', using ReLU instead"
            )
            return nn.ReLU()
        
        return activations[activation]
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the auxiliary head.
        
        Args:
            x: Input latent tensor, shape (batch_size, latent_dim, ...)
            
        Returns:
            Prediction tensor, shape (batch_size, output_dim)
        """
        # Global average pooling if input has spatial/temporal dimensions
        if x.dim() > 2:
            # Pool over all dimensions except batch and channel
            pool_dims = tuple(range(2, x.dim()))
            x = x.mean(dim=pool_dims)
        
        # Forward through network
        return self.network(x)
    
    def preprocess_target(self, target: Any) -> Tensor:
        """Preprocess target value for training.
        
        Handles different target types and applies configured preprocessing
        like normalization and log transforms.
        
        Args:
            target: Raw target value from MIDI metadata
            
        Returns:
            Preprocessed target tensor ready for loss computation
            
        Raises:
            ValueError: If target preprocessing fails
        """
        try:
            # Handle list/tuple targets
            if isinstance(target, (list, tuple)):
                if self.config.head_type == "multi_label":
                    # Convert list to multi-hot encoding
                    if self.config.num_classes is None:
                        raise ValueError("num_classes must be specified for multi_label heads")
                    target_tensor = torch.zeros(self.config.num_classes)
                    for idx in target:
                        if isinstance(idx, (int, float)) and 0 <= idx < self.config.num_classes:
                            target_tensor[int(idx)] = 1.0
                    return target_tensor
                else:
                    # Take first valid value for single predictions
                    if len(target) == 0:
                        target = 0.0
                    else:
                        target = target[0]
            
            # Convert to float for numerical processing
            if not isinstance(target, (int, float)):
                self.logger.warning(
                    f"Non-numeric target for {self.config.name}: {target}, using 0.0"
                )
                target = 0.0
            
            target = float(target)
            
            # Apply log transform if configured
            if self.config.log_transform:
                target = torch.log(torch.tensor(target + 1e-8))
            else:
                target = torch.tensor(target)
            
            # Apply normalization if configured
            if self.config.target_min is not None and self.config.target_max is not None:
                target = (target - self.config.target_min) / (
                    self.config.target_max - self.config.target_min
                )
                # Clamp to [0, 1] range
                target = torch.clamp(target, 0.0, 1.0)
            
            return target.float()
            
        except Exception as e:
            self.logger.error(
                f"Failed to preprocess target for {self.config.name}: {target}, error: {e}"
            )
            # Return zero tensor as fallback
            if self.config.head_type == "multi_label":
                num_classes = self.config.num_classes or 1  # Fallback to 1 if None
                return torch.zeros(num_classes, dtype=torch.float32)
            else:
                return torch.tensor(0.0, dtype=torch.float32)
    
    def denormalize_prediction(self, prediction: Tensor) -> Tensor:
        """Denormalize prediction back to original scale.
        
        Useful for logging and evaluation in original units.
        
        Args:
            prediction: Normalized prediction tensor
            
        Returns:
            Denormalized prediction tensor
        """
        if self.config.target_min is not None and self.config.target_max is not None:
            prediction = prediction * (self.config.target_max - self.config.target_min) + self.config.target_min
        
        if self.config.log_transform:
            prediction = torch.exp(prediction)
        
        return prediction 