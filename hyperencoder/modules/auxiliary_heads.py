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


class AuxiliaryHead(nn.Module):
    """A single auxiliary prediction head.
    
    This module takes latent representations as input and predicts a specific
    MIDI metadata target. It handles different prediction types (regression,
    classification, multi-label) and includes target preprocessing.
    
    Args:
        config: Configuration dict specifying architecture and target details
        input_dim: Dimension of input latent representations
        
    Examples:
        >>> config = {
        ...     "name": "tempo_predictor",
        ...     "target_key": "tempo_bpm", 
        ...     "head_type": "regression",
        ...     "loss_type": "mse"
        ... }
        >>> head = AuxiliaryHead(config, input_dim=128)
        >>> 
        >>> # Forward pass
        >>> latents = torch.randn(32, 128, 64)  # (batch, channels, time)
        >>> prediction = head(latents)  # (32, 1) for regression
    """
    
    def __init__(self, config: dict[str, Any], input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Build the network
        self.network = self._build_network()
        
        self.logger.info(
            f"Created auxiliary head '{config.get('name', 'unnamed')}' -> {config.get('target_key', 'unknown')} "
            f"({config.get('head_type', 'unknown')}, input_dim={input_dim})"
        )
    
    def _validate_config(self) -> None:
        """Validate the head configuration."""
        head_type = self.config.get("head_type", "regression")
        
        if head_type in ["classification", "multi_label"]:
            num_classes = self.config.get("num_classes")
            if num_classes is None:
                raise ValueError(
                    f"num_classes must be specified for {head_type} heads"
                )
            if num_classes <= 0:
                raise ValueError("num_classes must be positive")
        
        target_min = self.config.get("target_min")
        target_max = self.config.get("target_max")
        if target_min is not None and target_max is not None:
            if target_min >= target_max:
                raise ValueError("target_min must be less than target_max")
    
    def _build_network(self) -> nn.Module:
        """Build the prediction network."""
        layers = []
        prev_dim = self.input_dim
        
        # Hidden layers
        hidden_dims = self.config.get("hidden_dims", [256, 128])
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(self.config.get("activation", "relu")),
                nn.Dropout(self.config.get("dropout_rate", 0.1))
            ])
            prev_dim = hidden_dim
        
        # Output layer
        output_dim = self._get_output_dim()
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _get_output_dim(self) -> int:
        """Get the output dimension based on head type."""
        head_type = self.config.get("head_type", "regression")
        
        if head_type == "regression":
            return 1
        elif head_type in ["classification", "multi_label"]:
            num_classes = self.config.get("num_classes")
            if num_classes is None:
                raise ValueError(f"num_classes must be specified for {head_type}")
            return num_classes
        else:
            raise ValueError(f"Unknown head type: {head_type}")
    
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
                head_type = self.config.get("head_type", "regression")
                if head_type == "multi_label":
                    # Convert list to multi-hot encoding
                    num_classes = self.config.get("num_classes")
                    if num_classes is None:
                        raise ValueError("num_classes must be specified for multi_label heads")
                    target_tensor = torch.zeros(num_classes)
                    for idx in target:
                        if isinstance(idx, (int, float)) and 0 <= idx < num_classes:
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
                    f"Non-numeric target for {self.config.get('name', 'unnamed')}: {target}, using 0.0"
                )
                target = 0.0
            
            target = float(target)
            
            # Apply log transform if configured
            if self.config.get("log_transform", False):
                target = torch.log(torch.tensor(target + 1e-8))
            else:
                target = torch.tensor(target)
            
            # Apply normalization if configured
            target_min = self.config.get("target_min")
            target_max = self.config.get("target_max")
            if target_min is not None and target_max is not None:
                target = (target - target_min) / (target_max - target_min)
                # Clamp to [0, 1] range
                target = torch.clamp(target, 0.0, 1.0)
            
            return target.float()
        
        except Exception as e:
            self.logger.error(
                f"Error preprocessing target for {self.config.get('name', 'unnamed')}: {e}"
            )
            # Return zero tensor as fallback
            return torch.tensor(0.0).float()
    
    def denormalize_prediction(self, prediction: Tensor) -> Tensor:
        """Denormalize prediction to original scale.
        
        Args:
            prediction: Normalized prediction tensor
            
        Returns:
            Denormalized prediction tensor
        """
        target_min = self.config.get("target_min")
        target_max = self.config.get("target_max")
        
        if target_min is not None and target_max is not None:
            # Denormalize from [0, 1] to [target_min, target_max]
            prediction = prediction * (target_max - target_min) + target_min
        
        # Apply inverse log transform if configured
        if self.config.get("log_transform", False):
            prediction = torch.exp(prediction)
        
        return prediction 