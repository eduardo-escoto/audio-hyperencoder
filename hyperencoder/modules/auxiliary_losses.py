"""
Auxiliary loss functions for hyperencoder multi-task learning.

This module provides loss functions specifically designed for auxiliary prediction
heads, integrating with the stable_audio_tools loss system.
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from .auxiliary_heads import AuxiliaryHead


class AuxiliaryLoss(nn.Module):
    """Loss function for auxiliary prediction heads.
    
    This loss integrates auxiliary heads with the stable_audio_tools MultiLoss
    system by computing predictions and losses for MIDI metadata targets.
    
    Args:
        auxiliary_head: The auxiliary head to compute predictions
        latent_key: Key in loss_info dict containing latents
        info_key: Key in loss_info dict containing metadata
        weight: Weight for this loss in multi-loss training
        name: Name for logging and identification
        
    Examples:
        >>> from hyperencoder.modules.auxiliary_heads import AuxiliaryHead, AuxiliaryHeadConfig
        >>> 
        >>> # Create a tempo prediction head
        >>> config = AuxiliaryHeadConfig(
        ...     name="tempo_predictor",
        ...     target_key="tempo_bpm",
        ...     head_type="regression",
        ...     loss_type="mse"
        ... )
        >>> head = AuxiliaryHead(config, input_dim=128)
        >>> 
        >>> # Create loss for the head
        >>> loss_fn = AuxiliaryLoss(
        ...     auxiliary_head=head,
        ...     latent_key="inner_latents",
        ...     weight=0.1,
        ...     name="tempo_loss"
        ... )
    """
    
    def __init__(
        self,
        auxiliary_head: AuxiliaryHead,
        latent_key: str = "inner_latents",
        info_key: str = "info",
        weight: float = 1.0,
        name: str = "auxiliary_loss"
    ):
        super().__init__()
        self.auxiliary_head = auxiliary_head
        self.latent_key = latent_key
        self.info_key = info_key
        self.weight = weight
        self.name = name
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize loss function based on head configuration
        self.loss_fn = self._create_loss_function()
        
        self.logger.info(
            f"Created auxiliary loss '{name}' for head '{auxiliary_head.config.name}' "
            f"with weight {weight}"
        )
    
    def _create_loss_function(self) -> nn.Module:
        """Create the appropriate loss function based on head type and config."""
        head_config = self.auxiliary_head.config
        
        if head_config.head_type == "regression":
            if head_config.loss_type == "mse":
                return nn.MSELoss()
            elif head_config.loss_type == "mae" or head_config.loss_type == "l1":
                return nn.L1Loss()
            elif head_config.loss_type == "huber":
                return nn.HuberLoss()
            elif head_config.loss_type == "smooth_l1":
                return nn.SmoothL1Loss()
            else:
                self.logger.warning(
                    f"Unknown regression loss type '{head_config.loss_type}', using MSE"
                )
                return nn.MSELoss()
        
        elif head_config.head_type == "classification":
            if head_config.loss_type == "ce" or head_config.loss_type == "crossentropy":
                return nn.CrossEntropyLoss()
            elif head_config.loss_type == "focal":
                return FocalLoss()
            elif head_config.loss_type == "nll":
                return nn.NLLLoss()
            else:
                self.logger.warning(
                    f"Unknown classification loss type '{head_config.loss_type}', using CrossEntropy"
                )
                return nn.CrossEntropyLoss()
        
        elif head_config.head_type == "multi_label":
            if head_config.loss_type == "bce":
                return nn.BCEWithLogitsLoss()
            elif head_config.loss_type == "focal":
                return MultilabelFocalLoss()
            else:
                self.logger.warning(
                    f"Unknown multi-label loss type '{head_config.loss_type}', using BCE"
                )
                return nn.BCEWithLogitsLoss()
        
        else:
            raise ValueError(f"Unknown head type: {head_config.head_type}")
    
    def forward(self, loss_info: Dict[str, Any]) -> Tensor:
        """Compute auxiliary loss from loss_info dict.
        
        Args:
            loss_info: Dictionary containing latents and metadata information
            
        Returns:
            Computed loss tensor, weighted by self.weight
            
        Raises:
            KeyError: If required keys are missing from loss_info
            RuntimeError: If loss computation fails
        """
        try:
            # Extract latents (guaranteed to exist in training)
            if self.latent_key not in loss_info:
                raise KeyError(f"Latent key '{self.latent_key}' not found in loss_info")
            
            latents = loss_info[self.latent_key]
            
            # Extract info dict
            info = loss_info.get(self.info_key, loss_info)
            
            # Extract MIDI metadata (guaranteed to exist since dataset errors without it)
            if "midi_metadata" not in info:
                raise KeyError("MIDI metadata not found in info dict")
            
            midi_metadata = info["midi_metadata"]
            
            # Get target value
            target_key = self.auxiliary_head.config.target_key
            if target_key not in midi_metadata:
                # Log warning but return zero loss for missing targets
                self.logger.warning(
                    f"Target key '{target_key}' not found in MIDI metadata for {self.name}"
                )
                return torch.tensor(0.0, device=latents.device, requires_grad=True)
            
            target_value = midi_metadata[target_key]
            
            # Preprocess target
            target = self.auxiliary_head.preprocess_target(target_value)
            target = target.to(latents.device)
            
            # Forward pass through auxiliary head
            prediction = self.auxiliary_head(latents)
            
            # Compute loss based on head type
            loss = self._compute_loss(prediction, target)
            
            # Apply weight and return
            return loss * self.weight
            
        except Exception as e:
            self.logger.error(f"Failed to compute auxiliary loss {self.name}: {e}")
            # Return zero loss to avoid stopping training
            device = loss_info.get(self.latent_key, torch.tensor(0.0)).device
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute loss based on head type."""
        head_type = self.auxiliary_head.config.head_type
        
        if head_type == "regression":
            # Ensure prediction and target have compatible shapes
            if prediction.dim() > 1:
                prediction = prediction.squeeze()
            if target.dim() > 0 and target.numel() == 1:
                target = target.squeeze()
            
            # Expand target to match batch size if needed
            if target.dim() == 0 and prediction.dim() == 1:
                target = target.expand_as(prediction)
            
            return self.loss_fn(prediction, target)
            
        elif head_type == "classification":
            # Target should be class indices (long tensor)
            target = target.long()
            
            # Handle single target vs batch
            if target.dim() == 0:
                target = target.expand(prediction.size(0))
            
            return self.loss_fn(prediction, target)
            
        elif head_type == "multi_label":
            # Target should be binary (float tensor)
            target = target.float()
            
            # Handle single target vs batch
            if target.dim() == 1 and prediction.dim() == 2:
                target = target.unsqueeze(0).expand_as(prediction)
            
            return self.loss_fn(prediction, target)
            
        else:
            raise ValueError(f"Unknown head type: {head_type}")


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in classification.
    
    Focal Loss down-weights easy examples and focuses on hard examples,
    which is useful for imbalanced classification tasks.
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
        
    References:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Prediction logits (batch_size, num_classes)
            targets: Target class indices (batch_size,)
            
        Returns:
            Focal loss tensor
        """
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultilabelFocalLoss(nn.Module):
    """Focal Loss for multi-label classification.
    
    Applies focal loss independently to each label in multi-label setting.
    
    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute multi-label focal loss.
        
        Args:
            inputs: Prediction logits (batch_size, num_classes)
            targets: Target binary labels (batch_size, num_classes)
            
        Returns:
            Multi-label focal loss tensor
        """
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)
        
        # Compute focal weight
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Compute BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss 