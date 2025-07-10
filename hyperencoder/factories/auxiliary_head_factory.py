"""
Factory functions for creating auxiliary heads and losses.

This module provides factory functions for creating auxiliary heads and their
associated losses from configuration objects, following proper separation of concerns.
"""

from typing import Any
from torch.nn import ModuleDict

from ..modules.auxiliary_heads import AuxiliaryHead, AuxiliaryHeadConfig
from ..modules.auxiliary_losses import AuxiliaryLoss


def create_auxiliary_heads_from_config(
    auxiliary_heads_config: dict[str, Any],
    latent_dim: int,
) -> tuple[ModuleDict, list[AuxiliaryLoss]]:
    """Factory function to create auxiliary heads and losses from configuration.
    
    Args:
        auxiliary_heads_config: Dictionary containing auxiliary heads configuration
        latent_dim: Latent dimension for the auxiliary heads
        
    Returns:
        Tuple of (auxiliary_heads ModuleDict, auxiliary_losses list)
        
    Examples:
        >>> config = {"enabled": True, "heads": [{"name": "tempo", "head_type": "regression", ...}]}
        >>> heads, losses = create_auxiliary_heads_from_config(config, 64)
        >>> print(f"Created {len(heads)} auxiliary heads")
    """
    auxiliary_heads = ModuleDict()
    auxiliary_losses = []
    
    if not auxiliary_heads_config.get("enabled", False):
        return auxiliary_heads, auxiliary_losses
        
    heads_config = auxiliary_heads_config.get("heads", [])
    if not heads_config:
        return auxiliary_heads, auxiliary_losses
    
    for head_config in heads_config:
        # Create auxiliary head from config
        head = AuxiliaryHead(
            config=AuxiliaryHeadConfig(**head_config),
            input_dim=latent_dim
        )
        
        # Store head in ModuleDict
        auxiliary_heads[head_config["name"]] = head
        
        # Create auxiliary loss for the head
        aux_loss = AuxiliaryLoss(
            auxiliary_head=head,
            latent_key="inner_latents",
            weight=head_config.get("loss_weight", 1.0),
            name=f"{head_config['name']}_loss"
        )
        
        # Add to loss list
        auxiliary_losses.append(aux_loss)
        
    return auxiliary_heads, auxiliary_losses


def create_auxiliary_head(
    head_config: dict[str, Any],
    latent_dim: int,
) -> tuple[AuxiliaryHead, AuxiliaryLoss]:
    """Factory function to create a single auxiliary head and its loss.
    
    Args:
        head_config: Configuration dictionary for the auxiliary head
        latent_dim: Latent dimension for the auxiliary head
        
    Returns:
        Tuple of (auxiliary_head, auxiliary_loss)
        
    Examples:
        >>> config = {"name": "tempo", "head_type": "regression", "num_classes": 1}
        >>> head, loss = create_auxiliary_head(config, 64)
    """
    # Create auxiliary head from config
    head = AuxiliaryHead(
        config=AuxiliaryHeadConfig(**head_config),
        input_dim=latent_dim
    )
    
    # Create auxiliary loss for the head
    aux_loss = AuxiliaryLoss(
        auxiliary_head=head,
        latent_key="inner_latents",
        weight=head_config.get("loss_weight", 1.0),
        name=f"{head_config['name']}_loss"
    )
    
    return head, aux_loss 