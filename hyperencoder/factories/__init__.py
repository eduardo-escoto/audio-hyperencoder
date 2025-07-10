"""Factory functions for creating hyperencoder models and components.

This module provides centralized factory functions for creating models,
configurations, and other components. Factory functions are placed here
to avoid circular imports between models and datamodels.
"""

from .model_factory import (
    create_hyperencoder_from_config,
    create_hyperencoder,
)
from .auxiliary_head_factory import (
    create_auxiliary_heads_from_config,
    create_auxiliary_head,
)

__all__ = [
    "create_hyperencoder_from_config",
    "create_hyperencoder",
    "create_auxiliary_heads_from_config", 
    "create_auxiliary_head",
] 