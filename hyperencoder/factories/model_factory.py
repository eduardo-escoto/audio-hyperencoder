"""Factory functions for creating hyperencoder models.

This module contains factory functions for creating hyperencoder models from
configurations. These functions are placed here to avoid circular imports
between models and datamodels.
"""

from typing import Any

from omegaconf import DictConfig
from stable_audio_tools.models.autoencoders import (
    create_decoder_from_config,
    create_encoder_from_config,
    create_bottleneck_from_config,
)


def create_hyperencoder_from_config(config: DictConfig):
    """Create a HyperEncoder model from a Hydra configuration.

    Args:
        config: DictConfig containing all model parameters

    Returns:
        Configured HyperEncoder instance

    Examples:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({"latent_dim": 4, "in_channels": 64})
        >>> model = create_hyperencoder_from_config(config)
    """
    # Local import to avoid circular import
    from hyperencoder.models.hyperencoder import HyperEncoder
    
    # Create encoder from config
    encoder_config = config.get("encoder", {})
    encoder = create_encoder_from_config(encoder_config)

    # Create decoder from config
    decoder_config = config.get("decoder", {})
    decoder = create_decoder_from_config(decoder_config)

    # Create bottleneck if specified
    bottleneck = None
    bottleneck_config = config.get("bottleneck")
    if bottleneck_config is not None:
        bottleneck = create_bottleneck_from_config(bottleneck_config)

    return HyperEncoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=config.get("latent_dim", 4),
        bottleneck=bottleneck,
        input_channels=config.get("in_channels", 64),
        output_channels=config.get("out_channels", 64),
    )


def create_hyperencoder(
    latent_dim: int | None = None,
    in_channels: int | None = None,
    out_channels: int | None = None,
    encoder_config: dict[str, Any] | None = None,
    decoder_config: dict[str, Any] | None = None,
    bottleneck_config: dict[str, Any] | None = None,
):
    """Create a HyperEncoder model with programmatic parameters.

    This is a convenience function for users who want to create models
    programmatically without using configuration files. All parameters
    use sensible defaults.

    Args:
        latent_dim: Dimension of the latent space (default: 4)
        in_channels: Number of input channels (default: 64)
        out_channels: Number of output channels (default: 64)
        encoder_config: Optional encoder configuration dict
        decoder_config: Optional decoder configuration dict
        bottleneck_config: Optional bottleneck configuration dict

    Returns:
        Configured HyperEncoder instance

    Examples:
        >>> # Use all defaults
        >>> model = create_hyperencoder()
        >>>
        >>> # Custom latent dimension
        >>> model = create_hyperencoder(latent_dim=64)
        >>>
        >>> # Custom encoder config
        >>> model = create_hyperencoder(
        ...     encoder_config={"latent_dim": 128, "channels": 256}
        ... )
    """
    # Create a DictConfig with the provided parameters
    config_dict: dict[str, Any] = {}
    
    if latent_dim is not None:
        config_dict["latent_dim"] = latent_dim
    if in_channels is not None:
        config_dict["in_channels"] = in_channels
    if out_channels is not None:
        config_dict["out_channels"] = out_channels

    if encoder_config is not None:
        config_dict["encoder"] = encoder_config
    if decoder_config is not None:
        config_dict["decoder"] = decoder_config
    if bottleneck_config is not None:
        config_dict["bottleneck"] = bottleneck_config

    # Create DictConfig and delegate to config-based factory
    from omegaconf import DictConfig
    model_config = DictConfig(config_dict)
    return create_hyperencoder_from_config(model_config) 