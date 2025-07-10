"""Factory functions for creating hyperencoder models.

This module contains factory functions for creating hyperencoder models from
configurations. These functions are placed here to avoid circular imports
between models and datamodels.
"""

from typing import Any

from stable_audio_tools.models.autoencoders import (
    create_decoder_from_config,
    create_encoder_from_config,
    create_bottleneck_from_config,
)

from hyperencoder.datamodels import ModelConfig


def create_hyperencoder_from_config(config: ModelConfig):
    """Create a HyperEncoder model from a Pydantic configuration.

    Args:
        config: ModelConfig containing all model parameters

    Returns:
        Configured HyperEncoder instance

    Examples:
        >>> from hyperencoder.datamodels import ModelConfig
        >>> config = ModelConfig()  # Uses all defaults
        >>> model = create_hyperencoder_from_config(config)
        >>>
        >>> # Or with custom parameters
        >>> config = ModelConfig(latent_dim=32, in_channels=128)
        >>> model = create_hyperencoder_from_config(config)
    """
    # Local import to avoid circular import
    from hyperencoder.models.hyperencoder import HyperEncoder
    
    # Create encoder from config
    encoder = create_encoder_from_config(config.encoder.model_dump())

    # Create decoder from config
    decoder = create_decoder_from_config(config.decoder.model_dump())

    # Create bottleneck if specified
    bottleneck = None
    if config.bottleneck is not None:
        bottleneck = create_bottleneck_from_config(config.bottleneck.model_dump())

    return HyperEncoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=config.latent_dim,
        bottleneck=bottleneck,
        input_channels=config.in_channels,
        output_channels=config.out_channels,
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
    use the same defaults as defined in the ModelConfig Pydantic model.

    Args:
        latent_dim: Dimension of the latent space (None for default)
        in_channels: Number of input channels (None for default)
        out_channels: Number of output channels (None for default)
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
    # Import ModelConfig locally to avoid circular imports
    from hyperencoder.datamodels import ModelConfig
    
    # Create a ModelConfig with the provided parameters
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

    # Create ModelConfig and delegate to config-based factory
    model_config = ModelConfig(**config_dict)
    return create_hyperencoder_from_config(model_config) 