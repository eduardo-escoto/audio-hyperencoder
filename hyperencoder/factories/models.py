from typing import Any

from stable_audio_tools.models.autoencoders import (
    create_decoder_from_config,
    create_encoder_from_config,
    create_bottleneck_from_config,
)

from hyperencoder.datamodels.modeling import ModelingConfig
from hyperencoder.models.hyperencoder import HyperEncoder


def create_hyperencoder_from_config(cfg: ModelingConfig) -> HyperEncoder:
    encoder_config = (
        cfg.model.encoder.model_dump() if cfg.model.encoder is not None else None
    )
    decoder_config = (
        cfg.model.decoder.model_dump() if cfg.model.decoder is not None else None
    )
    bottleneck_config = (
        cfg.model.bottleneck.model_dump() if cfg.model.bottleneck is not None else None
    )

    match cfg.model_type:
        case "hyperencoder":
            return create_hyperencoder(
                latent_dim=cfg.model.latent_dim,
                in_channels=cfg.model.in_channels,
                out_channels=cfg.model.out_channels,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                bottleneck_config=bottleneck_config,
            )
        case _:
            raise ValueError(f"Unknown model type: {cfg.model_type}")


def create_hyperencoder(
    latent_dim: int | None = None,
    in_channels: int | None = None,
    out_channels: int | None = None,
    encoder_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    decoder_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    bottleneck_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
) -> HyperEncoder:
    if encoder_config is not None:
        encoder = create_encoder_from_config(encoder_config)
    else:
        encoder = None

    if decoder_config is not None:
        decoder = create_decoder_from_config(decoder_config)
    else:
        decoder = None

    if bottleneck_config is not None:
        bottleneck = create_bottleneck_from_config(bottleneck_config)
    else:
        bottleneck = None

    return HyperEncoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        bottleneck=bottleneck,
        input_channels=in_channels,
        output_channels=out_channels,
    )
