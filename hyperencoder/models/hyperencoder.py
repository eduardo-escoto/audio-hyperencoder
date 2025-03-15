from typing import Any

from torch.nn import Module
from stable_audio_tools.models.autoencoders import (
    create_decoder_from_config,
    create_encoder_from_config,
    create_bottleneck_from_config,
)


class HyperEncoder(Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        input_channels,
        output_channels,
        bottleneck=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.bottleneck = bottleneck
        self.input_channels = input_channels
        self.output_channels = output_channels

    def encode(
        self,
        outer_latents,
        skip_bottleneck: bool = False,
        return_info: bool = False,
        **kwargs,
    ):
        info = {}

        inner_latents = self.encoder(outer_latents)

        info["pre_bottleneck_inner_latents"] = inner_latents

        if self.bottleneck is not None and not skip_bottleneck:
            inner_latents, bottleneck_info = self.bottleneck.encode(
                inner_latents, return_info=True, **kwargs
            )

            info["post_bottleneck_inner_latents"] = inner_latents
            info.update(bottleneck_info)

        if return_info:
            return inner_latents, info

        return inner_latents

    def decode(self, inner_latents, skip_bottleneck: bool = False, **kwargs):
        latents = inner_latents

        if self.bottleneck is not None and not skip_bottleneck:
            latents = self.bottleneck.decode(latents)

        outer_latents = self.decoder(latents, **kwargs)

        return outer_latents


def create_hyperencoder_from_config(config: dict[str, Any]):
    ae_config = config["model"]

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])

    bottleneck_config = ae_config.get("bottleneck", None)

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    if bottleneck_config is not None:
        bottleneck = create_bottleneck_from_config(bottleneck_config)

    return HyperEncoder(
        encoder,
        decoder,
        latent_dim=latent_dim,
        bottleneck=bottleneck,
        input_channels=in_channels,
        output_channels=out_channels,
    )
