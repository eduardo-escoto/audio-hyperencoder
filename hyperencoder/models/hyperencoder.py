from torch.nn import Module


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
        # log = logging.getLogger()
        # log.info(f"Outer shape {outer_latents.shape}")
        # log.info(f"Outer dim: {outer_latents.dim()}")
        inner_latents = self.encoder(outer_latents)
        # log.info(f"Inner Latent shape before bottleneck: {inner_latents.shape}")
        # log.info(f"Inner latent before bottleneck: {inner_latents.dim()}")
        info["pre_bottleneck_inner_latents"] = inner_latents

        if self.bottleneck is not None and not skip_bottleneck:
            inner_latents, bottleneck_info = self.bottleneck.encode(
                inner_latents, return_info=True, **kwargs
            )
            # log.info(f"Inner Latent shape after bottleneck: {inner_latents.shape}")
            # log.info(f"Inner latent dim after bottleneck: {inner_latents.dim()}")

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
