from enum import Enum
from typing import Union

from torch.nn import Module as TorchModule
from lightning import LightningModule
from torch.optim import Adam
from torch.functional import F
from stable_audio_tools.models.bottleneck import (
    Bottleneck,
    FSQBottleneck,
    RVQBottleneck,
    RVQVAEBottleneck,
)
from stable_audio_tools.models.autoencoders import (
    DecoderBlock,
    EncoderBlock,
    OobleckDecoder,
    OobleckEncoder,
)

from ..models.hyperencoder import HyperEncoder


class BottleneckTypes(Enum):
    FSQ = FSQBottleneck
    RVQ = RVQBottleneck
    RVQVAE = RVQVAEBottleneck

    def get_bottleneck(self) -> Bottleneck:
        return self.value


class EncoderDecoderTypes(Enum):
    OOBLECK = (OobleckEncoder, OobleckDecoder)
    VANILLA = (EncoderBlock, DecoderBlock)

    def get_encoder_decoder(self) -> tuple[TorchModule, TorchModule]:
        return self.value


class LatentHyperencoder(LightningModule):
    def __init__(self, hyperencoder: HyperEncoder):
        super().__init__()
        self.hyperencoder = hyperencoder

    @staticmethod
    def factory(
        target_dim: int,
        hyper_latent_dim: int = 4,
        autoencoder_type: EncoderDecoderTypes = EncoderDecoderTypes.OOBLECK,
        bottleneck_type: BottleneckTypes = BottleneckTypes.FSQ,
        bottleneck_kwargs: Union[dict, None] = None,
        autoencoder_kwargs: Union[dict, None] = None,
    ):
        Encoder, Decoder = autoencoder_type.get_encoder_decoder()
        Bottleneck = bottleneck_type.get_bottleneck()

        encoder = Encoder(
            in_channels=target_dim,
            latent_dim=hyper_latent_dim,
            **autoencoder_kwargs if autoencoder_kwargs else {},
        )
        decoder = Decoder(
            out_channels=target_dim,
            latent_dim=hyper_latent_dim,
            **autoencoder_kwargs if autoencoder_kwargs else {},
        )

        bottleneck = Bottleneck(**bottleneck_kwargs if bottleneck_kwargs else {})

        return LatentHyperencoder(HyperEncoder(encoder, decoder, bottleneck))

    def __reconstruction_loss__(self, latents, reconstructed_latents):
        return F.mse_loss(latents, reconstructed_latents)

    def training_step(self, batch, batch_idx):
        latents = batch
        reconstructed_latents = self.hyperencoder(latents)
        loss = self.__reconstruction_loss__(latents, reconstructed_latents)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}
