from enum import Enum
from typing import Union

from lightning import LightningModule
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


class SupportedBottleneckTypes(Enum):
    FSQ = FSQBottleneck
    RVQ = RVQBottleneck
    RVQVAE = RVQVAEBottleneck

    def get_bottleneck(self) -> Bottleneck:
        return self.value


class SupportedEncoderDecoderTypes(Enum):
    OOBLECK = (OobleckEncoder, OobleckDecoder)
    VANILLA = (EncoderBlock, DecoderBlock)

    def get_encoder_decoder(self):
        return self.value


class LatentHyperencoder(LightningModule):
    def __init__(self, autoencoder, bottleneck):
        super().__init__()
        self.auto_enc = autoencoder
        self.bottleneck = bottleneck

    @staticmethod
    def from_config(
        model_config: dict,
        autoencoder_type: str,
        bottleneck_type: str,
        hyper_latent_dim: int = 10,
        bottleneck_kwargs: Union[dict, None] = None,
        autoencoder_kwargs: Union[dict, None] = None,
    ):
        """
        This function builds the underlying encoder model from the stability audio open
        model config, and instantiates the LatentHyperencoder with the given encoder
        model and bottleneck type.
        """
        target_dim = model_config["model"]["pretransform"]["config"]["latent_dim"]

        pass

    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        return super().configure_optimizers()
