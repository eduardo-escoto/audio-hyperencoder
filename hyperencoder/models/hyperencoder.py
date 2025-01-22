from typing import Union

from torch.nn import Module
from stable_audio_tools.models.bottleneck import Bottleneck
from stable_audio_tools.models.autoencoders import (
    DecoderBlock,
    EncoderBlock,
    OobleckDecoder,
    OobleckEncoder,
)


class HyperEncoder(Module):
    def __init__(
        self,
        encoder: Union[EncoderBlock, OobleckEncoder],
        decoder: Union[DecoderBlock, OobleckDecoder],
        bottleneck: Bottleneck,
    ):
        super().__init__()
        self.encoder: Union[EncoderBlock, OobleckEncoder] = encoder
        self.decoder: Union[DecoderBlock, OobleckDecoder] = decoder
        self.bottleneck: Bottleneck = bottleneck

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.process_latent_list_for_hyperencoder(x)
        x = self.encoder(x)
        x = self.bottleneck.encode(x)
        return x

    def decode(self, x):
        x = self.bottleneck.decode(x)
        x = self.decoder(x)
        return x

    def process_latent_list_for_hyperencoder(self, latent_list):
        # Need to implement a similar method to proceess_audio_list_for_autoencoder so
        # that we can pad to max length and ensure that the latent tensors are the
        # same size and that they are the same shape even if we have a batch of 1
        pass
