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
        x = self.encode(x, return_info=True)
        x = self.decode(x)
        return x

    def encode(self, x, return_info=False):
        x = self.encoder(x)
        x = self.bottleneck.encode(x)
        return x

    def decode(self, x):
        x = self.bottleneck.decode(x)
        x = self.decoder(x)
        return x


# Load up PreEncodedDataset in the data module you made
# see if the inputs look right
# try the FSQ implementation after as well.
# Will probably need to use a notbeook to understand how the
# tensors work

# real audio
# audio to vae latent to hyperencoder latents and comparison

# checkout demo callback -
# gotta figure out their pre-encoding scheme.
# reconstruction
# hubert loss in the semantic losses
# or l1
