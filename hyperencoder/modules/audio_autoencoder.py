from lightning import LightningModule
from stable_audio_tools.models.autoencoders import AudioAutoencoder


class AudioAutoEncoder(LightningModule):
    def __init__(self, autoencoder: AudioAutoencoder, encode_only: bool = False):
        super().__init__()
        self.autoencoder = autoencoder
        self.encode_only = encode_only

    def forward(self, x):
        return self.autoencoder(x)

    def preprocess_batch(self, batch):
        waveforms, sample_rates = batch
        return self.autoencoder.preprocess_audio_list_for_encoder(
            waveforms, sample_rates
        )

    def encode_batch(self, batch):
        preprocessed_batch = self.preprocess_batch(batch)
        encoded_batch = self.autoencoder.encode_audio(preprocessed_batch)
        return encoded_batch

    def decode_batch(self, batch):
        decoded_batch = self.autoencoder.decode_audio(batch)
        return decoded_batch

    def predict_step(self, batch, batch_idx, dataloader_idx):
        encoded_batch = self.encode_batch(batch)

        if self.encode_only:
            return encoded_batch

        decoded_batch = self.decode_batch(encoded_batch)
        return decoded_batch
