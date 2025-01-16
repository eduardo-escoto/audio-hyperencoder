import json

import argparse

import lightning as L

from os import environ

from pathlib import Path

from stable_audio_tools.models.autoencoders import AudioAutoencoder
from stable_audio_tools.models.bottleneck import RVQVAEBottleneck
from stable_audio_tools import get_pretrained_model


argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--model-name",
    type=str,
    required=True,
    help="Huggingface model name for stable-audio-open model",
)
argparser.add_argument(
    "--autoencoder", type=str, required=True, help="stable-audio-open autoencoder name"
)
argparser.add_argument(
    "--bottleneck", type=str, required=True, help="stable-audio-open bottleneck name"
)
argparser.add_argument(
    "--hf-token",
    type=str,
    required=False,
    help="Huggingface token for downloading models",
)


class LatentHyperencoder(L.LightningModule):
    def __init__(self, autoencoder):
        super().__init__()
        self.auto_enc = autoencoder

    @staticmethod
    def from_config(model_config, autoencoder, bottleneck):
        target_latent_dim = model_config['model']['pretransform']['config']['latent_dim']
        

        pass
        
    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        return super().configure_optimizers()


def get_model_config(
    model_name, cache_dir="./model_cache", force_download=False, hf_token=None
):
    # huggingface names usually have a slash to indicate repo + model name
    sanitized_model_name = model_name.replace("/", "-")
    config_path = Path(f"{cache_dir}/{sanitized_model_name}.json")

    model_config = None

    # load from cache or download and create cache
    if config_path.exists() and not force_download:
        with open(config_path, "r") as config_file:
            model_config = json.load(config_file)
    else:
        from huggingface_hub import login

        # login to huggingface if token is provided, else it'll use env_variable
        if hf_token is not None:
            login(token=hf_token)

        # download model + model_config from huggingface
        _, model_config = get_pretrained_model(model_name)

        # create the file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.touch()

        # dump model config into cache
        with open(config_path, "w") as config_file:
            json.dump(model_config, config_file, indent=4)

    return model_config


def main(model_name, autoencoder, bottleneck, hf_token=None):
    """
    Step 1: Model Config
        Get model config such that we can get tensor shape and channels
        from the config to properly configure latent space encoder/decoder
    """
    model_config = get_model_config(model_name, hf_token=hf_token)
    """
    Step 2: Create the hyperencoder and bottleneck
        We'll have to ensure the autoencoder and bottlenecks are configured
        properly to receive the latent tensors properly
    """
    hyperencoder = LatentHyperencoder.from_config(
        model_config, autoencoder, bottleneck
    )
    """
    Step 3: Create latent dataloader
        Load our pre-encoded latents from directory
    """
    latent_dataloader = create_latent_dataloader()

    """
    Step 4: Run Training
        - Configure checkpoints and log reconstructed outputs along with 
            saved model checkpoint for spotchecking
        - Save all metrics along with model checkpoint
    """
    run_training(hyperencoder, latent_dataloader)


if __name__ == "__main__":
    args = argparser.parse_args()
    main(args.model_name, args.autoencoder, args.bottleneck, args.hf_token)
