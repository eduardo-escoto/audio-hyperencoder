import argparse

from modules import LatentHyperencoder
from models.utils import get_model_config

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--model-name",
    type=str,
    required=True,
    default="stable-audio-open/stable-audio-open",
    help="Huggingface model name for stable-audio-open model",
)
argparser.add_argument(
    "--autoencoder",
    type=str,
    required=True,
    help="stable-audio-open autoencoder name",
    default="oobleck",
)
argparser.add_argument(
    "--bottleneck",
    type=str,
    required=True,
    help="stable-audio-open bottleneck name",
    default="fsq",
)
argparser.add_argument(
    "--hf-token",
    type=str,
    required=False,
    help="Huggingface token for downloading models",
)


def create_latent_dataloader():
    # Placeholder function for creating the latent dataloader
    # Replace with actual implementation
    return None


def run_training(hyperencoder, latent_dataloader):
    # Placeholder function for running the training
    # Replace with actual implementation
    pass


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
    hyperencoder = LatentHyperencoder.from_config(model_config, autoencoder, bottleneck)

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
