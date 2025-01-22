import logging
import argparse
import warnings

from torch import set_float32_matmul_precision
from lightning import Trainer

from .data import LatentDataModule
from .modules import LatentHyperencoder
from .models.utils import get_model_config
from .modules.latent_autoencoder import BottleneckTypes, EncoderDecoderTypes

valid_bottlenecks = BottleneckTypes._member_map_
valid_encoders = EncoderDecoderTypes._member_map_

# Turn off future warnings for vector_quantize_pytorch and torch
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="vector_quantize_pytorch"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--model-name",
    type=str,
    required=False,
    default="stabilityai/stable-audio-open-1.0",
    help="Huggingface model name for stable-audio-open model",
)
argparser.add_argument(
    "--autoencoder",
    type=str,
    required=True,
    help="stable-audio-open autoencoder name",
    choices=list(valid_encoders.keys()),
)
argparser.add_argument(
    "--bottleneck",
    type=str,
    required=True,
    help="stable-audio-open bottleneck name",
    choices=list(valid_bottlenecks.keys()),
)
argparser.add_argument(
    "--hf-token",
    type=str,
    required=False,
    help="Huggingface token for downloading models",
)

argparser.add_argument(
    "--input-dir",
    type=str,
    required=True,
    help="Directory containing pre-encoded latent tensors",
)

argparser.add_argument(
    "--checkpoint-dir",
    type=str,
    required=True,
    help="Directory to save model checkpoints and associated data",
)


def run_training(hyperencoder, latent_dataloader, checkoint_dir):
    set_float32_matmul_precision("medium")
    trainer = Trainer(
        max_epochs=100, default_root_dir=checkoint_dir, log_every_n_steps=10
    )

    trainer.fit(hyperencoder, latent_dataloader)


def main(model_name, input_dir, checkpoint_dir, autoencoder, bottleneck, hf_token=None):
    """
    Step 1: Model Config
        Get model config such that we can get tensor shape and channels
        from the config to properly configure latent space encoder/decoder
    """
    logging.info(f"Getting model config for {model_name}")
    model_config = get_model_config(model_name, hf_token=hf_token)
    target_dim = model_config["model"]["pretransform"]["config"]["latent_dim"]

    """
    Step 2: Create the hyperencoder and bottleneck
        We'll have to ensure the autoencoder and bottlenecks are configured
        properly to receive the latent tensors properly
    """
    logging.info(f"Creating latent hyperencoder with {autoencoder} and {bottleneck}")
    hyperencoder = LatentHyperencoder.factory(
        target_dim,
        autoencoder_type=autoencoder,
        bottleneck_type=bottleneck,
        bottleneck_kwargs={"levels": [8, 5, 5, 5]},
    )

    """
    Step 3: Create latent dataloader
        Load our pre-encoded latents from directory
    """
    logging.info(f"Creating latent dataloader from {input_dir}")
    latent_dataloader = LatentDataModule(input_dir, batch_size=10)
    """
    Step 4: Run Training
        - Configure checkpoints and log reconstructed outputs along with 
            saved model checkpoint for spotchecking
        - Save all metrics along with model checkpoint
    """
    logging.info("Running training")
    run_training(hyperencoder, latent_dataloader, checkpoint_dir)


def initialize_logger(log_file: str = "training.log"):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


if __name__ == "__main__":
    """
uv run python -m hyperencoder.train_encoder \
--autoencoder OOBLECK --bottleneck FSQ  \
--input-dir ./data/babyslakh_16k_latents \
--checkpoint-dir=./data/hyperencoder_checkpoints
    """
    args = argparser.parse_args()
    initialize_logger(f"{args.checkpoint_dir}/training.log")

    logging.info(f"Training hyperencoder with {args.autoencoder} and {args.bottleneck}")
    main(
        args.model_name,
        args.input_dir,
        args.checkpoint_dir,
        valid_encoders[args.autoencoder],
        valid_bottlenecks[args.bottleneck],
        args.hf_token,
    )
