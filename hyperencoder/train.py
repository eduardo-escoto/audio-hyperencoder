import json
import logging
import warnings
from os import path, environ

from torch import set_float32_matmul_precision
from lightning import Trainer, seed_everything
from prefigure import get_all_args
from torch.multiprocessing import set_sharing_strategy
from lightning.pytorch.loggers import CometLogger, WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from stable_audio_tools.training import (
    create_model_from_config,
    create_training_wrapper_from_config,
)
from stable_audio_tools.data.dataset import create_dataloader_from_config

# from .modules.latent_autoencoder import BottleneckTypes, EncoderDecoderTypes

# valid_bottlenecks = BottleneckTypes._member_map_
# valid_encoders = EncoderDecoderTypes._member_map_


class ExceptionCallback(Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


class ModelConfigEmbedderCallback(Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


# Turn off future warnings for vector_quantize_pytorch and torch
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="vector_quantize_pytorch"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def run_training(hyperencoder, latent_dataloader, checkoint_dir):
    set_float32_matmul_precision("medium")
    trainer = Trainer(
        max_epochs=100, default_root_dir=checkoint_dir, log_every_n_steps=10
    )

    trainer.fit(hyperencoder, latent_dataloader)


# def main(
# model_name, input_dir, checkpoint_dir, autoencoder, bottleneck, hf_token=None):
#     """
#     Step 1: Model Config
#         Get model config such that we can get tensor shape and channels
#         from the config to properly configure latent space encoder/decoder
#     """
#     logging.info(f"Getting model config for {model_name}")
#     model_config = get_model_config(model_name, hf_token=hf_token)
#     target_dim = model_config["model"]["pretransform"]["config"]["latent_dim"]

#     """
#     Step 2: Create the hyperencoder and bottleneck
#         We'll have to ensure the autoencoder and bottlenecks are configured
#         properly to receive the latent tensors properly
#     """
#     logging.info(f"Creating latent hyperencoder with {autoencoder} and {bottleneck}")
#     hyperencoder = LatentHyperencoder.factory(
#         target_dim,
#         autoencoder_type=autoencoder,
#         bottleneck_type=bottleneck,
#         bottleneck_kwargs={"levels": [8, 5, 5, 5]},
#     )

#     """
#     Step 3: Create latent dataloader
#         Load our pre-encoded latents from directory
#     """
#     logging.info(f"Creating latent dataloader from {input_dir}")
#     latent_dataloader = LatentDataModule(input_dir, batch_size=2)
#     """
#     Step 4: Run Training
#         - Configure checkpoints and log reconstructed outputs along with
#             saved model checkpoint for spotchecking
#         - Save all metrics along with model checkpoint
#     """
#     logging.info("Running training")
#     run_training(hyperencoder, latent_dataloader, checkpoint_dir)


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


def main():
    args = get_all_args(
        defaults_file="/home/eduardo/Projects/pre_encode_audio/hyperencoder/defaults/train_defaults.ini"
    )
    seed = args.seed

    initialize_logger(f"{args.save_dir}/training.log")

    # logging.info(f"Training hyperencoder with {args.autoencoder}
    # and {args.bottleneck}")
    logging.info(f"CUDE_VISIBLE_DEVICES={environ.get('CUDA_VISIBLE_DEVICES')}")

    set_sharing_strategy("file_system")

    seed_everything(seed, workers=True)

    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    # Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy

            strategy = DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=5e8,
                allgather_bucket_size=5e8,
                load_full_weights=True,
            )
        else:
            strategy = args.strategy
    else:
        strategy = "ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto"

    model = create_model_from_config(model_config)
    training_wrapper = create_training_wrapper_from_config(model_config, model)
    if args.logger == "wandb":
        logger = WandbLogger(project=args.name)
        logger.watch(training_wrapper)

        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = path.join(
                args.save_dir,
                logger.experiment.project,
                logger.experiment.id,
                "checkpoints",
            )
        else:
            checkpoint_dir = None
    elif args.logger == "comet":
        logger = CometLogger(project_name=args.name)
        checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None

    ckpt_callback = ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1
    )
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    val_args = {}

    trainer = Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[
            ckpt_callback,
            # demo_callback,
            # exc_callback,
            save_model_config_callback,
        ],
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,  # If you need to debug validation, change this line
        **val_args,
    )

    trainer.fit(model, train_dl, ckpt_path=args.save_dir)


if __name__ == "__main__":
    """
uv run python -m hyperencoder.train_encoder \
--autoencoder OOBLECK --bottleneck FSQ  \
--input-dir ./data/babyslakh_16k_latents \
--checkpoint-dir=./data/hyperencoder_checkpoints
    """
    main()
