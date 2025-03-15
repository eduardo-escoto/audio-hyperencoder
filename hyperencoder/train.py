import json
import logging
import warnings
from os import path, environ
from pathlib import Path

from torch import set_float32_matmul_precision
from lightning import Trainer, seed_everything
from prefigure import get_all_args, push_wandb_config
from torch.multiprocessing import set_sharing_strategy
from lightning.pytorch.loggers import CometLogger, WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from .data import create_datamodule_from_config
from .models import create_hyperencoder_from_config
from .training import AutoencoderDemoCallback, create_he_training_wrapper_from_config

module_base_path = Path(__file__).parent

# Turn off future warnings for vector_quantize_pytorch and torch
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="vector_quantize_pytorch"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


set_float32_matmul_precision("medium")


class ExceptionCallback(Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


class ModelConfigEmbedderCallback(Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


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
    set_sharing_strategy("file_system")
    args = get_all_args(
        defaults_file=(module_base_path / "./defaults/train_defaults.ini").resolve()
    )
    seed = args.seed

    initialize_logger(f"{args.save_dir}/training.log")

    # logging.info(f"Training hyperencoder with {args.autoencoder}
    # and {args.bottleneck}")
    logging.info(f"CUDE_VISIBLE_DEVICES={environ.get('CUDA_VISIBLE_DEVICES')}")

    seed_everything(seed, workers=True)

    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    pre_enc_datamodule = create_datamodule_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed,
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

    model = create_hyperencoder_from_config(model_config)
    training_wrapper = create_he_training_wrapper_from_config(model_config, model)

    if args.logger == "wandb":
        logger = WandbLogger(
            project=args.project, name=args.name, save_dir=args.save_dir
        )
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

    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})

    pre_enc_datamodule.setup("validate")
    demo_callback = AutoencoderDemoCallback(
        pre_enc_datamodule.val_dataloader(),
        demo_every=model_config.get("demo_every", 10),
        max_demos=model_config.get("max_demos", 10),
    )

    if args.logger == "wandb":
        push_wandb_config(logger, args_dict)
    elif args.logger == "comet":
        logger.log_hyperparams(args_dict)

    val_args = {}
    if args.val_every > 0:
        val_args.update(
            {
                "check_val_every_n_epoch": None,
                "val_check_interval": args.val_every,
            }
        )

    exc_callback = ExceptionCallback()
    trainer = Trainer(
        devices="auto",
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[
            ckpt_callback,
            demo_callback,
            exc_callback,
            save_model_config_callback,
        ],
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        # reload_dataloaders_every_n_epochs=0,
        # num_sanity_val_steps=0,  # If you need to debug validation, change this line
        **val_args,
    )

    trainer.fit(
        model,
        datamodule=pre_enc_datamodule,
        ckpt_path=args.ckpt_path if args.ckpt_path else None,
    )


if __name__ == "__main__":
    """
    """
    main()
