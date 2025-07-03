import sys
import json
import logging
import warnings
from os import path, environ
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from torch import set_float32_matmul_precision
from lightning import Trainer, seed_everything
from stable_audio_tools import get_pretrained_model
from torch.multiprocessing import set_sharing_strategy
from lightning.pytorch.loggers import CometLogger, WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichProgressBar
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.utils import copy_state_dict
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from .data import create_datamodule_from_config
from .models import create_hyperencoder_from_config
from .training import (
    AutoencoderDemoCallback,
    HyperEncoderTrainingWrapper,
    create_he_training_wrapper_from_config,
    reload_he_training_wrapper_from_config_and_ckpt,
)
from .logging_utils import initialize_logger
from .config import TrainingConfig
from .config.hydra_integration import (
    load_training_config,
    validate_and_resolve_paths,
    print_config_summary,
)

# module_base_path = Path(__file__).parent

# Turn off future warnings for vector_quantize_pytorch and torch
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="vector_quantize_pytorch"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


set_float32_matmul_precision("medium")


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def load_model(
    model_config=None,
    model_ckpt_path=None,
    pretrained_name=None,
    pretransform_ckpt_path=None,
    model_half=False,
):
    logger = logging.getLogger()
    logger.info(f"Loading pretrained model {pretrained_name}")
    model = None
    if pretrained_name is not None:
        model, model_config = get_pretrained_model(pretrained_name)

    logger.info("Done loading model")

    return model, model_config


class ExceptionCallback(Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


class ModelConfigEmbedderCallback(Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Replace sys.stdout and sys.stderr with the logger
class LoggerWriter:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        if message.rstrip() != "":
            self.logger.log(self.log_level, message.rstrip())

    def flush(self):
        pass


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration."""
    set_sharing_strategy("file_system")

    # Load and validate training configuration
    training_config = load_training_config(cfg)
    training_config = validate_and_resolve_paths(training_config)
    
    # Print configuration summary
    print_config_summary(training_config)
    
    seed = training_config.seed

    # Initialize the wandb or comet logger first to get the experiment ID
    if training_config.logger == "wandb":
        logger = WandbLogger(
            project=training_config.project, 
            name=training_config.name, 
            save_dir=str(training_config.save_dir) if training_config.save_dir else None, 
            id=training_config.run_id, 
            log_model="all"
        )
        # logger.watch(None)  # Watch can be updated later when the model is created

        if training_config.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = path.join(
                str(training_config.save_dir),
                logger.experiment.project,
                logger.experiment.id,
                "checkpoints",
            )
            log_dir = path.join(
                str(training_config.save_dir),
                logger.experiment.project,
                logger.experiment.id,
                "logs",
            )
        else:
            checkpoint_dir = None
            log_dir = str(training_config.save_dir) if training_config.save_dir else None
    elif training_config.logger == "comet":
        logger = CometLogger(project_name=training_config.name)
        checkpoint_dir = str(training_config.save_dir) if training_config.save_dir else None
        log_dir = str(training_config.save_dir) if training_config.save_dir else None
    else:
        logger = None
        checkpoint_dir = str(training_config.save_dir) if training_config.save_dir else None
        log_dir = str(training_config.save_dir) if training_config.save_dir else None

    # Initialize the logger with the dynamically determined log directory
    training_logger = initialize_logger(
        log_dir=log_dir,
        experiment_id=logger.experiment.id
        if logger and training_config.logger == "wandb"
        else "default",
    )

    # Redirect stdout and stderr to the logger
    # sys.stdout = LoggerWriter(training_logger, logging.INFO)
    # sys.stderr = LoggerWriter(training_logger, logging.ERROR)

    training_logger.info(f"CUDA_VISIBLE_DEVICES={environ.get('CUDA_VISIBLE_DEVICES')}")
    training_logger.info("Running training script with configuration:")
    training_logger.info(json.dumps(training_config.model_dump(), indent=2, default=str))
    seed_everything(seed, workers=True)

    # Load model and dataset configurations
    if training_config.model_config_path:
        with open(training_config.model_config_path) as f:
            model_config = json.load(f)
    else:
        raise ValueError("model_config_path must be specified in training configuration")

    if training_config.dataset_config:
        with open(training_config.dataset_config) as f:
            dataset_config = json.load(f)
    else:
        raise ValueError("dataset_config must be specified in training configuration")
    training_logger.info("Creating the pre_encoded data module")
    training_logger.info(f"persistent workers: {training_config.persistent_workers}, {bool(training_config.persistent_workers)}")
    pre_enc_datamodule = create_datamodule_from_config(
        dataset_config,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        random_seed=training_config.seed,
        persistent_workers=bool(training_config.persistent_workers),
    )
    training_logger.info("Setting up the validation fold for demos")
    pre_enc_datamodule.setup("validate")
    training_logger.info("Validation fold for demos setup")
    # Set multi-GPU strategy if specified
    if training_config.strategy != "auto":  # Only use custom strategy if not auto
        if training_config.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy

            strategy = DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=int(5e8),
                allgather_bucket_size=int(5e8),
                load_full_weights=True,
            )
        else:
            strategy = training_config.strategy
    else:
        strategy = "auto"  # Use Lightning's auto strategy selection
    training_logger.info("Loading Hyperencoder")
    
    model = create_hyperencoder_from_config(model_config)
    
    if training_config.pretrained_ckpt_path:
        training_logger.info("LOADING FROM CHECKPOINT!!")
        training_logger.info(args.pretrained_ckpt_path)
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))
        training_wrapper =  reload_he_training_wrapper_from_config_and_ckpt(model_config, model, args.pretrained_ckpt_path)
    else:
        training_wrapper = create_he_training_wrapper_from_config(model_config, model)

    training_logger.info("Loaded Hyperencoder")
    if args.logger == "wandb":
        logger.watch(training_wrapper)

    ckpt_callback = ModelCheckpoint(
        every_n_epochs=args.checkpoint_every, dirpath=checkpoint_dir, save_last=True, 
        save_top_k=args.save_top_k, monitor="train/loss"
    )
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})

    pre_trained_model, pre_trained_model_config = load_model(
        pretrained_name="stabilityai/stable-audio-open-1.0"
    )

    demo_callback = AutoencoderDemoCallback(
        pre_enc_datamodule.val_dataloader(),
        pre_trained_model.pretransform,
        demo_every=model_config['demo'].get("demo_every", 10),
        max_demos=model_config['demo'].get("max_demos", 10),
    )

    if args.logger == "wandb":
        if args.ckpt_path is None:
            push_wandb_config(logger, args_dict)
        else:
            # If we're resuming a run on wandb, we don't want to push a new config or
            #  change anything. Just reload from the old one, which we do by providing 
            # the run id and the ckpt id.
            pass
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
    training_logger.info("Creating Trainer")

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        ),
        leave=True,
    )

    trainer = Trainer(
        devices=args.devices,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[
            progress_bar,
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
        enable_progress_bar=True,
        # reload_dataloaders_every_n_epochs=0,
        # num_sanity_val_steps=0,  # If you need to debug validation, change this line
        **val_args,
    )

    training_logger.info("Started Training")
    trainer.fit(
        training_wrapper,
        datamodule=pre_enc_datamodule,
        ckpt_path=args.ckpt_path if args.ckpt_path else None,
    )
    training_logger.info("Finished Training")


if __name__ == "__main__":
    main()
