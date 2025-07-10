"""Training task implementation for the CLI.

This module contains the training task that is dispatched from the main CLI.
It fixes the issues with the original training script including:
- Proper use of Hydra composition for model/data configs
- Fixed variable references (training_config instead of args)
- Clean separation of concerns
"""

import sys
import json
import logging
import warnings
from os import path, environ
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch import set_float32_matmul_precision
from lightning import Trainer, seed_everything
from stable_audio_tools import get_pretrained_model
from torch.multiprocessing import set_sharing_strategy
from lightning.pytorch.loggers import CometLogger, WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichProgressBar
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.models.utils import copy_state_dict
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from hyperencoder.data import create_datamodule_from_config
from hyperencoder.models import create_hyperencoder_from_config
from hyperencoder.training import (
    AutoencoderDemoCallback,
    HyperEncoderTrainingWrapper,
    create_he_training_wrapper_from_config,
    reload_he_training_wrapper_from_config_and_ckpt,
)
from hyperencoder.logging_utils import initialize_logger
from hyperencoder.config import TrainingConfig, ModelConfig, DataConfig
from hyperencoder.config.hydra_integration import (
    load_training_config,
    validate_and_resolve_paths,
    print_config_summary,
)

# Turn off future warnings for vector_quantize_pytorch and torch
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="vector_quantize_pytorch"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

set_float32_matmul_precision("medium")


class TqdmHandler(logging.StreamHandler):
    """Custom logging handler for tqdm compatibility."""
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
    """Load a pretrained model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading pretrained model {pretrained_name}")
    model = None
    if pretrained_name is not None:
        model, model_config = get_pretrained_model(pretrained_name)

    logger.info("Done loading model")
    return model, model_config


class ExceptionCallback(Callback):
    """Callback to handle exceptions during training."""
    def on_exception(self, trainer, pl_module, exception):
        print(f"{type(exception).__name__}: {exception}")


class ModelConfigEmbedderCallback(Callback):
    """Callback to embed model config in checkpoint."""
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


def push_wandb_config(logger, config_dict):
    """Push configuration to wandb."""
    # This function might need to be implemented based on your wandb setup
    # For now, we'll just log the config
    if hasattr(logger, 'experiment'):
        logger.experiment.config.update(config_dict)


class LoggerWriter:
    """Writer class to redirect stdout/stderr to logger."""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        if message.rstrip() != "":
            self.logger.log(self.log_level, message.rstrip())

    def flush(self):
        pass


def train_task(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object containing all training settings
    """
    logger = logging.getLogger(__name__)
    logger.info("🎯 Starting training task")
    
    set_sharing_strategy("file_system")

    # Extract training config from the main config
    # This assumes the training config is properly composed by Hydra
    training_config = load_training_config(cfg)
    training_config = validate_and_resolve_paths(training_config)
    
    # Extract model and data configs from Hydra composition
    # These should be available in the main config through defaults
    if 'model' not in cfg:
        raise ValueError("Model configuration not found in config. Make sure to specify model in defaults.")
    if 'data' not in cfg:
        raise ValueError("Data configuration not found in config. Make sure to specify data in defaults.")
    
    # Convert OmegaConf to dict for compatibility with existing functions
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    dataset_config = OmegaConf.to_container(cfg.data, resolve=True)
    
    # Print configuration summary
    print_config_summary(training_config)
    
    seed = training_config.seed

    # Initialize the wandb or comet logger first to get the experiment ID
    if training_config.logger == "wandb":
        wandb_logger = WandbLogger(
            project=training_config.project, 
            name=training_config.name, 
            save_dir=str(training_config.save_dir) if training_config.save_dir else None, 
            id=training_config.run_id, 
            log_model="all"
        )

        if training_config.save_dir and isinstance(wandb_logger.experiment.id, str):
            checkpoint_dir = path.join(
                str(training_config.save_dir),
                wandb_logger.experiment.project,
                wandb_logger.experiment.id,
                "checkpoints",
            )
            log_dir = path.join(
                str(training_config.save_dir),
                wandb_logger.experiment.project,
                wandb_logger.experiment.id,
                "logs",
            )
        else:
            checkpoint_dir = None
            log_dir = str(training_config.save_dir) if training_config.save_dir else None
    elif training_config.logger == "comet":
        wandb_logger = CometLogger(project_name=training_config.name)
        checkpoint_dir = str(training_config.save_dir) if training_config.save_dir else None
        log_dir = str(training_config.save_dir) if training_config.save_dir else None
    else:
        wandb_logger = None
        checkpoint_dir = str(training_config.save_dir) if training_config.save_dir else None
        log_dir = str(training_config.save_dir) if training_config.save_dir else None

    # Initialize the logger with the dynamically determined log directory
    training_logger = initialize_logger(
        log_dir=log_dir,
        experiment_id=wandb_logger.experiment.id
        if wandb_logger and training_config.logger == "wandb"
        else "default",
    )

    training_logger.info(f"CUDA_VISIBLE_DEVICES={environ.get('CUDA_VISIBLE_DEVICES')}")
    training_logger.info("Running training script with configuration:")
    training_logger.info(json.dumps(training_config.model_dump(), indent=2, default=str))
    seed_everything(seed, workers=True)

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
    
    # Fixed: Use training_config instead of undefined args
    if training_config.pretrained_ckpt_path:
        training_logger.info("LOADING FROM CHECKPOINT!!")
        training_logger.info(training_config.pretrained_ckpt_path)
        copy_state_dict(model, load_ckpt_state_dict(training_config.pretrained_ckpt_path))
        training_wrapper = reload_he_training_wrapper_from_config_and_ckpt(
            model_config, model, training_config.pretrained_ckpt_path
        )
    else:
        training_wrapper = create_he_training_wrapper_from_config(model_config, model)

    training_logger.info("Loaded Hyperencoder")
    if training_config.logger == "wandb":
        wandb_logger.watch(training_wrapper)

    ckpt_callback = ModelCheckpoint(
        every_n_epochs=training_config.checkpoint_every, 
        dirpath=checkpoint_dir, 
        save_last=True, 
        save_top_k=training_config.save_top_k, 
        monitor="train/loss"
    )
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    # Create config dict for logging
    config_dict = training_config.model_dump()
    config_dict.update({"model_config": model_config})
    config_dict.update({"dataset_config": dataset_config})

    pre_trained_model, pre_trained_model_config = load_model(
        pretrained_name="stabilityai/stable-audio-open-1.0"
    )

    demo_callback = AutoencoderDemoCallback(
        pre_enc_datamodule.val_dataloader(),
        pre_trained_model.pretransform,
        demo_every=model_config['demo'].get("demo_every", 10),
        max_demos=model_config['demo'].get("max_demos", 10),
    )

    if training_config.logger == "wandb":
        if training_config.ckpt_path is None:
            push_wandb_config(wandb_logger, config_dict)
        else:
            # If we're resuming a run on wandb, we don't want to push a new config
            pass
    elif training_config.logger == "comet":
        wandb_logger.log_hyperparams(config_dict)

    val_args = {}
    if training_config.val_every > 0:
        val_args.update(
            {
                "check_val_every_n_epoch": None,
                "val_check_interval": training_config.val_every,
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
        devices=training_config.devices,
        accelerator="gpu",
        num_nodes=training_config.num_nodes,
        strategy=strategy,
        precision=training_config.precision,
        accumulate_grad_batches=training_config.accum_batches,
        callbacks=[
            progress_bar,
            ckpt_callback,
            demo_callback,
            exc_callback,
            save_model_config_callback,
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=training_config.save_dir,
        gradient_clip_val=training_config.gradient_clip_val,
        enable_progress_bar=True,
        **val_args,
    )

    training_logger.info("Started Training")
    trainer.fit(
        training_wrapper,
        datamodule=pre_enc_datamodule,
        ckpt_path=training_config.ckpt_path if training_config.ckpt_path else None,
    )
    training_logger.info("Finished Training")
    
    logger.info("✅ Training task completed successfully!") 