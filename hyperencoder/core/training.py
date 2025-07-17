import sys
import json
import logging
import warnings
from os import path, environ
from pathlib import Path
from argparse import Namespace, ArgumentParser

from tqdm import tqdm
from torch import set_float32_matmul_precision
from lightning import Trainer, seed_everything
from omegaconf import DictConfig
from stable_audio_tools import get_pretrained_model
from torch.multiprocessing import set_sharing_strategy
from lightning.pytorch.loggers import CometLogger, WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichProgressBar
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from hyperencoder.core.utils import get_checkpoint_dir, get_strategy_from_config
from hyperencoder.data.utils import create_datamodule_from_config
from hyperencoder.models.utils import load_model
from hyperencoder.core.callbacks import ExceptionCallback, ModelConfigEmbedderCallback
from hyperencoder.datamodels.tasks import TrainingTaskConfig
from hyperencoder.factories.models import create_hyperencoder_from_config
from hyperencoder.factories.logging import create_experiment_logger_from_config

# from from import (
#     AutoencoderDemoCallback,
#     create_he_training_wrapper_from_config,
#     reload_he_training_wrapper_from_config_and_ckpt,
# )


def train(cfg: TrainingTaskConfig) -> None:
    logger = logging.getLogger(__name__)
    experiment_logger = create_experiment_logger_from_config(cfg.experiment_logging)
    checkpoint_dir = get_checkpoint_dir(cfg.experiment_logging, experiment_logger)
    
    # logger.info(f"CUDA_VISIBLE_DEVICES={environ.get('CUDA_VISIBLE_DEVICES')}")
    # logger.info("Running training script with defaults:")
    # logger.info(json.dumps(args.__dict__, indent=2))
    
    set_sharing_strategy("file_system")
    set_float32_matmul_precision("medium")
    _ = seed_everything(cfg.seed, workers=True)


    # Initialize the wandb or comet logger first to get the experiment ID
    # logger.watch(None)  # Watch can be updated later when the model is created

        # if args.save_dir and isinstance(logger.experiment.id, str):
        #     checkpoint_dir = path.join(
        #         args.save_dir,
        #         logger.experiment.project,
        #         logger.experiment.id,
        #         "checkpoints",
        #     )
        #     log_dir = path.join(
        #         args.save_dir,
        #         logger.experiment.project,
        #         logger.experiment.id,
        #         "logs",
        #     )
    #     else:
    #         checkpoint_dir = None
    #         log_dir = args.save_dir
    # else:
    #     logger = None
    #     checkpoint_dir = args.save_dir if args.save_dir else None
    #     log_dir = args.save_dir

    # Initialize the logger with the dynamically determined log directory
    # training_logger = initialize_logger(
    #     log_dir=log_dir,
    #     experiment_id=logger.experiment.id
    #     if logger and args.logger == "wandb"
    #     else "default",
    # )

    # Redirect stdout and stderr to the logger
    # sys.stdout = LoggerWriter(training_logger, logging.INFO)
    # sys.stderr = LoggerWriter(training_logger, logging.ERROR)


    

    # with open(args.model_config) as f:
    #     model_config = json.load(f)

    # with open(args.dataset_config) as f:
    #     dataset_config = json.load(f)
    logger.info("Creating the pre_encoded data module")
    # logger.info(
    #     f"persistent workers: {args.persistent_workers}, {bool(args.persistent_workers)}"
    # )

    pre_enc_datamodule = create_datamodule_from_config(cfg.data)

    logger.info("Setting up the validation fold for demos")
    pre_enc_datamodule.setup("validate")
    logger.info("Validation fold for demos setup")

    # Set multi-GPU strategy if specified
    strategy = get_strategy_from_config(cfg.training)
    
    logger.info("Loading Hyperencoder")

    # This should make a get_model_from_config function util to be able to load from a checkpoint
    # idk
    model = create_hyperencoder_from_config(cfg.modeling)

    # if cfg.training.pretrained_ckpt_path:
    #     logger.info("LOADING FROM CHECKPOINT!!")
    #     logger.info(cfg.training.pretrained_ckpt_path)
    #     copy_state_dict(model, load_ckpt_state_dict(cfg.training.pretrained_ckpt_path))
    #     training_wrapper = reload_he_training_wrapper_from_config_and_ckpt(
    #         cfg.modeling, model, cfg.training.pretrained_ckpt_path
    #     )
    # else:
        # training_wrapper = create_he_training_wrapper_from_config(model_config, model)

    logger.info("Loaded Hyperencoder")

    # if cfg.experiment_logging.type == "wandb":
        # experiment_logger.watch(training_wrapper)

    ckpt_callback = ModelCheckpoint(
        every_n_epochs=cfg.training.checkpoint_every,
        dirpath=checkpoint_dir,
        save_last=True,
        save_top_k=cfg.training.save_top_k,
        monitor="train/loss",
    )
    save_model_config_callback = ModelConfigEmbedderCallback(cfg.modeling)

    # args_dict = vars(args)
    # args_dict.update({"model_config": model_config})
    # args_dict.update({"dataset_config": dataset_config})

    pre_trained_model, _ = load_model(
        pretrained_name="stabilityai/stable-audio-open-1.0"
    )

    demo_callback = AutoencoderDemoCallback(
        pre_enc_datamodule.val_dataloader(),
        pre_trained_model.pretransform,
        demo_every=model_config["demo"].get("demo_every", 10),
        max_demos=model_config["demo"].get("max_demos", 10),
    )

    if args.logger == "wandb":
        if args.ckpt_path is None:
            push_wandb_config(experiment_logger, args_dict)
        else:
            # If we're resuming a run on wandb, we don't want to push a new config or
            #  change anything. Just reload from the old one, which we do by providing
            # the run id and the ckpt id.
            pass
    elif args.logger == "comet":
        experiment_logger.log_hyperparams(args_dict)

    val_args = {}
    if args.val_every > 0:
        val_args.update(
            {
                "check_val_every_n_epoch": None,
                "val_check_interval": args.val_every,
            }
        )

    exc_callback = ExceptionCallback()
    logger.info("Creating Trainer")

    # progress_bar = RichProgressBar(
    #     theme=RichProgressBarTheme(
    #         description="green_yellow",
    #         progress_bar="green1",
    #         progress_bar_finished="green1",
    #         progress_bar_pulse="#6206E0",
    #         batch_progress="green_yellow",
    #         time="grey82",
    #         processing_speed="grey82",
    #         metrics="grey82",
    #         metrics_text_delimiter="\n",
    #         metrics_format=".3e",
    #     ),
    #     leave=True,
    # )

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
        logger=experiment_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        enable_progress_bar=True,
        # reload_dataloaders_every_n_epochs=0,
        # num_sanity_val_steps=0,  # If you need to debug validation, change this line
        **val_args,
    )

    logger.info("Started Training")
    trainer.fit(
        training_wrapper,
        datamodule=pre_enc_datamodule,
        ckpt_path=args.ckpt_path if args.ckpt_path else None,
    )
    logger.info("Finished Training")
