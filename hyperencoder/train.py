import sys
import json
import logging
import warnings
from os import path, environ
from pathlib import Path

from tqdm import tqdm
from torch import set_float32_matmul_precision
from lightning import Trainer, seed_everything
from prefigure import get_all_args, push_wandb_config
from stable_audio_tools import get_pretrained_model
from torch.multiprocessing import set_sharing_strategy
from lightning.pytorch.loggers import CometLogger, WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from .data import create_datamodule_from_config
from .models import create_hyperencoder_from_config
from .training import AutoencoderDemoCallback, create_he_training_wrapper_from_config
from .logging_utils import initialize_logger

module_base_path = Path(__file__).parent

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


def main():
    set_sharing_strategy("file_system")
    args = get_all_args(
        defaults_file=str(
            (module_base_path / "./defaults/train_defaults.ini").resolve()
        )
    )
    seed = args.seed

    # Initialize the wandb or comet logger first to get the experiment ID
    if args.logger == "wandb":
        logger = WandbLogger(
            project=args.project, name=args.name, save_dir=args.save_dir
        )
        # logger.watch(None)  # Watch can be updated later when the model is created

        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = path.join(
                args.save_dir,
                logger.experiment.project,
                logger.experiment.id,
                "checkpoints",
            )
            log_dir = path.join(
                args.save_dir,
                logger.experiment.project,
                logger.experiment.id,
                "logs",
            )
        else:
            checkpoint_dir = None
            log_dir = args.save_dir
    elif args.logger == "comet":
        logger = CometLogger(project_name=args.name)
        checkpoint_dir = args.save_dir if args.save_dir else None
        log_dir = args.save_dir
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None
        log_dir = args.save_dir

    # Initialize the logger with the dynamically determined log directory
    training_logger = initialize_logger(
        log_dir=log_dir,
        experiment_id=logger.experiment.id
        if logger and args.logger == "wandb"
        else "default",
    )

    # Redirect stdout and stderr to the logger
    # sys.stdout = LoggerWriter(training_logger, logging.INFO)
    # sys.stderr = LoggerWriter(training_logger, logging.ERROR)

    training_logger.info(f"CUDA_VISIBLE_DEVICES={environ.get('CUDA_VISIBLE_DEVICES')}")
    training_logger.info("Running training script with defaults:")
    training_logger.info(json.dumps(args.__dict__, indent=2))
    seed_everything(seed, workers=True)

    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)
    training_logger.info("Creating the pre_encoded data module")
    pre_enc_datamodule = create_datamodule_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed,
        persistent_workers=args.persistent_workers,
    )
    training_logger.info("Setting up the validation fold for demos")
    pre_enc_datamodule.setup("validate")
    training_logger.info("Validation fold for demos setup")
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
    training_logger.info("Loading Hyperencoder")
    model = create_hyperencoder_from_config(model_config)
    training_wrapper = create_he_training_wrapper_from_config(model_config, model)
    training_logger.info("Loaded Hyperencoder")
    if args.logger == "wandb":
        logger.watch(training_wrapper)

    ckpt_callback = ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1
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
