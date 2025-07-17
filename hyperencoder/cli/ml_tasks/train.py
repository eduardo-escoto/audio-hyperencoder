"""Training task implementation for the CLI.

This module contains the training task that is dispatched from the main CLI.
Updated to use proper Hydra logging and the new Pydantic configuration system.
"""

import logging
import warnings
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CometLogger
from omegaconf import DictConfig
from torch import set_float32_matmul_precision
from torch.multiprocessing import set_sharing_strategy

from hyperencoder.data import create_datamodule_from_config
from hyperencoder.models import create_hyperencoder_from_config
from hyperencoder.training import (
    AutoencoderDemoCallback,
    create_he_training_wrapper_from_config,
    reload_he_training_wrapper_from_config_and_ckpt,
)
from hyperencoder.datamodels.hydra_integration import (
    create_training_config_from_hydra,
    create_model_config_from_hydra,
    create_data_config_from_hydra,
    get_experiment_output_dir,
    validate_and_resolve_paths,
    print_config_summary,
)

# Configure warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="vector_quantize_pytorch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
set_float32_matmul_precision("medium")


def train_task(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration.

    Args:
        cfg: Hydra configuration object containing all training settings
    """
    import time
    
    logger = logging.getLogger(__name__)
    logger.info("🎯 Starting training task")
    start_time = time.time()

    try:
        # Set multiprocessing strategy
        logger.debug("🔧 Setting multiprocessing strategy to 'file_system'")
        set_sharing_strategy("file_system")

        # Create configuration objects from Hydra config
        logger.info("📋 Creating configuration objects from Hydra config")
        config_start = time.time()
        
        training_config = create_training_config_from_hydra(cfg)
        model_config = create_model_config_from_hydra(cfg)
        data_config = create_data_config_from_hydra(cfg)
        
        logger.debug(f"⏱️ Configuration creation took {time.time() - config_start:.2f}s")
        logger.info(f"📊 Training config: {training_config.name} (seed: {training_config.seed})")
        logger.info(f"🧠 Model config created successfully")
        logger.info(f"💾 Data config created successfully")

        # Validate and resolve paths
        training_config = validate_and_resolve_paths(training_config)

        # Print configuration summary
        print_config_summary(training_config)

        # Set random seed
        seed_everything(training_config.seed, workers=True)
        logger.info(f"🎲 Set random seed to {training_config.seed}")

        # Create experiment logger
        experiment_logger = None
        experiment_dir = get_experiment_output_dir(cfg, training_config.logger)
        
        if training_config.logger == "wandb":
            experiment_logger = WandbLogger(
                project=training_config.project,
                name=training_config.name,
                save_dir=str(experiment_dir),
                id=training_config.run_id,
                log_model="all",
            )
            # Update experiment directory with actual wandb run ID
            if hasattr(experiment_logger.experiment, 'id'):
                experiment_dir = experiment_dir / experiment_logger.experiment.id
                
        elif training_config.logger == "comet":
            experiment_logger = CometLogger(
                project_name=training_config.project,
                experiment_name=training_config.name,
                save_dir=str(experiment_dir),
            )
            
        # Create output directories
        checkpoint_dir = experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Experiment directory: {experiment_dir}")
        logger.info(f"💾 Checkpoint directory: {checkpoint_dir}")

        # Create data module
        logger.info("🔧 Creating data module")
        data_start = time.time()
        datamodule = create_datamodule_from_config(data_config)
        logger.debug(f"⏱️ Data module creation took {time.time() - data_start:.2f}s")
        
        # Log data module info
        try:
            # Try to get dataset info if available
            if hasattr(datamodule, 'train_dataloader'):
                train_dl = datamodule.train_dataloader()
                if hasattr(train_dl, 'dataset') and hasattr(train_dl.dataset, '__len__'):
                    logger.info(f"📊 Training dataset size: {len(train_dl.dataset):,} samples")
            if hasattr(datamodule, 'val_dataloader'):
                val_dl = datamodule.val_dataloader()
                if hasattr(val_dl, 'dataset') and hasattr(val_dl.dataset, '__len__'):
                    logger.info(f"📊 Validation dataset size: {len(val_dl.dataset):,} samples")
        except Exception as e:
            logger.debug(f"Could not get dataset size info: {e}")

        # Create model
        logger.info("🔧 Creating hyperencoder model")
        model_start = time.time()
        hyperencoder = create_hyperencoder_from_config(model_config)
        logger.debug(f"⏱️ Model creation took {time.time() - model_start:.2f}s")
        
        # Log model info
        if hasattr(hyperencoder, 'parameters'):
            total_params = sum(p.numel() for p in hyperencoder.parameters())
            trainable_params = sum(p.numel() for p in hyperencoder.parameters() if p.requires_grad)
            logger.info(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log memory usage if available
        try:
            import psutil
            import torch
            memory = psutil.virtual_memory()
            logger.debug(f"💾 System memory: {memory.percent}% used ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_cached = torch.cuda.memory_reserved(i) / 1024**3
                    logger.debug(f"🎮 GPU {i} memory: {mem_allocated:.1f}GB allocated, {mem_cached:.1f}GB cached")
        except ImportError:
            logger.debug("💾 Memory monitoring not available (psutil not installed)")
        except Exception as e:
            logger.debug(f"💾 Could not get memory info: {e}")

        # Create/reload training wrapper
        if training_config.pretrained_ckpt_path:
            logger.info(f"📥 Loading from checkpoint: {training_config.pretrained_ckpt_path}")
            training_wrapper = reload_he_training_wrapper_from_config_and_ckpt(
                model_config,
                hyperencoder,
                str(training_config.pretrained_ckpt_path),
            )
        else:
            logger.info("🔧 Creating new training wrapper")
            training_wrapper = create_he_training_wrapper_from_config(
                training_config,
                model_config,
                hyperencoder,
            )

        # Set up wandb model watching
        if training_config.logger == "wandb" and isinstance(experiment_logger, WandbLogger):
            experiment_logger.watch(training_wrapper)

        # Create callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=training_config.checkpoint_every,
            dirpath=checkpoint_dir,
            save_last=True,
            save_top_k=training_config.save_top_k,
            monitor="train/loss",
        )
        callbacks.append(checkpoint_callback)

        # Demo callback if enabled
        if model_config.demo:
            try:
                from stable_audio_tools import get_pretrained_model
                
                pretrained_model, _ = get_pretrained_model("stabilityai/stable-audio-open-1.0")
                if (hasattr(pretrained_model, 'pretransform') and 
                    pretrained_model.pretransform is not None and 
                    hasattr(pretrained_model.pretransform, 'model')):
                    demo_callback = AutoencoderDemoCallback(
                        datamodule.val_dataloader(),
                        pretrained_model.pretransform.model,  # type: ignore[report-argument-type]
                        demo_every=model_config.demo.demo_every,
                        sample_rate=model_config.demo.sample_rate,
                        max_demos=model_config.demo.max_demos,
                    )
                    callbacks.append(demo_callback)
                    logger.info("✅ Added demo callback")
                else:
                    logger.warning("⚠️ Pretrained model does not have valid pretransform.model")
            except Exception as e:
                logger.warning(f"⚠️ Could not create demo callback: {e}")

        # Validation settings
        val_kwargs = {}
        if training_config.val_every > 0:
            val_kwargs.update({
                "check_val_every_n_epoch": None,
                "val_check_interval": training_config.val_every,
            })

        # Create trainer
        logger.info("🏋️ Creating trainer")
        trainer = Trainer(
            devices=training_config.devices,
            accelerator="gpu",
            num_nodes=training_config.num_nodes,
            strategy=training_config.strategy,
            precision=training_config.precision,
            accumulate_grad_batches=training_config.accum_batches,
            callbacks=callbacks,
            logger=experiment_logger,
            log_every_n_steps=training_config.log_every_n_steps,
            max_epochs=training_config.max_epochs,
            default_root_dir=str(experiment_dir),
            gradient_clip_val=training_config.gradient_clip_val,
            enable_progress_bar=True,
            **val_kwargs,
        )

        # Start training
        logger.info("🚀 Starting training")
        logger.info(f"📊 Training setup: {training_config.max_epochs} epochs, {training_config.devices} devices")
        logger.info(f"📊 Logging every {training_config.log_every_n_steps} steps")
        
        training_start = time.time()
        trainer.fit(
            training_wrapper,
            datamodule=datamodule,
            ckpt_path=str(training_config.ckpt_path) if training_config.ckpt_path else None,
        )
        training_duration = time.time() - training_start
        
        logger.info(f"✅ Training completed successfully in {training_duration:.2f}s ({training_duration/60:.1f}m)")
        logger.info(f"📊 Total pipeline time: {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.exception("Full traceback:")
        raise
