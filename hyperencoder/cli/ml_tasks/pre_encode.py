"""Pre-encoding task implementation for the CLI.

This module contains the pre-encoding task that is dispatched from the main CLI.
It processes audio files and saves encoded latents using a pretrained model.
Updated to use proper Hydra logging and the new Pydantic configuration system.
"""

import logging
import pathlib
import warnings
from typing import Any

import torch
import torchaudio
from lightning import Trainer
from omegaconf import DictConfig
from safetensors.torch import save_file as sf_save_file
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.models.pretransforms import AutoencoderPretransform

from hyperencoder.modules import AudioAutoEncoder
from hyperencoder.data.audio import AudioDataModule
from hyperencoder.datamodels.hydra_integration import create_pre_encode_config_from_hydra


def login_to_hf(token: str | None = None) -> None:
    """Login to HuggingFace Hub."""
    try:
        from huggingface_hub import login
        
        if token is None:
            login()
        else:
            login(token)
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("HuggingFace Hub not available, skipping login")


def load_model(model_name: str, hf_token: str | None = None) -> Any:
    """Load pretrained model from HuggingFace Hub."""
    logger = logging.getLogger(__name__)
    
    try:
        if hf_token is not None:
            login_to_hf(token=hf_token)

        logger.info(f"📥 Loading model: {model_name}")
        model, pretrained_model_config = get_pretrained_model(model_name)
        logger.info(f"✅ Successfully loaded model: {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_name}: {e}")
        raise


def get_input_files(
    input_dir: str,
    file_name: str = "*.wav",
    batch_pattern: str = r"Track\d*",
    path_file: str | None = None,
) -> dict[str, list[pathlib.Path]]:
    """Get input files organized by batch."""
    logger = logging.getLogger(__name__)
    
    try:
        if path_file is not None:
            logger.info(f"📋 Loading file paths from: {path_file}")
            with open(path_file) as f:
                file_paths = f.read().strip().split("\n")
            valid_paths = [pathlib.Path(p) for p in file_paths if p.strip()]
            logger.info(f"📁 Found {len(valid_paths)} files in path file")
            return {"batch_0": valid_paths}

        input_path = pathlib.Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_path}")
            
        logger.info(f"🔍 Scanning directory: {input_path}")
        files = [p.resolve() for p in sorted(list(input_path.glob(f"**/{file_name}")))]
        logger.info(f"📁 Found {len(files)} files matching pattern: {file_name}")

        from collections import defaultdict
        import regex

        batch_dict = defaultdict(list)
        r = regex.compile(batch_pattern)

        for file in files:
            matches = r.findall(str(file))
            batch_folder = matches[0] if matches else "default_batch"
            batch_dict[batch_folder].append(file)

        logger.info(f"📊 Organized files into {len(batch_dict)} batches")
        return dict(batch_dict)
        
    except Exception as e:
        logger.error(f"❌ Failed to get input files: {e}")
        raise


def process_batches(
    device: torch.device,
    reload_pretransform: Any,
    batches: dict[str, list[pathlib.Path]],
    output_dir_path: pathlib.Path,
    batch_size: int = 1,
    log_failures: bool = True,
) -> None:
    """Process batches of audio files and save encoded latents."""
    logger = logging.getLogger(__name__)
    
    try:
        failure_log_path = output_dir_path / "failures.log" if log_failures else None
        total_processed = 0
        total_failed = 0

        with (
            open(failure_log_path, "w") if failure_log_path else open("/dev/null", "w")
        ) as fail_file:
            
            for batch_name, file_paths in batches.items():
                logger.info(f"🔄 Processing batch: {batch_name} ({len(file_paths)} files)")

                batch_tensors = []
                sample_rates = []
                failed_files = []

                # Load audio files
                for file in file_paths:
                    try:
                        waveform, sample_rate = torchaudio.load(file)
                        batch_tensors.append(waveform)
                        sample_rates.append(sample_rate)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load file {file}: {e}")
                        failed_files.append(str(file))
                        if log_failures:
                            fail_file.write(f"{str(file)}\n")

                total_failed += len(failed_files)

                if not batch_tensors:
                    logger.warning(f"⚠️ No valid files in batch {batch_name}, skipping...")
                    continue

                logger.info(f"📊 Successfully loaded {len(batch_tensors)} files, {len(failed_files)} failed")

                # Process in sub-batches
                sub_batches = []
                for i in range(0, len(batch_tensors), batch_size):
                    sub_batch_tensors = batch_tensors[i : i + batch_size]
                    sub_sample_rates = sample_rates[i : i + batch_size]

                    try:
                        preprocessed_audio = (
                            reload_pretransform.model.preprocess_audio_list_for_encoder(
                                sub_batch_tensors, sub_sample_rates
                            )
                        )
                        preprocessed_audio = preprocessed_audio.to(device)

                        latents = reload_pretransform.model.encode_audio(preprocessed_audio)
                        cpu_latents = latents.to("cpu")
                        torch.cuda.empty_cache()

                        sub_batches.append(cpu_latents)
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process sub-batch {i//batch_size + 1}: {e}")
                        raise

                if not sub_batches:
                    logger.warning(f"⚠️ No valid latents generated for batch {batch_name}, skipping...")
                    continue

                # Concatenate all sub-batches
                cpu_latents = torch.cat(sub_batches, dim=0)
                total_processed += cpu_latents.shape[0]

                # Create output dictionary
                out_dict = {}
                for i in range(min(cpu_latents.shape[0], len(file_paths))):
                    if i < len(file_paths):
                        out_dict[file_paths[i].stem] = cpu_latents[i]

                # Save to file
                output_fp = pathlib.Path(f"{batch_name}_latent.safetensors")
                out_path = output_dir_path / output_fp

                logger.info(f"💾 Saving latents to: {out_path}")
                if not out_path.parent.exists():
                    logger.info(f"📁 Creating directory: {out_path.parent}")
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                sf_save_file(out_dict, out_path.absolute())
                logger.info(f"✅ Saved {len(out_dict)} latents for batch {batch_name}")

        logger.info(f"🎉 Processing complete! {total_processed} files processed, {total_failed} failed")
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        raise


def pre_encode_task(cfg: DictConfig) -> None:
    """Main pre-encoding function using Hydra configuration.

    Args:
        cfg: Hydra configuration object containing all pre-encoding settings
    """
    logger = logging.getLogger(__name__)
    logger.info("🔄 Starting pre-encoding task")

    try:
        # Configure warnings
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", module="torch")
        warnings.filterwarnings("ignore", module="stable_audio_tools")
        warnings.filterwarnings("ignore", module="x_transformers")
        warnings.filterwarnings("ignore", module="vector_quantize_pytorch")
        torch.set_float32_matmul_precision("medium")

        # Create configuration from Hydra config
        pre_encode_config = create_pre_encode_config_from_hydra(cfg)
        
        logger.info("🔧 Pre-encoding Configuration:")
        logger.info(f"  📁 Input Directory: {pre_encode_config.input_dir}")
        logger.info(f"  📁 Output Directory: {pre_encode_config.output_dir}")
        logger.info(f"  🤖 Model: {pre_encode_config.model_name}")
        logger.info(f"  📊 Batch Size: {pre_encode_config.batch_size}")
        logger.info(f"  🔍 File Pattern: {pre_encode_config.file_pattern}")
        logger.info(f"  📋 Batch Pattern: {pre_encode_config.batch_pattern}")

        # Validate required paths
        input_path = pathlib.Path(pre_encode_config.input_dir)
        output_path = pathlib.Path(pre_encode_config.output_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_path}")

        if pre_encode_config.create_output_dir and not output_path.exists():
            logger.info(f"📁 Creating output directory: {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)
        elif not output_path.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_path}")

        # Load model
        model = load_model(
            model_name=pre_encode_config.model_name,
            hf_token=pre_encode_config.hf_token
        )

        # Get input files
        logger.info("🔍 Scanning for input files...")
        if pre_encode_config.use_path_file and pre_encode_config.path_file:
            batches = get_input_files(
                input_dir=str(input_path),
                file_name=pre_encode_config.file_pattern,
                batch_pattern=pre_encode_config.batch_pattern,
                path_file=pre_encode_config.path_file,
            )
        else:
            batches = get_input_files(
                input_dir=str(input_path),
                file_name=pre_encode_config.file_pattern,
                batch_pattern=pre_encode_config.batch_pattern,
            )

        if not batches:
            logger.warning("⚠️ No audio files found to process!")
            return

        total_files = sum(len(files) for files in batches.values())
        logger.info(f"📊 Found {len(batches)} batches with {total_files} total files")

        # Process batches
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ Using device: {device}")

        process_batches(
            device=device,
            reload_pretransform=model.pretransform,
            batches=batches,
            output_dir_path=output_path,
            batch_size=pre_encode_config.batch_size,
            log_failures=pre_encode_config.log_failures,
        )

        logger.info("✅ Pre-encoding task completed successfully!")

    except Exception as e:
        logger.error(f"❌ Pre-encoding failed: {e}")
        logger.exception("Full traceback:")
        raise
