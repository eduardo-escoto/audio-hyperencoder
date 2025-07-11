"""
Pre-encoding task for hyperencoder using Hydra configuration.

This module provides the pre-encoding entry point for processing audio files
into latent representations using a pre-trained autoencoder.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from stable_audio_tools import get_pretrained_model
from torch.utils.data import DataLoader
import torchaudio

from hyperencoder.data.audio import AudioDataset


@hydra.main(version_base=None, config_path="../../configs", config_name="pre_encode")
def main(cfg: DictConfig) -> None:
    """Main pre-encoding function configured with Hydra.
    
    Args:
        cfg: Hydra configuration object containing all pre-encoding parameters
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting pre-encoding task")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Get pre-encoding configuration
    pre_encode_config = cfg.get("pre_encode", {})
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load pre-trained model
    logger.info("Loading pre-trained autoencoder model")
    model_name = pre_encode_config.get("model_name", "stabilityai/stable-audio-open-1.0")
    model, model_config = get_pretrained_model(model_name)
    model = model.to(device)
    model.eval()
    
    # Set up input and output paths
    input_dir = Path(pre_encode_config.get("input_dir", "data/audio"))
    output_dir = Path(pre_encode_config.get("output_dir", "data/latents"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create audio dataset
    logger.info("Creating audio dataset")
    audio_extensions = pre_encode_config.get("audio_extensions", [".wav", ".mp3", ".flac"])
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"**/*{ext}"))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        logger.warning("No audio files found!")
        return
    
    # Process files
    batch_size = pre_encode_config.get("batch_size", 1)
    sample_rate = pre_encode_config.get("sample_rate", 44100)
    crop_length = pre_encode_config.get("crop_length", 32768)
    
    logger.info(f"Processing {len(audio_files)} files with batch size {batch_size}")
    
    processed_count = 0
    
    try:
        for audio_file in audio_files:
            try:
                # Load audio
                waveform, sr = torchaudio.load(audio_file)
                
                # Resample if needed
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = resampler(waveform)
                
                # Crop or pad to desired length
                if waveform.shape[-1] > crop_length:
                    # Random crop
                    start_idx = torch.randint(0, waveform.shape[-1] - crop_length + 1, (1,)).item()
                    waveform = waveform[:, start_idx:start_idx + crop_length]
                elif waveform.shape[-1] < crop_length:
                    # Pad with zeros
                    pad_length = crop_length - waveform.shape[-1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_length))
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Add batch dimension and move to device
                waveform = waveform.unsqueeze(0).to(device)
                
                # Encode to latents
                with torch.no_grad():
                    if hasattr(model, 'encode'):
                        latents = model.encode(waveform)
                    elif hasattr(model, 'pretransform') and hasattr(model.pretransform, 'encode'):
                        latents = model.pretransform.encode(waveform)
                    else:
                        logger.error("Model does not have encode method")
                        continue
                
                # Save latents
                output_file = output_dir / f"{audio_file.stem}.pt"
                torch.save(latents.cpu(), output_file)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(audio_files)} files")
                    
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                continue
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    
    logger.info(f"✅ Pre-encoding completed. Processed {processed_count}/{len(audio_files)} files")


if __name__ == "__main__":
    main()
