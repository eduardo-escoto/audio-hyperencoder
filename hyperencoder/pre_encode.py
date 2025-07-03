import json
import pathlib
import warnings
from typing import Dict, List, Optional, Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torchaudio
from tqdm import tqdm
from lightning import Trainer
from safetensors.torch import save_file as sf_save_file
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.models.pretransforms import AutoencoderPretransform

from .data.audio import AudioDataModule
from .modules import AudioAutoEncoder


def login_to_hf(token: Optional[str] = None):
    """Login to HuggingFace Hub."""
    from huggingface_hub import login

    if token is None:
        login()
    else:
        login(token)


def load_model_config(path):
    """Load model configuration from JSON file."""
    with open(path) as f:
        pretransform_config = json.load(f)
        return pretransform_config


def load_model(model_name: str, hf_token: Optional[str] = None):
    """Load pretrained model from HuggingFace Hub."""
    if hf_token is not None:
        login_to_hf(token=hf_token)

    # Download model
    model, pretrained_model_config = get_pretrained_model(model_name)
    return model


def get_input_files(
    input_dir: str,
    file_name: str = "*.wav",
    batch_pattern: str = r"Track\d*",
    batched: bool = True,
    path_file: Optional[str] = None,
) -> Dict[str, List[pathlib.Path]]:
    """Get input files organized by batch."""
    if path_file is not None:
        with open(path_file) as f:
            file_paths = f.read().strip().split("\n")
        return {"batch_0": [pathlib.Path(p) for p in file_paths if p.strip()]}

    input_path = pathlib.Path(input_dir)
    files = [p.resolve() for p in sorted(list(input_path.glob(f"**/{file_name}")))]
    
    from collections import defaultdict
    import regex

    batch_dict = defaultdict(list)
    r = regex.compile(batch_pattern)

    for file in files:
        matches = r.findall(str(file))
        batch_folder = matches[0] if matches else "default_batch"
        batch_dict[batch_folder].append(file)
    
    return dict(batch_dict)


def get_path_up_to_n_parents(path: pathlib.Path, n: int) -> pathlib.Path:
    """Get the file path up to a certain number of parent directories."""
    out_path = "/"
    for _ in range(n):
        path = path.parent
        out_path = "/" + path.name + out_path
    return pathlib.Path(out_path)


def get_path_up_to_regex(path: pathlib.Path, regex_str: str = r"Track\d*") -> pathlib.Path:
    """Get the file path up to a certain number of parent directories."""
    import regex

    r = regex.compile(regex_str)
    full_path = path.resolve()
    iter_path = path.resolve()

    out_path = ""
    while out_path != str(full_path.parent) and r.search(out_path) is None:
        iter_path = iter_path.parent
        out_path = "/" + iter_path.name + out_path

    return pathlib.Path(out_path)


def process_batches(
    device: torch.device,
    reload_pretransform: Any,
    batches: Dict[str, List[pathlib.Path]],
    output_dir_path: pathlib.Path,
    batch_size: int = 1,
    log_failures: bool = True,
    loop_offset: Optional[int] = None,
    n_jobs: Optional[int] = None,
    parent_level: int = 1,
) -> None:
    """Process batches of audio files and save encoded latents."""
    failure_log_path = output_dir_path / "failures.log" if log_failures else None
    
    with open(failure_log_path, "w") if failure_log_path else open("/dev/null", "w") as fail_file:
        for batch_name, file_paths in tqdm(
            batches.items(), total=len(list(batches.keys()))
        ):
            print(f"Processing: {batch_name}")

            batch_tensors = []
            sample_rates = []

            for file in file_paths:
                try:
                    waveform, sample_rate = torchaudio.load(file)
                    batch_tensors.append(waveform)
                    sample_rates.append(sample_rate)
                except Exception as e:
                    print(f"File Failed: {file}")
                    print(e)
                    if log_failures:
                        fail_file.write(f"{str(file)}\r\n")

            if not batch_tensors:
                print(f"No valid files in batch {batch_name}, skipping...")
                continue

            # Process in sub-batches
            sub_batches = []
            for i in range(0, len(batch_tensors), batch_size):
                sub_batch_tensors = batch_tensors[i:i + batch_size]
                sub_sample_rates = sample_rates[i:i + batch_size]

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

            if not sub_batches:
                print(f"No valid latents generated for batch {batch_name}, skipping...")
                continue

            cpu_latents = torch.cat(sub_batches, dim=0)

            # Create output dictionary
            out_dict = {}
            for i in range(min(cpu_latents.shape[0], len(file_paths))):
                out_dict[file_paths[i].stem] = cpu_latents[i]

            # Save to file
            output_fp = pathlib.Path(f"{batch_name}_latent.safetensors")
            out_path = output_dir_path / output_fp

            print(f"Outputting: {out_path.resolve()}")
            if not out_path.parent.exists():
                print(f"Creating directory: {out_path.parent}")
                out_path.parent.mkdir(parents=True, exist_ok=True)

            sf_save_file(out_dict, out_path.absolute())


def audio_encoding_pipeline(
    audio_pretransform: AutoencoderPretransform, 
    input_path: pathlib.Path, 
    n_devices: int = 1
) -> Optional[Any]:
    """Run the audio encoding pipeline using Lightning."""
    dm = AudioDataModule(
        input_path, 
        batch_size=1, 
        file_pattern=r".*\.wav$", 
        group_pattern=r"Track\d*"
    )
    model = AudioAutoEncoder(audio_pretransform.model, encode_only=True)

    trainer = Trainer(devices=n_devices, accelerator="gpu")
    encoded_audio = trainer.predict(model, dm)
    return encoded_audio


@hydra.main(version_base=None, config_path="../configs", config_name="pre_encode")
def main(cfg: DictConfig) -> None:
    """Main pre-encoding function using Hydra configuration."""
    # Configure warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="torch")
    warnings.filterwarnings("ignore", module="stable_audio_tools")
    warnings.filterwarnings("ignore", module="x_transformers")
    warnings.filterwarnings("ignore", module="vector_quantize_pytorch")
    torch.set_float32_matmul_precision("medium")

    # Print configuration
    print("Pre-encoding Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Extract configuration
    pre_encode_cfg = cfg.pre_encode
    
    # Validate required paths
    input_path = pathlib.Path(pre_encode_cfg.input_dir)
    output_path = pathlib.Path(pre_encode_cfg.output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_path}")
    
    if pre_encode_cfg.create_output_dir and not output_path.exists():
        print(f"Creating output directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    elif not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_path}")

    # Load model
    print(f"Loading model: {pre_encode_cfg.model_name}")
    model = load_model(
        model_name=pre_encode_cfg.model_name,
        hf_token=pre_encode_cfg.hf_token
    )

    # Get input files
    print("Scanning for input files...")
    if pre_encode_cfg.use_path_file and pre_encode_cfg.path_file:
        batches = get_input_files(
            input_dir=str(input_path),
            file_name=pre_encode_cfg.file_pattern,
            batch_pattern=pre_encode_cfg.batch_pattern,
            path_file=pre_encode_cfg.path_file
        )
    else:
        batches = get_input_files(
            input_dir=str(input_path),
            file_name=pre_encode_cfg.file_pattern,
            batch_pattern=pre_encode_cfg.batch_pattern
        )
    
    if not batches:
        print("No audio files found to process!")
        return
    
    print(f"Found {len(batches)} batches with {sum(len(files) for files in batches.values())} total files")

    # Process batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    process_batches(
        device=device,
        reload_pretransform=model.pretransform,
        batches=batches,
        output_dir_path=output_path,
        batch_size=pre_encode_cfg.batch_size,
        log_failures=pre_encode_cfg.log_failures
    )
    
    print("Pre-encoding completed successfully!")


if __name__ == "__main__":
    main()
