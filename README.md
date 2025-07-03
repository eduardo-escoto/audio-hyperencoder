# Audio Hyperencoder

This project uses Hydra for configuration management with semantically meaningful config files:

- `configs/train.yaml` - Training configuration (used by `hyperencoder.train`)
- `configs/pre_encode.yaml` - Pre-encoding configuration (used by `hyperencoder.pre_encode`)

## Training

To run training:
```bash
uv run --env-file .env python -m hyperencoder.train
```

## Pre-encoding

The pre-encoding script converts audio files to latent representations using stable-audio-tools models. It now uses Hydra for configuration management.

### Basic Usage

```bash
# Basic pre-encoding with required paths
uv run python -m hyperencoder.pre_encode \
  pre_encode.input_dir="/path/to/audio/files" \
  pre_encode.output_dir="/path/to/encoded/output"

# With additional options
uv run python -m hyperencoder.pre_encode \
  pre_encode.input_dir="/path/to/audio/files" \
  pre_encode.output_dir="/path/to/encoded/output" \
  pre_encode.n_devices=2 \
  pre_encode.batch_size=4 \
  pre_encode.file_pattern="*.wav" \
  pre_encode.hf_token="your_hf_token_here"
```

### Configuration Files

You can also create custom configuration files:

```bash
# Use a specific pre-encode config variant
uv run python -m hyperencoder.pre_encode --config-name=pre_encode \
  --config-path=configs/pre_encode \
  pre_encode.input_dir="/path/to/audio"
```

### Environment Variables

Set your HuggingFace token via environment variable:
```bash
export HF_TOKEN="your_token_here"
uv run python -m hyperencoder.pre_encode \
  pre_encode.input_dir="/path/to/audio" \
  pre_encode.output_dir="/path/to/output"
```

### Configuration Options

- `model_name`: HuggingFace model name (default: "stabilityai/stable-audio-open-1.0")
- `input_dir`: Directory containing audio files (required)
- `output_dir`: Directory to save encoded latents (required)
- `file_pattern`: Pattern to match audio files (default: "*.wav")
- `batch_pattern`: Regex pattern to group files into batches (default: "Track\\d*")
- `n_devices`: Number of GPU devices to use (default: 1)
- `batch_size`: Batch size for processing (default: 1)
- `hf_token`: HuggingFace token for gated models (optional)
- `create_output_dir`: Create output directory if it doesn't exist (default: true)
- `log_failures`: Log failed files to failures.log (default: true)