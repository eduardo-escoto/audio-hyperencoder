# Audio Hyperencoder

A research project for learning semantic representations of music and audio using nested auto-encoders (hyperencoders). Built on top of Stability AI's `stable-audio-tools` with modern configuration management.

## 🎯 **Modern Configuration Architecture**

This project uses **Hydra** for configuration management with a clean task-based system:

- `train.yaml` - Complete training configuration that composes data + model + training + hydra
- `pre_encode.yaml` - Complete pre-encoding configuration that composes data + model + pre_encode + hydra
- `data/default.yaml` - Dataset configuration
- `model/default.yaml` - Model architecture configuration  
- `training/default.yaml` - Training process configuration
- `pre_encode/default.yaml` - Pre-encoding process configuration
- `hydra/default.yaml` - Hydra framework configuration

## 🚀 **Quick Start**

### Training

```bash
# Basic training with default configuration
uv run python -m hyperencoder.cli.main

# Training with component overrides
uv run python -m hyperencoder.cli.main model=custom_model
uv run python -m hyperencoder.cli.main data.batch_size=64
```

### Pre-encoding

```bash
# Basic pre-encoding
uv run python -m hyperencoder.cli.main --config-name=pre_encode \
  pre_encode.input_dir="/path/to/audio/files" \
  pre_encode.output_dir="/path/to/encoded/output"

# Pre-encoding with overrides
uv run python -m hyperencoder.cli.main --config-name=pre_encode \
  pre_encode.input_dir="/path/to/audio/files" \
  pre_encode.output_dir="/path/to/encoded/output" \
  pre_encode.n_devices=2 \
  pre_encode.batch_size=4
```

## 🛠️ **Development Tools**

### Configuration Management

```bash
# Generate configuration files for customization
uv run python -m hyperencoder.cli.utils generate configs

# Generate JSON schemas for IDE support
uv run python -m hyperencoder.cli.utils generate schemas

# Show project information
uv run python -m hyperencoder.cli.utils info
```

### Configuration Customization

1. **Generate configs**: `uv run python -m hyperencoder.cli.utils generate configs`
2. **Edit generated files**: Modify configs in `./hyperencoder-configs/`
3. **Use custom configs**: `uv run python -m hyperencoder.cli.main --config-path=./hyperencoder-configs`

## 📋 **Configuration Details**

### Training Configuration (`train.yaml`)
```yaml
defaults:
- /data: default
- /model: default
- /training: default
- /hydra: default
- _self_
experiment_name: baseline_experiment
description: Basic hyperencoder experiment with default settings
project_name: audio-hyperencoder
seed: 42
task: train
```

### Pre-encoding Configuration (`pre_encode.yaml`)
```yaml
defaults:
- /data: default
- /model: default
- /pre_encode: default
- /hydra: default
- _self_
run_name: baseline_run
description: run of baseline pre encoder over dataset
project_name: audio-hyperencoder
seed: 42
task: pre_encode
```

### Common Configuration Options

**Pre-encoding Options**:
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

**Training Options**:
- `batch_size`: Training batch size (default: 32)
- `num_workers`: Data loading workers (default: 8)
- `devices`: GPU devices to use (default: auto)
- `precision`: Training precision (default: 16-mixed)
- `project`: WandB project name (default: hyperencoder)

## 🎵 **Auxiliary Heads: Multi-Task Learning**

The hyperencoder supports **auxiliary heads** for multi-task learning on MIDI metadata, enabling semantic understanding of musical structure and composition.

### Quick Start with Auxiliary Heads

```bash
# Training with basic song-level features (tempo, velocity, duration, etc.)
uv run python -m hyperencoder.cli.main auxiliary_heads=basic_song_level

# Training with just tempo and velocity prediction
uv run python -m hyperencoder.cli.main auxiliary_heads=tempo_velocity

# Enable auxiliary heads with custom settings
uv run python -m hyperencoder.cli.main auxiliary_heads.enabled=true auxiliary_heads.logging_interval=50
```

### Auxiliary Head Types

**Regression Heads** (continuous values):
- `tempo_bpm`: Music tempo in beats per minute
- `duration_seconds`: Song duration in seconds
- `average_velocity`: Average MIDI velocity (volume)
- `note_density`: Number of notes per second

**Classification Heads** (discrete categories):
- `time_signature_numerator`: Time signature (4/4, 3/4, etc.)
- `key_signature`: Musical key (C major, F# minor, etc.)
- `num_tracks`: Number of MIDI tracks

**Multi-Label Heads** (multiple categories):
- `unique_programs`: MIDI instruments used in the song

### Configuration Presets

#### Basic Song-Level Features (`basic_song_level.yaml`)
```yaml
auxiliary_heads:
  enabled: true
  heads:
    - name: tempo_predictor
      target_key: tempo_bpm
      head_type: regression
      loss_weight: 0.1
    - name: duration_predictor
      target_key: duration_seconds
      head_type: regression
      loss_weight: 0.1
    # ... 7 total heads
```

#### Tempo & Velocity Only (`tempo_velocity.yaml`)
```yaml
auxiliary_heads:
  enabled: true
  heads:
    - name: tempo_predictor
      target_key: tempo_bpm
      head_type: regression
      loss_weight: 0.2
    - name: velocity_predictor
      target_key: average_velocity
      head_type: regression
      loss_weight: 0.2
```

### Custom Auxiliary Head Configuration

Create your own auxiliary heads configuration:

```yaml
# configs/auxiliary_heads/custom.yaml
auxiliary_heads:
  enabled: true
  validation_enabled: true
  logging_interval: 100
  
  heads:
    - name: tempo_predictor
      target_key: tempo_bpm
      head_type: regression
      loss_type: mse
      loss_weight: 0.1
      target_min: 60.0
      target_max: 200.0
      hidden_dims: [512, 256]
      dropout_rate: 0.1
      activation: relu
      
    - name: key_predictor
      target_key: key_signature
      head_type: classification
      loss_type: ce
      loss_weight: 0.1
      num_classes: 24  # 12 major + 12 minor
      hidden_dims: [512, 256]
      dropout_rate: 0.1
      activation: relu
```

### Monitoring and Metrics

During training, auxiliary heads provide additional metrics:

**Training Logs**:
- `train/tempo_predictor_loss`: Loss for tempo prediction
- `train/velocity_predictor_loss`: Loss for velocity prediction
- `train/loss`: Total loss (reconstruction + auxiliary)

**Validation Metrics**:
- `aux/tempo_predictor_mae`: Mean Absolute Error for tempo
- `aux/tempo_predictor_mse`: Mean Squared Error for tempo
- `aux/key_predictor_accuracy`: Classification accuracy for key signature
- `aux/tempo_predictor_mae_denorm`: Interpretable MAE in original units

### Architecture Details

- **Input**: Auxiliary heads process `inner_latents` (post-encoder, pre-decoder)
- **Independence**: Each head is completely independent with no shared layers
- **Integration**: Losses are automatically combined with reconstruction loss via `MultiLoss`
- **Validation**: Comprehensive evaluation metrics computed during validation
- **Error Handling**: Graceful degradation if MIDI metadata is missing

### Benefits

1. **Semantic Learning**: Forces the encoder to capture meaningful musical structure
2. **Improved Latent Space**: Multi-task optimization improves latent representation quality
3. **Interpretability**: Provides insight into what the model learns about music
4. **Flexible Training**: Easy to enable/disable different auxiliary tasks
5. **Research Tool**: Enables systematic study of musical feature learning

## 🔧 **Environment Setup**

### HuggingFace Token
Set your HuggingFace token via environment variable:
```bash
export HF_TOKEN="your_token_here"
```

### Environment File
Create a `.env` file for persistent environment variables:
```
HF_TOKEN=your_token_here
WANDB_API_KEY=your_wandb_key
```

## 🎯 **Advanced Usage**

### Hydra Features

```bash
# Multirun experiments
uv run python -m hyperencoder.cli.main --multirun model=basic,vqvae data.batch_size=32,64

# Tab completion (after shell setup)
uv run python -m hyperencoder.cli.main model=<TAB>

# Configuration composition
uv run python -m hyperencoder.cli.main model=custom_model training.batch_size=64
```

### Output Management

Hydra automatically manages output directories:
- Default: `outputs/YYYY-MM-DD/HH-MM-SS/`
- Contains: logs, configs, model checkpoints
- Access via `${hydra:runtime.output_dir}` in configs

## 🏗️ **Architecture**

- **Hyperencoder Models**: Nested autoencoders in `hyperencoder/models/`
- **Data Loading**: Audio and latent data modules in `hyperencoder/data/`
- **Configuration**: Pydantic models with Hydra integration in `hyperencoder/datamodels/`
- **CLI**: Modern CLI with task dispatch in `hyperencoder/cli/`
- **Training**: PyTorch Lightning training framework in `hyperencoder/training/`