# Hyperencoder Hydra & OmegaConf Migration Plan

## Overview
This plan details the migration from `prefigure` (INI-based) configuration to Hydra + OmegaConf (YAML-based) configuration management for the hyperencoder deep learning project.

## Current Configuration Analysis

### Current Structure:
1. **Training Config**: `prefigure` + INI file (`hyperencoder/defaults/train_defaults.ini`)
2. **Model Config**: JSON files (`configs/models/*.json`)  
3. **Data Config**: JSON files (`configs/data/*.json`)
4. **Pre-encoding**: Basic function signature, no proper arg parsing

### Current Issues:
- Hard-coded absolute paths
- Mixed configuration formats (INI + JSON)
- Manual JSON loading in multiple places
- Limited composition/inheritance capabilities
- Prefigure dependency for simple argument parsing

## Migration Strategy

### Phase 1: Dependency & Structure Setup

#### 1.1 Update Dependencies
- **Remove**: `prefigure>=0.0.9` from `pyproject.toml`
- **Add**: `hydra-core>=1.3.0`, `omegaconf>=2.3.0`

#### 1.2 Create New Configuration Structure
```
configs/
├── config.yaml                    # Main config entry point
├── experiment/                    # Experiment-specific overrides
│   ├── hyperencoder_basic.yaml
│   ├── hyperencoder_vae.yaml
│   └── hyperencoder_fsq.yaml
├── model/                         # Model architecture configs
│   ├── hyperencoder.yaml         # Default hyperencoder
│   ├── hyperencoder_basic.yaml   # Basic variant
│   └── hyperencoder_large.yaml   # Future: larger model
├── data/                          # Dataset configurations  
│   ├── hyperencoder.yaml         # Default dataset
│   ├── pre_encoded.yaml          # Pre-encoded latents
│   └── babyslakh.yaml            # Specific dataset variant
├── training/                      # Training-specific configs
│   ├── default.yaml              # Default training params
│   ├── distributed.yaml          # Multi-GPU settings
│   └── debug.yaml                # Debug/dev settings
├── pre_encode/                    # Pre-encoding configs
│   ├── default.yaml              # Default pre-encoding
│   └── batch_processing.yaml     # Batch processing variant
└── hydra/                         # Hydra runtime configs
    └── default.yaml              # Hydra behavior config
```

### Phase 2: Configuration File Migration

#### 2.1 Convert INI to YAML (`hyperencoder/defaults/train_defaults.ini` → `configs/training/default.yaml`)
```yaml
# configs/training/default.yaml
# Training configuration
name: hyperencoder_vae
project: hyperencoder

# Training hyperparameters
batch_size: 256
num_workers: 8
seed: 42
accum_batches: 1

# Checkpointing
checkpoint_every: 10
val_every: -1
save_top_k: 10

# Hardware & Optimization  
num_nodes: 1
strategy: auto
precision: "16-mixed"
gradient_clip_val: 0.0

# Paths (now relative)
save_dir: ???  # Must be specified by user
ckpt_path: null
pretrained_ckpt_path: null
pretransform_ckpt_path: null

# Logging
logger: wandb

# Recovery
recover: false
```

#### 2.2 Convert Model JSON to YAML (`configs/models/*.json` → `configs/model/*.yaml`)
```yaml
# configs/model/hyperencoder.yaml
_target_: hyperencoder.models.hyperencoder.HyperEncoder

encoder:
  _target_: hyperencoder.models.encoders.OobleckEncoder
  in_channels: 64
  channels: 4
  latent_dim: 4
  c_mults: [16, 8, 4, 2, 2]
  strides: [8, 8, 4, 4, 1]
  use_snake: false

decoder:
  _target_: hyperencoder.models.decoders.OobleckDecoder
  out_channels: 64
  channels: 4
  latent_dim: 4
  c_mults: [16, 8, 4, 2, 2]
  strides: [8, 8, 4, 4, 1]  
  use_snake: false
  final_tanh: false

bottleneck:
  _target_: hyperencoder.models.bottlenecks.FSQBottleneck
  levels: [8, 5, 5, 5]

# Model dimensions
latent_dim: 4
in_channels: 64
out_channels: 64

# Training configuration
training:
  optimizer_configs:
    hyperencoder:
      optimizer:
        _target_: torch.optim.AdamW
        lr: 5e-5
        betas: [0.9, 0.999]
        weight_decay: 1e-3
      scheduler:
        _target_: stable_audio_tools.training.lr_schedulers.InverseLR
        inv_gamma: 1000000
        power: 0.5
        warmup: 0.99

# Demo settings
demo:
  demo_every: 20
  max_demos: 10
```

#### 2.3 Convert Data JSON to YAML (`configs/data/*.json` → `configs/data/*.yaml`)
```yaml
# configs/data/hyperencoder.yaml
_target_: hyperencoder.data.create_datamodule_from_config

dataset_type: latents_for_hyperencoder
split_type: auto
loading_strategy: lazy

# Split ratios
train_split_pct: 0.8
val_split_pct: 0.1  
test_split_pct: 0.1

# Datasets
datasets:
  - path: ${oc.env:DATA_ROOT}/pre_encoded_babyslakh
```

#### 2.4 Create Main Config (`configs/config.yaml`)
```yaml
# @package _global_
defaults:
  - model: hyperencoder
  - data: hyperencoder  
  - training: default
  - _self_

# Experiment settings
experiment_name: ???
run_name: ${training.name}

# Hydra configuration
hydra:
  run:
    dir: ${training.save_dir}/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: ${training.save_dir}/${experiment_name}
    subdir: ${hydra:job.num}

# Enable config resolution
defaults:
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled
```

### Phase 3: Code Migration

#### 3.1 Update Training Script (`hyperencoder/train.py`)

**Key Changes:**
- Replace `from prefigure import get_all_args, push_wandb_config`
- Add `@hydra.main(version_base=None, config_path="../configs", config_name="config")`
- Replace `args = get_all_args(...)` with Hydra config injection
- Use `OmegaConf.to_yaml(cfg)` for logging instead of `json.dumps(args.__dict__)`
- Replace manual JSON loading with direct config access

**Before:**
```python
def main():
    args = get_all_args(defaults_file=str((module_base_path / "./defaults/train_defaults.ini").resolve()))
    with open(args.model_config) as f:
        model_config = json.load(f)
```

**After:**
```python
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Config is already loaded and composed
    model_config = cfg.model
    training_config = cfg.training
```

#### 3.2 Create Pre-encoding Script Integration (`hyperencoder/pre_encode.py`)

Add proper Hydra integration:
```python
@hydra.main(version_base=None, config_path="../configs", config_name="pre_encode")
def main(cfg: DictConfig) -> None:
    # Implementation with proper config handling
```

#### 3.3 Update Configuration Loading Utilities

Create `hyperencoder/config_utils.py`:
```python
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

def instantiate_model(model_cfg: DictConfig):
    """Instantiate model from config using Hydra's instantiate."""
    return instantiate(model_cfg)

def instantiate_datamodule(data_cfg: DictConfig, **kwargs):
    """Instantiate datamodule with overrides."""
    return instantiate(data_cfg, **kwargs)
```

### Phase 4: Advanced Features

#### 4.1 Experiment Configuration
```yaml
# configs/experiment/hyperencoder_basic.yaml
# @package _global_
defaults:
  - override /model: hyperencoder_basic
  - override /training: default

experiment_name: hyperencoder_basic_experiment
training:
  name: hyperencoder_basic
  batch_size: 128  # Override
```

#### 4.2 Multi-run Configuration Support
```yaml
# configs/training/sweep.yaml
defaults:
  - default

# Hydra sweep parameters
hydra:
  mode: MULTIRUN
  sweeper:
    _target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
    max_batch_size: null
    params:
      training.batch_size: 128,256,512
      model.training.optimizer_configs.hyperencoder.optimizer.lr: 1e-5,5e-5,1e-4
```

### Phase 5: Migration Steps

#### 5.1 Implementation Order:
1. Update `pyproject.toml` dependencies
2. Create new YAML config structure
3. Migrate training script (`train.py`)
4. Test training script with new configs
5. Migrate pre-encoding script (`pre_encode.py`)  
6. Create experiment configs
7. Update documentation
8. Remove old config files and prefigure

#### 5.2 Testing Strategy:
1. **Unit Tests**: Test config loading and instantiation
2. **Integration Tests**: Test full training pipeline with new configs
3. **Backwards Compatibility**: Ensure existing model checkpoints work
4. **Multi-GPU**: Test distributed training configurations

### Phase 6: Benefits After Migration

#### 6.1 Immediate Benefits:
- **Structured Configuration**: Clear separation of concerns
- **Type Safety**: Better validation and IDE support
- **Composition**: Easy config inheritance and overrides
- **Environment Integration**: Easy integration with env variables
- **Experiment Management**: Built-in experiment tracking

#### 6.2 Advanced Capabilities:
- **Hyperparameter Sweeps**: Native support for parameter sweeps
- **Config Validation**: Structured config validation
- **Dynamic Configuration**: Runtime config modification
- **Multi-run Experiments**: Parallel experiment execution

### Phase 7: Usage Examples

#### 7.1 Basic Training:
```bash
python -m hyperencoder.train experiment=hyperencoder_basic training.batch_size=512
```

#### 7.2 Custom Experiment:
```bash
python -m hyperencoder.train \
  model=hyperencoder_large \
  data=babyslakh \
  training=distributed \
  experiment_name=large_model_experiment
```

#### 7.3 Hyperparameter Sweep:
```bash
python -m hyperencoder.train -m \
  model.training.optimizer_configs.hyperencoder.optimizer.lr=1e-5,5e-5,1e-4 \
  training.batch_size=128,256
```

## Next Steps

1. **Review this plan** - Modify based on your preferences
2. **Approve dependency changes** - Confirm Hydra/OmegaConf versions
3. **Start with Phase 1** - Update dependencies and create basic structure
4. **Iterate phase by phase** - Test each phase before proceeding

## Questions for Review

1. Do you prefer the proposed directory structure for configs?
2. Should we maintain some backwards compatibility during transition?
3. Any specific Hydra features you want to emphasize (e.g., config groups, structured configs)?
4. Do you want to add config validation using Hydra's structured configs feature?

This plan provides a comprehensive migration path while maintaining the existing functionality and improving configuration management significantly. 