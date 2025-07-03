# Hyperencoder Hydra & OmegaConf Migration Plan

## Overview
This plan details the migration from `prefigure` (INI-based) configuration to a modern configuration stack using **Hydra + OmegaConf + Pydantic** for the hyperencoder deep learning project.

### **Enhanced Goals:**
- **Hydra + OmegaConf**: Powerful configuration composition and CLI overrides
- **Pydantic**: Type-safe configuration models with automatic JSON schema generation
- **JSON Schema Integration**: YAML files with `$schema` references for IDE validation and autocomplete
- **Embedded Documentation**: Rich descriptions and examples directly in configuration models
- **Automated Workflows**: Pre-commit hooks for schema generation and validation

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
```bash
# Remove prefigure dependency
uv remove prefigure

# Add core configuration stack
uv add hydra-core omegaconf pydantic

# Add development dependencies for schema generation
uv add --group dev pydantic[email] pre-commit
```

**New Dependencies:**
- **hydra-core**: Configuration composition and CLI
- **omegaconf**: YAML/config management 
- **pydantic**: Type validation and schema generation
- **pre-commit**: Automated hooks for schema generation

#### 1.2 Create New Configuration Structure
```
# Project Structure
hyperencoder/
├── configs/                       # YAML configuration files
│   ├── config.yaml               # Main config entry point  
│   ├── experiment/               # Experiment-specific overrides
│   │   ├── hyperencoder_basic.yaml
│   │   ├── hyperencoder_vae.yaml
│   │   └── hyperencoder_fsq.yaml
│   ├── model/                    # Model architecture configs
│   │   ├── hyperencoder.yaml    # Default hyperencoder
│   │   ├── hyperencoder_basic.yaml   # Basic variant
│   │   └── hyperencoder_large.yaml   # Future: larger model
│   ├── data/                     # Dataset configurations  
│   │   ├── hyperencoder.yaml    # Default dataset
│   │   ├── pre_encoded.yaml     # Pre-encoded latents
│   │   └── babyslakh.yaml       # Specific dataset variant
│   ├── training/                 # Training-specific configs
│   │   ├── default.yaml         # Default training params
│   │   ├── distributed.yaml     # Multi-GPU settings
│   │   └── debug.yaml           # Debug/dev settings
│   ├── pre_encode/               # Pre-encoding configs
│   │   ├── default.yaml         # Default pre-encoding
│   │   └── batch_processing.yaml # Batch processing variant
│   └── hydra/                    # Hydra runtime configs
│       └── default.yaml         # Hydra behavior config
├── schemas/                       # Generated JSON schemas
│   ├── model_config.schema.json  # Model configuration schema
│   ├── training_config.schema.json # Training configuration schema
│   ├── data_config.schema.json   # Data configuration schema
│   └── experiment_config.schema.json # Experiment configuration schema
└── hyperencoder/
    └── config/                    # Pydantic configuration models
        ├── __init__.py
        ├── model_config.py        # Model configuration models
        ├── training_config.py     # Training configuration models  
        ├── data_config.py         # Data configuration models
        └── base.py                # Base configuration classes
```

### Phase 2: Pydantic Configuration Models

#### 2.1 Create Pydantic Configuration Models

First, create the Pydantic models that will define our configuration structure:

```python
# hyperencoder/config/base.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, List, Union
from pathlib import Path

class BaseConfig(BaseModel):
    """Base configuration with common settings."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "examples": [{}]
        }
    )
```

```python
# hyperencoder/config/training_config.py
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from pathlib import Path
from .base import BaseConfig

class TrainingConfig(BaseConfig):
    """Training configuration for hyperencoder models.
    
    This configuration controls all aspects of the training process including
    hyperparameters, checkpointing, hardware settings, and logging.
    """
    
    # Experiment identification
    name: str = Field(
        default="hyperencoder_vae",
        description="Name of the training run for identification and logging",
        examples=["hyperencoder_vae", "hyperencoder_fsq_experiment"]
    )
    project: str = Field(
        default="hyperencoder", 
        description="Project name for grouping related experiments",
        examples=["hyperencoder", "audio_compression"]
    )
    
    # Training hyperparameters
    batch_size: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Batch size for training. Larger values may improve stability but require more memory",
        examples=[128, 256, 512]
    )
    num_workers: int = Field(
        default=8,
        ge=0,
        le=32,
        description="Number of CPU workers for data loading. Should be <= number of CPU cores",
        examples=[4, 8, 16]
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility across runs",
        examples=[42, 1337, 2023]
    )
    accum_batches: int = Field(
        default=1,
        ge=1,
        description="Number of batches to accumulate gradients over (effective batch size = batch_size * accum_batches)",
        examples=[1, 2, 4]
    )
    
    # Checkpointing and validation
    checkpoint_every: int = Field(
        default=10,
        ge=1,
        description="Save checkpoint every N training steps",
        examples=[10, 100, 1000]
    )
    val_every: int = Field(
        default=-1,
        description="Run validation every N steps (-1 to disable)",
        examples=[-1, 50, 100]
    )
    save_top_k: int = Field(
        default=10,
        ge=-1,
        description="Number of best checkpoints to keep (-1 for unlimited)",
        examples=[1, 5, 10, -1]
    )
    
    # Hardware and optimization
    num_nodes: int = Field(
        default=1,
        ge=1,
        description="Number of compute nodes for distributed training",
        examples=[1, 2, 4]
    )
    strategy: str = Field(
        default="auto",
        description="PyTorch Lightning strategy for multi-GPU training",
        examples=["auto", "ddp", "deepspeed"]
    )
    precision: Literal["32", "16-mixed", "bf16-mixed"] = Field(
        default="16-mixed",
        description="Training precision. 16-mixed uses automatic mixed precision for efficiency"
    )
    gradient_clip_val: float = Field(
        default=0.0,
        ge=0.0,
        description="Gradient clipping value (0.0 to disable)",
        examples=[0.0, 0.5, 1.0]
    )
    
    # Paths
    save_dir: Optional[str] = Field(
        default=None,
        description="Directory to save checkpoints and logs. Must be specified at runtime",
        examples=["/path/to/experiments", "./outputs"]
    )
    ckpt_path: Optional[Path] = Field(
        default=None,
        description="Path to checkpoint file to resume training from",
        examples=["./checkpoints/last.ckpt"]
    )
    pretrained_ckpt_path: Optional[Path] = Field(
        default=None,
        description="Path to pretrained checkpoint to initialize from",
        examples=["./pretrained/model.ckpt"]
    )
    pretransform_ckpt_path: Optional[Path] = Field(
        default=None,
        description="Path to pretransform checkpoint if needed"
    )
    
    # Logging
    logger: Literal["wandb", "comet", "none"] = Field(
        default="wandb",
        description="Logging backend for experiment tracking"
    )
    
    # Recovery
    recover: bool = Field(
        default=False,
        description="Whether to attempt recovery from latest checkpoint"
    )
```

#### 2.2 Convert INI to YAML with Schema (`hyperencoder/defaults/train_defaults.ini` → `configs/training/default.yaml`)
```yaml
# configs/training/default.yaml
# yaml-language-server: $schema=../schemas/training_config.schema.json

# Training configuration - see hyperencoder.config.training_config.TrainingConfig for full documentation

# Experiment identification
name: hyperencoder_vae  # Name of this training run
project: hyperencoder   # Project for grouping experiments

# Training hyperparameters
batch_size: 256         # Batch size (higher = more memory, potentially more stable)
num_workers: 8          # CPU workers for data loading
seed: 42               # Random seed for reproducibility
accum_batches: 1       # Gradient accumulation steps

# Checkpointing and validation  
checkpoint_every: 10   # Save checkpoint every N steps
val_every: -1         # Validation frequency (-1 = disabled)
save_top_k: 10        # Number of best checkpoints to keep

# Hardware and optimization
num_nodes: 1          # Number of compute nodes
strategy: auto        # Multi-GPU strategy
precision: "16-mixed" # Training precision (16-mixed = automatic mixed precision)
gradient_clip_val: 0.0 # Gradient clipping (0.0 = disabled)

# Paths (resolved at runtime)
save_dir: ???         # Must be specified - where to save outputs
ckpt_path: null       # Optional: checkpoint to resume from
pretrained_ckpt_path: null # Optional: pretrained weights to start from
pretransform_ckpt_path: null # Optional: pretransform checkpoint

# Logging
logger: wandb         # Experiment tracking (wandb/comet/none)

# Recovery
recover: false        # Attempt to recover from latest checkpoint
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

#### 2.3 Schema Generation and Pre-commit Integration

Create a script to generate JSON schemas from Pydantic models:

```python
# scripts/generate_schemas.py
#!/usr/bin/env python3
"""Generate JSON schemas from Pydantic configuration models."""

import json
from pathlib import Path
from typing import Dict, Any

from hyperencoder.config.training_config import TrainingConfig
from hyperencoder.config.model_config import ModelConfig  
from hyperencoder.config.data_config import DataConfig

def generate_schema(config_class, output_path: Path) -> None:
    """Generate JSON schema for a Pydantic model."""
    schema = config_class.model_json_schema()
    
    # Add custom properties for better IDE support
    schema["$id"] = f"https://hyperencoder.ai/schemas/{output_path.name}"
    schema["$comment"] = f"Generated from {config_class.__module__}.{config_class.__name__}"
    
    # Write schema with pretty formatting
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2, sort_keys=True)
    
    print(f"Generated schema: {output_path}")

def main():
    """Generate all JSON schemas."""
    schemas_dir = Path("schemas")
    schemas_dir.mkdir(exist_ok=True)
    
    # Generate schemas for each configuration type
    configs = [
        (TrainingConfig, "training_config.schema.json"),
        (ModelConfig, "model_config.schema.json"),
        (DataConfig, "data_config.schema.json"),
    ]
    
    for config_class, filename in configs:
        output_path = schemas_dir / filename
        generate_schema(config_class, output_path)
    
    print(f"\n✅ Generated {len(configs)} JSON schemas in {schemas_dir}/")

if __name__ == "__main__":
    main()
```

**Pre-commit Hook Configuration:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: generate-schemas
        name: Generate JSON Schemas
        entry: python scripts/generate_schemas.py
        language: system
        files: ^hyperencoder/config/.*\.py$
        pass_filenames: false
        
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
        args: ['--unsafe']  # Allow custom YAML tags
      - id: check-json
        files: ^schemas/.*\.json$
        
  - repo: https://github.com/adrienverge/yamllint.git
    rev: v1.32.0
    hooks:
      - id: yamllint
        files: ^configs/.*\.yaml$
        args: [-c=.yamllint.yaml]
```

**YAML Lint Configuration:**
```yaml
# .yamllint.yaml
extends: default
rules:
  line-length:
    max: 120
  comments:
    min-spaces-from-content: 1
  truthy:
    allowed-values: ['true', 'false', 'yes', 'no']
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

## Questions for Review ✅

1. **Do you prefer the proposed directory structure for configs?**
   - ✅ Ed: I like the proposed structure. Great job!
2. **Should we maintain some backwards compatibility during transition?**
   - ✅ Ed: No need for backwards compatibility, I'd like to completely lose dependency on prefigure.
3. **Any specific Hydra features you want to emphasize?**
   - ✅ Ed: I think using both structured configs and config groups would be beneficial here
4. **Do you want to add config validation using Hydra's structured configs feature?**
   - ✅ Ed: Yes please! Both Pydantic schema generation and Hydra validation

## Answers to Ed's Extra Questions

### **Q: Where does Pydantic fit in outside of just JSON schema generation?**

Pydantic can be incredibly useful throughout your hyperencoder project in several ways:

#### **1. Runtime Configuration Validation**
```python
# hyperencoder/config/loader.py
from omegaconf import DictConfig
from pydantic import ValidationError
from .training_config import TrainingConfig

def load_and_validate_config(cfg: DictConfig) -> TrainingConfig:
    """Load Hydra config and validate with Pydantic."""
    try:
        # Convert OmegaConf to dict and validate with Pydantic
        config_dict = OmegaConf.to_container(cfg.training, resolve=True)
        return TrainingConfig(**config_dict)
    except ValidationError as e:
        print(f"❌ Configuration validation failed:")
        for error in e.errors():
            print(f"  - {error['loc']}: {error['msg']}")
        raise
```

#### **2. API Input/Output Validation**
```python
# hyperencoder/api/training_api.py
from pydantic import BaseModel
from .config.training_config import TrainingConfig

class TrainingRequest(BaseModel):
    """API request for starting training."""
    config: TrainingConfig
    experiment_id: str
    tags: List[str] = []

class TrainingResponse(BaseModel):
    """API response for training status."""
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    metrics: Dict[str, float] = {}
```

#### **3. Model Checkpoint Metadata**
```python
# hyperencoder/checkpoints/metadata.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any

class CheckpointMetadata(BaseModel):
    """Metadata stored with model checkpoints."""
    model_config: Dict[str, Any]
    training_config: Dict[str, Any] 
    created_at: datetime = Field(default_factory=datetime.now)
    metrics: Dict[str, float] = {}
    git_commit: Optional[str] = None
    
    # Automatically embed in checkpoint files for full reproducibility
```

#### **4. Experiment Tracking Integration**
```python
# hyperencoder/logging/experiment_logger.py
from .config.training_config import TrainingConfig

class ExperimentLogger:
    def log_config(self, config: TrainingConfig):
        """Log validated configuration to W&B/Comet."""
        # Pydantic ensures all configs are valid before logging
        config_dict = config.model_dump(mode='json')
        wandb.config.update(config_dict)
```

#### **5. Data Pipeline Validation**
```python
# hyperencoder/data/validation.py
from pydantic import BaseModel, validator
from pathlib import Path

class DatasetInfo(BaseModel):
    """Validate dataset paths and properties."""
    path: Path
    num_files: int
    total_size_gb: float
    
    @validator('path')
    def path_exists(cls, v):
        if not v.exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        return v
```

### **Q: Can you convert all my existing json configurations and ini configurations to the proper yaml ones?**

**Absolutely!** Here's a conversion script and the converted configurations:

#### **Automated Conversion Script**
```python
# scripts/convert_legacy_configs.py
#!/usr/bin/env python3
"""Convert existing JSON and INI configs to new YAML format."""

import json
import configparser
from pathlib import Path
import yaml
from typing import Dict, Any

def convert_ini_to_yaml(ini_path: Path, output_path: Path) -> None:
    """Convert INI file to YAML with proper structure."""
    config = configparser.ConfigParser()
    config.read(ini_path)
    
    # Convert INI sections to nested dict
    yaml_config = {}
    for section in config.sections():
        yaml_config.update(dict(config[section]))
    
    # Clean up boolean and numeric values
    for key, value in yaml_config.items():
        if value.lower() in ('true', 'false'):
            yaml_config[key] = value.lower() == 'true'
        elif value.isdigit():
            yaml_config[key] = int(value)
        elif value.replace('.', '').isdigit():
            yaml_config[key] = float(value)
        elif value in ('', 'null', 'None'):
            yaml_config[key] = None
    
    # Add schema reference
    yaml_config = {
        "$schema": f"../schemas/{output_path.stem.replace('_', '')}_config.schema.json",
        **yaml_config
    }
    
    # Write YAML with comments
    with open(output_path, 'w') as f:
        f.write(f"# {output_path.name}\n")
        f.write(f"# Converted from {ini_path}\n")
        f.write(f"# yaml-language-server: $schema={yaml_config['$schema']}\n\n")
        yaml.dump({k: v for k, v in yaml_config.items() if k != '$schema'}, 
                 f, default_flow_style=False, indent=2)

def convert_json_to_yaml(json_path: Path, output_path: Path) -> None:
    """Convert JSON file to YAML with schema reference."""
    with open(json_path) as f:
        data = json.load(f)
    
    # Add schema reference at top
    yaml_content = f"""# {output_path.name}
# Converted from {json_path}
# yaml-language-server: $schema=../schemas/{output_path.stem.replace('_', '')}_config.schema.json

"""
    
    # Write YAML
    with open(output_path, 'w') as f:
        f.write(yaml_content)
        yaml.dump(data, f, default_flow_style=False, indent=2)

def main():
    """Convert all legacy configuration files."""
    # Create output directories
    Path("configs/training").mkdir(parents=True, exist_ok=True)
    Path("configs/model").mkdir(parents=True, exist_ok=True)
    Path("configs/data").mkdir(parents=True, exist_ok=True)
    
    # Convert INI files
    ini_files = [
        ("hyperencoder/defaults/train_defaults.ini", "configs/training/default.yaml"),
        ("hyperencoder/defaults/train_vqvae.ini", "configs/training/vqvae.yaml"),
    ]
    
    for ini_path, yaml_path in ini_files:
        if Path(ini_path).exists():
            convert_ini_to_yaml(Path(ini_path), Path(yaml_path))
            print(f"✅ Converted {ini_path} → {yaml_path}")
    
    # Convert JSON files
    json_files = [
        ("configs/models/hyperencoder.json", "configs/model/hyperencoder.yaml"),
        ("configs/models/hyperencoder_basic.json", "configs/model/hyperencoder_basic.yaml"),
        ("configs/models/hyperencoder_vqvae.json", "configs/model/hyperencoder_vqvae.yaml"),
        ("configs/data/hyperencoder.json", "configs/data/hyperencoder.yaml"),
        ("configs/data/pre_encoded.json", "configs/data/pre_encoded.yaml"),
    ]
    
    for json_path, yaml_path in json_files:
        if Path(json_path).exists():
            convert_json_to_yaml(Path(json_path), Path(yaml_path))
            print(f"✅ Converted {json_path} → {yaml_path}")

if __name__ == "__main__":
    main()
```

This comprehensive approach gives you:
- **Type-safe configurations** at runtime
- **IDE support** with autocompletion and validation
- **API validation** for any future web interfaces
- **Checkpoint reproducibility** with embedded config metadata
- **Automatic conversion** of all your existing configurations

The Pydantic integration provides validation at every level of your pipeline, not just schema generation! 🚀

---

## Updated Migration Plan Summary

### **Enhanced Architecture:**
✅ **Hydra + OmegaConf + Pydantic** - Complete configuration stack  
✅ **JSON Schema Integration** - IDE validation and autocomplete  
✅ **Automated Workflows** - Pre-commit hooks for schema generation  
✅ **Type Safety** - Runtime validation at every level  
✅ **Rich Documentation** - Embedded examples and descriptions  

### **Key Improvements from Ed's Feedback:**
1. **Use `uv add`** for dependency management instead of hardcoded versions
2. **Pydantic Integration** throughout the entire pipeline (not just schemas)
3. **Automated Schema Generation** with pre-commit hooks
4. **Complete Legacy Conversion** - script to convert all existing configs
5. **Enhanced IDE Support** - YAML language server integration
6. **No Backwards Compatibility** - clean break from prefigure

### **Ready to Implement:**
- **Phase 1**: Dependency updates with `uv add`
- **Phase 2**: Pydantic models + schema generation 
- **Phase 3**: Code migration with type-safe validation
- **Conversion Script**: Automated legacy config conversion

This plan now provides a world-class configuration system that will scale beautifully with your deep learning workflows! 🎯