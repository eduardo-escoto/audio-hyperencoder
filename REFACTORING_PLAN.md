# Configuration Refactoring Plan

## Executive Summary

This document outlines the plan to refactor the pydantic datamodel configuration system to properly separate concerns between task orchestration and domain-specific configurations (training, model, data, etc.).

## Current State Analysis

### Problems Identified

1. **Naming Confusion**: `TrainingConfig` in `training.py` is actually a task orchestration config, not a training-specific config
2. **Mixed Responsibilities**: Current configs mix task orchestration with domain-specific concerns
3. **Improper Composition**: The main config is composing all other configs instead of being a pure task orchestrator
4. **Scattered Domain Logic**: Training-specific logic is mixed with task management logic

### Current Config Classes Distribution

#### `hyperencoder/datamodels/training.py` (195 lines)
**Current `TrainingConfig` - Actually Task Orchestration:**
- `name`, `project` - Task identification
- `batch_size`, `num_workers` - Data loading (should be in DataConfig)
- `seed` - Task-level setting
- `num_nodes`, `devices`, `strategy`, `precision` - Infrastructure/hardware
- `max_epochs`, `log_every_n_steps` - Training loop control
- `checkpoint_every`, `val_every`, `save_top_k`, `recover` - Checkpointing
- `learning_rate`, `gradient_clip_val` - Training optimization
- `model_config_path`, `dataset_config`, `save_dir` - Task file paths
- `ckpt_path`, `pretrained_ckpt_path` - Task resume/initialization
- `run_id`, `ckpt_name`, `logger` - Task metadata/logging

#### `hyperencoder/datamodels/model_config.py` (430 lines)
**Current `TrainingConfig` - Actually Model-Specific Training:**
- `optimizer_configs` - Model-specific optimization (should be in proper TrainingConfig)

#### Other Files Distribution
- `data_config.py` - 4 config classes (DataConfig, DatasetEntry, CropConfig, MidiMetadataConfig)
- `hydra_config.py` - 8 config classes (various Hydra and task configs)
- `model_config.py` - 10 config classes (model architecture and components)
- `auxiliary_heads.py` - 1 config class (AuxiliaryHeadConfig)
- `midi_metadata.py` - 1 config class (MidiMetadata)
- `pre_encode_config.py` - 1 config class (PreEncodeConfig)

---

## Your Specifications

**Please fill in this section with your desired architecture:**

### Top-Level Abstractions/Config Classes
I will fill in the list below with the config objects and what parameters they have. As of now, I don't want to over-model all of the sub-objects, since I think I made that mistake before, so lets just do the top level ones.
```
[Please list the main config classes you want to create]

Example:
- HydraConfig: This model will represent and hold all the configuration options related to hydra. Check the Hydra documentation for which ones are available.
- WandBConfig: This shoudld be a model that specifies the objects which configure the WandB logger. We should use the torch or lightning interface to instantiate, so lets also look at the documentation for what is available.
- TrainingConfig: This model will hold configuration for defining any options related to the lightning trainer, including the trainer constructor, the fit method, and any trainer callbacks. This will have the following:
    - trainer (dict)
    - fit (dict)
    - callbacks (list of dicts)
- ModelConfig: Holds deep learning model specification that goes to the model constructor.
    - model_type (string)
    - sample_size (int, optional)
    - sample_rate (int, optional)
    - audio_channels (int, optional)
    - model: dict
    

- TaskConfig (task orchestration)
- TaskConfig:
- RunConfig:
- ExperimentConfig: 
```

### Specific Config Items by Category

```
[Please specify which config items should go into each class]

Example:
TaskConfig:
- name, project, seed
- file paths (model_config_path, dataset_config, save_dir)
- checkpointing strategy
- etc.

TrainingConfig:
- learning_rate, gradient_clip_val
- optimizer configs
- loss configs
- etc.
```

---

## Migration Analysis

### Current Config Items Inventory

| **Category** | **Current Items** | **Current Location** | **Proposed New Location** |
|---|---|---|---|
| **Task Identity** | `name`, `project`, `seed` | `training.py:TrainingConfig` | TBD |
| **Hardware/Infrastructure** | `num_nodes`, `devices`, `strategy`, `precision` | `training.py:TrainingConfig` | TBD |
| **Data Loading** | `batch_size`, `num_workers`, `persistent_workers` | `training.py:TrainingConfig` | TBD |
| **Training Loop** | `max_epochs`, `log_every_n_steps` | `training.py:TrainingConfig` | TBD |
| **Checkpointing** | `checkpoint_every`, `val_every`, `save_top_k`, `recover` | `training.py:TrainingConfig` | TBD |
| **Optimization** | `learning_rate`, `gradient_clip_val` | `training.py:TrainingConfig` | TBD |
| **Model Optimization** | `optimizer_configs` | `model_config.py:TrainingConfig` | TBD |
| **Task Paths** | `model_config_path`, `dataset_config`, `save_dir` | `training.py:TrainingConfig` | TBD |
| **Resume/Init** | `ckpt_path`, `pretrained_ckpt_path`, `pretransform_ckpt_path` | `training.py:TrainingConfig` | TBD |
| **Logging/Metadata** | `run_id`, `ckpt_name`, `logger` | `training.py:TrainingConfig` | TBD |

### Salvageable Components

#### Well-Structured Classes (Keep As-Is)
- `BaseConfig` - Solid foundation
- `DataConfig` and sub-configs - Well-organized data concerns
- `ModelConfig` architecture components - Good separation
- `HydraConfig` - Proper Hydra integration

#### Needs Refactoring
- `training.py:TrainingConfig` - Split into multiple classes
- `model_config.py:TrainingConfig` - Rename and merge with new TrainingConfig
- Task-related configs in `hydra_config.py` - Consolidate and clarify

#### Rename/Reorganization Candidates
- `auxiliary_heads.py` → Could move to `model_config.py`
- `midi_metadata.py` → Could move to `data_config.py`
- Some hydra configs might be consolidated

## Implementation Strategy

### Phase 1: Create New Architecture (Breaking Changes OK)

1. **Create new config classes** based on your specifications
2. **Migrate config items** from current locations to new classes
3. **Update imports** in `__init__.py`
4. **Update factory functions** to use new config structure

### Phase 2: Update Usage Points

1. **Update hydra integration** (`hydra_integration.py`)
2. **Update factories** (`factories/`)
3. **Update training wrapper** (`training/hyperencoder.py`)
4. **Update CLI** (`cli/`)
5. **Update data loading** (`data/`)

### Phase 3: Update Generated Artifacts

1. **Update config generation** (`cli/core/config_generation.py`)
2. **Update schema generation** (`cli/core/schema_generation.py`)
3. **Regenerate YAML configs** and JSON schemas
4. **Update documentation**

### Phase 4: Cleanup

1. **Remove old config classes**
2. **Remove unused imports**
3. **Update tests**
4. **Final validation**

## File Reorganization Plan

### New File Structure (TBD based on your specs)

```
hyperencoder/datamodels/
├── __init__.py                 # Updated exports
├── base.py                     # Keep as-is
├── task_config.py              # New: Task orchestration
├── training_config.py          # New: Pure training concerns
├── infrastructure_config.py    # New: Hardware/distributed settings
├── model_config.py             # Keep: Model architecture (cleaned up)
├── data_config.py              # Keep: Data loading/processing
├── hydra_config.py             # Keep: Hydra integration (cleaned up)
├── pre_encode_config.py        # Keep: Pre-encoding settings
└── [other specialized configs as needed]
```

### Import Changes Required

| **Module** | **Current Import** | **New Import** | **Impact** |
|---|---|---|---|
| `factories/model_factory.py` | `from hyperencoder.datamodels import ModelConfig` | TBD | Update factory functions |
| `training/hyperencoder.py` | `from hyperencoder.datamodels import ModelConfig, TrainingConfig, DemoConfig` | TBD | Update training wrapper |
| `cli/ml_tasks/train.py` | `from hyperencoder.datamodels.hydra_integration import create_training_config_from_hydra` | TBD | Update task functions |
| `data/latent.py` | `from hyperencoder.datamodels import MidiMetadataConfig, MidiMetadata` | TBD | Update data loading |

## Risk Assessment

### Low Risk
- Creating new config classes
- Moving config items between classes
- Updating imports in `__init__.py`

### Medium Risk
- Updating factory functions
- Modifying hydra integration
- Updating CLI commands

### High Risk
- Changing training wrapper interfaces
- Modifying data loading patterns
- Breaking existing YAML configs

## Success Criteria

1. **Clear Separation of Concerns**: Each config class has a single, well-defined responsibility
2. **Proper Naming**: Config class names accurately reflect their purpose
3. **Maintainable**: Easy to add new config options without confusion
4. **Type Safe**: All pydantic validation and type safety preserved
5. **Backward Compatible Configs**: Existing YAML configs work with adapters if needed
6. **Documentation**: Clear documentation of new architecture

## Next Steps

1. **Fill in your specifications** in the designated section above
2. **Review and approve** the migration analysis
3. **Create new config classes** based on your specifications
4. **Implement the migration** in phases
5. **Test thoroughly** with existing workflows
6. **Update documentation** and examples

---

**Note**: This plan assumes breaking changes are acceptable. All existing functionality will be preserved, but the internal organization and some import paths will change. 