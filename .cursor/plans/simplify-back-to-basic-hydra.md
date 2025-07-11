# Simplify Back to Basic Hydra

## Overview
Remove all Pydantic datamodels and schema generation code to return to basic Hydra configuration. The current system over-abstracted too early and has become bloated. We'll simplify to use plain YAML configs with Hydra's DictConfig.

## Current State Analysis

### What to Remove
1. **All Pydantic datamodels** in `hyperencoder/datamodels/`
   - `base.py` - BaseConfig class
   - `training.py` - TrainingConfig
   - `data_config.py` - DataConfig, CropConfig, etc.
   - `model_config.py` - ModelConfig and all sub-configs
   - `hydra_config.py` - All Hydra integration configs
   - `pre_encode_config.py` - PreEncodeConfig
   - `auxiliary_heads.py` - AuxiliaryHeadConfig
   - `midi_metadata.py` - MidiMetadata
   - `hydra_integration.py` - Hydra-Pydantic bridge code
   - `__init__.py` - Exports all the configs

2. **Schema generation code**
   - `hyperencoder/cli/core/schema_generation.py` - Entire file
   - `hyperencoder/cli/core/config_generation.py` - Entire file
   - `schemas/` directory - All JSON schema files

3. **CLI utilities that depend on Pydantic**
   - Update `hyperencoder/cli/dev_commands/generate.py` to remove schema generation
   - Update `hyperencoder/cli/dev_commands/clean.py` to remove schema cleanup
   - Simplify `hyperencoder/cli/dev_cli.py` to remove schema commands

### What to Update
1. **Factory methods** - Convert from Pydantic models to DictConfig
   - `hyperencoder/factories/model_factory.py`
   - `hyperencoder/factories/auxiliary_head_factory.py`

2. **Data loading** - Convert from Pydantic to DictConfig
   - `hyperencoder/data/utils.py`
   - `hyperencoder/data/latent.py`
   - `hyperencoder/data/midi_extractor.py`

3. **Training code** - Convert from Pydantic to DictConfig
   - `hyperencoder/training/hyperencoder.py`
   - `hyperencoder/cli/ml_tasks/train.py`
   - `hyperencoder/cli/ml_tasks/pre_encode.py`

4. **Configuration files** - Ensure they work with basic Hydra
   - Keep existing YAML files in `configs/`
   - Remove any Pydantic-specific features

### Code Impact Analysis
Files that import from `hyperencoder.datamodels`:
- `hyperencoder/data/utils.py` - Uses DataConfig
- `hyperencoder/data/latent.py` - Uses MidiMetadataConfig, MidiMetadata
- `hyperencoder/factories/model_factory.py` - Uses ModelConfig
- `hyperencoder/data/midi_extractor.py` - Uses MidiMetadata
- `hyperencoder/cli/ml_tasks/train.py` - Uses hydra_integration
- `hyperencoder/cli/core/config_generation.py` - Uses all configs
- `hyperencoder/cli/core/schema_generation.py` - Uses all configs
- `hyperencoder/training/hyperencoder.py` - Uses ModelConfig
- `hyperencoder/cli/ml_tasks/pre_encode.py` - Uses hydra_integration

## Implementation Plan

### Phase 1: Remove Pydantic Infrastructure
1. **Delete entire `hyperencoder/datamodels/` directory**
2. **Delete entire `schemas/` directory**
3. **Delete schema generation files**
   - `hyperencoder/cli/core/schema_generation.py`
   - `hyperencoder/cli/core/config_generation.py`

### Phase 2: Update Factory Methods
1. **Update `hyperencoder/factories/model_factory.py`**
   - Replace `ModelConfig` with `DictConfig`
   - Remove Pydantic-specific validation
   - Use `.get()` methods for accessing config values

2. **Update `hyperencoder/factories/auxiliary_head_factory.py`**
   - Replace `AuxiliaryHeadConfig` with `DictConfig`
   - Simplify factory logic

### Phase 3: Update Data Loading
1. **Update `hyperencoder/data/utils.py`**
   - Replace `DataConfig` with `DictConfig`
   - Remove Pydantic-specific methods
   - Use basic dict access patterns

2. **Update `hyperencoder/data/latent.py`**
   - Replace `MidiMetadataConfig` with `DictConfig`
   - Replace `MidiMetadata` with basic dict structures

3. **Update `hyperencoder/data/midi_extractor.py`**
   - Replace `MidiMetadata` with basic dict structures

### Phase 4: Update Training Code
1. **Update `hyperencoder/training/hyperencoder.py`**
   - Replace Pydantic imports with DictConfig
   - Remove Pydantic-specific validation
   - Use basic dict access patterns

2. **Update `hyperencoder/cli/ml_tasks/train.py`**
   - Remove hydra_integration imports
   - Work directly with Hydra's DictConfig
   - Simplify config passing

3. **Update `hyperencoder/cli/ml_tasks/pre_encode.py`**
   - Remove hydra_integration imports
   - Work directly with Hydra's DictConfig

### Phase 5: Update CLI Utilities
1. **Update `hyperencoder/cli/dev_commands/generate.py`**
   - Remove schema generation functionality
   - Keep only essential config generation if needed

2. **Update `hyperencoder/cli/dev_commands/clean.py`**
   - Remove schema cleanup functionality

3. **Update `hyperencoder/cli/dev_cli.py`**
   - Remove schema-related commands

### Phase 6: Update Configuration Management
1. **Update `hyperencoder/cli/core/` directory**
   - Remove schema generation files
   - Keep only essential CLI core functionality

2. **Update main package `__init__.py`**
   - Remove datamodels exports
   - Keep only essential exports

### Phase 7: Update Development Tools
1. **Update `Makefile`**
   - Remove schema generation targets
   - Remove pydantic-specific targets

2. **Update `.pre-commit-config.yaml`**
   - Remove schema generation hooks
   - Remove pydantic-specific validation

## Benefits of Simplification

1. **Reduced Complexity** - No more Pydantic-Hydra bridge code
2. **Faster Development** - No schema generation or validation overhead
3. **Better Hydra Integration** - Use Hydra's native features directly
4. **Easier Configuration** - Simple YAML files without type constraints
5. **Less Maintenance** - No need to keep Pydantic models in sync with configs

## Risks & Considerations

1. **Loss of Type Safety** - No more Pydantic validation
   - Mitigation: Use good defaults and document config structure
2. **Runtime Errors** - Config errors only caught at runtime
   - Mitigation: Good error handling and early validation
3. **Less IDE Support** - No schema-based autocomplete
   - Mitigation: Good documentation and examples

## Success Criteria

- [ ] All Pydantic datamodels removed
- [ ] All schema generation code removed
- [ ] All imports updated to use DictConfig
- [ ] Training still works with basic Hydra configs
- [ ] Pre-encoding still works with basic Hydra configs
- [ ] CLI utilities still work (without schema features)
- [ ] Configuration files still work with Hydra
- [ ] All tests pass (if any exist)

## Next Steps

1. Get approval for this simplification approach
2. Create a backup branch before starting
3. Execute Phase 1 (removal of Pydantic infrastructure)
4. Iteratively update code to work with DictConfig
5. Test at each phase to ensure nothing breaks
6. Update documentation to reflect simplified approach 