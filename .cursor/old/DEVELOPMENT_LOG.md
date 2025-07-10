# Development Log

## 2024-12-19 - Auxiliary Heads System: Multi-Task Learning Implementation
- **Major Feature**: Implemented comprehensive auxiliary heads system for multi-task learning on MIDI metadata
- **Core Modules**: Created `auxiliary_heads.py` and `auxiliary_losses.py` with complete Pydantic configuration
- **Neural Architecture**: Built configurable prediction heads supporting regression, classification, and multi-label tasks
- **Loss Integration**: Seamlessly integrated auxiliary losses with `stable_audio_tools.training.losses.MultiLoss`
- **Training Integration**: Updated `HyperEncoderTrainingWrapper` to support auxiliary heads with proper batch metadata handling
- **Validation Metrics**: Implemented comprehensive evaluation metrics (MAE/MSE for regression, accuracy for classification, F1 for multi-label)
- **Configuration System**: Extended `ModelConfig` with `AuxiliaryHeadsConfig` including validation and defaults
- **Preset Configurations**: Created `basic_song_level.yaml` (7 heads) and `tempo_velocity.yaml` (2 heads) presets
- **Target Features**: Supports tempo, duration, velocity, note density, time signature, key signature, and track count prediction
- **Error Handling**: Robust error handling with graceful degradation and comprehensive logging
- **Factory Integration**: Updated all factory functions to pass auxiliary heads configuration
- **Architecture Decision**: Auxiliary heads process `inner_latents` (post-encoder, pre-decoder) for semantic understanding
- **Bug Fix**: Fixed critical training bug where batch metadata was discarded, preventing auxiliary losses from working
- **Impact**: Enables semantic learning of musical structure through multi-task optimization, improving latent space quality

## 2024-12-19 - CLI Utility Enhancements: Complete Development Toolchain
- **Major CLI Restructuring**: Refactored CLI utilities to use proper typer subcommands (`generate`, `setup`, `clean`)
- **Comprehensive Schema Generation**: Extended schema generation to include all 20 Pydantic datamodels
- **VSCode Integration**: Added `setup vscode` command that creates proper JSON schema mappings in `.vscode/settings.json`
- **VSCode Teardown**: Added `teardown-vscode` command to safely remove hyperencoder schema mappings while preserving other settings
- **Advanced Cleaning**: Implemented sophisticated cleaning commands that safely remove only auto-generated files
- **Bug Fix**: Fixed cleaning command to properly detect `pre_encode.yaml` file (was missing from detection logic)
- **Makefile Updates**: Updated all Makefile commands to use the new CLI structure with correct paths, including teardown targets
- **Schema Types**: Now generating schemas for all config types including task configs, sub-configs, and specialized models
- **Pattern Matching**: VSCode integration maps schemas to appropriate YAML file patterns for perfect IDE support
- **Safety Features**: Clean commands include dry-run mode, confirmation prompts, and safe file detection
- **Backup Creation**: Teardown commands automatically create backups before modifying VSCode settings
- **Complete Workflow**: Full development lifecycle from `generate all` → `setup vscode` → `teardown vscode` → `clean all` works seamlessly
- **Impact**: Developers now have a complete, safe, and beautiful CLI toolchain for configuration management

## 2024-12-19 - Configuration Architecture Cleanup: Task-Based System Complete
- **Major Restructuring**: Completed comprehensive cleanup of configuration architecture
- **Task-Based Approach**: Replaced complex "experiment" concept with direct task composition
- **Main Entry Points**: Created clean `train.yaml` and `pre_encode.yaml` as primary configuration files
- **Legacy Cleanup**: Removed unnecessary configs: `experiment/default.yaml`, `config.yaml`, `training.yaml`
- **Pydantic Models**: Introduced `TrainTaskConfig` and `PreEncodeTaskConfig` as single source of truth
- **YAML Generation Fixes**: Fixed Hydra defaults format with proper absolute paths (`/data: default`)
- **Configuration Groups**: Maintained clean config groups: data, model, training, pre_encode, hydra
- **CLI Testing**: Verified both `--config-name=train` and `--config-name=pre_encode` work correctly
- **Directory Structure**: Final clean structure with only 7 essential configuration files
- **Impact**: Dramatically simplified configuration system - from 10+ configs to 7 essential ones

## 2024-12-19 - CLI Config Generation Fixes: Pure Pydantic Implementation
- **Hydra Defaults Format**: Fixed YAML generation to use proper Hydra defaults list format
- **Custom YAML Dumper**: Created `CustomYAMLDumper` to handle `Union[str, Dict[str, str]]` types correctly
- **Output Directory Fix**: Changed config generation from project root to `hyperencoder/cli/configs/`
- **Hardcoded Values Removal**: Eliminated all hardcoded values - everything now comes from Pydantic models
- **Hydra Config Separation**: Moved hydra configuration to dedicated `HydraConfig` model
- **Path Resolution**: Fixed absolute path issues in config composition with `/data: default` syntax
- **Validation Integration**: All configuration values now validated through Pydantic models
- **Impact**: Achieved pure Pydantic approach with zero hardcoded values and proper Hydra integration

## 2024-12-19 - Hydra Migration Complete: Manual Config Loading Solution
- **Major Achievement**: Successfully completed the full Hydra migration with a novel manual config loading approach
- **Key Technical Decision**: Chose manual config extraction over Hydra structured configs due to Pydantic incompatibility
- **Directory Refactor**: Renamed `hyperencoder/config` → `hyperencoder/datamodels` for better clarity
- **INI Migration**: Successfully incorporated defaults from `train_vqvae.ini` into Pydantic models
- **Default Update**: Changed training name default from `hyperencoder_experiment` to `vqvae_hyperencoder`
- **Package Distribution**: Fixed config path to use bundled configs (`hyperencoder/configs/`) for pip compatibility
- **Legacy Cleanup**: Removed INI files and old JSON configs after successful migration
- **Schema Updates**: Regenerated JSON schemas with correct module paths
- **Import Fixes**: Updated all imports throughout codebase from `hyperencoder.config` to `hyperencoder.datamodels`
- **Impact**: Complete modernization of configuration system with type safety, validation, and beautiful CLI

## 2024-12-19 - Structured Configs Challenge and Solution Discovery
- **Problem Identified**: Hydra's structured configs require dataclasses, not Pydantic models
- **Error Resolved**: `ValidationError: Input class 'TrainingConfig' is not a structured config`
- **Solution Implemented**: Created manual config loading in `hyperencoder/datamodels/hydra_integration.py`
- **Architecture Decision**: Maintain separation between Hydra (composition) and Pydantic (validation)
- **Benefits Preserved**: Full Pydantic validation + all Hydra features (multirun, tab completion, output dirs)
- **Impact**: Avoided framework conflicts while maintaining benefits of both systems

## 2024-12-19 - CLI Architecture and Task Dispatch Implementation
- **CLI Infrastructure**: Created beautiful CLI with Hydra + colorlog integration
- **Task Modules**: Implemented `hyperencoder/cli/tasks/train.py` and `hyperencoder/cli/tasks/pre_encode.py`
- **Utilities CLI**: Built `hyperencoder-utils` with typer + rich for development tools
- **Configuration Composition**: Fixed main config to use proper Hydra defaults structure
- **Variable References**: Fixed all `args.*` → `training_config.*` references in training code
- **Package Entrypoints**: Added CLI commands to pyproject.toml for easy access
- **Impact**: Production-ready CLI with beautiful UX and full Hydra feature support

## 2024-12-19 - Factory Methods and Pydantic Integration
- **Factory Updates**: Updated `create_hyperencoder_from_config` to accept ModelConfig instead of dict
- **Data Architecture Fix**: Moved data loading parameters from TrainingConfig to DataConfig
- **Programmatic Interface**: Created factory methods using Pydantic defaults as parameter defaults
- **Bridge Method Elimination**: Removed `dictconfig_to_pydantic` conversion methods
- **Training Integration**: Updated training task to use new Pydantic-based factories
- **Impact**: Eliminated hardcoded defaults, improved type safety, cleaner architecture

## Previous Development History
*Note: This development log was created to track progress going forward. Previous development history exists in git commits and planning documents.* 