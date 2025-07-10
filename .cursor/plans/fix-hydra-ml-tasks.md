# Fix Hydra ML Tasks Integration

## Overview
Fix the ML tasks in `hyperencoder/cli/ml_tasks/` to properly integrate with the new Pydantic configuration system and use Hydra's built-in logging instead of the broken custom logging setup.

## Issues Identified

### 1. Broken Logging Import
- `train.py` imports `from hyperencoder.logging_utils import initialize_logger` 
- This module only exists in `hyperencoder/legacy/logging_utils.py`
- The import fails because there's no `logging_utils.py` in the root hyperencoder package

### 2. Custom Logging vs Hydra Logging
- Both tasks use custom logging setup instead of Hydra's logging system
- Hydra provides excellent logging out of the box with proper formatting and file handling
- Custom logging duplicates functionality and creates maintenance overhead

### 3. Configuration Integration Issues
- Tasks manually extract configs from Hydra DictConfig
- They convert to Pydantic models in complex ways
- This bypasses the clean configuration system you've built

### 4. Missing Error Handling
- Tasks don't follow your established error handling patterns
- No proper structured error reporting or logging

## Technical Analysis

### Current State
- `train.py`: 348 lines with complex manual config extraction
- `pre_encode.py`: 292 lines with basic hydra integration
- Both use `logging.getLogger(__name__)` but set up custom loggers
- Complex manual conversion between DictConfig and Pydantic models

### Target State
- Clean, simple task functions that use Hydra's built-in logging
- Proper integration with your Pydantic configuration system
- Consistent error handling and logging patterns
- Removal of custom logging setup

## Implementation Plan

### Phase 1: Remove Custom Logging
1. **Remove broken import** from `train.py`
2. **Update both tasks** to use Hydra's logger exclusively
3. **Remove custom logging setup** (TqdmHandler, LoggerWriter classes)
4. **Use hydra.core.global_hydra.GlobalHydra.instance().hydra_cfg** for logging config

### Phase 2: Simplify Configuration Loading
1. **Create configuration factory functions** in `hyperencoder/datamodels/`
2. **Update tasks to use factories** instead of manual conversion
3. **Remove complex config extraction logic**
4. **Use structured configs with Hydra properly**

### Phase 3: Improve Error Handling
1. **Add consistent error handling** following your established patterns
2. **Improve logging messages** with proper context
3. **Add validation for required paths and configs**
4. **Implement graceful failure modes**

### Phase 4: Clean Up and Test
1. **Remove unused imports and classes**
2. **Simplify task function signatures**
3. **Test both tasks with your configuration system**
4. **Verify logging works correctly**

## Specific Changes

### train.py Changes
1. Remove imports: `initialize_logger`, `TqdmHandler`, `LoggerWriter`, `ModelConfigEmbedderCallback`
2. Replace custom logging with: `hydra.core.global_hydra.GlobalHydra.instance().hydra_cfg.job_logging`
3. Create config factory: `create_training_config_from_hydra(cfg)`
4. Simplify task function to ~100 lines vs current 348

### pre_encode.py Changes
1. Already has better structure, just needs logging update
2. Add proper error handling for file operations
3. Use structured logging instead of print statements
4. Add configuration validation

### New Utility Functions
1. `create_training_config_from_hydra(cfg: DictConfig) -> TrainingConfig`
2. `create_model_config_from_hydra(cfg: DictConfig) -> ModelConfig`
3. `create_data_config_from_hydra(cfg: DictConfig) -> DataConfig`
4. `setup_hydra_logging() -> logging.Logger`

## Success Criteria
- [x] Both tasks run without import errors
- [x] Logging uses Hydra's system exclusively
- [x] Configuration integration is clean and simple
- [x] Error handling follows project patterns
- [x] Task functions are significantly simplified
- [x] All existing functionality preserved

## Testing Approach
1. Test training task with default configuration
2. Test pre-encoding task with default configuration  
3. Verify logging output goes to correct files
4. Test error handling with invalid configurations
5. Verify wandb/comet logging still works properly

## Dependencies
- Requires your existing Pydantic configuration system
- Needs hydra logging configuration in place
- Should work with your current auxiliary heads system

## Risks & Mitigations
- **Risk**: Breaking existing training workflows
  - **Mitigation**: Preserve all existing functionality, just clean up implementation
- **Risk**: Logging format changes
  - **Mitigation**: Use Hydra's standard logging which is more robust
- **Risk**: Configuration loading issues
  - **Mitigation**: Create factory functions that handle edge cases properly

## Questions & Clarifications
1. Do you want to keep the custom progress bar (RichProgressBar) or use standard Lightning progress bars?
2. Should we maintain backward compatibility with the old logging format?
3. Any specific Hydra logging configuration preferences (file patterns, log levels)?
4. Do you want to preserve the experiment ID-based logging directory structure?

## Next Steps
1. Review and approve this plan
2. Create branch `cursor/fix-hydra-ml-tasks`
3. Implement Phase 1 (logging fixes)
4. Test and validate changes
5. Proceed with remaining phases 