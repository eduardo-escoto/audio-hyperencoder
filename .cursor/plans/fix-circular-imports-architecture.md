# Fix Circular Imports - Architectural Solution

## Current Circular Dependency Chain

```
hyperencoder/models/hyperencoder.py
  ↓ needs ModelConfig from
hyperencoder/datamodels/model_config.py  
  ↓ needs AuxiliaryHeadConfig from
hyperencoder/modules/auxiliary_heads.py
  ↑ imported by
hyperencoder/modules/latent_autoencoder.py
  ↓ imports HyperEncoder from
hyperencoder/models/hyperencoder.py
```

## Root Cause Analysis

### The Real Problem
The circular dependency exists because:
1. **Factory functions are in the wrong place**: `create_hyperencoder_from_config()` and `create_hyperencoder()` are in `models/hyperencoder.py`
2. **ModelConfig needs AuxiliaryHeadConfig**: But AuxiliaryHeadConfig is in modules, creating a cross-dependency
3. **Modules import models**: `latent_autoencoder.py` imports `HyperEncoder` directly

### Current Band-Aid "Fixes"
- `TYPE_CHECKING` imports ❌ (hides problem, doesn't solve it)
- String type annotations ❌ (syntax workaround, not architectural fix)
- Local imports inside functions ❌ (runtime import overhead and complexity)

## Architectural Solution

### Phase 1: Create Factories Module
**Goal**: Move factory functions to a neutral location that can import from both models and datamodels

#### 1.1 Create `hyperencoder/factories/`
```
hyperencoder/factories/
├── __init__.py
├── model_factory.py      # Model creation functions
└── config_factory.py     # Configuration creation functions
```

#### 1.2 Move Factory Functions
- Move `create_hyperencoder_from_config()` from `models/hyperencoder.py` to `factories/model_factory.py`
- Move `create_hyperencoder()` from `models/hyperencoder.py` to `factories/model_factory.py`
- Update imports in `models/hyperencoder.py` to re-export from factories for backward compatibility

#### 1.3 Update Dependencies
- `factories/model_factory.py` can safely import from both `models` and `datamodels`
- `models/hyperencoder.py` becomes a pure model definition (no config dependencies)
- `datamodels/model_config.py` no longer needs to be imported by models

### Phase 2: Fix AuxiliaryHeadConfig Location
**Goal**: Move AuxiliaryHeadConfig to proper location following separation of concerns

#### 2.1 Move AuxiliaryHeadConfig to datamodels (CHOSEN APPROACH)
- Move `AuxiliaryHeadConfig` from `modules/auxiliary_heads.py` to `datamodels/auxiliary_heads.py`
- Keep `AuxiliaryHead` class in `modules/auxiliary_heads.py`
- Update imports: `from ..datamodels.auxiliary_heads import AuxiliaryHeadConfig`

**Why this is the right approach:**
- `AuxiliaryHeadConfig` is a Pydantic configuration model → belongs in `datamodels/`
- `AuxiliaryHead` is a PyTorch implementation → belongs in `modules/`
- Clean separation: configuration definitions vs. implementation
- No cross-module imports: `datamodels/model_config.py` → `datamodels/auxiliary_heads.py`
- Proper dependency direction: `modules/` → `datamodels/`

### Phase 3: Clean Up Model Imports
**Goal**: Remove direct model imports from modules where possible

#### 3.1 Review `latent_autoencoder.py`
- Check if it actually needs to import `HyperEncoder` directly
- Consider using factory pattern or dependency injection
- If it needs HyperEncoder, keep the import (this is acceptable)

#### 3.2 Use Dependency Injection
- Pass model instances to modules instead of having modules create models
- This is cleaner and more testable

### Phase 4: Remove Band-Aids
**Goal**: Remove all TYPE_CHECKING workarounds and string annotations

#### 4.1 Remove TYPE_CHECKING
- Remove `if TYPE_CHECKING:` blocks
- Use direct imports now that circular dependencies are resolved

#### 4.2 Remove String Annotations
- Change `config: "ModelConfig"` back to `config: ModelConfig`
- Remove local imports inside functions

## Implementation Plan

### Step 1: Create Factories Module
```bash
mkdir -p hyperencoder/factories
touch hyperencoder/factories/__init__.py
touch hyperencoder/factories/model_factory.py
```

### Step 2: Move Factory Functions
1. Copy `create_hyperencoder_from_config()` and `create_hyperencoder()` to `factories/model_factory.py`
2. Add proper imports to `factories/model_factory.py`
3. Update `models/hyperencoder.py` to import and re-export from factories
4. Test that existing code still works

### Step 3: Move AuxiliaryHeadConfig to datamodels
1. Create `datamodels/auxiliary_heads.py` 
2. Move `AuxiliaryHeadConfig` from `modules/auxiliary_heads.py` to `datamodels/auxiliary_heads.py`
3. Update `datamodels/model_config.py` to import from `datamodels/auxiliary_heads.py`
4. Update `modules/auxiliary_heads.py` to import from `datamodels/auxiliary_heads.py`
5. Remove TYPE_CHECKING import
6. Test configuration loading

### Step 4: Clean Up Band-Aids
1. Remove TYPE_CHECKING blocks
2. Change string annotations back to direct types
3. Remove local imports inside functions
4. Test all imports work correctly

### Step 5: Integration Testing
1. Test model creation with `create_hyperencoder_from_config()`
2. Test CLI commands (train, pre_encode)
3. Test auxiliary heads functionality
4. Verify no circular import errors

## Directory Structure After Fix

```
hyperencoder/
├── factories/
│   ├── __init__.py
│   └── model_factory.py          # Factory functions (can import models + datamodels)
├── models/
│   └── hyperencoder.py           # Pure model definition (no config imports)
├── datamodels/
│   ├── model_config.py           # Config definitions (imports from datamodels only)
│   └── auxiliary_heads.py        # AuxiliaryHeadConfig (Pydantic model)
├── modules/
│   ├── auxiliary_heads.py        # AuxiliaryHead (PyTorch module)
│   └── latent_autoencoder.py     # Can import models (acceptable)
```

## Benefits of This Solution

### Technical Benefits
1. **No circular dependencies**: Clean module structure
2. **Proper separation of concerns**: Models, configs, and factories are separate
3. **No runtime overhead**: No local imports or TYPE_CHECKING workarounds
4. **Type safety**: Full type checking without string annotations
5. **Testable**: Easier to test individual components

### Developer Experience
1. **Clear architecture**: Obvious where to find factory functions
2. **No magic**: No hidden imports or TYPE_CHECKING tricks
3. **Maintainable**: Easy to understand and modify
4. **Extensible**: Easy to add new factory functions

### Research Benefits
1. **Reliable**: No hidden import issues that could break during research
2. **Debuggable**: Clear error messages without import workarounds
3. **Reusable**: Factory pattern makes it easy to create models programmatically

## Success Criteria

- [x] All imports work without TYPE_CHECKING
- [x] No circular import errors
- [x] All existing functionality still works
- [x] CLI commands work correctly
- [x] Model creation works with configs
- [x] Auxiliary heads integration works
- [x] No string type annotations needed
- [x] Clean, understandable code structure

## Timeline

- **Phase 1**: 30 minutes (create factories, move functions)
- **Phase 2**: 15 minutes (fix AuxiliaryHeadConfig import) 
- **Phase 3**: 15 minutes (clean up modules)
- **Phase 4**: 15 minutes (remove band-aids)
- **Phase 5**: 30 minutes (testing)

**Total**: ~2 hours for a proper architectural fix

This is a fundamental architectural improvement that will eliminate technical debt and create a maintainable codebase structure. 