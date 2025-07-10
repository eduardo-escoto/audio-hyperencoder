# Feature: MIDI Metadata Injection for PreEncodedLatentDataset

## Overview
Add MIDI metadata injection capabilities to the existing `PreEncodedLatentDataset` or create a new dataset class that extends it. This will enable the dataset to automatically extract and inject metadata from `.midi` files into the `info` dict that gets passed through the training pipeline.

## Technical Analysis

### Current State Analysis
- **Current Dataset**: `PreEncodedLatentDataset` loads `.safetensors` files and creates `info` dict from `EncodedDirectoryInfo` and safetensor content
- **File Discovery**: Uses suffix mapping (`DEFAULT_SUFFIX_MAPPING`) to discover related files in directories
- **Info Dict**: Contains file paths, prefixes, and safetensor content merged together
- **Configuration**: Uses `DataConfig` with `DatasetEntry` for path specifications

### Key Requirements
1. **MIDI Directory**: Allow users to specify a directory containing MIDI files
2. **Filename Mapping**: Support transformation mapping between latent files and MIDI files
3. **Metadata Extraction**: Extract musical metadata from MIDI files using `miditoolkit`
4. **Backward Compatibility**: Maintain compatibility with existing dataset usage
5. **Optional Feature**: MIDI metadata injection should be optional and configurable

### MIDI Metadata to Extract
Using `miditoolkit`, we should extract:
- **Basic Info**: Tempo, time signature, key signature, total duration
- **Note Statistics**: Number of notes, note range (min/max), average velocity
- **Track Information**: Number of tracks, track names, instrument programs
- **Harmonic Analysis**: Chord progressions, key centers (if detectable)
- **Rhythmic Patterns**: Beat patterns, syncopation measures
- **Structural Elements**: Sections, repetitions, variations

## Implementation Plan

### 1. Configuration Updates

#### New Configuration Model
Create `MidiMetadataConfig` in `hyperencoder/datamodels/data_config.py`:
```python
class MidiMetadataConfig(BaseConfig):
    """Configuration for MIDI metadata injection."""
    
    enabled: bool = Field(
        default=False,
        description="Whether to enable MIDI metadata injection"
    )
    
    midi_dir: str | None = Field(
        default=None,
        description="Directory containing MIDI files"
    )
    
    filename_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping transformations from latent filenames to MIDI filenames"
    )
    
    filename_transform_pattern: str | None = Field(
        default=None,
        description="Regex pattern for transforming latent filenames to MIDI filenames"
    )
    
    filename_transform_replacement: str | None = Field(
        default=None,
        description="Replacement string for regex transformation"
    )
    
    default_midi_extension: str = Field(
        default=".mid",
        description="Default MIDI file extension"
    )
    
    fail_on_missing_midi: bool = Field(
        default=False,
        description="Whether to fail if MIDI file is not found"
    )
    
    extract_harmony: bool = Field(
        default=True,
        description="Whether to extract harmonic analysis"
    )
    
    extract_rhythm: bool = Field(
        default=True,
        description="Whether to extract rhythmic patterns"
    )
```

#### Update DataConfig
Add MIDI metadata config to `DataConfig`:
```python
class DataConfig(BaseConfig):
    # ... existing fields ...
    
    midi_metadata: MidiMetadataConfig | None = Field(
        default=None,
        description="Configuration for MIDI metadata injection"
    )
```

### 2. MIDI Metadata Extraction Module

#### Create `hyperencoder/data/midi_metadata.py`
```python
from typing import Dict, Any, Optional
from pathlib import Path
import miditoolkit
from pydantic import BaseModel

class MidiMetadata(BaseModel):
    """Extracted MIDI metadata."""
    
    # Basic info
    tempo: float
    time_signature: tuple[int, int]
    key_signature: str | None
    duration_seconds: float
    
    # Note statistics
    total_notes: int
    note_range_min: int
    note_range_max: int
    average_velocity: float
    
    # Track information
    num_tracks: int
    track_names: list[str]
    instrument_programs: list[int]
    
    # Optional harmonic/rhythmic analysis
    chord_progressions: list[str] | None = None
    key_centers: list[str] | None = None
    rhythmic_patterns: dict[str, Any] | None = None

class MidiMetadataExtractor:
    """Extract metadata from MIDI files using miditoolkit."""
    
    def __init__(self, extract_harmony: bool = True, extract_rhythm: bool = True):
        self.extract_harmony = extract_harmony
        self.extract_rhythm = extract_rhythm
    
    def extract_metadata(self, midi_path: Path) -> MidiMetadata:
        """Extract metadata from a MIDI file."""
        # Implementation using miditoolkit
        pass
    
    def _extract_basic_info(self, midi_obj) -> dict:
        """Extract basic MIDI information."""
        pass
    
    def _extract_note_statistics(self, midi_obj) -> dict:
        """Extract note-level statistics."""
        pass
    
    def _extract_track_info(self, midi_obj) -> dict:
        """Extract track and instrument information."""
        pass
    
    def _extract_harmonic_analysis(self, midi_obj) -> dict:
        """Extract harmonic analysis (optional)."""
        pass
    
    def _extract_rhythmic_patterns(self, midi_obj) -> dict:
        """Extract rhythmic patterns (optional)."""
        pass
```

### 3. Filename Transformation System

#### Create `hyperencoder/data/filename_mapping.py`
```python
import re
from typing import Dict, Optional
from pathlib import Path

class FilenameMapper:
    """Handle filename transformations between latent and MIDI files."""
    
    def __init__(
        self,
        mapping: Dict[str, str] = None,
        transform_pattern: str = None,
        transform_replacement: str = None,
        default_extension: str = ".mid"
    ):
        self.mapping = mapping or {}
        self.transform_pattern = transform_pattern
        self.transform_replacement = transform_replacement
        self.default_extension = default_extension
        
        if transform_pattern:
            self.pattern_regex = re.compile(transform_pattern)
    
    def map_latent_to_midi(self, latent_filename: str) -> str:
        """Map a latent filename to corresponding MIDI filename."""
        
        # Check explicit mapping first
        if latent_filename in self.mapping:
            return self.mapping[latent_filename]
        
        # Apply regex transformation if configured
        if self.transform_pattern and self.transform_replacement:
            midi_filename = self.pattern_regex.sub(
                self.transform_replacement, 
                latent_filename
            )
            return midi_filename
        
        # Default: replace extension
        stem = Path(latent_filename).stem
        return f"{stem}{self.default_extension}"
    
    def find_midi_file(self, latent_filename: str, midi_dir: Path) -> Optional[Path]:
        """Find the corresponding MIDI file for a latent filename."""
        midi_filename = self.map_latent_to_midi(latent_filename)
        midi_path = midi_dir / midi_filename
        
        if midi_path.exists():
            return midi_path
        
        # Try alternative extensions
        for ext in [".mid", ".midi", ".MID", ".MIDI"]:
            alt_path = midi_dir / f"{Path(midi_filename).stem}{ext}"
            if alt_path.exists():
                return alt_path
        
        return None
```

### 4. Dataset Extension

#### Option A: Extend PreEncodedLatentDataset
Create `MidiEnhancedLatentDataset` that inherits from `PreEncodedLatentDataset`:

```python
class MidiEnhancedLatentDataset(PreEncodedLatentDataset):
    """PreEncodedLatentDataset with MIDI metadata injection."""
    
    def __init__(
        self,
        file_tuples: list[tuple[str, EncodedDirectoryInfo]],
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
        crop_config=None,
        midi_metadata_config: MidiMetadataConfig = None,
    ):
        super().__init__(file_tuples, loading_strategy, crop_config)
        
        self.midi_config = midi_metadata_config
        self.midi_extractor = None
        self.filename_mapper = None
        
        if midi_metadata_config and midi_metadata_config.enabled:
            self._setup_midi_processing()
    
    def _setup_midi_processing(self):
        """Initialize MIDI processing components."""
        self.midi_extractor = MidiMetadataExtractor(
            extract_harmony=self.midi_config.extract_harmony,
            extract_rhythm=self.midi_config.extract_rhythm
        )
        
        self.filename_mapper = FilenameMapper(
            mapping=self.midi_config.filename_mapping,
            transform_pattern=self.midi_config.filename_transform_pattern,
            transform_replacement=self.midi_config.filename_transform_replacement,
            default_extension=self.midi_config.default_midi_extension
        )
    
    def __getitem__(self, idx):
        # Get the original item
        latents, info = super().__getitem__(idx)
        
        # Inject MIDI metadata if configured
        if self.midi_config and self.midi_config.enabled:
            midi_metadata = self._extract_midi_metadata(info)
            if midi_metadata:
                info["midi_metadata"] = midi_metadata.dict()
        
        return latents, info
    
    def _extract_midi_metadata(self, info: dict) -> Optional[MidiMetadata]:
        """Extract MIDI metadata for the current item."""
        if not self.midi_config.midi_dir:
            return None
        
        midi_dir = Path(self.midi_config.midi_dir)
        if not midi_dir.exists():
            return None
        
        # Get the latent filename from info
        latent_filename = Path(info.get("latents", "")).name
        if not latent_filename:
            return None
        
        # Find corresponding MIDI file
        midi_path = self.filename_mapper.find_midi_file(latent_filename, midi_dir)
        if not midi_path:
            if self.midi_config.fail_on_missing_midi:
                raise FileNotFoundError(f"MIDI file not found for {latent_filename}")
            return None
        
        # Extract metadata
        try:
            return self.midi_extractor.extract_metadata(midi_path)
        except Exception as e:
            if self.midi_config.fail_on_missing_midi:
                raise RuntimeError(f"Failed to extract MIDI metadata from {midi_path}: {e}")
            return None
```

#### Option B: Composition Approach
Alternatively, create a `MidiMetadataInjector` that can be composed with any dataset:

```python
class MidiMetadataInjector:
    """Inject MIDI metadata into dataset items."""
    
    def __init__(self, midi_config: MidiMetadataConfig):
        self.config = midi_config
        self.extractor = MidiMetadataExtractor(...)
        self.mapper = FilenameMapper(...)
    
    def inject_metadata(self, info: dict) -> dict:
        """Inject MIDI metadata into info dict."""
        # Implementation here
        pass
```

### 5. Configuration Schema Updates

#### Update JSON Schema
Update `schemas/data_config.schema.json` to include MIDI metadata configuration options.

#### Update Default Configurations
Add MIDI metadata options to `configs/data/default.yaml`:
```yaml
# ... existing config ...

midi_metadata:
  enabled: false
  midi_dir: null
  filename_mapping: {}
  filename_transform_pattern: null
  filename_transform_replacement: null
  default_midi_extension: ".mid"
  fail_on_missing_midi: false
  extract_harmony: true
  extract_rhythm: true
```

### 6. Factory Method Updates

#### Update Data Factory
Modify the data factory method to handle MIDI-enhanced datasets:

```python
def create_dataset(config: DataConfig) -> Dataset:
    """Create dataset based on configuration."""
    
    if config.midi_metadata and config.midi_metadata.enabled:
        # Use MIDI-enhanced dataset
        return MidiEnhancedLatentDataset(...)
    else:
        # Use standard dataset
        return PreEncodedLatentDataset(...)
```

### 7. Integration Points

#### Update DataModule
Ensure `PreEncodedLatentDataModule` can handle MIDI metadata configuration:

```python
class PreEncodedLatentDataModule(LightningDataModule):
    def __init__(
        self,
        # ... existing params ...
        midi_metadata_config: MidiMetadataConfig = None,
    ):
        # ... existing code ...
        self.midi_metadata_config = midi_metadata_config
    
    def setup(self, stage: str | None = None):
        # Use appropriate dataset class based on configuration
        dataset_class = (
            MidiEnhancedLatentDataset 
            if self.midi_metadata_config and self.midi_metadata_config.enabled
            else PreEncodedLatentDataset
        )
        
        # ... rest of setup ...
```

## Testing Strategy

### Unit Tests
1. **MidiMetadataExtractor Tests**: Test metadata extraction with various MIDI files
2. **FilenameMapper Tests**: Test filename transformation logic
3. **Dataset Integration Tests**: Test MIDI metadata injection in dataset

### Integration Tests
1. **End-to-End Pipeline**: Test complete data loading with MIDI metadata
2. **Configuration Validation**: Test Pydantic validation of MIDI config
3. **Error Handling**: Test behavior with missing MIDI files

### Example Test MIDI Files
Create test MIDI files with known metadata for reproducible testing.

## Example Usage

### Configuration
```yaml
# configs/data/with_midi_metadata.yaml
defaults:
  - default

midi_metadata:
  enabled: true
  midi_dir: "/path/to/midi/files"
  filename_transform_pattern: "(.*)_latent\\.safetensors"
  filename_transform_replacement: "\\1.mid"
  fail_on_missing_midi: false
  extract_harmony: true
  extract_rhythm: true
```

### Command Line Usage
```bash
# Train with MIDI metadata
uv run python -m hyperencoder.cli.main \
  data=with_midi_metadata \
  data.midi_metadata.midi_dir="/path/to/midi/files"

# Override specific MIDI settings
uv run python -m hyperencoder.cli.main \
  data.midi_metadata.enabled=true \
  data.midi_metadata.midi_dir="/path/to/midi/files" \
  data.midi_metadata.fail_on_missing_midi=true
```

### In Training Code
```python
# The info dict will now contain MIDI metadata
for latents, info in dataloader:
    if "midi_metadata" in info:
        midi_data = info["midi_metadata"]
        tempo = midi_data["tempo"]
        key_signature = midi_data["key_signature"]
        # Use MIDI metadata for conditioning or analysis
```

## Questions & Clarifications

1. **Filename Mapping Complexity**: Should we support more complex filename mappings (e.g., multiple patterns, conditional mappings)?

2. **Metadata Caching**: Should we cache extracted MIDI metadata to avoid reprocessing?

3. **Memory Usage**: For large datasets, should we implement lazy loading of MIDI metadata?

4. **Metadata Validation**: Should we validate extracted MIDI metadata for consistency?

5. **Alternative Libraries**: Are there other MIDI processing libraries we should consider besides miditoolkit?

6. **Advanced Analysis**: Should we include more sophisticated musical analysis (e.g., chord recognition, genre classification)?

## Dependencies

- **miditoolkit**: ✅ Added for MIDI file processing
- **Existing dataset infrastructure**: Uses current `PreEncodedLatentDataset`
- **Configuration system**: Leverages existing Pydantic configuration

## Risks & Considerations

1. **Performance Impact**: MIDI metadata extraction could slow down data loading
   - **Mitigation**: Implement caching and lazy loading options

2. **File Path Complexity**: Complex filename mappings could be error-prone
   - **Mitigation**: Provide comprehensive validation and clear error messages

3. **MIDI File Quality**: Varied MIDI file quality could affect metadata extraction
   - **Mitigation**: Robust error handling and optional metadata extraction

4. **Memory Usage**: Large datasets with rich MIDI metadata could consume significant memory
   - **Mitigation**: Implement efficient metadata storage and lazy loading

5. **Backward Compatibility**: Changes could break existing workflows
   - **Mitigation**: Make MIDI metadata optional and maintain existing API

## Success Criteria

1. **Functional**: MIDI metadata is successfully extracted and injected into dataset items
2. **Configurable**: Users can easily enable/disable and configure MIDI metadata extraction
3. **Robust**: System handles missing MIDI files and extraction errors gracefully
4. **Performant**: Minimal impact on data loading performance
5. **Maintainable**: Clean, documented code that integrates well with existing codebase
6. **Tested**: Comprehensive test coverage for all MIDI metadata functionality

This implementation will provide a flexible, configurable system for enriching audio latent datasets with musical metadata, enabling more sophisticated model training and analysis workflows. 