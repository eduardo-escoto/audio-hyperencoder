"""
Unit tests for Hydra integration module.

Tests the bridge functions that convert between Hydra/OmegaConf DictConfig
and our Pydantic configuration models.
"""

import pytest
import io
import sys
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from hyperencoder.config import TrainingConfig
from hyperencoder.config.hydra_integration import (
    dictconfig_to_pydantic,
    pydantic_to_dictconfig,
    load_training_config,
    validate_and_resolve_paths,
    print_config_summary,
)


class TestDictConfigToPydantic:
    """Test the dictconfig_to_pydantic function."""

    def test_basic_conversion(self, sample_dictconfig):
        """Test basic DictConfig to Pydantic conversion."""
        config = dictconfig_to_pydantic(sample_dictconfig, TrainingConfig)
        
        assert isinstance(config, TrainingConfig)
        assert config.name == "test_experiment"
        assert config.batch_size == 64
        assert config.num_workers == 4
        assert config.strategy == "auto"

    def test_empty_dictconfig(self):
        """Test conversion with empty DictConfig."""
        empty_cfg = OmegaConf.create({})
        config = dictconfig_to_pydantic(empty_cfg, TrainingConfig)
        
        # Should use defaults
        assert config.name == "hyperencoder_experiment"
        assert config.batch_size == 32
        assert config.seed == 42

    def test_partial_dictconfig(self):
        """Test conversion with partial configuration."""
        partial_cfg = OmegaConf.create({
            "name": "partial_test",
            "batch_size": 128,
            "precision": "bf16-mixed"
        })
        
        config = dictconfig_to_pydantic(partial_cfg, TrainingConfig)
        
        assert config.name == "partial_test"
        assert config.batch_size == 128
        assert config.precision == "bf16-mixed"
        # Should use defaults for missing fields
        assert config.num_workers == 8
        assert config.seed == 42

    def test_invalid_dictconfig(self):
        """Test conversion with invalid configuration."""
        invalid_cfg = OmegaConf.create({
            "name": "invalid_test",
            "batch_size": 0,  # Invalid: must be >= 1
            "precision": "invalid_precision"
        })
        
        with pytest.raises(ValidationError):
            dictconfig_to_pydantic(invalid_cfg, TrainingConfig)

    def test_type_conversion(self):
        """Test that types are properly converted."""
        cfg = OmegaConf.create({
            "name": "type_test",
            "batch_size": "64",  # String should convert to int
            "persistent_workers": "true",  # String should convert to bool
            "gradient_clip_val": "1.5"  # String should convert to float
        })
        
        config = dictconfig_to_pydantic(cfg, TrainingConfig)
        
        assert config.batch_size == 64
        assert isinstance(config.batch_size, int)
        assert config.persistent_workers is True
        assert isinstance(config.persistent_workers, bool)
        assert config.gradient_clip_val == 1.5
        assert isinstance(config.gradient_clip_val, float)

    def test_path_conversion(self):
        """Test that path fields are properly handled."""
        cfg = OmegaConf.create({
            "name": "path_test",
            "model_config_path": "/path/to/model.json",
            "save_dir": "outputs",
            "dataset_config": None
        })
        
        config = dictconfig_to_pydantic(cfg, TrainingConfig)
        
        assert isinstance(config.model_config_path, Path)
        assert isinstance(config.save_dir, Path)
        assert config.dataset_config is None


class TestPydanticToDictConfig:
    """Test the pydantic_to_dictconfig function."""

    def test_basic_conversion(self, sample_training_config):
        """Test basic Pydantic to DictConfig conversion."""
        dictconfig = pydantic_to_dictconfig(sample_training_config)
        
        assert isinstance(dictconfig, DictConfig)
        assert dictconfig.name == "test_experiment"
        assert dictconfig.batch_size == 64
        assert dictconfig.num_workers == 4
        assert dictconfig.strategy == "auto"

    def test_path_objects_to_strings(self):
        """Test that Path objects are converted to strings."""
        config = TrainingConfig(
            name="path_test",
            model_config_path=Path("/path/to/model.json"),
            save_dir=Path("/path/to/outputs")
        )
        
        dictconfig = pydantic_to_dictconfig(config)
        
        assert isinstance(dictconfig.model_config_path, str)
        assert isinstance(dictconfig.save_dir, str)
        assert dictconfig.model_config_path == "/path/to/model.json"
        assert dictconfig.save_dir == "/path/to/outputs"

    def test_none_values(self):
        """Test that None values are preserved."""
        config = TrainingConfig(
            name="none_test",
            model_config_path=None,
            dataset_config=None,
            run_id=None
        )
        
        dictconfig = pydantic_to_dictconfig(config)
        
        assert dictconfig.model_config_path is None
        assert dictconfig.dataset_config is None
        assert dictconfig.run_id is None

    def test_nested_structures(self):
        """Test that nested structures are properly converted."""
        # Create a config with nested path structures in a hypothetical field
        config = TrainingConfig(name="nested_test", batch_size=32)
        
        dictconfig = pydantic_to_dictconfig(config)
        
        # Should be able to access all fields
        assert hasattr(dictconfig, 'name')
        assert hasattr(dictconfig, 'batch_size')
        assert hasattr(dictconfig, 'strategy')

    def test_round_trip_conversion(self, sample_training_config):
        """Test that Pydantic → DictConfig → Pydantic conversion preserves data."""
        # Convert to DictConfig
        dictconfig = pydantic_to_dictconfig(sample_training_config)
        
        # Convert back to Pydantic
        restored_config = dictconfig_to_pydantic(dictconfig, TrainingConfig)
        
        # Should be identical (except for Path object comparison)
        original_dict = sample_training_config.model_dump()
        restored_dict = restored_config.model_dump()
        
        # Compare field by field (handling Path objects)
        for key, original_value in original_dict.items():
            restored_value = restored_dict[key]
            if isinstance(original_value, Path):
                assert isinstance(restored_value, Path)
                assert str(original_value) == str(restored_value)
            else:
                assert original_value == restored_value


class TestLoadTrainingConfig:
    """Test the load_training_config function."""

    def test_load_from_hydra_config(self):
        """Test loading training config from a full Hydra configuration."""
        hydra_cfg = OmegaConf.create({
            "app": {"name": "hyperencoder", "version": "0.1.0"},
            "training": {
                "name": "hydra_test",
                "batch_size": 128,
                "num_workers": 6,
                "precision": "32-true",
                "logger": "tensorboard"
            }
        })
        
        training_config = load_training_config(hydra_cfg)
        
        assert isinstance(training_config, TrainingConfig)
        assert training_config.name == "hydra_test"
        assert training_config.batch_size == 128
        assert training_config.num_workers == 6
        assert training_config.precision == "32-true"
        assert training_config.logger == "tensorboard"

    def test_load_missing_training_section(self):
        """Test loading when training section is missing."""
        hydra_cfg = OmegaConf.create({
            "app": {"name": "hyperencoder"},
            # No training section
        })
        
        training_config = load_training_config(hydra_cfg)
        
        # Should use all defaults
        assert training_config.name == "hyperencoder_experiment"
        assert training_config.batch_size == 32
        assert training_config.num_workers == 8

    def test_load_empty_training_section(self):
        """Test loading with empty training section."""
        hydra_cfg = OmegaConf.create({
            "training": {}
        })
        
        training_config = load_training_config(hydra_cfg)
        
        # Should use all defaults
        assert training_config.name == "hyperencoder_experiment"
        assert training_config.batch_size == 32

    def test_load_invalid_training_config(self):
        """Test loading with invalid training configuration."""
        hydra_cfg = OmegaConf.create({
            "training": {
                "name": "invalid_test",
                "batch_size": 0,  # Invalid
                "precision": "invalid_precision"  # Invalid
            }
        })
        
        with pytest.raises(ValidationError):
            load_training_config(hydra_cfg)


class TestValidateAndResolvePaths:
    """Test the validate_and_resolve_paths function."""

    def test_resolve_relative_paths(self, temp_dir: Path):
        """Test resolving relative paths."""
        config = TrainingConfig(
            name="path_test",
            model_config_path="configs/model.json",
            save_dir="outputs",
            dataset_config="../data/dataset.json"
        )
        
        resolved_config = validate_and_resolve_paths(config, temp_dir)
        
        assert resolved_config.model_config_path.is_absolute()
        assert resolved_config.save_dir.is_absolute()
        assert resolved_config.dataset_config.is_absolute()
        
        # Check that paths are resolved relative to base_path
        # Handle macOS symlink /var -> /private/var
        temp_dir_resolved = temp_dir.resolve()
        model_path_str = str(resolved_config.model_config_path)
        save_dir_str = str(resolved_config.save_dir)
        
        assert (model_path_str.startswith(str(temp_dir)) or 
                model_path_str.startswith(str(temp_dir_resolved)))
        assert (save_dir_str.startswith(str(temp_dir)) or 
                save_dir_str.startswith(str(temp_dir_resolved)))

    def test_preserve_absolute_paths(self, temp_dir):
        """Test that absolute paths are preserved."""
        absolute_path = Path("/absolute/path/to/model.json")
        
        config = TrainingConfig(
            name="absolute_test",
            model_config_path=absolute_path
        )
        
        resolved_config = validate_and_resolve_paths(config, temp_dir)
        
        assert resolved_config.model_config_path == absolute_path

    def test_handle_none_paths(self, temp_dir):
        """Test that None paths are preserved."""
        config = TrainingConfig(
            name="none_test",
            model_config_path=None,
            save_dir=None
        )
        
        resolved_config = validate_and_resolve_paths(config, temp_dir)
        
        assert resolved_config.model_config_path is None
        assert resolved_config.save_dir is None

    def test_default_base_path(self):
        """Test that current directory is used when no base_path provided."""
        config = TrainingConfig(
            name="default_base_test",
            model_config_path="model.json"
        )
        
        resolved_config = validate_and_resolve_paths(config)
        
        assert resolved_config.model_config_path.is_absolute()
        # Should be resolved relative to current working directory
        assert str(resolved_config.model_config_path).startswith(str(Path.cwd()))

    def test_string_path_conversion(self, temp_dir):
        """Test that string paths are properly converted."""
        config = TrainingConfig(
            name="string_test",
            model_config_path="configs/model.json"
        )
        
        resolved_config = validate_and_resolve_paths(config, temp_dir)
        
        assert isinstance(resolved_config.model_config_path, Path)
        assert resolved_config.model_config_path.is_absolute()

    def test_immutability(self, temp_dir):
        """Test that original config is not modified."""
        original_config = TrainingConfig(
            name="immutable_test",
            model_config_path="configs/model.json"
        )
        
        original_path = original_config.model_config_path
        
        resolved_config = validate_and_resolve_paths(original_config, temp_dir)
        
        # Original should be unchanged
        assert original_config.model_config_path == original_path
        # Resolved should be different
        assert resolved_config.model_config_path != original_path


class TestPrintConfigSummary:
    """Test the print_config_summary function."""

    def test_summary_output(self, sample_training_config):
        """Test that config summary prints correctly."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            print_config_summary(sample_training_config)
            output = captured_output.getvalue()
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
        
        # Check that key information is in the output
        assert "Training Configuration Summary" in output
        assert "test_experiment" in output
        assert "test_project" in output
        assert "64" in output  # batch_size
        assert "4" in output   # num_workers
        assert "42" in output  # seed
        assert "auto" in output  # strategy
        assert "16-mixed" in output  # precision
        assert "wandb" in output  # logger

    def test_summary_with_paths(self, temp_dir):
        """Test summary output with path fields."""
        config = TrainingConfig(
            name="path_summary_test",
            model_config_path=temp_dir / "model.json",
            save_dir=temp_dir / "outputs"
        )
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            print_config_summary(config)
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        
        # Check that paths are displayed
        assert str(temp_dir) in output
        assert "model.json" in output
        assert "outputs" in output

    def test_summary_with_none_values(self):
        """Test summary output with None values."""
        config = TrainingConfig(
            name="none_summary_test",
            model_config_path=None,
            save_dir=None
        )
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            print_config_summary(config)
            output = captured_output.getvalue()
        finally:
            sys.stdout = sys.__stdout__
        
        # Check that None values are handled gracefully
        assert "none_summary_test" in output
        assert "None" in output


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_complete_workflow(self, temp_dir):
        """Test a complete workflow from DictConfig to resolved Pydantic."""
        # Start with a Hydra-style configuration
        hydra_cfg = OmegaConf.create({
            "training": {
                "name": "workflow_test",
                "batch_size": 64,
                "model_config_path": "configs/model.json",
                "save_dir": "outputs",
                "precision": "bf16-mixed",
                "logger": "tensorboard"
            }
        })
        
        # Load training config
        training_config = load_training_config(hydra_cfg)
        
        # Resolve paths
        resolved_config = validate_and_resolve_paths(training_config, temp_dir)
        
        # Verify the complete workflow
        assert resolved_config.name == "workflow_test"
        assert resolved_config.batch_size == 64
        assert resolved_config.precision == "bf16-mixed"
        assert resolved_config.logger == "tensorboard"
        assert resolved_config.model_config_path.is_absolute()
        assert resolved_config.save_dir.is_absolute()
        # Handle macOS symlink /var -> /private/var  
        temp_dir_resolved = temp_dir.resolve()
        model_path_str = str(resolved_config.model_config_path)
        assert (model_path_str.startswith(str(temp_dir)) or 
                model_path_str.startswith(str(temp_dir_resolved)))

    def test_error_handling_workflow(self):
        """Test error handling in the complete workflow."""
        # Invalid Hydra configuration
        hydra_cfg = OmegaConf.create({
            "training": {
                "name": "error_test",
                "batch_size": 0,  # Invalid
                "precision": "invalid"  # Invalid
            }
        })
        
        # Should fail at load_training_config step
        with pytest.raises(ValidationError):
            load_training_config(hydra_cfg)

    def test_backwards_conversion(self, sample_training_config):
        """Test converting back from Pydantic to DictConfig."""
        # Start with Pydantic config
        original_config = sample_training_config
        
        # Convert to DictConfig
        dictconfig = pydantic_to_dictconfig(original_config)
        
        # Should be usable as OmegaConf config
        assert isinstance(dictconfig, DictConfig)
        assert dictconfig.name == original_config.name
        assert dictconfig.batch_size == original_config.batch_size
        
        # Should be able to access with OmegaConf syntax
        assert OmegaConf.select(dictconfig, "name") == original_config.name
        assert OmegaConf.select(dictconfig, "batch_size") == original_config.batch_size

    def test_schema_generation_after_conversion(self, sample_dictconfig):
        """Test that Pydantic features work after conversion."""
        # Convert from DictConfig
        config = dictconfig_to_pydantic(sample_dictconfig, TrainingConfig)
        
        # Should still have full Pydantic functionality
        schema = config.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "name" in schema["properties"]
        
        # Should be able to serialize
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test_experiment" 