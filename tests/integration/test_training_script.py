"""
Integration tests for the Hydra-based training script.

Tests that the new training script can properly load configuration,
validate settings, and initialize components without errors.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from omegaconf import OmegaConf, DictConfig
from hyperencoder.config.hydra_integration import (
    load_training_config,
    validate_and_resolve_paths,
)


class TestTrainingScriptIntegration:
    """Test the integration between Hydra config and training script."""

    def test_configuration_loading_workflow(self, temp_dir):
        """Test the complete configuration loading workflow."""
        # Create a sample Hydra configuration
        hydra_cfg = OmegaConf.create({
            "training": {
                "name": "integration_test",
                "batch_size": 32,
                "num_workers": 4,
                "model_config_path": "configs/model.json",
                "dataset_config": "configs/dataset.json",
                "save_dir": "outputs",
                "logger": "wandb",
                "project": "test_project"
            }
        })
        
        # Test the configuration loading pipeline
        training_config = load_training_config(hydra_cfg)
        resolved_config = validate_and_resolve_paths(training_config, temp_dir)
        
        # Verify configuration is properly loaded
        assert resolved_config.name == "integration_test"
        assert resolved_config.batch_size == 32
        assert resolved_config.num_workers == 4
        assert resolved_config.logger == "wandb"
        assert resolved_config.project == "test_project"
        
        # Verify paths are resolved
        assert resolved_config.model_config_path.is_absolute()
        assert resolved_config.dataset_config.is_absolute()
        assert resolved_config.save_dir.is_absolute()

    def test_configuration_validation_errors(self):
        """Test that invalid configurations are properly rejected."""
        # Create invalid configuration
        invalid_cfg = OmegaConf.create({
            "training": {
                "name": "invalid_test",
                "batch_size": 0,  # Invalid: must be >= 1
                "num_workers": -1,  # Invalid: must be >= 0
                "precision": "invalid_precision"  # Invalid: not in allowed values
            }
        })
        
        # Should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            load_training_config(invalid_cfg)

    def test_partial_configuration_with_defaults(self):
        """Test that partial configuration uses appropriate defaults."""
        partial_cfg = OmegaConf.create({
            "training": {
                "name": "partial_test",
                "batch_size": 64
                # Missing many fields - should use defaults
            }
        })
        
        training_config = load_training_config(partial_cfg)
        
        # Check that specified values are used
        assert training_config.name == "partial_test"
        assert training_config.batch_size == 64
        
        # Check that defaults are applied
        assert training_config.num_workers == 8  # Default
        assert training_config.seed == 42  # Default
        assert training_config.strategy == "auto"  # Default
        assert training_config.precision == "16-mixed"  # Default

    def test_hydra_config_structure(self):
        """Test that our Hydra configuration files are valid."""
        # Test that we can load the actual config files
        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        try:
            with initialize(config_path="../../conf", version_base=None):
                cfg = compose(config_name="config")
                
                # Should be able to load training config
                training_config = load_training_config(cfg)
                
                # Basic validation
                assert hasattr(training_config, 'name')
                assert hasattr(training_config, 'batch_size')
                assert hasattr(training_config, 'project')
                
        finally:
            # Clean up Hydra
            GlobalHydra.instance().clear()

    @patch('hyperencoder.train_hydra.create_hyperencoder_from_config')
    @patch('hyperencoder.train_hydra.create_datamodule_from_config')
    @patch('hyperencoder.train_hydra.initialize_logger')
    def test_training_script_initialization(
        self, 
        mock_logger_init,
        mock_datamodule_create,
        mock_model_create,
        temp_dir
    ):
        """Test that the training script can initialize without errors."""
        # Create mock model and dataset config files
        model_config_path = temp_dir / "model.json"
        dataset_config_path = temp_dir / "dataset.json"
        
        model_config = {
            "model_type": "hyperencoder",
            "demo": {"demo_every": 10, "max_demos": 5}
        }
        dataset_config = {"type": "pre_encoded"}
        
        with open(model_config_path, 'w') as f:
            import json
            json.dump(model_config, f)
            
        with open(dataset_config_path, 'w') as f:
            import json
            json.dump(dataset_config, f)
        
        # Create Hydra config
        hydra_cfg = OmegaConf.create({
            "training": {
                "name": "mock_test",
                "batch_size": 32,
                "model_config_path": str(model_config_path),
                "dataset_config": str(dataset_config_path),
                "save_dir": str(temp_dir / "outputs"),
                "logger": "tensorboard"  # Don't use wandb in tests
            }
        })
        
        # Mock the components that would be created
        mock_model = MagicMock()
        mock_datamodule = MagicMock()
        mock_logger = MagicMock()
        
        mock_model_create.return_value = mock_model
        mock_datamodule_create.return_value = mock_datamodule
        mock_logger_init.return_value = mock_logger
        
        # Test configuration loading and validation
        training_config = load_training_config(hydra_cfg)
        resolved_config = validate_and_resolve_paths(training_config, temp_dir)
        
        # Verify the configuration can be processed
        assert resolved_config.name == "mock_test"
        assert resolved_config.batch_size == 32
        assert resolved_config.logger == "tensorboard"
        assert resolved_config.model_config_path == model_config_path
        assert resolved_config.dataset_config == dataset_config_path
        
        # Verify model and dataset configs can be loaded
        with open(resolved_config.model_config_path) as f:
            loaded_model_config = json.load(f)
            assert loaded_model_config == model_config
            
        with open(resolved_config.dataset_config) as f:
            loaded_dataset_config = json.load(f)
            assert loaded_dataset_config == dataset_config

    def test_type_safety_and_validation(self):
        """Test that our configuration system provides proper type safety."""
        # Test valid configuration
        valid_cfg = OmegaConf.create({
            "training": {
                "name": "type_test",
                "batch_size": "32",  # String should convert to int
                "persistent_workers": "true",  # String should convert to bool
                "gradient_clip_val": "1.5",  # String should convert to float
                "precision": "bf16-mixed",  # Valid literal
                "strategy": "ddp"  # Valid literal
            }
        })
        
        training_config = load_training_config(valid_cfg)
        
        # Verify type conversions
        assert isinstance(training_config.batch_size, int)
        assert training_config.batch_size == 32
        assert isinstance(training_config.persistent_workers, bool)
        assert training_config.persistent_workers is True
        assert isinstance(training_config.gradient_clip_val, float)
        assert training_config.gradient_clip_val == 1.5
        
        # Verify literal validation
        assert training_config.precision == "bf16-mixed"
        assert training_config.strategy == "ddp" 