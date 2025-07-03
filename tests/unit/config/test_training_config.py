"""
Unit tests for TrainingConfig class.

Tests the training configuration class with all its validation rules,
constraints, path handling, and integration with BaseConfig.
"""

import pytest
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError
from hyperencoder.config import TrainingConfig


class TestTrainingConfig:
    """Test suite for TrainingConfig class."""

    def test_training_config_defaults(self):
        """Test that TrainingConfig has correct default values."""
        config = TrainingConfig()
        
        # Run identification defaults
        assert config.name == "hyperencoder_experiment"
        assert config.project == "hyperencoder"
        
        # Training hyperparameters defaults
        assert config.batch_size == 32
        assert config.num_workers == 8
        assert config.seed == 42
        assert config.accum_batches == 1
        
        # Hardware configuration defaults
        assert config.num_nodes == 1
        assert config.devices == "auto"
        assert config.strategy == "auto"
        assert config.precision == "16-mixed"
        assert config.persistent_workers is False
        
        # Checkpointing defaults
        assert config.checkpoint_every == 1
        assert config.val_every == -1
        assert config.save_top_k == 20
        assert config.recover is False
        
        # Optimization defaults
        assert config.gradient_clip_val == 0.0
        
        # Path defaults (should be None)
        assert config.model_config_path is None
        assert config.dataset_config is None
        assert config.val_dataset_config is None
        assert config.save_dir is None
        assert config.ckpt_path is None
        assert config.pretrained_ckpt_path is None
        assert config.pretransform_ckpt_path is None
        
        # Metadata defaults
        assert config.run_id is None
        assert config.ckpt_name is None
        
        # Logging defaults
        assert config.logger == "wandb"
        
        # Legacy defaults
        assert config.remove_pretransform_weight_norm is False

    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            name="custom_experiment",
            project="custom_project",
            batch_size=64,
            num_workers=4,
            seed=123,
            precision="bf16-mixed",
            logger="tensorboard"
        )
        
        assert config.name == "custom_experiment"
        assert config.project == "custom_project"
        assert config.batch_size == 64
        assert config.num_workers == 4
        assert config.seed == 123
        assert config.precision == "bf16-mixed"
        assert config.logger == "tensorboard"

    def test_batch_size_validation(self):
        """Test batch_size field validation."""
        # Valid batch sizes
        valid_sizes = [1, 16, 32, 64, 128, 256, 512]
        for size in valid_sizes:
            config = TrainingConfig(batch_size=size)
            assert config.batch_size == size
        
        # Invalid batch sizes - too small
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)
        
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=-1)
        
        # Invalid batch sizes - too large
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=1000)

    def test_num_workers_validation(self):
        """Test num_workers field validation."""
        # Valid num_workers
        valid_workers = [0, 1, 4, 8, 16, 32]
        for workers in valid_workers:
            config = TrainingConfig(num_workers=workers)
            assert config.num_workers == workers
        
        # Invalid num_workers - negative
        with pytest.raises(ValidationError):
            TrainingConfig(num_workers=-1)
        
        # Invalid num_workers - too large
        with pytest.raises(ValidationError):
            TrainingConfig(num_workers=100)

    def test_seed_validation(self):
        """Test seed field validation."""
        # Valid seeds
        valid_seeds = [0, 42, 123, 999999]
        for seed in valid_seeds:
            config = TrainingConfig(seed=seed)
            assert config.seed == seed
        
        # Invalid seeds - negative
        with pytest.raises(ValidationError):
            TrainingConfig(seed=-1)

    def test_accum_batches_validation(self):
        """Test accum_batches field validation."""
        # Valid values
        valid_values = [1, 2, 4, 8, 16]
        for value in valid_values:
            config = TrainingConfig(accum_batches=value)
            assert config.accum_batches == value
        
        # Invalid values - too small
        with pytest.raises(ValidationError):
            TrainingConfig(accum_batches=0)

    def test_num_nodes_validation(self):
        """Test num_nodes field validation."""
        # Valid values
        valid_values = [1, 2, 4, 8]
        for value in valid_values:
            config = TrainingConfig(num_nodes=value)
            assert config.num_nodes == value
        
        # Invalid values - too small
        with pytest.raises(ValidationError):
            TrainingConfig(num_nodes=0)

    def test_strategy_validation(self):
        """Test strategy field validation (Literal type)."""
        # Valid strategies
        valid_strategies = ["auto", "ddp", "ddp_find_unused_parameters_true", "fsdp"]
        for strategy in valid_strategies:
            config = TrainingConfig(strategy=strategy)
            assert config.strategy == strategy
        
        # Invalid strategy
        with pytest.raises(ValidationError):
            TrainingConfig(strategy="invalid_strategy")

    def test_precision_validation(self):
        """Test precision field validation (Literal type)."""
        # Valid precisions
        valid_precisions = ["16-mixed", "bf16-mixed", "32-true", "64-true"]
        for precision in valid_precisions:
            config = TrainingConfig(precision=precision)
            assert config.precision == precision
        
        # Invalid precision
        with pytest.raises(ValidationError):
            TrainingConfig(precision="invalid_precision")

    def test_logger_validation(self):
        """Test logger field validation (Literal type)."""
        # Valid loggers
        valid_loggers = ["wandb", "tensorboard", "csv"]
        for logger in valid_loggers:
            config = TrainingConfig(logger=logger)
            assert config.logger == logger
        
        # Invalid logger
        with pytest.raises(ValidationError):
            TrainingConfig(logger="invalid_logger")

    def test_checkpoint_every_validation(self):
        """Test checkpoint_every field validation."""
        # Valid values
        valid_values = [1, 5, 10, 100]
        for value in valid_values:
            config = TrainingConfig(checkpoint_every=value)
            assert config.checkpoint_every == value
        
        # Invalid values - too small
        with pytest.raises(ValidationError):
            TrainingConfig(checkpoint_every=0)

    def test_val_every_validation(self):
        """Test val_every field validation."""
        # Valid values (including -1 for disabled)
        valid_values = [-1, 1, 10, 100, 1000]
        for value in valid_values:
            config = TrainingConfig(val_every=value)
            assert config.val_every == value
        
        # Invalid values - too small
        with pytest.raises(ValidationError):
            TrainingConfig(val_every=-2)

    def test_save_top_k_validation(self):
        """Test save_top_k field validation."""
        # Valid values (including -1 for all)
        valid_values = [-1, 1, 5, 10, 20]
        for value in valid_values:
            config = TrainingConfig(save_top_k=value)
            assert config.save_top_k == value
        
        # Invalid values - too small
        with pytest.raises(ValidationError):
            TrainingConfig(save_top_k=-2)

    def test_gradient_clip_val_validation(self):
        """Test gradient_clip_val field validation."""
        # Valid values
        valid_values = [0.0, 0.5, 1.0, 5.0]
        for value in valid_values:
            config = TrainingConfig(gradient_clip_val=value)
            assert config.gradient_clip_val == value
        
        # Invalid values - negative
        with pytest.raises(ValidationError):
            TrainingConfig(gradient_clip_val=-0.1)

    def test_path_field_types(self):
        """Test that path fields accept Path objects and strings."""
        config = TrainingConfig(
            model_config_path="/path/to/model.json",
            dataset_config=Path("/path/to/dataset.json"),
            save_dir="/path/to/save"
        )
        
        # All should be converted to Path objects
        assert isinstance(config.model_config_path, Path)
        assert isinstance(config.dataset_config, Path)
        assert isinstance(config.save_dir, Path)

    def test_path_resolution_validator(self, temp_dir):
        """Test the resolve_paths field validator."""
        # Create test files
        model_config_file = temp_dir / "model.json"
        model_config_file.touch()
        
        # Test relative path resolution
        config = TrainingConfig(
            model_config_path="model.json",
            save_dir="outputs"
        )
        
        # Paths should be resolved to absolute paths
        assert config.model_config_path.is_absolute()
        assert config.save_dir.is_absolute()

    def test_none_path_handling(self):
        """Test that None values are handled correctly for path fields."""
        config = TrainingConfig(
            model_config_path=None,
            dataset_config=None,
            save_dir=None
        )
        
        assert config.model_config_path is None
        assert config.dataset_config is None
        assert config.save_dir is None

    def test_empty_string_path_handling(self):
        """Test that empty strings are converted to None for path fields."""
        config = TrainingConfig(
            model_config_path="",
            dataset_config="",
            save_dir=""
        )
        
        assert config.model_config_path is None
        assert config.dataset_config is None
        assert config.save_dir is None

    def test_type_validation(self):
        """Test type validation for various fields."""
        # String fields
        with pytest.raises(ValidationError):
            TrainingConfig(name=123)  # Should be string
        
        with pytest.raises(ValidationError):
            TrainingConfig(project=123)  # Should be string
        
        # Integer fields
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size="not_int")
        
        with pytest.raises(ValidationError):
            TrainingConfig(num_workers="not_int")
        
        # Boolean fields
        with pytest.raises(ValidationError):
            TrainingConfig(persistent_workers="not_bool")
        
        with pytest.raises(ValidationError):
            TrainingConfig(recover="not_bool")
        
        # Float fields
        with pytest.raises(ValidationError):
            TrainingConfig(gradient_clip_val="not_float")

    def test_serialization_to_dict(self):
        """Test serialization to dictionary."""
        config = TrainingConfig(
            name="test_experiment",
            batch_size=64,
            model_config_path="/path/to/model.json"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test_experiment"
        assert config_dict["batch_size"] == 64
        assert isinstance(config_dict["model_config_path"], Path)

    def test_deserialization_from_dict(self):
        """Test deserialization from dictionary."""
        config_dict = {
            "name": "test_from_dict",
            "batch_size": 128,
            "num_workers": 2,
            "precision": "32-true",
            "model_config_path": "/path/to/model.json"
        }
        
        config = TrainingConfig.from_dict(config_dict)
        
        assert config.name == "test_from_dict"
        assert config.batch_size == 128
        assert config.num_workers == 2
        assert config.precision == "32-true"
        assert isinstance(config.model_config_path, Path)

    def test_yaml_serialization(self, temp_dir):
        """Test YAML serialization and deserialization."""
        config = TrainingConfig(
            name="yaml_test",
            batch_size=64,
            model_config_path=temp_dir / "model.json",
            save_dir=temp_dir / "outputs"
        )
        
        yaml_path = temp_dir / "training_config.yaml"
        
        # Save to YAML
        config.save_yaml(yaml_path)
        assert yaml_path.exists()
        
        # Load from YAML
        loaded_config = TrainingConfig.from_yaml(yaml_path)
        
        assert loaded_config.name == "yaml_test"
        assert loaded_config.batch_size == 64
        assert loaded_config.model_config_path == config.model_config_path
        assert loaded_config.save_dir == config.save_dir

    def test_configuration_inheritance(self):
        """Test that TrainingConfig properly inherits from BaseConfig."""
        config = TrainingConfig()
        
        # Should have BaseConfig methods
        assert hasattr(config, 'to_dict')
        assert hasattr(config, 'from_dict')
        assert hasattr(config, 'save_yaml')
        assert hasattr(config, 'from_yaml')
        
        # Should have BaseConfig model_config settings
        assert config.model_config['extra'] == 'forbid'
        assert config.model_config['validate_assignment'] is True

    def test_field_descriptions_present(self):
        """Test that field descriptions are present in schema."""
        schema = TrainingConfig.model_json_schema()
        
        # Check that key fields have descriptions
        properties = schema.get('properties', {})
        
        assert 'description' in properties.get('name', {})
        assert 'description' in properties.get('batch_size', {})
        assert 'description' in properties.get('num_workers', {})
        assert 'description' in properties.get('strategy', {})
        assert 'description' in properties.get('precision', {})

    def test_field_examples_present(self):
        """Test that field examples are present in schema."""
        schema = TrainingConfig.model_json_schema()
        properties = schema.get('properties', {})
        
        # Check that some fields have examples
        assert 'examples' in properties.get('name', {})
        assert 'examples' in properties.get('batch_size', {})
        assert 'examples' in properties.get('devices', {})

    def test_comprehensive_valid_config(self):
        """Test a comprehensive valid configuration."""
        config = TrainingConfig(
            # Run identification
            name="comprehensive_test",
            project="test_project",
            
            # Training hyperparameters
            batch_size=64,
            num_workers=4,
            seed=123,
            accum_batches=2,
            
            # Hardware configuration
            num_nodes=2,
            devices="2",
            strategy="ddp",
            precision="bf16-mixed",
            persistent_workers=True,
            
            # Checkpointing
            checkpoint_every=5,
            val_every=100,
            save_top_k=10,
            recover=True,
            
            # Optimization
            gradient_clip_val=1.0,
            
            # Logging
            logger="tensorboard",
            
            # Legacy
            remove_pretransform_weight_norm=True
        )
        
        # Verify all values are set correctly
        assert config.name == "comprehensive_test"
        assert config.batch_size == 64
        assert config.strategy == "ddp"
        assert config.precision == "bf16-mixed"
        assert config.logger == "tensorboard"
        assert config.remove_pretransform_weight_norm is True

    def test_edge_case_combinations(self):
        """Test edge case combinations of field values."""
        # Minimum valid values
        config_min = TrainingConfig(
            batch_size=1,
            num_workers=0,
            seed=0,
            accum_batches=1,
            num_nodes=1,
            checkpoint_every=1,
            val_every=-1,
            save_top_k=-1,
            gradient_clip_val=0.0
        )
        
        assert config_min.batch_size == 1
        assert config_min.num_workers == 0
        assert config_min.val_every == -1
        assert config_min.save_top_k == -1
        
        # Maximum valid values
        config_max = TrainingConfig(
            batch_size=512,
            num_workers=32,
            seed=999999,
            accum_batches=100,
            num_nodes=100,
            checkpoint_every=1000,
            val_every=10000,
            save_top_k=100,
            gradient_clip_val=10.0
        )
        
        assert config_max.batch_size == 512
        assert config_max.num_workers == 32
        assert config_max.gradient_clip_val == 10.0 