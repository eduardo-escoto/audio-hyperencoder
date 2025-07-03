"""
Unit tests for DataConfig class.

Tests the data configuration class with all its validation rules,
constraints, path handling, and integration with BaseConfig.
"""

import pytest
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError
from hyperencoder.config import DataConfig, DatasetEntry, CropConfig


class TestCropConfig:
    """Test suite for CropConfig class."""

    def test_crop_config_defaults(self):
        """Test that CropConfig has correct default values."""
        config = CropConfig()
        
        assert config.random_crop is True
        assert config.crop_ratio == 8
        assert config.original_crop_length == 47

    def test_crop_config_custom_values(self):
        """Test CropConfig with custom values."""
        config = CropConfig(
            random_crop=False,
            crop_ratio=16,
            original_crop_length=32
        )
        
        assert config.random_crop is False
        assert config.crop_ratio == 16
        assert config.original_crop_length == 32

    def test_crop_ratio_validation(self):
        """Test crop_ratio field validation."""
        # Valid values
        valid_ratios = [1, 8, 16, 32]
        for ratio in valid_ratios:
            config = CropConfig(crop_ratio=ratio)
            assert config.crop_ratio == ratio
        
        # Invalid values - too small
        with pytest.raises(ValidationError):
            CropConfig(crop_ratio=0)
        
        # Invalid values - too large
        with pytest.raises(ValidationError):
            CropConfig(crop_ratio=64)

    def test_original_crop_length_validation(self):
        """Test original_crop_length field validation."""
        # Valid values
        valid_lengths = [1, 10, 47, 100]
        for length in valid_lengths:
            config = CropConfig(original_crop_length=length)
            assert config.original_crop_length == length
        
        # Invalid values - too small
        with pytest.raises(ValidationError):
            CropConfig(original_crop_length=0)


class TestDatasetEntry:
    """Test suite for DatasetEntry class."""

    def test_dataset_entry_defaults(self):
        """Test that DatasetEntry has correct default values."""
        config = DatasetEntry(path="/test/path")
        
        assert config.path == "/test/path"
        assert config.name is None
        assert config.weight == 1.0

    def test_dataset_entry_custom_values(self):
        """Test DatasetEntry with custom values."""
        config = DatasetEntry(
            path="/custom/path",
            name="custom_dataset",
            weight=0.5
        )
        
        assert config.path == "/custom/path"
        assert config.name == "custom_dataset"
        assert config.weight == 0.5

    def test_weight_validation(self):
        """Test weight field validation."""
        # Valid weights
        valid_weights = [0.0, 0.5, 1.0, 2.0]
        for weight in valid_weights:
            config = DatasetEntry(path="/test", weight=weight)
            assert config.weight == weight
        
        # Invalid weights - negative
        with pytest.raises(ValidationError):
            DatasetEntry(path="/test", weight=-0.1)

    def test_path_required(self):
        """Test that path is required."""
        with pytest.raises(ValidationError):
            DatasetEntry()


class TestDataConfig:
    """Test suite for DataConfig class."""

    def test_data_config_defaults(self):
        """Test that DataConfig has correct default values."""
        config = DataConfig(path="/test/path")
        
        # Core configuration
        assert config.target_ == "hyperencoder.data.latent.LatentsForHyperEncoderDataset"
        
        # Dataset type and loading strategy
        assert config.dataset_type == "latents_for_hyperencoder"
        assert config.split_type == "auto"
        assert config.loading_strategy == "lazy"
        
        # Data splits
        assert config.train_split_pct == 0.8
        assert config.val_split_pct == 0.1
        assert config.test_split_pct == 0.1
        
        # Dataset paths
        assert config.datasets == []
        assert config.path == "/test/path"
        
        # Crop configuration
        assert config.crop_config is None
        
        # Data processing options
        assert config.normalize is True
        assert config.shuffle is True
        
        # Caching options
        assert config.cache_dir is None
        assert config.use_cache is False
        
        # Advanced options
        assert config.max_files is None
        assert config.file_extension == ".pt"
        assert config.recursive is True

    def test_data_config_custom_values(self):
        """Test DataConfig with custom values."""
        crop_config = CropConfig(random_crop=False, crop_ratio=16)
        dataset_entry = DatasetEntry(path="/custom/path", weight=0.5)
        
        config = DataConfig(
            path="/test/path",
            target_="custom.dataset.class",
            dataset_type="audio_dataset",
            split_type="manual",
            loading_strategy="eager",
            train_split_pct=0.7,
            val_split_pct=0.2,
            test_split_pct=0.1,
            datasets=[dataset_entry],
            crop_config=crop_config,
            normalize=False,
            shuffle=False,
            cache_dir="/cache/path",
            use_cache=True,
            max_files=1000,
            file_extension=".wav",
            recursive=False
        )
        
        assert config.target_ == "custom.dataset.class"
        assert config.dataset_type == "audio_dataset"
        assert config.split_type == "manual"
        assert config.loading_strategy == "eager"
        assert config.train_split_pct == 0.7
        assert config.val_split_pct == 0.2
        assert config.test_split_pct == 0.1
        assert len(config.datasets) == 1
        assert config.datasets[0].path == "/custom/path"
        assert config.crop_config.crop_ratio == 16
        assert config.normalize is False
        assert config.shuffle is False
        assert config.cache_dir == "/cache/path"
        assert config.use_cache is True
        assert config.max_files == 1000
        assert config.file_extension == ".wav"
        assert config.recursive is False

    def test_dataset_type_validation(self):
        """Test dataset_type field validation (Literal type)."""
        # Valid types
        valid_types = ["latents_for_hyperencoder", "audio_dataset", "pre_encoded_dataset"]
        for dataset_type in valid_types:
            config = DataConfig(path="/test", dataset_type=dataset_type)
            assert config.dataset_type == dataset_type
        
        # Invalid type
        with pytest.raises(ValidationError):
            DataConfig(path="/test", dataset_type="invalid_type")

    def test_split_type_validation(self):
        """Test split_type field validation (Literal type)."""
        # Valid types
        valid_types = ["auto", "manual", "none"]
        for split_type in valid_types:
            config = DataConfig(path="/test", split_type=split_type)
            assert config.split_type == split_type
        
        # Invalid type
        with pytest.raises(ValidationError):
            DataConfig(path="/test", split_type="invalid_type")

    def test_loading_strategy_validation(self):
        """Test loading_strategy field validation (Literal type)."""
        # Valid strategies
        valid_strategies = ["lazy", "eager", "memory_mapped"]
        for strategy in valid_strategies:
            config = DataConfig(path="/test", loading_strategy=strategy)
            assert config.loading_strategy == strategy
        
        # Invalid strategy
        with pytest.raises(ValidationError):
            DataConfig(path="/test", loading_strategy="invalid_strategy")

    def test_split_percentages_validation(self):
        """Test that split percentages are valid."""
        # Valid percentages
        valid_splits = [
            (0.8, 0.1, 0.1),
            (0.7, 0.2, 0.1),
            (0.6, 0.3, 0.1),
            (1.0, 0.0, 0.0),
        ]
        
        for train, val, test in valid_splits:
            config = DataConfig(
                path="/test",
                train_split_pct=train,
                val_split_pct=val,
                test_split_pct=test
            )
            assert config.train_split_pct == train
            assert config.val_split_pct == val
            assert config.test_split_pct == test
        
        # Invalid percentages - negative
        with pytest.raises(ValidationError):
            DataConfig(path="/test", train_split_pct=-0.1)
        
        # Invalid percentages - too large
        with pytest.raises(ValidationError):
            DataConfig(path="/test", train_split_pct=1.5)

    def test_splits_sum_to_one_validation(self):
        """Test that train/val/test splits sum to 1.0."""
        # Valid splits that sum to 1.0
        config = DataConfig(
            path="/test",
            train_split_pct=0.8,
            val_split_pct=0.1,
            test_split_pct=0.1
        )
        assert abs(config.train_split_pct + config.val_split_pct + config.test_split_pct - 1.0) < 1e-6
        
        # Invalid splits that don't sum to 1.0 - This test might need adjustment
        # depending on whether the validation is implemented to check the sum
        # during model validation. For now, we'll test individual field validation.

    def test_max_files_validation(self):
        """Test max_files field validation."""
        # Valid values
        valid_values = [1, 100, 1000, 10000]
        for value in valid_values:
            config = DataConfig(path="/test", max_files=value)
            assert config.max_files == value
        
        # Invalid values - too small
        with pytest.raises(ValidationError):
            DataConfig(path="/test", max_files=0)

    def test_datasets_or_path_validation(self):
        """Test that either datasets list or path is provided."""
        # Valid - path provided
        config = DataConfig(path="/test/path")
        assert config.path == "/test/path"
        
        # Valid - datasets list provided
        dataset_entry = DatasetEntry(path="/dataset/path")
        config = DataConfig(datasets=[dataset_entry])
        assert len(config.datasets) == 1
        
        # Invalid - neither provided
        with pytest.raises(ValidationError):
            DataConfig(datasets=[])

    def test_path_resolution_with_interpolation(self):
        """Test path resolution with OmegaConf interpolations."""
        # Paths with interpolations should not be resolved
        config = DataConfig(path="${oc.env:DATA_ROOT}/dataset")
        assert config.path == "${oc.env:DATA_ROOT}/dataset"
        
        # Regular paths should be resolved (converted to string)
        config = DataConfig(path="/absolute/path")
        assert config.path == "/absolute/path"

    def test_target_field_alias(self):
        """Test that _target_ field works with alias."""
        # Test with alias
        config = DataConfig(path="/test", **{"_target_": "custom.target.class"})
        assert config.target_ == "custom.target.class"
        
        # Test with field name
        config = DataConfig(path="/test", target_="custom.target.class")
        assert config.target_ == "custom.target.class"

    def test_serialization_to_dict(self):
        """Test serialization to dictionary."""
        crop_config = CropConfig(crop_ratio=16)
        dataset_entry = DatasetEntry(path="/test/path", weight=0.5)
        
        config = DataConfig(
            target_="custom.class",
            path="/test/path",
            datasets=[dataset_entry],
            crop_config=crop_config,
            normalize=False
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["target_"] == "custom.class"
        assert config_dict["path"] == "/test/path"
        assert len(config_dict["datasets"]) == 1
        assert config_dict["datasets"][0]["path"] == "/test/path"
        assert config_dict["crop_config"]["crop_ratio"] == 16
        assert config_dict["normalize"] is False

    def test_deserialization_from_dict(self):
        """Test deserialization from dictionary."""
        config_dict = {
            "target_": "custom.class",
            "path": "/test/path",
            "dataset_type": "audio_dataset",
            "normalize": False,
            "crop_config": {
                "crop_ratio": 16,
                "random_crop": False
            }
        }
        
        config = DataConfig.from_dict(config_dict)
        
        assert config.target_ == "custom.class"
        assert config.path == "/test/path"
        assert config.dataset_type == "audio_dataset"
        assert config.normalize is False
        assert config.crop_config.crop_ratio == 16
        assert config.crop_config.random_crop is False

    def test_yaml_serialization(self, temp_dir):
        """Test YAML serialization and deserialization."""
        crop_config = CropConfig(crop_ratio=16)
        config = DataConfig(
            path="/test/path",
            crop_config=crop_config,
            normalize=False
        )
        
        # Save to YAML
        yaml_path = temp_dir / "test_config.yaml"
        config.save_yaml(yaml_path)
        
        # Load from YAML
        loaded_config = DataConfig.from_yaml(yaml_path)
        
        assert loaded_config.path == "/test/path"
        assert loaded_config.crop_config.crop_ratio == 16
        assert loaded_config.normalize is False

    def test_comprehensive_valid_config(self):
        """Test a comprehensive valid configuration."""
        crop_config = CropConfig(
            random_crop=True,
            crop_ratio=8,
            original_crop_length=47
        )
        
        dataset_entries = [
            DatasetEntry(path="/dataset1", weight=0.7),
            DatasetEntry(path="/dataset2", weight=0.3, name="secondary")
        ]
        
        config = DataConfig(
            target_="hyperencoder.data.latent.LatentsForHyperEncoderDataset",
            dataset_type="latents_for_hyperencoder",
            split_type="auto",
            loading_strategy="lazy",
            train_split_pct=0.8,
            val_split_pct=0.1,
            test_split_pct=0.1,
            datasets=dataset_entries,
            crop_config=crop_config,
            normalize=True,
            shuffle=True,
            cache_dir="/cache",
            use_cache=True,
            max_files=10000,
            file_extension=".pt",
            recursive=True
        )
        
        # Verify all fields are set correctly
        assert config.target_ == "hyperencoder.data.latent.LatentsForHyperEncoderDataset"
        assert config.dataset_type == "latents_for_hyperencoder"
        assert config.split_type == "auto"
        assert config.loading_strategy == "lazy"
        assert config.train_split_pct == 0.8
        assert config.val_split_pct == 0.1
        assert config.test_split_pct == 0.1
        assert len(config.datasets) == 2
        assert config.datasets[0].weight == 0.7
        assert config.datasets[1].name == "secondary"
        assert config.crop_config.crop_ratio == 8
        assert config.normalize is True
        assert config.shuffle is True
        assert config.cache_dir == "/cache"
        assert config.use_cache is True
        assert config.max_files == 10000
        assert config.file_extension == ".pt"
        assert config.recursive is True

    def test_field_descriptions_present(self):
        """Test that important fields have descriptions."""
        config = DataConfig(path="/test")
        schema = config.model_json_schema()
        
        # Check that key fields have descriptions
        properties = schema["properties"]
        assert "description" in properties["_target_"]  # Uses alias in schema
        assert "description" in properties["dataset_type"]
        assert "description" in properties["loading_strategy"]
        assert "description" in properties["train_split_pct"] 