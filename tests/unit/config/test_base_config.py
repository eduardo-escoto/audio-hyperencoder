"""
Unit tests for BaseConfig class.

Tests the foundational configuration class that all other configs inherit from.
"""

import pytest
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, ValidationError
from hyperencoder.config import BaseConfig


class TestBaseConfig:
    """Test suite for BaseConfig class."""

    def test_base_config_instantiation(self):
        """Test that BaseConfig can be instantiated with defaults."""
        config = BaseConfig()
        assert config is not None
        assert isinstance(config, BaseConfig)

    def test_base_config_with_subclass(self):
        """Test that BaseConfig can be subclassed properly."""
        
        class TestConfig(BaseConfig):
            name: str = Field(default="test")
            value: int = Field(default=42)
        
        config = TestConfig()
        assert config.name == "test"
        assert config.value == 42
        
        # Test with custom values
        config2 = TestConfig(name="custom", value=100)
        assert config2.name == "custom"
        assert config2.value == 100

    def test_base_config_validation_behavior(self):
        """Test that validation works correctly."""
        
        class ValidatedConfig(BaseConfig):
            positive_int: int = Field(ge=1, description="Must be positive")
            limited_string: str = Field(min_length=1, max_length=10)
        
        # Valid configuration
        config = ValidatedConfig(positive_int=5, limited_string="hello")
        assert config.positive_int == 5
        assert config.limited_string == "hello"
        
        # Invalid configurations should raise ValidationError
        with pytest.raises(ValidationError):
            ValidatedConfig(positive_int=0, limited_string="hello")  # positive_int too small
            
        with pytest.raises(ValidationError):
            ValidatedConfig(positive_int=5, limited_string="")  # string too short
            
        with pytest.raises(ValidationError):
            ValidatedConfig(positive_int=5, limited_string="a" * 20)  # string too long

    def test_to_dict_method(self):
        """Test the to_dict() method."""
        
        class TestConfig(BaseConfig):
            name: str = "test"
            count: int = 42
            enabled: bool = True
        
        config = TestConfig()
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["enabled"] is True

    def test_from_dict_method(self):
        """Test the from_dict() class method."""
        
        class TestConfig(BaseConfig):
            name: str = Field(default="default")
            count: int = Field(default=0)
        
        data = {"name": "from_dict", "count": 123}
        config = TestConfig.from_dict(data)
        
        assert config.name == "from_dict"
        assert config.count == 123

    def test_round_trip_dict_conversion(self):
        """Test that to_dict() and from_dict() work together."""
        
        class TestConfig(BaseConfig):
            name: str = "original"
            values: list[int] = [1, 2, 3]
            nested: dict[str, str] = {"key": "value"}
        
        original = TestConfig()
        dict_repr = original.to_dict()
        reconstructed = TestConfig.from_dict(dict_repr)
        
        assert original.model_dump() == reconstructed.model_dump()

    def test_save_yaml_method(self, temp_dir):
        """Test saving configuration to YAML file."""
        
        class TestConfig(BaseConfig):
            name: str = "yaml_test"
            settings: dict[str, Any] = {"debug": True, "timeout": 30}
        
        config = TestConfig()
        yaml_path = temp_dir / "test_config.yaml"
        
        config.save_yaml(yaml_path)
        
        # Verify file was created
        assert yaml_path.exists()
        
        # Verify content
        with open(yaml_path, 'r') as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["name"] == "yaml_test"
        assert loaded_data["settings"]["debug"] is True
        assert loaded_data["settings"]["timeout"] == 30

    def test_from_yaml_method(self, temp_dir):
        """Test loading configuration from YAML file."""
        
        class TestConfig(BaseConfig):
            name: str = Field(default="default")
            count: int = Field(default=0)
            enabled: bool = Field(default=False)
        
        # Create test YAML file
        yaml_path = temp_dir / "test_load.yaml"
        test_data = {
            "name": "loaded_from_yaml",
            "count": 999,
            "enabled": True
        }
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(test_data, f)
        
        # Load configuration
        config = TestConfig.from_yaml(yaml_path)
        
        assert config.name == "loaded_from_yaml"
        assert config.count == 999
        assert config.enabled is True

    def test_round_trip_yaml_conversion(self, temp_dir):
        """Test that save_yaml() and from_yaml() work together."""
        
        class TestConfig(BaseConfig):
            name: str = "round_trip_test"
            numbers: list[int] = [10, 20, 30]
            metadata: dict[str, Any] = {"version": "1.0", "author": "test"}
        
        original = TestConfig()
        yaml_path = temp_dir / "round_trip.yaml"
        
        # Save and load
        original.save_yaml(yaml_path)
        loaded = TestConfig.from_yaml(yaml_path)
        
        # Compare
        assert original.model_dump() == loaded.model_dump()

    def test_yaml_with_none_values(self, temp_dir):
        """Test YAML serialization with None values."""
        
        class TestConfig(BaseConfig):
            name: str = "test"
            optional_field: Optional[str] = None
            optional_path: Optional[Path] = None
        
        config = TestConfig()
        yaml_path = temp_dir / "none_values.yaml"
        
        config.save_yaml(yaml_path)
        loaded = TestConfig.from_yaml(yaml_path)
        
        assert loaded.name == "test"
        assert loaded.optional_field is None
        assert loaded.optional_path is None

    def test_config_dict_settings(self):
        """Test that ConfigDict settings work correctly."""
        
        class StrictConfig(BaseConfig):
            name: str = "test"
        
        config = StrictConfig(name="allowed")
        assert config.name == "allowed"
        
        # Test that extra fields are forbidden (based on our BaseConfig.model_config)
        with pytest.raises(ValidationError):
            StrictConfig(name="test", extra_field="not_allowed")

    def test_validate_assignment(self):
        """Test that validate_assignment works."""
        
        class TestConfig(BaseConfig):
            count: int = Field(ge=0)
        
        config = TestConfig(count=5)
        assert config.count == 5
        
        # This should work - valid assignment
        config.count = 10
        assert config.count == 10
        
        # This should fail - invalid assignment
        with pytest.raises(ValidationError):
            config.count = -1

    def test_inheritance_behavior(self):
        """Test that configuration inheritance works correctly."""
        
        class BaseTestConfig(BaseConfig):
            name: str = "base"
            base_field: int = 100
        
        class DerivedTestConfig(BaseTestConfig):
            name: str = "derived"  # Override default
            derived_field: str = "derived_value"
        
        config = DerivedTestConfig()
        assert config.name == "derived"
        assert config.base_field == 100
        assert config.derived_field == "derived_value"
        
        # Test that derived config can be converted to dict
        config_dict = config.to_dict()
        assert "name" in config_dict
        assert "base_field" in config_dict
        assert "derived_field" in config_dict

    def test_error_handling(self, temp_dir):
        """Test error handling in various scenarios."""
        
        class TestConfig(BaseConfig):
            name: str = "test"
        
        # Test loading from non-existent file
        non_existent = temp_dir / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            TestConfig.from_yaml(non_existent)
        
        # Test loading from invalid YAML
        invalid_yaml = temp_dir / "invalid.yaml"
        with open(invalid_yaml, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        with pytest.raises(yaml.YAMLError):
            TestConfig.from_yaml(invalid_yaml)

    def test_path_handling(self, temp_dir):
        """Test configuration with Path objects."""
        
        class PathConfig(BaseConfig):
            config_path: Optional[Path] = None
            data_dir: Optional[Path] = None
        
        config = PathConfig(
            config_path=temp_dir / "config.yaml",
            data_dir=temp_dir / "data"
        )
        
        assert isinstance(config.config_path, Path)
        assert isinstance(config.data_dir, Path)
        
        # Test serialization handles Path objects
        config_dict = config.to_dict()
        assert isinstance(config_dict["config_path"], Path)
        assert isinstance(config_dict["data_dir"], Path)
        
        # Test YAML serialization (should convert Path to string)
        yaml_path = temp_dir / "path_config.yaml"
        config.save_yaml(yaml_path)
        
        # Load and verify
        loaded = PathConfig.from_yaml(yaml_path)
        assert loaded.config_path == config.config_path
        assert loaded.data_dir == config.data_dir 