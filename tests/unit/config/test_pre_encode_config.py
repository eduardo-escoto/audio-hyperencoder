"""
Unit tests for PreEncodeConfig class.

Tests the pre-encoding configuration class and its validation behavior.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError
from hyperencoder.config import PreEncodeConfig


class TestPreEncodeConfig:
    """Test suite for PreEncodeConfig class."""

    def test_pre_encode_config_instantiation(self):
        """Test that PreEncodeConfig can be instantiated with required fields."""
        config = PreEncodeConfig(
            input_dir="/path/to/input",
            output_dir="/path/to/output"
        )
        assert config is not None
        assert isinstance(config, PreEncodeConfig)

    def test_pre_encode_config_defaults(self):
        """Test that default values are set correctly."""
        config = PreEncodeConfig(
            input_dir="/path/to/input",
            output_dir="/path/to/output"
        )
        
        # Test default values
        assert config.model_name == "stabilityai/stable-audio-open-1.0"
        assert config.hf_token is None
        assert config.file_pattern == "*.wav"
        assert config.batch_pattern == r"Track\d*"
        assert config.n_devices == 1
        assert config.batch_size == 1
        assert config.create_output_dir is True
        assert config.log_failures is True
        assert config.use_path_file is False
        assert config.path_file is None

    def test_pre_encode_config_required_fields(self):
        """Test that required fields are validated."""
        # Missing input_dir should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            PreEncodeConfig(output_dir="/path/to/output")  # type: ignore
        assert "input_dir" in str(exc_info.value)
        
        # Missing output_dir should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            PreEncodeConfig(input_dir="/path/to/input")  # type: ignore
        assert "output_dir" in str(exc_info.value)

    def test_pre_encode_config_validation_constraints(self):
        """Test that validation constraints work correctly."""
        # Test n_devices validation (should be >= 1)
        with pytest.raises(ValidationError):
            PreEncodeConfig(
                input_dir="/path/to/input",
                output_dir="/path/to/output",
                n_devices=0
            )
            
        with pytest.raises(ValidationError):
            PreEncodeConfig(
                input_dir="/path/to/input",
                output_dir="/path/to/output",
                n_devices=-1
            )
            
        # Test batch_size validation (should be >= 1)
        with pytest.raises(ValidationError):
            PreEncodeConfig(
                input_dir="/path/to/input",
                output_dir="/path/to/output",
                batch_size=0
            )
            
        with pytest.raises(ValidationError):
            PreEncodeConfig(
                input_dir="/path/to/input",
                output_dir="/path/to/output",
                batch_size=-1
            )
            
        # Valid values should work
        config = PreEncodeConfig(
            input_dir="/path/to/input",
            output_dir="/path/to/output",
            n_devices=2,
            batch_size=4
        )
        assert config.n_devices == 2
        assert config.batch_size == 4

    def test_pre_encode_config_with_all_fields(self):
        """Test configuration with all fields specified."""
        config = PreEncodeConfig(
            model_name="custom/model",
            hf_token="test_token",
            input_dir="/custom/input",
            output_dir="/custom/output",
            file_pattern="*.mp3",
            batch_pattern=r"Song\d*",
            n_devices=2,
            batch_size=8,
            create_output_dir=False,
            log_failures=False,
            use_path_file=True,
            path_file="/path/to/file_list.txt"
        )
        
        assert config.model_name == "custom/model"
        assert config.hf_token == "test_token"
        assert config.input_dir == "/custom/input"
        assert config.output_dir == "/custom/output"
        assert config.file_pattern == "*.mp3"
        assert config.batch_pattern == r"Song\d*"
        assert config.n_devices == 2
        assert config.batch_size == 8
        assert config.create_output_dir is False
        assert config.log_failures is False
        assert config.use_path_file is True
        assert config.path_file == "/path/to/file_list.txt"

    def test_pre_encode_config_to_dict(self):
        """Test the to_dict() method."""
        config = PreEncodeConfig(
            input_dir="/path/to/input",
            output_dir="/path/to/output",
            n_devices=2,
            batch_size=4
        )
        
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result["input_dir"] == "/path/to/input"
        assert result["output_dir"] == "/path/to/output"
        assert result["n_devices"] == 2
        assert result["batch_size"] == 4
        assert result["model_name"] == "stabilityai/stable-audio-open-1.0"

    def test_pre_encode_config_from_dict(self):
        """Test the from_dict() class method."""
        data = {
            "input_dir": "/test/input",
            "output_dir": "/test/output",
            "model_name": "test/model",
            "n_devices": 3,
            "batch_size": 6
        }
        
        config = PreEncodeConfig.from_dict(data)
        
        assert config.input_dir == "/test/input"
        assert config.output_dir == "/test/output"
        assert config.model_name == "test/model"
        assert config.n_devices == 3
        assert config.batch_size == 6

    def test_pre_encode_config_round_trip_dict_conversion(self):
        """Test that to_dict() and from_dict() work together."""
        original = PreEncodeConfig(
            input_dir="/original/input",
            output_dir="/original/output",
            model_name="original/model",
            n_devices=4,
            batch_size=8,
            hf_token="original_token"
        )
        
        dict_repr = original.to_dict()
        reconstructed = PreEncodeConfig.from_dict(dict_repr)
        
        assert original.model_dump() == reconstructed.model_dump()

    def test_pre_encode_config_save_yaml(self, temp_dir):
        """Test saving configuration to YAML file."""
        config = PreEncodeConfig(
            input_dir="/yaml/input",
            output_dir="/yaml/output",
            model_name="yaml/model",
            n_devices=2,
            batch_size=4
        )
        
        yaml_path = temp_dir / "pre_encode_config.yaml"
        config.save_yaml(yaml_path)
        
        # Verify file was created
        assert yaml_path.exists()
        
        # Verify content by loading it back
        loaded = PreEncodeConfig.from_yaml(yaml_path)
        assert loaded.input_dir == "/yaml/input"
        assert loaded.output_dir == "/yaml/output"
        assert loaded.model_name == "yaml/model"
        assert loaded.n_devices == 2
        assert loaded.batch_size == 4

    def test_pre_encode_config_from_yaml(self, temp_dir):
        """Test loading configuration from YAML file."""
        yaml_path = temp_dir / "test_pre_encode.yaml"
        
        # Create test YAML file
        yaml_content = """
input_dir: "/yaml/load/input"
output_dir: "/yaml/load/output"
model_name: "yaml/load/model"
n_devices: 3
batch_size: 6
hf_token: "yaml_token"
file_pattern: "*.flac"
batch_pattern: "Track\\\\d+"
create_output_dir: false
log_failures: false
"""
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        # Load configuration
        config = PreEncodeConfig.from_yaml(yaml_path)
        
        assert config.input_dir == "/yaml/load/input"
        assert config.output_dir == "/yaml/load/output"
        assert config.model_name == "yaml/load/model"
        assert config.n_devices == 3
        assert config.batch_size == 6
        assert config.hf_token == "yaml_token"
        assert config.file_pattern == "*.flac"
        assert config.batch_pattern == r"Track\d+"
        assert config.create_output_dir is False
        assert config.log_failures is False

    def test_pre_encode_config_round_trip_yaml_conversion(self, temp_dir):
        """Test that save_yaml() and from_yaml() work together."""
        original = PreEncodeConfig(
            input_dir="/round/trip/input",
            output_dir="/round/trip/output",
            model_name="round/trip/model",
            n_devices=4,
            batch_size=8,
            hf_token="round_trip_token",
            file_pattern="*.ogg",
            batch_pattern=r"Audio\d*",
            create_output_dir=False,
            log_failures=True,
            use_path_file=True,
            path_file="/path/to/files.txt"
        )
        
        yaml_path = temp_dir / "round_trip_pre_encode.yaml"
        
        # Save and load
        original.save_yaml(yaml_path)
        loaded = PreEncodeConfig.from_yaml(yaml_path)
        
        # Compare
        assert original.model_dump() == loaded.model_dump()

    def test_pre_encode_config_field_constraints(self):
        """Test specific field constraints and edge cases."""
        base_config = {
            "input_dir": "/test/input",
            "output_dir": "/test/output"
        }
        
        # Test upper bounds for n_devices
        with pytest.raises(ValidationError):
            PreEncodeConfig(**base_config, n_devices=9)  # Should be <= 8
            
        # Test upper bounds for batch_size
        with pytest.raises(ValidationError):
            PreEncodeConfig(**base_config, batch_size=33)  # Should be <= 32
            
        # Test valid upper bounds
        config = PreEncodeConfig(**base_config, n_devices=8, batch_size=32)
        assert config.n_devices == 8
        assert config.batch_size == 32

    def test_pre_encode_config_path_file_usage(self):
        """Test path_file usage scenarios."""
        # Test with use_path_file=True and path_file specified
        config = PreEncodeConfig(
            input_dir="/test/input",
            output_dir="/test/output",
            use_path_file=True,
            path_file="/test/file_list.txt"
        )
        
        assert config.use_path_file is True
        assert config.path_file == "/test/file_list.txt"
        
        # Test with use_path_file=False and no path_file
        config2 = PreEncodeConfig(
            input_dir="/test/input",
            output_dir="/test/output",
            use_path_file=False
        )
        
        assert config2.use_path_file is False
        assert config2.path_file is None 