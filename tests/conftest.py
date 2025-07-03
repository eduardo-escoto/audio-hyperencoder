"""
Pytest configuration and shared fixtures for hyperencoder tests.

This file contains common test fixtures, utilities, and pytest configuration
that can be used across all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, cast

from omegaconf import OmegaConf, DictConfig
from hyperencoder.config import BaseConfig, TrainingConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "name": "test_experiment",
        "project": "test_project",
        "batch_size": 64,
        "num_workers": 4,
        "seed": 42,
        "devices": "auto",
        "strategy": "auto",
        "precision": "16-mixed",
        "logger": "wandb",
    }


@pytest.fixture
def sample_training_config(sample_config_dict) -> TrainingConfig:
    """Sample TrainingConfig instance for testing."""
    return TrainingConfig(**sample_config_dict)


@pytest.fixture
def sample_dictconfig(sample_config_dict) -> DictConfig:
    """Sample OmegaConf DictConfig for testing."""
    return cast(DictConfig, OmegaConf.create(sample_config_dict))


@pytest.fixture
def config_files_dir(temp_dir) -> Path:
    """Create a temporary directory with test config files."""
    config_dir = temp_dir / "configs"
    config_dir.mkdir()
    
    # Create a sample YAML config file
    config_file = config_dir / "test_config.yaml"
    config_data = {
        "name": "test_from_file",
        "batch_size": 32,
        "num_workers": 2,
    }
    
    with open(config_file, 'w') as f:
        import yaml
        yaml.safe_dump(config_data, f)
    
    return config_dir


@pytest.fixture
def invalid_config_dict() -> Dict[str, Any]:
    """Invalid configuration dictionary for testing validation."""
    return {
        "name": "test_invalid",
        "batch_size": 0,  # Invalid: must be >= 1
        "num_workers": -1,  # Invalid: must be >= 0
        "precision": "invalid_precision",  # Invalid: not in allowed values
    }


# Test utilities
def assert_config_equals(config1: BaseConfig, config2: BaseConfig) -> None:
    """Assert that two configuration objects are equal."""
    assert config1.model_dump() == config2.model_dump()


def create_test_config_file(path: Path, config_data: Dict[str, Any]) -> None:
    """Create a test configuration file."""
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(config_data, f)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    ) 