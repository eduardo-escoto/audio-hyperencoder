"""
Base configuration classes for the hyperencoder project.

This module provides the foundational Pydantic models that all other
configuration classes inherit from.
"""

from typing import Any
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """
    Base configuration class that all other config classes inherit from.

    Provides common functionality like:
    - JSON schema generation
    - Validation
    - Documentation embedding
    - Path resolution
    """

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="forbid",
        # Validate assignments
        validate_assignment=True,
        # Use enum values instead of raw values
        use_enum_values=True,
        # Populate by name (allows both field names and aliases)
        populate_by_name=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfig":
        """Create a config instance from a dictionary."""
        return cls(**data)

    def save_yaml(self, path: Path) -> None:
        """Save the config to a YAML file."""
        import yaml

        # Convert Path objects to strings for YAML compatibility
        def path_to_str(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: path_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [path_to_str(item) for item in obj]
            return obj

        config_dict = path_to_str(self.to_dict())

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)

    @classmethod
    def from_yaml(cls, path: Path) -> "BaseConfig":
        """Load a config from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
