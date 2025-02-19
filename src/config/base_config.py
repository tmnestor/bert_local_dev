import json
from copy import deepcopy
from dataclasses import Field, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, get_type_hints

import yaml

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base configuration class providing validation and serialization.

    This class serves as the foundation for all configuration classes in the project,
    providing common functionality for validation, serialization, and path management.

    The class uses Python's dataclass functionality combined with type hints for
    automatic validation and serialization of configuration parameters.

    Attributes:
        No default attributes - this is a base class

    Methods:
        validate(): Validates all configuration parameters
        merge(other): Merges another config into this one
        to_dict(): Converts config to dictionary
        from_dict(config_dict): Creates instance from dictionary
        load_yaml(path): Loads config from YAML file
        load_json(path): Loads config from JSON file
        save_yaml(path): Saves config as YAML file
        save_json(path): Saves config as JSON file
    """

    def __init__(self, **kwargs):
        """Initialize allowing unknown kwargs."""
        # Only set attributes that are defined in the class
        known_fields = {f.name for f in fields(self)}
        for name, value in kwargs.items():
            if name in known_fields:
                setattr(self, name, value)

    def validate(self) -> None:
        """Validates all configuration parameters.

        Performs type checking and validation for all fields in the configuration.
        Validates paths exist for file fields and creates directories for directory fields.

        Raises:
            ValueError: If any configuration parameter is invalid
            FileNotFoundError: If required files don't exist
        """
        type_hints = get_type_hints(self.__class__)
        for field in fields(self):
            value = getattr(self, field.name)
            field_type = type_hints.get(field.name)
            self._validate_field(field, value, field_type, field.name)

    def _validate_field(
        self, field: Field, value: Any, field_type: Any, field_name: str
    ) -> None:
        """Validates a single configuration field.

        Args:
            field: Field descriptor from dataclass
            value: Value to validate
            field_type: Expected type of the field
            field_name: Name of the field being validated

        Raises:
            ValueError: If field validation fails
        """
        # Skip validation for explicitly optional fields that are None
        if value is None and self._is_optional(field_type):
            return

        if value is None and not self._is_optional(field_type):
            raise ValueError(f"{field_name} cannot be None")

        if isinstance(value, Path):
            self._validate_path_field(field_name, value)

        # Centralized validation logic
        if field_name == "num_classes" and value is not None and value < 1:
            raise ValueError("num_classes must be positive if specified")
        if field_name == "max_seq_len" and value < 1:
            raise ValueError("max_seq_len must be positive")
        if field_name == "metrics" and not isinstance(value, list):
            raise ValueError("metrics must be a list")

    def _validate_path_field(self, name: str, path: Path) -> None:
        """Validate path fields"""
        if name.endswith("_file"):
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
        elif name.endswith("_dir"):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_optional(field_type: Any) -> bool:
        """Check if field type is Optional"""
        # Check for both Optional[T] and Union[T, None] patterns
        origin = getattr(field_type, "__origin__", None)
        args = getattr(field_type, "__args__", ())
        return origin is Optional or (origin is Union and type(None) in args)

    def merge(self, other: "BaseConfig") -> None:
        """Merge another config into this one"""
        for field in fields(other):
            value = getattr(other, field.name)
            if value is not None:
                setattr(self, field.name, deepcopy(value))

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create instance from dictionary"""
        # Get all fields that should be initialized
        init_fields = {f.name for f in fields(cls) if f.init}
        filtered_dict = {k: v for k, v in config_dict.items() if k in init_fields}
        return cls(**filtered_dict)

    @classmethod
    def load_yaml(cls: Type[T], path: Path) -> T:
        """Load config from YAML file"""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def load_json(cls: Type[T], path: Path) -> T:
        """Load config from JSON file"""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def save_yaml(self, path: Path) -> None:
        """Save config as YAML file"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f)

    def save_json(self, path: Path) -> None:
        """Save config as JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
