import json
from copy import deepcopy
from dataclasses import Field, asdict, dataclass, fields, MISSING
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, get_type_hints

import yaml

T = TypeVar('T', bound='BaseConfig')

@dataclass
class BaseConfig:
    """Base configuration class with validation and serialization support"""
    
    def __init__(self, **kwargs):
        """Initialize configuration with proper field default handling"""
        # Get all fields including inherited ones
        class_fields = fields(self.__class__)
        
        # First set defaults
        for field in class_fields:
            if field.default_factory is not MISSING:
                setattr(self, field.name, field.default_factory())
            elif field.default is not MISSING:
                setattr(self, field.name, field.default)
            else:
                setattr(self, field.name, None)
        
        # Then update with provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> None:
        """Validate configuration parameters"""
        type_hints = get_type_hints(self.__class__)
        for field in fields(self):
            value = getattr(self, field.name)
            field_type = type_hints.get(field.name)
            self._validate_field(field, value, field_type)

    def _validate_field(self, field: Field, value: Any, field_type: Any) -> None:
        """Validate a single field"""
        # Skip validation for explicitly optional fields that are None
        if value is None and self._is_optional(field_type):
            return
            
        if value is None and not self._is_optional(field_type):
            raise ValueError(f"{field.name} cannot be None")
            
        if isinstance(value, Path):
            self._validate_path_field(field.name, value)
            
        if hasattr(self, f"_validate_{field.name}"):
            validator = getattr(self, f"_validate_{field.name}")
            validator(value)

    def _validate_path_field(self, name: str, path: Path) -> None:
        """Validate path fields"""
        if name.endswith('_file'):
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
        elif name.endswith('_dir'):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_optional(field_type: Any) -> bool:
        """Check if field type is Optional"""
        # Check for both Optional[T] and Union[T, None] patterns
        origin = getattr(field_type, "__origin__", None)
        args = getattr(field_type, "__args__", ())
        return origin is Optional or (origin is Union and type(None) in args)

    def merge(self, other: 'BaseConfig') -> None:
        """Merge another config into this one"""
        for field in fields(other):
            value = getattr(other, field.name)
            if value is not None:
                setattr(self, field.name, deepcopy(value))

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create instance from dictionary"""
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)

    @classmethod
    def load_yaml(cls: Type[T], path: Path) -> T:
        """Load config from YAML file"""
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def load_json(cls: Type[T], path: Path) -> T:
        """Load config from JSON file"""
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def save_yaml(self, path: Path) -> None:
        """Save config as YAML file"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f)

    def save_json(self, path: Path) -> None:
        """Save config as JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
