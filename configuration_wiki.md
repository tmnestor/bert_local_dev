#### /Users/tod/Desktop/_bert_local_dev/configuration_wiki.md
# Configuration Management in BERT Classifier Project

This document outlines how configurations are managed within the BERT Classifier project. The system is designed to provide flexibility, maintainability, and reproducibility across different environments and tasks.

## Overview

The configuration system relies on a combination of YAML files, command-line arguments, and dataclasses to define and manage settings for various aspects of the project, including:

*   Data loading and preprocessing
*   Model architecture
*   Training parameters
*   Optimization settings
*   Paths and directory structures

## Key Components

1.  **`config.yml`**: The primary configuration file in YAML format. It defines default values for all configurable parameters.

    *   Uses YAML anchors and aliases for reuse and consistency.
    *   Supports custom YAML tags (e.g., `!join`) for dynamic path construction.

2.  **`configuration.py`**: This module handles loading, validating, and providing access to configuration settings.

    *   Defines dataclasses (e.g., `ModelConfig`, `PredictionConfig`) to represent different configuration profiles.
    *   Loads `config.yml` and processes its values.
    *   Provides a central `get_config()` function to retrieve configuration instances.
    *   Includes helper functions for processing paths and optimizer settings.

3.  **BaseConfig**: A base class for all configuration dataclasses, providing common functionality like validation and serialization.

4.  **Command-Line Arguments**: Used to override settings defined in `config.yml`.

    *   Each configuration dataclass has an `add_argparse_args()` method to define command-line arguments.
    *   The `from_args()` method creates a configuration instance from parsed arguments.

## Configuration Loading and Processing

1.  **Loading `config.yml`**: The `load_yaml_config()` function loads the `config.yml` file.

    *   Handles custom YAML tags for path construction.
    *   Performs initial processing of configuration values (e.g., converting paths to `Path` objects).

2.  **Creating Configuration Instances**: The `get_config()` function is the main entry point for obtaining configuration instances.

    *   Loads the base configuration from `config.yml`.
    *   Overrides settings with command-line arguments, if provided.
    *   Creates an instance of the appropriate configuration dataclass (e.g., `ModelConfig`, `PredictionConfig`).
    *   Calls the `validate()` method to ensure the configuration is valid.

3.  **Dataclass Initialization (`__post_init__`)**: The `__post_init__()` method in each configuration dataclass performs additional initialization steps.

    *   Sets up directory paths.
    *   Resolves relative paths to absolute paths.

## Configuration Dataclasses

The project uses several configuration dataclasses, each tailored to a specific task:

*   **`ModelConfig`**: Base configuration for model training and evaluation.

    *   Defines settings for data loading, model architecture, training parameters, and paths.
        ```python
        @dataclass
        class ModelConfig(BaseConfig):
            """Strongly typed configuration for model training.

            Attributes:
                bert_encoder_path (Path): Path to the BERT encoder model.
                bert_model_name (str): Name or path of the BERT model.
                num_classes (Optional[int]): Number of output classes.
                # ... more attributes ...
            """
            bert_encoder_path: Path = field(
                default_factory=lambda: Path(CONFIG["model_paths"]["bert_encoder"]),
                metadata={"help": "Path to the BERT encoder model"},
            )
            # ... more fields ...
        ```

*   **`PredictionConfig`**: Configuration for prediction tasks.

    *   Inherits from `ModelConfig`.
    *   Adds settings specific to prediction, such as the path to the trained model and the output file.

## Overriding Configuration Values

Configuration values can be overridden in several ways:

1.  **Command-Line Arguments**: Provide values directly when running a script.

    ```bash
    python -m src.training.train --output_root "/path/to/output" --num_epochs 10
    ```

2.  **Modifying `config.yml`**: Change the default values in the configuration file.

## Best Practices

*   **Use Absolute Paths**:  Wherever possible, use absolute paths in `config.yml` to avoid issues with relative paths when running scripts from different directories.
*   **Validate Configurations**:  Always call the `validate()` method on configuration instances to catch errors early.
*   **Document Configuration Parameters**:  Provide clear descriptions for all configuration parameters in the code and in this document. Use docstrings in the configuration dataclasses and their fields to provide more context and usage information directly in the code.
*   **Centralize Configuration Access**:  Use the `get_config()` function as the single entry point for obtaining configuration instances.
*   **Centralized Validation**: Validation logic is consolidated into the `_validate_field` method within `BaseConfig` for consistency and maintainability.
*   **Explicit Path Handling**: The `resolve_path` function is used to resolve paths relative to a root directory, improving code readability.

## Example

Here's an example of how to use the configuration system in a script:

```python
from src.config.configuration import get_config
from src.utils.logging_manager import setup_logging

def main():
    config = get_config()
    setup_logging(config)

    logger.info(f"Output root: {config.output_root}")
    logger.info(f"Data file: {config.data_file}")

if __name__ == "__main__":
    main()