"""Default configuration values."""

from pathlib import Path

import yaml


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Try project root first, then environment variable
        config_paths = [
            Path.cwd() / "config.yml",
            # Path(os.environ.get('BERT_CONFIG', 'config/config.yml'))
        ]
        config_path = next((p for p in config_paths if p.exists()), None)
        if not config_path:
            raise FileNotFoundError("No configuration file found")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert paths to Path objects
    config["output_root"] = Path(config["output_root"])

    # Convert directory paths
    if "dirs" in config:
        config["dirs"] = {k: str(v) for k, v in config["dirs"].items()}

    # Convert model paths
    if "model_paths" in config:
        config["model_paths"] = {k: str(v) for k, v in config["model_paths"].items()}

    # Convert string tuple to actual tuple for betas
    if "optimizer" in config:
        if "betas" in config["optimizer"]:
            beta_str = config["optimizer"]["betas"]
            if isinstance(beta_str, str):
                # Remove parentheses and split
                beta_str = beta_str.strip("()").split(",")
                config["optimizer"]["betas"] = tuple(float(x.strip()) for x in beta_str)

    return config


# Load configuration
CONFIG = load_config()

# Export configuration sections
DIR_DEFAULTS = {"output_root": CONFIG["output_root"], "dirs": CONFIG["dirs"]}

DATA_DEFAULTS = CONFIG.get("data", {"default_file": "bbc-text.csv"})

MODEL_PATHS = CONFIG["model_paths"]

# Update MODEL_DEFAULTS to only include model parameters (no learning rate)
MODEL_DEFAULTS = {
    "max_seq_len": CONFIG["model"]["max_seq_len"],
    "batch_size": CONFIG["model"]["batch_size"],
    "num_epochs": CONFIG["model"]["num_epochs"],
    "device": CONFIG["model"]["device"],
    "metric": CONFIG["model"]["metric"],
    "metrics": CONFIG["model"]["metrics"],
}

# Update CLASSIFIER_DEFAULTS to match new config.yml structure
CLASSIFIER_DEFAULTS = {
    "bert_hidden_size": CONFIG["classifier"]["bert_hidden_size"],
    "hidden_dims": CONFIG["classifier"]["hidden_dims"],
    "dropout_rate": CONFIG["classifier"]["dropout_rate"],  # Changed from hidden_dropout
    "activation": CONFIG["classifier"]["activation"],
    "num_classes": CONFIG["classifier"]["num_classes"],
}

# Add OPTIMIZER_DEFAULTS for all optimizer settings
OPTIMIZER_DEFAULTS = {
    "optimizer_choice": CONFIG["optimizer"]["optimizer_choice"],
    "lr": CONFIG["optimizer"]["lr"],
    "weight_decay": CONFIG["optimizer"]["weight_decay"],
    "warmup_ratio": CONFIG["optimizer"]["warmup_ratio"],
    "momentum": CONFIG["optimizer"]["momentum"],
    "alpha": CONFIG["optimizer"]["alpha"],
    "nesterov": CONFIG["optimizer"]["nesterov"],
    "betas": CONFIG["optimizer"]["betas"],  # Use betas tuple directly
    "eps": CONFIG["optimizer"]["eps"],
}

# Update OPTIM_SEARCH_SPACE to use dynamic hidden dimensions
OPTIM_SEARCH_SPACE = {
    "batch_size": [16, 32, 64],
    "num_hidden_layers": (1, 4),  # Now controlling number of layers directly
    "dropout_rate": (0.1, 0.5),
    "learning_rate": (1e-5, 1e-3),  # log scale
    "weight_decay": (1e-8, 1e-3),  # log scale
    "warmup_ratio": (0.0, 0.2),
}
