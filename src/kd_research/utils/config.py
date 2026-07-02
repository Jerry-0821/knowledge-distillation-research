"""Safe YAML configuration loading for experiment configuration files."""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file cannot be interpreted safely."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file whose top-level value is a mapping.

    This helper only parses configuration data. It does not assign defaults or
    treat an example configuration as a final experiment decision.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Config path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in config file: {config_path}") from error
    except OSError as error:
        raise ConfigError(f"Could not read config file: {config_path}") from error

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("Config top level must be a mapping.")

    return loaded
