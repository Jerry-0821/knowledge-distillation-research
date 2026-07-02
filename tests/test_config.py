from pathlib import Path

import pytest

from kd_research.utils.config import ConfigError, load_yaml_config


def test_load_yaml_config_returns_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "example.yaml"
    config_path.write_text("experiment:\n  seed: 7\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert config == {"experiment": {"seed": 7}}


def test_load_yaml_config_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        load_yaml_config(missing_path)


def test_load_yaml_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- first\n- second\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_yaml_config(config_path)


def test_load_yaml_config_uses_safe_yaml_loader(tmp_path: Path) -> None:
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(
        "value: !!python/object/apply:os.system ['echo unsafe']\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_yaml_config(config_path)
