# the same code from the sample project is used
import typing as t
from pathlib import Path

from pydantic import BaseModel
from yaml import safe_load

# Project Directories
PARENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = PARENT_DIR.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    training_data_file: str
    test_data_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    var_pre_rename: dict
    var_groups_post_rename: t.Sequence[str]
    vars_to_drop: t.Sequence[str]
    var_groups_to_drop: t.Sequence[str]
    vars_to_extract: t.Sequence[str]
    features: t.Sequence[str]
    cat_vars: t.Sequence[str]
    num_vars: t.Sequence[str]
    test_size: float
    random_state: int
    drop_tol: float
    xgboost_params: dict


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> t.Dict:
    """Parse YAML containing the package configuration."""
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = safe_load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: t.Optional[t.Dict] = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config),
        model_config=ModelConfig(**parsed_config),
    )

    return _config


config = create_and_validate_config()
