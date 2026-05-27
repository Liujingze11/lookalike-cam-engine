"""应用级全局配置。"""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AppConfig:
    project_root: Path = Path(__file__).parent.parent
    pipeline_config_path: str = "configs/pipeline.yaml"
    debug: bool = False


_app_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
    return _app_config


def set_config(config: AppConfig) -> None:
    global _app_config
    _app_config = config
