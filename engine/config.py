"""Pipeline configuration loader — YAML to typed dataclasses."""
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigLoadError(Exception):
    """Raised when a configuration file cannot be loaded or parsed."""


@dataclass
class DetectorConfig:
    class_path: str
    kwargs: dict = field(default_factory=dict)


@dataclass
class EmbedderConfig:
    name: str
    class_path: str
    model_name: str
    pretrained: str
    device: str
    source: str


@dataclass
class RetrieverConfig:
    name: str
    class_path: str
    index_path: str
    embedding_type: str
    dimension: int = 512


@dataclass
class RankerConfig:
    class_path: str


@dataclass
class PipelineConfig:
    detectors: list[DetectorConfig] = field(default_factory=list)
    embedders: list[EmbedderConfig] = field(default_factory=list)
    retrievers: list[RetrieverConfig] = field(default_factory=list)
    ranker: RankerConfig | None = None


@dataclass
class Preset:
    name: str
    weights: dict[str, float]


@dataclass
class SafetyConfig:
    min_final_score: float = 0.0
    blocked_character_tags: list[str] = field(default_factory=list)
    save_audience_images: bool = False
    save_audience_embeddings: bool = False
    embedding_ttl_minutes: int = 60
    require_human_approval: bool = False


def load_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> PipelineConfig:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigLoadError(f"Pipeline config not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML in pipeline config: {e}") from e

    if data is None:
        raise ConfigLoadError(f"Pipeline config is empty: {path}")

    raw = data.get("pipeline", {})

    try:
        detectors = [DetectorConfig(**d) for d in raw.get("detectors", [])]
        embedders = [EmbedderConfig(**e) for e in raw.get("embedders", [])]
        retrievers = [RetrieverConfig(**r) for r in raw.get("retrievers", [])]
        ranker = RankerConfig(**raw["ranker"]) if raw.get("ranker") else None
    except (TypeError, KeyError) as e:
        raise ConfigLoadError(f"Missing or invalid field in pipeline config: {e}") from e

    return PipelineConfig(
        detectors=detectors,
        embedders=embedders,
        retrievers=retrievers,
        ranker=ranker,
    )


def load_presets(path: str | Path = "configs/presets.yaml") -> dict[str, Preset]:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigLoadError(f"Presets config not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML in presets config: {e}") from e

    if data is None:
        raise ConfigLoadError(f"Presets config is empty: {path}")

    raw = data.get("presets", {})
    result = {}
    for name, item in raw.items():
        try:
            result[name] = Preset(name=name, weights=item["weights"])
        except (KeyError, TypeError) as e:
            raise ConfigLoadError(
                f"Missing 'weights' in preset '{name}': {e}"
            ) from e
    return result


def load_safety_config(path: str | Path = "configs/safety.yaml") -> SafetyConfig:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigLoadError(f"Safety config not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML in safety config: {e}") from e

    if data is None:
        raise ConfigLoadError(f"Safety config is empty: {path}")

    # Only pass known fields to SafetyConfig, ignoring extra keys
    known_fields = {f.name for f in dataclasses.fields(SafetyConfig)}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return SafetyConfig(**filtered)
