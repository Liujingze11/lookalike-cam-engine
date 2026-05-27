"""加载管线配置、preset 权重、安全规则。"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml


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
    detectors: list[dict] = field(default_factory=list)
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
    with open(path) as f:
        raw = yaml.safe_load(f)["pipeline"]

    embedders = [
        EmbedderConfig(**e) for e in raw.get("embedders", [])
    ]
    retrievers = [
        RetrieverConfig(**r) for r in raw.get("retrievers", [])
    ]
    ranker = RankerConfig(**raw["ranker"]) if raw.get("ranker") else None

    return PipelineConfig(
        detectors=raw.get("detectors", []),
        embedders=embedders,
        retrievers=retrievers,
        ranker=ranker,
    )


def load_presets(path: str | Path = "configs/presets.yaml") -> dict[str, Preset]:
    with open(path) as f:
        raw = yaml.safe_load(f)["presets"]
    return {
        name: Preset(name=name, weights=data["weights"])
        for name, data in raw.items()
    }


def load_safety_config(path: str | Path = "configs/safety.yaml") -> SafetyConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SafetyConfig(**raw)
