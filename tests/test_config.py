from pathlib import Path

from engine.config import (
    load_pipeline_config,
    load_presets,
    load_safety_config,
    PipelineConfig,
    EmbedderConfig,
    RetrieverConfig,
)


def test_load_pipeline_config():
    cfg = load_pipeline_config()
    assert isinstance(cfg, PipelineConfig)
    assert len(cfg.embedders) == 1
    assert cfg.embedders[0].name == "clip_global"
    assert cfg.embedders[0].model_name == "ViT-B-32"
    assert len(cfg.retrievers) == 1
    assert cfg.retrievers[0].name == "global_retriever"
    assert cfg.retrievers[0].dimension == 512
    assert cfg.ranker is not None


def test_load_presets():
    presets = load_presets()
    assert "default" in presets
    assert presets["default"].weights["global"] == 1.0


def test_load_safety_config():
    cfg = load_safety_config()
    assert "political" in cfg.blocked_character_tags
    assert cfg.save_audience_images is False
