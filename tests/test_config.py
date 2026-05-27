import pytest
import yaml
from pathlib import Path

from engine.config import (
    ConfigLoadError,
    DetectorConfig,
    load_pipeline_config,
    load_presets,
    load_safety_config,
    PipelineConfig,
    EmbedderConfig,
    RetrieverConfig,
    SafetyConfig,
    Preset,
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
    assert len(cfg.detectors) == 0


def test_load_presets():
    presets = load_presets()
    assert "default" in presets
    assert presets["default"].weights["global"] == 1.0


def test_load_safety_config():
    cfg = load_safety_config()
    assert "political" in cfg.blocked_character_tags
    assert cfg.save_audience_images is False


# Edge case tests using tmp_path

def test_pipeline_config_missing_file(tmp_path):
    path = tmp_path / "nonexistent.yaml"
    with pytest.raises(ConfigLoadError, match="not found"):
        load_pipeline_config(str(path))


def test_pipeline_config_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ConfigLoadError, match="empty"):
        load_pipeline_config(str(path))


def test_pipeline_config_missing_ranker(tmp_path):
    path = tmp_path / "no_ranker.yaml"
    path.write_text(yaml.dump({
        "pipeline": {
            "detectors": [],
            "embedders": [],
            "retrievers": [],
        }
    }))
    cfg = load_pipeline_config(str(path))
    assert cfg.ranker is None


def test_pipeline_config_with_detector(tmp_path):
    path = tmp_path / "with_detector.yaml"
    path.write_text(yaml.dump({
        "pipeline": {
            "detectors": [
                {"class_path": "engine.detectors.face_detector.FaceDetector", "kwargs": {"threshold": 0.5}}
            ],
            "embedders": [],
            "retrievers": [],
        }
    }))
    cfg = load_pipeline_config(str(path))
    assert len(cfg.detectors) == 1
    assert isinstance(cfg.detectors[0], DetectorConfig)
    assert cfg.detectors[0].class_path == "engine.detectors.face_detector.FaceDetector"
    assert cfg.detectors[0].kwargs == {"threshold": 0.5}


def test_pipeline_config_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("{bad: [unclosed")
    with pytest.raises(ConfigLoadError, match="Invalid YAML"):
        load_pipeline_config(str(path))


def test_pipeline_config_missing_required_field(tmp_path):
    path = tmp_path / "missing_field.yaml"
    path.write_text(yaml.dump({
        "pipeline": {
            "embedders": [
                {"name": "bad_embedder"}  # missing class_path, model_name, etc.
            ],
            "retrievers": [],
        }
    }))
    with pytest.raises(ConfigLoadError, match="Missing or invalid field"):
        load_pipeline_config(str(path))


def test_presets_missing_file(tmp_path):
    path = tmp_path / "no_presets.yaml"
    with pytest.raises(ConfigLoadError, match="not found"):
        load_presets(str(path))


def test_presets_missing_weights_field(tmp_path):
    path = tmp_path / "bad_presets.yaml"
    path.write_text(yaml.dump({
        "presets": {
            "bad_one": {}  # missing 'weights'
        }
    }))
    with pytest.raises(ConfigLoadError, match="Missing 'weights'"):
        load_presets(str(path))


def test_presets_from_tmp_path(tmp_path):
    path = tmp_path / "test_presets.yaml"
    path.write_text(yaml.dump({
        "presets": {
            "my_preset": {"weights": {"global": 0.8, "face": 0.2}}
        }
    }))
    presets = load_presets(str(path))
    assert "my_preset" in presets
    assert presets["my_preset"].weights["global"] == 0.8


def test_safety_config_missing_file(tmp_path):
    path = tmp_path / "no_safety.yaml"
    with pytest.raises(ConfigLoadError, match="not found"):
        load_safety_config(str(path))


def test_safety_config_ignores_unknown_keys(tmp_path):
    path = tmp_path / "extra_keys.yaml"
    path.write_text(yaml.dump({
        "min_final_score": 0.5,
        "save_audience_images": True,
        "extra_unknown_field": "should be ignored",
        "another_extra": 123,
    }))
    cfg = load_safety_config(str(path))
    assert cfg.min_final_score == 0.5
    assert cfg.save_audience_images is True
    # unknown keys are silently ignored, defaults used for missing fields
    assert cfg.save_audience_embeddings is False


def test_safety_config_empty_file(tmp_path):
    path = tmp_path / "empty_safety.yaml"
    path.write_text("")
    with pytest.raises(ConfigLoadError, match="empty"):
        load_safety_config(str(path))


def test_safety_config_defaults(tmp_path):
    path = tmp_path / "minimal_safety.yaml"
    path.write_text(yaml.dump({
        "min_final_score": 0.7,
        "blocked_character_tags": ["tag1"],
    }))
    cfg = load_safety_config(str(path))
    assert cfg.min_final_score == 0.7
    assert cfg.blocked_character_tags == ["tag1"]
    assert cfg.save_audience_images is False  # default
    assert cfg.embedding_ttl_minutes == 60    # default
