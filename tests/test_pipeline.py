import json
from pathlib import Path

from engine.config import (
    PipelineConfig,
    EmbedderConfig,
    RetrieverConfig,
    RankerConfig,
)
from engine.pipeline import LookalikePipeline


def test_pipeline_initializes_with_config():
    """验证管线能从配置初始化（无真实模型加载时不测试 embed）。"""
    config = PipelineConfig(
        detectors=[],
        embedders=[],  # 空 embedder 列表跳过模型加载
        retrievers=[],
        ranker=RankerConfig(
            class_path="engine.rankers.simple_ranker.SimpleRanker"
        ),
    )
    pipeline = LookalikePipeline(config=config)
    assert pipeline._ranker is not None


def test_pipeline_loads_character_names(tmp_path, monkeypatch):
    """验证管线正确读取角色名和安全标记。"""
    chars_dir = tmp_path / "characters"
    chars_dir.mkdir()
    mario_dir = chars_dir / "mario"
    mario_dir.mkdir()
    (mario_dir / "card.json").write_text(json.dumps({
        "id": "mario",
        "name": "Mario",
        "type": "game_character",
        "safety": {"allowed": True, "risk_level": "low"},
    }))

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "characters").symlink_to(chars_dir)

    pipeline = LookalikePipeline(config=PipelineConfig(
        detectors=[], embedders=[], retrievers=[],
        ranker=RankerConfig(class_path="engine.rankers.simple_ranker.SimpleRanker"),
    ))

    names = pipeline._load_character_names()
    assert names.get("mario") == "Mario"

    safety = pipeline._load_character_safety()
    assert safety.get("mario") is True
