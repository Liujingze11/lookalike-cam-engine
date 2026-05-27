import base64
import json
import numpy as np
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def test_app(tmp_path, monkeypatch):
    """构造一个不加载真实模型的测试应用。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    monkeypatch.chdir(tmp_path)

    # 创建最小配置文件
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "pipeline.yaml").write_text("""
pipeline:
  detectors: []
  embedders: []
  retrievers: []
  ranker:
    class_path: engine.rankers.simple_ranker.SimpleRanker
""")
    (configs_dir / "presets.yaml").write_text("""
presets:
  default:
    weights:
      global: 1.0
""")
    (configs_dir / "safety.yaml").write_text("""
min_final_score: 0.0
blocked_character_tags: []
save_audience_images: false
save_audience_embeddings: false
embedding_ttl_minutes: 60
require_human_approval: false
""")

    # 创建角色数据
    chars_dir = tmp_path / "data" / "characters" / "mario"
    chars_dir.mkdir(parents=True)
    (chars_dir / "card.json").write_text(json.dumps({
        "id": "mario",
        "name": "Mario",
        "type": "game_character",
        "safety": {"allowed": True, "risk_level": "low"},
    }))

    from app.main import app
    # 替换 lifespan，避免启动时加载真实模型
    from app.api.match import set_pipeline
    from engine.pipeline import LookalikePipeline

    pipeline = LookalikePipeline(config_path=str(configs_dir / "pipeline.yaml"))
    set_pipeline(pipeline)

    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_match_image_invalid_data(client):
    resp = client.post("/api/match/image", json={
        "image_base64": "not-valid-base64!!!",
    })
    assert resp.status_code == 400


def _make_test_image_base64():
    """生成一张测试图片的 base64。"""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_match_image_returns_empty_when_no_index(client):
    """没有建索引时，返回空结果。"""
    resp = client.post("/api/match/image", json={
        "image_base64": _make_test_image_base64(),
        "preset": "default",
        "top_k": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["query_id"].startswith("q_")
