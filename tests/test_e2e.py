"""端到端测试：完整链路 — 建索引 → 加载 → 匹配。"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engine.embedders.clip_embedder import ClipEmbedder
from engine.retrievers.faiss_retriever import FaissRetriever


def _create_sample_characters(chars_dir: Path):
    """创建 4 个测试角色，每个角色有 1 张纯色测试图。"""
    characters = [
        {"id": "mario", "name": "Mario", "color": (255, 0, 0)},       # 红
        {"id": "luigi", "name": "Luigi", "color": (0, 255, 0)},       # 绿
        {"id": "pikachu", "name": "Pikachu", "color": (255, 255, 0)}, # 黄
        {"id": "sailor_moon", "name": "Sailor Moon", "color": (255, 200, 200)},  # 粉
    ]

    for char in characters:
        char_dir = chars_dir / char["id"]
        char_dir.mkdir(parents=True)

        card = {
            "id": char["id"],
            "name": char["name"],
            "type": "test",
            "safety": {"allowed": True, "risk_level": "low"},
        }
        (char_dir / "card.json").write_text(json.dumps(card))

        images_dir = char_dir / "images"
        images_dir.mkdir()
        img = Image.new("RGB", (224, 224), char["color"])
        img.save(images_dir / "ref.jpg")


@pytest.mark.slow
def test_e2e_build_index_and_match(tmp_path):
    """端到端测试：建索引 → 检索 → 用同色图查询应返回对应角色。

    此测试加载真实 OpenCLIP 模型（CPU）。"""
    # 1. 准备角色数据
    chars_dir = tmp_path / "characters"
    _create_sample_characters(chars_dir)

    # 2. 加载 embedder
    embedder = ClipEmbedder(device="cpu")
    assert embedder.dimension == 512

    # 3. 为每个角色建 embedding
    all_vectors = []
    all_ids = []
    for card_path in sorted(chars_dir.glob("*/card.json")):
        card = json.loads(card_path.read_text())
        char_dir = card_path.parent
        vecs = []
        for img_path in sorted((char_dir / "images").iterdir()):
            if img_path.name.startswith("."):
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            fv = embedder.embed(img)
            vecs.append(fv.vector)
        mean_vec = np.mean(vecs, axis=0)
        mean_vec = mean_vec / np.linalg.norm(mean_vec)
        all_vectors.append(mean_vec)
        all_ids.append(card["id"])

    vec_matrix = np.stack(all_vectors, axis=0)

    # 4. 建 FAISS 索引
    retriever = FaissRetriever(dimension=512)
    retriever.build_index(vec_matrix, all_ids)
    assert retriever.size == 4

    # 5. 用红色图查询，应返回 mario
    red_img = np.full((224, 224, 3), (255, 0, 0), dtype=np.uint8)
    query_vec = embedder.embed(red_img).vector
    results = retriever.search(query_vec, top_k=3)

    assert len(results) >= 1
    # 红色图最像红色角色 mario
    assert results[0].character_id == "mario"
    assert results[0].similarity > 0.5
