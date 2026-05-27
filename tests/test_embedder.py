import numpy as np
import pytest

from engine.embedders.base import FeatureVector


def test_feature_vector_dataclass():
    vec = FeatureVector(
        embedding_type="global",
        vector=np.zeros(512, dtype=np.float32),
        metadata={"dim": 512},
    )
    assert vec.embedding_type == "global"
    assert vec.vector.shape == (512,)
    assert vec.metadata["dim"] == 512


@pytest.mark.slow
def test_clip_embedder_loads_and_embeds(sample_image_path):
    """需要 GPU 或 CPU 且有 open_clip 模型。标记为 slow 方便 CI 跳过。"""
    from PIL import Image

    from engine.embedders.clip_embedder import ClipEmbedder

    embedder = ClipEmbedder(device="cpu")
    assert embedder.embedding_type == "global"
    assert embedder.dimension == 512

    img = np.array(Image.open(sample_image_path).convert("RGB"))
    result = embedder.embed(img)
    assert result.embedding_type == "global"
    assert result.vector.shape == (512,)
    assert result.vector.dtype == np.float32
    # 归一化后向量模长应接近 1
    assert abs(np.linalg.norm(result.vector) - 1.0) < 0.01
