import numpy as np
from engine.retrievers.faiss_retriever import FaissRetriever


def test_build_and_search():
    dim = 64
    retriever = FaissRetriever(dimension=dim)

    # 创建 5 个随机归一化向量
    rng = np.random.RandomState(42)
    vecs = rng.randn(5, dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"char_{i}" for i in range(5)]

    retriever.build_index(vecs, ids)
    assert retriever.size == 5

    # 用第一个向量搜自己，自己应该是 Top-1
    results = retriever.search(vecs[0], top_k=3)
    assert len(results) == 3
    assert results[0].character_id == "char_0"
    assert results[0].similarity > 0.99  # 归一化后内积≈余弦


def test_save_and_load(tmp_path):
    dim = 64
    retriever = FaissRetriever(dimension=dim)
    rng = np.random.RandomState(42)
    vecs = rng.randn(5, dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"char_{i}" for i in range(5)]
    retriever.build_index(vecs, ids)

    path = tmp_path / "test.index"
    retriever.save(str(path))

    # 重新加载
    loaded = FaissRetriever()
    loaded.load(str(path))
    assert loaded.size == 5

    results = loaded.search(vecs[2], top_k=1)
    assert results[0].character_id == "char_2"


def test_empty_index_returns_empty():
    retriever = FaissRetriever(dimension=64)
    results = retriever.search(np.random.randn(64).astype(np.float32))
    assert results == []
