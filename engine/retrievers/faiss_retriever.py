"""基于 FAISS 的向量检索引擎。"""
import pickle
from pathlib import Path

import faiss
import numpy as np

from engine.retrievers.base import BaseRetriever, Candidate


class FaissRetriever(BaseRetriever):
    """FAISS IndexFlatIP 检索器，使用内积（等价于归一化后的余弦相似度）。"""

    def __init__(self, dimension: int = 512):
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._id_map: list[str] = []  # index_position -> character_id

    def search(self, vector: np.ndarray, top_k: int = 50) -> list[Candidate]:
        if self._index.ntotal == 0:
            return []

        query = vector.reshape(1, -1).astype(np.float32)
        # FAISS 要求对 IndexFlatIP 的查询向量也做归一化
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append(
                Candidate(
                    character_id=self._id_map[idx],
                    embedding_type="global",  # 后续版本从配置读取
                    similarity=float(score),
                    index=int(idx),
                )
            )
        return results

    def build_index(self, vectors: np.ndarray, ids: list[str]) -> None:
        """构建新索引。vectors 应当是已归一化的。"""
        vecs = vectors.astype(np.float32).copy()
        # 再次归一化确保
        faiss.normalize_L2(vecs)
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vecs)
        self._id_map = list(ids)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p))
        meta_path = str(p) + ".meta"
        with open(meta_path, "wb") as f:
            pickle.dump({"id_map": self._id_map, "dimension": self._dimension}, f)

    def load(self, path: str) -> None:
        self._index = faiss.read_index(str(path))
        meta_path = str(path) + ".meta"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._id_map = meta["id_map"]
        self._dimension = meta["dimension"]

    @property
    def size(self) -> int:
        return self._index.ntotal
