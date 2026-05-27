"""检索器抽象接口。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Candidate:
    """检索召回的候选角色。"""
    character_id: str
    embedding_type: str     # 从哪个索引召回: "global" | "face" | ...
    similarity: float        # 余弦相似度 [0, 1]
    index: int               # 在索引中的位置


class BaseRetriever(ABC):
    """检索器抽象基类。"""

    @abstractmethod
    def search(
        self, vector: "np.ndarray", top_k: int = 50
    ) -> list[Candidate]:
        """在索引中搜索与查询向量最相似的候选项。

        Args:
            vector: 查询向量 (D,)，float32。
            top_k: 返回前 K 个结果。

        Returns:
            候选角色列表，按相似度降序排列。
        """
        ...

    @abstractmethod
    def build_index(
        self, vectors: "np.ndarray", ids: list[str]
    ) -> None:
        """构建/重建索引。

        Args:
            vectors: (N, D) 矩阵，每行一个向量。
            ids: 对应的角色 ID 列表，长度 N。
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """将索引持久化到磁盘。"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """从磁盘加载索引。"""
        ...
