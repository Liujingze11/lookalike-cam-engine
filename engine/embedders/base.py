"""特征提取器抽象接口。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeatureVector:
    """特征向量及其元数据。"""
    embedding_type: str          # "global" | "face" | "outfit" | "color" | "head"
    vector: np.ndarray           # 特征向量 (D,)，float32
    metadata: dict = field(default_factory=dict)  # 额外信息，如 "dim": 512


class BaseEmbedder(ABC):
    """特征提取器抽象基类。所有 embedder 必须实现 embedding_type 和 embed。"""

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        """返回此 embedder 产出的 embedding 类型标识。"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回 embedding 向量维度。"""
        ...

    @abstractmethod
    def embed(self, image: np.ndarray) -> FeatureVector:
        """对输入图像提取特征向量。

        Args:
            image: RGB 图像 (H, W, C)，numpy array。

        Returns:
            特征向量。
        """
        ...
