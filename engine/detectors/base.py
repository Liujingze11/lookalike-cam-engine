"""检测器抽象接口和公共数据类型。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedRegion:
    """检测到的图像区域。"""
    region_type: str           # "full_image" | "face" | "upper_body" | "head" | ...
    image: np.ndarray          # 裁剪后的区域图像 (H, W, C)
    bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h)，全图时为 None
    confidence: float = 1.0
    embedder_type: str = "global"  # 该区域应使用哪个 embedder


class BaseDetector(ABC):
    """检测器抽象基类。所有检测器必须实现 detect 方法。"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectedRegion]:
        """检测图像中的感兴趣区域。

        Args:
            image: BGR 图像 (H, W, C)，numpy array。

        Returns:
            检测到的区域列表。
        """
        ...
