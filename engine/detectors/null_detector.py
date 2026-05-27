"""M1 空检测器：不检测任何东西，把整张图作为一个区域返回。"""
import numpy as np

from engine.detectors.base import BaseDetector, DetectedRegion


class NullDetector(BaseDetector):
    """M1 默认检测器。M2 之后会被真实的 FaceDetector 等替换。"""

    def detect(self, image: np.ndarray) -> list[DetectedRegion]:
        return [
            DetectedRegion(
                region_type="full_image",
                image=image,
                bbox=None,
                confidence=1.0,
                embedder_type="global",
            )
        ]
