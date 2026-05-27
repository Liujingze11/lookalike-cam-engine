import numpy as np
from engine.detectors.base import DetectedRegion
from engine.detectors.null_detector import NullDetector


def test_null_detector_returns_full_image():
    detector = NullDetector()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    regions = detector.detect(img)

    assert len(regions) == 1
    region = regions[0]
    assert region.region_type == "full_image"
    assert region.embedder_type == "global"
    assert region.bbox is None
    assert region.confidence == 1.0
    assert np.array_equal(region.image, img)


def test_detected_region_dataclass():
    region = DetectedRegion(
        region_type="face",
        image=np.zeros((112, 112, 3), dtype=np.uint8),
        bbox=(10, 20, 112, 112),
        embedder_type="face",
    )
    assert region.region_type == "face"
    assert region.bbox == (10, 20, 112, 112)
