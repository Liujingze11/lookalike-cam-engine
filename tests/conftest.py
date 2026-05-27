import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture
def project_root():
    return Path(__file__).parent.parent


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a simple test image."""
    import numpy as np
    from PIL import Image

    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return str(path)
