# Look-Alike Cam Engine M1 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建插件化流水线骨架，实现单图上传→Top-K 角色匹配的完整链路。

**Architecture:** 四层插件化流水线（检测→特征提取→检索→排序），每层通过抽象基类定义接口。M1 只实现 NullDetector + ClipEmbedder + FaissRetriever + SimpleRanker，但目录结构和接口为后续 M2-M7 升级预留完整空间。

**Tech Stack:** Python 3.10, FastAPI, OpenCLIP (ViT-B/32), FAISS, PyTorch, PyYAML

---

## 文件映射

| 文件 | 职责 |
|------|------|
| `requirements.txt` | 依赖声明 |
| `configs/pipeline.yaml` | 管线插件配置 |
| `configs/presets.yaml` | 预设权重 |
| `configs/safety.yaml` | 安全规则 |
| `engine/detectors/base.py` | 检测器抽象接口 + 公共类型 |
| `engine/detectors/null_detector.py` | M1: 不检测，全图作为输入 |
| `engine/embedders/base.py` | 特征提取器抽象接口 |
| `engine/embedders/clip_embedder.py` | OpenCLIP 全图语义特征 |
| `engine/retrievers/base.py` | 检索器抽象接口 |
| `engine/retrievers/faiss_retriever.py` | FAISS 向量检索 |
| `engine/rankers/base.py` | 排序器抽象接口 + 公共类型 |
| `engine/rankers/simple_ranker.py` | 单维度分数排序 |
| `engine/pipeline.py` | 管线调度器，编排四层 |
| `engine/config.py` | 管线配置加载 |
| `app/schemas/requests.py` | API 请求 Pydantic 模型 |
| `app/schemas/responses.py` | API 响应 Pydantic 模型 |
| `app/api/match.py` | POST /api/match/image 路由 |
| `app/config.py` | 应用配置（路径、设备等） |
| `app/main.py` | FastAPI 应用入口 |
| `scripts/build_index.py` | 构建 FAISS 索引 |
| `scripts/import_characters.py` | 导入角色到角色库 |
| `data/characters/{id}/card.json` | 角色元数据 |
| `tests/test_pipeline.py` | 管线集成测试 |
| `tests/test_embedder.py` | 特征提取器单元测试 |
| `tests/test_retriever.py` | 检索器单元测试 |
| `tests/test_api.py` | API 接口测试 |

---

### Task 1: 项目初始化

**Files:**
- Create: `requirements.txt`
- Create: `engine/__init__.py`, `engine/detectors/__init__.py`, `engine/embedders/__init__.py`, `engine/retrievers/__init__.py`, `engine/rankers/__init__.py`
- Create: `app/__init__.py`, `app/api/__init__.py`, `app/schemas/__init__.py`
- Create: `configs/` (目录), `data/characters/` (目录), `indexes/` (目录), `scripts/` (目录)
- Create: `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Write requirements.txt**

```txt
torch>=2.0.0
open-clip-torch>=2.24.0
faiss-cpu>=1.7.4
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
pyyaml>=6.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.8.0
pytest>=8.0.0
httpx>=0.27.0
```

- [ ] **Step 2: 创建目录结构和空 __init__ 文件**

Run:
```bash
cd /home/ljz/vibe_coding/lookalike-cam-engine
mkdir -p engine/detectors engine/embedders engine/retrievers engine/rankers
mkdir -p app/api app/schemas
mkdir -p configs data/characters indexes scripts tests
touch engine/__init__.py engine/detectors/__init__.py engine/embedders/__init__.py
touch engine/retrievers/__init__.py engine/rankers/__init__.py
touch app/__init__.py app/api/__init__.py app/schemas/__init__.py
touch tests/__init__.py
```

- [ ] **Step 3: 创建 tests/conftest.py**

```python
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
```

- [ ] **Step 4: 安装依赖**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && pip3 install -r requirements.txt`

- [ ] **Step 5: Init git and set remote**

```bash
cd /home/ljz/vibe_coding/lookalike-cam-engine
git init
git remote add origin https://github.com/Liujingze11/lookalike-cam-engine.git
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: initialize project structure and dependencies"
```

---

### Task 2: 配置系统

**Files:**
- Create: `configs/pipeline.yaml`
- Create: `configs/presets.yaml`
- Create: `configs/safety.yaml`
- Create: `engine/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write configs/pipeline.yaml**

```yaml
# Pipeline 插件配置
# 后续版本在此文件追加新的 detector/embedder/retriever 即可

pipeline:
  detectors: []  # M1 无需检测器; M2 追加 face_detector

  embedders:
    - name: clip_global
      class_path: engine.embedders.clip_embedder.ClipEmbedder
      model_name: ViT-B-32
      pretrained: laion2b_s34b_b79k
      device: cuda
      source: full_image

  retrievers:
    - name: global_retriever
      class_path: engine.retrievers.faiss_retriever.FaissRetriever
      index_path: indexes/global.index
      embedding_type: global
      dimension: 512

  ranker:
    class_path: engine.rankers.simple_ranker.SimpleRanker
```

- [ ] **Step 2: Write configs/presets.yaml**

```yaml
presets:
  default:
    weights:
      global: 1.0
  # 后续版本追加 celebrity_face, anime_vibe, outfit_match 等 preset
```

- [ ] **Step 3: Write configs/safety.yaml**

```yaml
# 安全配置
min_final_score: 0.0          # M1 不设阈值，M3+ 启用
save_audience_images: false
save_audience_embeddings: false
embedding_ttl_minutes: 60
require_human_approval: false

blocked_character_tags:
  - political
  - criminal
  - disease
  - body_shaming
  - racial_stereotype
```

- [ ] **Step 4: Write engine/config.py**

```python
"""加载管线配置、preset 权重、安全规则。"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EmbedderConfig:
    name: str
    class_path: str
    model_name: str
    pretrained: str
    device: str
    source: str


@dataclass
class RetrieverConfig:
    name: str
    class_path: str
    index_path: str
    embedding_type: str
    dimension: int = 512


@dataclass
class RankerConfig:
    class_path: str


@dataclass
class PipelineConfig:
    detectors: list[dict] = field(default_factory=list)
    embedders: list[EmbedderConfig] = field(default_factory=list)
    retrievers: list[RetrieverConfig] = field(default_factory=list)
    ranker: RankerConfig | None = None


@dataclass
class Preset:
    name: str
    weights: dict[str, float]


@dataclass
class SafetyConfig:
    min_final_score: float = 0.0
    blocked_character_tags: list[str] = field(default_factory=list)
    save_audience_images: bool = False
    save_audience_embeddings: bool = False
    embedding_ttl_minutes: int = 60
    require_human_approval: bool = False


def load_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)["pipeline"]

    embedders = [
        EmbedderConfig(**e) for e in raw.get("embedders", [])
    ]
    retrievers = [
        RetrieverConfig(**r) for r in raw.get("retrievers", [])
    ]
    ranker = RankerConfig(**raw["ranker"]) if raw.get("ranker") else None

    return PipelineConfig(
        detectors=raw.get("detectors", []),
        embedders=embedders,
        retrievers=retrievers,
        ranker=ranker,
    )


def load_presets(path: str | Path = "configs/presets.yaml") -> dict[str, Preset]:
    with open(path) as f:
        raw = yaml.safe_load(f)["presets"]
    return {
        name: Preset(name=name, weights=data["weights"])
        for name, data in raw.items()
    }


def load_safety_config(path: str | Path = "configs/safety.yaml") -> SafetyConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SafetyConfig(**raw)
```

- [ ] **Step 5: Write tests/test_config.py**

```python
from pathlib import Path

from engine.config import (
    load_pipeline_config,
    load_presets,
    load_safety_config,
    PipelineConfig,
    EmbedderConfig,
    RetrieverConfig,
)


def test_load_pipeline_config():
    cfg = load_pipeline_config()
    assert isinstance(cfg, PipelineConfig)
    assert len(cfg.embedders) == 1
    assert cfg.embedders[0].name == "clip_global"
    assert cfg.embedders[0].model_name == "ViT-B-32"
    assert len(cfg.retrievers) == 1
    assert cfg.retrievers[0].name == "global_retriever"
    assert cfg.retrievers[0].dimension == 512
    assert cfg.ranker is not None


def test_load_presets():
    presets = load_presets()
    assert "default" in presets
    assert presets["default"].weights["global"] == 1.0


def test_load_safety_config():
    cfg = load_safety_config()
    assert "political" in cfg.blocked_character_tags
    assert cfg.save_audience_images is False
```

- [ ] **Step 6: Run tests**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/test_config.py -v`
Expected: 3 PASS

- [ ] **Step 7: Commit**

```bash
git add configs/ engine/config.py tests/test_config.py
git commit -m "feat: add configuration system (pipeline, presets, safety)"
```

---

### Task 3: 检测层 — 抽象基类 + NullDetector

**Files:**
- Create: `engine/detectors/base.py`
- Create: `engine/detectors/null_detector.py`
- Create: `tests/test_detectors.py`

- [ ] **Step 1: Write engine/detectors/base.py**

```python
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
```

- [ ] **Step 2: Write engine/detectors/null_detector.py**

```python
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
```

- [ ] **Step 3: Write tests/test_detectors.py**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/test_detectors.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add engine/detectors/ tests/test_detectors.py
git commit -m "feat: add detector base class and NullDetector for M1"
```

---

### Task 4: 特征提取层 — 抽象基类 + ClipEmbedder

**Files:**
- Create: `engine/embedders/base.py`
- Create: `engine/embedders/clip_embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write engine/embedders/base.py**

```python
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
```

- [ ] **Step 2: Write engine/embedders/clip_embedder.py**

```python
"""OpenCLIP 全图语义特征提取器。"""
import numpy as np
import torch
from PIL import Image

from engine.embedders.base import BaseEmbedder, FeatureVector


class ClipEmbedder(BaseEmbedder):
    """使用 OpenCLIP 提取图像整体语义特征。"""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
    ):
        import open_clip

        self._device = device if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def embedding_type(self) -> str:
        return "global"

    @property
    def dimension(self) -> int:
        return 512

    @torch.no_grad()
    def embed(self, image: np.ndarray) -> FeatureVector:
        """提取图像特征。image 为 RGB numpy array (H, W, C)。"""
        pil_img = Image.fromarray(image)
        img_tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)

        features = self._model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        vec = features.cpu().numpy().flatten().astype(np.float32)

        return FeatureVector(
            embedding_type=self.embedding_type,
            vector=vec,
            metadata={"dim": self.dimension, "model": "ViT-B-32"},
        )
```

- [ ] **Step 3: Write tests/test_embedder.py**

```python
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
```

- [ ] **Step 4: Run fast tests**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/test_embedder.py::test_feature_vector_dataclass -v`
Expected: 1 PASS

- [ ] **Step 5: Commit**

```bash
git add engine/embedders/ tests/test_embedder.py
git commit -m "feat: add embedder base class and ClipEmbedder for M1"
```

---

### Task 5: 检索层 — 抽象基类 + FaissRetriever

**Files:**
- Create: `engine/retrievers/base.py`
- Create: `engine/retrievers/faiss_retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write engine/retrievers/base.py**

```python
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
```

- [ ] **Step 2: Write engine/retrievers/faiss_retriever.py**

```python
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
```

- [ ] **Step 3: Write tests/test_retriever.py**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/test_retriever.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add engine/retrievers/ tests/test_retriever.py
git commit -m "feat: add retriever base class and FaissRetriever for M1"
```

---

### Task 6: 排序层 — 抽象基类 + SimpleRanker

**Files:**
- Create: `engine/rankers/base.py`
- Create: `engine/rankers/simple_ranker.py`
- Create: `tests/test_ranker.py`

- [ ] **Step 1: Write engine/rankers/base.py**

```python
"""排序器抽象接口。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from engine.retrievers.base import Candidate
from engine.config import Preset, SafetyConfig


@dataclass
class MatchResult:
    """最终匹配结果。"""
    rank: int
    character_id: str
    name: str
    score: float
    feature_scores: dict[str, float]
    match_reasons: list[str] = field(default_factory=list)
    safe: bool = True


class BaseRanker(ABC):
    """排序器抽象基类。"""

    @abstractmethod
    def rank(
        self,
        candidates: list[Candidate],
        character_names: dict[str, str],
        character_safety: dict[str, bool],
        preset: Preset,
        safety: SafetyConfig,
        top_k: int = 5,
    ) -> list[MatchResult]:
        """对候选角色进行融合排序、安全过滤，返回最终 Top-K。

        Args:
            candidates: 各检索器召回的候选列表。
            character_names: character_id -> display_name 映射。
            character_safety: character_id -> is_safe 映射。
            preset: 当前预设权重配置。
            safety: 安全配置。
            top_k: 返回数量。

        Returns:
            排序后的匹配结果。
        """
        ...
```

- [ ] **Step 2: Write engine/rankers/simple_ranker.py**

```python
"""M1 简单排序器：按单维度相似度降序排列。"""
from engine.rankers.base import BaseRanker, MatchResult
from engine.retrievers.base import Candidate
from engine.config import Preset, SafetyConfig


class SimpleRanker(BaseRanker):
    """M1 排序器。单维度分数即为总分，无融合逻辑。"""

    def rank(
        self,
        candidates: list[Candidate],
        character_names: dict[str, str],
        character_safety: dict[str, bool],
        preset: Preset,
        safety: SafetyConfig,
        top_k: int = 5,
    ) -> list[MatchResult]:
        # 安全过滤
        blocked_tags = set(safety.blocked_character_tags)
        filtered = [
            c
            for c in candidates
            if character_safety.get(c.character_id, True)
        ]

        # 去重（同一角色可能从多个索引召回，保留最高分）
        seen: dict[str, float] = {}
        for c in filtered:
            cid = c.character_id
            if cid not in seen or c.similarity > seen[cid]:
                seen[cid] = c.similarity

        # 按分数降序
        sorted_ids = sorted(seen.items(), key=lambda x: x[1], reverse=True)

        # 构建结果
        results = []
        for rank, (cid, score) in enumerate(sorted_ids[:top_k], start=1):
            if score < safety.min_final_score:
                continue
            results.append(
                MatchResult(
                    rank=rank,
                    character_id=cid,
                    name=character_names.get(cid, cid),
                    score=round(float(score), 4),
                    feature_scores={"global": round(float(score), 4)},
                    match_reasons=[],
                    safe=character_safety.get(cid, True),
                )
            )

        return results
```

- [ ] **Step 3: Write tests/test_ranker.py**

```python
from engine.retrievers.base import Candidate
from engine.rankers.simple_ranker import SimpleRanker
from engine.config import Preset, SafetyConfig


def make_candidate(char_id: str, sim: float) -> Candidate:
    return Candidate(
        character_id=char_id,
        embedding_type="global",
        similarity=sim,
        index=0,
    )


def test_simple_ranker_sorts_by_similarity():
    ranker = SimpleRanker()
    candidates = [
        make_candidate("mario", 0.65),
        make_candidate("luigi", 0.92),
        make_candidate("peach", 0.78),
    ]
    names = {"mario": "Mario", "luigi": "Luigi", "peach": "Peach"}
    safety_map = {"mario": True, "luigi": True, "peach": True}

    results = ranker.rank(
        candidates, names, safety_map,
        preset=Preset(name="default", weights={"global": 1.0}),
        safety=SafetyConfig(),
        top_k=5,
    )

    assert len(results) == 3
    assert results[0].character_id == "luigi"
    assert results[0].score == 0.92
    assert results[1].character_id == "peach"
    assert results[2].character_id == "mario"


def test_simple_ranker_deduplicates():
    ranker = SimpleRanker()
    candidates = [
        make_candidate("mario", 0.9),
        make_candidate("mario", 0.7),  # 重复，低分
        make_candidate("mario", 0.8),  # 重复，中分
    ]
    names = {"mario": "Mario"}
    safety_map = {"mario": True}

    results = ranker.rank(
        candidates, names, safety_map,
        preset=Preset(name="default", weights={"global": 1.0}),
        safety=SafetyConfig(),
    )

    assert len(results) == 1
    assert results[0].score == 0.9  # 保留最高分


def test_simple_ranker_filters_unsafe():
    ranker = SimpleRanker()
    candidates = [
        make_candidate("safe", 0.9),
        make_candidate("unsafe", 0.95),
    ]
    names = {"safe": "Safe", "unsafe": "Unsafe"}
    safety_map = {"safe": True, "unsafe": False}

    results = ranker.rank(
        candidates, names, safety_map,
        preset=Preset(name="default", weights={"global": 1.0}),
        safety=SafetyConfig(),
    )

    assert len(results) == 1
    assert results[0].character_id == "safe"


def test_simple_ranker_respects_min_score():
    ranker = SimpleRanker()
    candidates = [
        make_candidate("a", 0.3),
        make_candidate("b", 0.8),
    ]
    names = {"a": "A", "b": "B"}
    safety_map = {"a": True, "b": True}

    results = ranker.rank(
        candidates, names, safety_map,
        preset=Preset(name="default", weights={"global": 1.0}),
        safety=SafetyConfig(min_final_score=0.5),
    )

    assert len(results) == 1
    assert results[0].character_id == "b"
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/test_ranker.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add engine/rankers/ tests/test_ranker.py
git commit -m "feat: add ranker base class and SimpleRanker for M1"
```

---

### Task 7: Pipeline 管线调度器

**Files:**
- Create: `engine/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write engine/pipeline.py**

```python
"""管线调度器：串联检测→特征提取→检索→排序四层。"""
import importlib
from pathlib import Path

import numpy as np

from engine.config import (
    PipelineConfig,
    Preset,
    SafetyConfig,
    load_pipeline_config,
    load_presets,
    load_safety_config,
)
from engine.detectors.base import BaseDetector
from engine.embedders.base import BaseEmbedder
from engine.retrievers.base import BaseRetriever
from engine.rankers.base import BaseRanker, MatchResult


def _import_class(class_path: str):
    """动态导入类，如 'engine.embedders.clip_embedder.ClipEmbedder'。"""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LookalikePipeline:
    """插件化相似度匹配管线。不依赖具体模型实现，全部通过配置驱动。"""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        config_path: str = "configs/pipeline.yaml",
    ):
        self._config = config or load_pipeline_config(config_path)
        self._detectors: list[BaseDetector] = []
        self._embedders: dict[str, BaseEmbedder] = {}   # embedding_type -> embedder
        self._retrievers: dict[str, BaseRetriever] = {}  # embedding_type -> retriever
        self._ranker: BaseRanker | None = None

        self._setup()

    def _setup(self):
        """按配置初始化各层插件。"""
        # 检测器
        for det_cfg in self._config.detectors:
            cls = _import_class(det_cfg["class_path"])
            self._detectors.append(cls(**det_cfg.get("kwargs", {})))

        # 特征提取器
        for emb_cfg in self._config.embedders:
            cls = _import_class(emb_cfg.class_path)
            instance = cls(
                model_name=emb_cfg.model_name,
                pretrained=emb_cfg.pretrained,
                device=emb_cfg.device,
            )
            self._embedders[instance.embedding_type] = instance

        # 检索器
        for ret_cfg in self._config.retrievers:
            cls = _import_class(ret_cfg.class_path)
            instance = cls(dimension=ret_cfg.dimension)
            index_path = Path(ret_cfg.index_path)
            if index_path.exists():
                instance.load(str(index_path))
            self._retrievers[ret_cfg.embedding_type] = instance

        # 排序器
        if self._config.ranker:
            cls = _import_class(self._config.ranker.class_path)
            self._ranker = cls()

    def run(
        self,
        image: np.ndarray,
        preset_name: str = "default",
        top_k: int = 5,
    ) -> list[MatchResult]:
        """执行完整匹配管线。

        Args:
            image: RGB 图像 (H, W, C)，numpy array。
            preset_name: 预设名称。
            top_k: 返回结果数量。

        Returns:
            排序后的 Top-K 匹配结果。
        """
        presets = load_presets()
        safety = load_safety_config()
        preset = presets.get(preset_name, presets["default"])

        # 第 1 层：检测
        if self._detectors:
            regions = []
            for detector in self._detectors:
                regions.extend(detector.detect(image))
        else:
            # 无检测器时回退到 NullDetector 逻辑
            from engine.detectors.null_detector import NullDetector
            regions = NullDetector().detect(image)

        # 第 2 层：特征提取
        vectors: dict[str, "np.ndarray"] = {}
        for region in regions:
            embedder = self._embedders.get(region.embedder_type)
            if embedder is None:
                continue
            fv = embedder.embed(region.image)
            vectors[fv.embedding_type] = fv.vector

        # 第 3 层：多路检索
        all_candidates = []
        for etype, vec in vectors.items():
            retriever = self._retrievers.get(etype)
            if retriever is None:
                continue
            all_candidates.extend(retriever.search(vec, top_k=max(top_k * 10, 50)))

        # 第 4 层：排序
        if self._ranker is None:
            return []

        # 从角色库加载名字和安全标记
        character_names = self._load_character_names()
        character_safety = self._load_character_safety()

        return self._ranker.rank(
            all_candidates, character_names, character_safety,
            preset, safety, top_k,
        )

    def _load_character_names(self) -> dict[str, str]:
        """扫描 data/characters/ 获取角色名映射。"""
        import json

        chars_dir = Path("data/characters")
        if not chars_dir.exists():
            return {}

        names = {}
        for card_path in chars_dir.glob("*/card.json"):
            try:
                card = json.loads(card_path.read_text())
                names[card["id"]] = card.get("name", card["id"])
            except (json.JSONDecodeError, KeyError):
                continue
        return names

    def _load_character_safety(self) -> dict[str, bool]:
        """扫描角色卡获取安全标记。"""
        import json

        chars_dir = Path("data/characters")
        if not chars_dir.exists():
            return {}

        safety = {}
        for card_path in chars_dir.glob("*/card.json"):
            try:
                card = json.loads(card_path.read_text())
                safety[card["id"]] = card.get("safety", {}).get("allowed", True)
            except (json.JSONDecodeError, KeyError):
                continue
        return safety

    @property
    def retrievers(self) -> dict[str, BaseRetriever]:
        return self._retrievers
```

- [ ] **Step 2: Write tests/test_pipeline.py**

```python
import json
import numpy as np
from pathlib import Path

from engine.config import (
    PipelineConfig,
    EmbedderConfig,
    RetrieverConfig,
    RankerConfig,
)
from engine.pipeline import LookalikePipeline


def _make_mock_embedder_config():
    return EmbedderConfig(
        name="clip_global",
        class_path="engine.embedders.clip_embedder.ClipEmbedder",
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
        source="full_image",
    )


def _make_mock_retriever_config(tmp_path: Path):
    return RetrieverConfig(
        name="global_retriever",
        class_path="engine.retrievers.faiss_retriever.FaissRetriever",
        index_path=str(tmp_path / "test.index"),
        embedding_type="global",
        dimension=64,
    )


def test_pipeline_initializes_with_config(tmp_path):
    """验证管线能从配置初始化（无真实模型加载时不测试 embed）。"""
    config = PipelineConfig(
        detectors=[],
        embedders=[],  # 空 embedder 列表跳过模型加载
        retrievers=[],
        ranker=RankerConfig(
            class_path="engine.rankers.simple_ranker.SimpleRanker"
        ),
    )
    pipeline = LookalikePipeline(config=config)
    assert pipeline._ranker is not None


def test_pipeline_loads_character_names(tmp_path, monkeypatch):
    """验证管线正确读取角色名。"""
    chars_dir = tmp_path / "characters"
    chars_dir.mkdir()
    mario_dir = chars_dir / "mario"
    mario_dir.mkdir()
    (mario_dir / "card.json").write_text(json.dumps({
        "id": "mario",
        "name": "Mario",
        "type": "game_character",
        "safety": {"allowed": True, "risk_level": "low"},
    }))

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "characters").symlink_to(chars_dir)

    pipeline = LookalikePipeline(config=PipelineConfig(
        detectors=[], embedders=[], retrievers=[],
        ranker=RankerConfig(class_path="engine.rankers.simple_ranker.SimpleRanker"),
    ))

    names = pipeline._load_character_names()
    assert names.get("mario") == "Mario"

    safety = pipeline._load_character_safety()
    assert safety.get("mario") is True
```

- [ ] **Step 3: Run tests**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/test_pipeline.py -v`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add engine/pipeline.py tests/test_pipeline.py
git commit -m "feat: add LookalikePipeline orchestrator"
```

---

### Task 8: API 数据模型

**Files:**
- Create: `app/schemas/requests.py`
- Create: `app/schemas/responses.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write app/schemas/requests.py**

```python
"""API 请求模型。"""
from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 编码的图片数据")
    preset: str = Field(default="default", description="预设名称")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
```

- [ ] **Step 2: Write app/schemas/responses.py**

```python
"""API 响应模型。"""
from pydantic import BaseModel, Field


class FeatureScores(BaseModel):
    global_: float = Field(default=0.0, alias="global")
    # 后续版本追加: face, outfit, color, hair, accessory, pose

    class Config:
        populate_by_name = True


class MatchResultItem(BaseModel):
    rank: int
    character_id: str
    name: str
    score: float
    feature_scores: dict[str, float]
    match_reasons: list[str] = []
    safe: bool = True


class MatchResponse(BaseModel):
    query_id: str
    results: list[MatchResultItem]


class ErrorResponse(BaseModel):
    detail: str
```

- [ ] **Step 3: Write tests/test_schemas.py**

```python
from app.schemas.requests import MatchRequest
from app.schemas.responses import MatchResponse, MatchResultItem


def test_match_request_defaults():
    req = MatchRequest(image_base64="aaa")
    assert req.preset == "default"
    assert req.top_k == 5


def test_match_request_validation():
    req = MatchRequest(image_base64="bbb", preset="anime_vibe", top_k=10)
    assert req.preset == "anime_vibe"
    assert req.top_k == 10


def test_match_response_serialization():
    resp = MatchResponse(
        query_id="q_001",
        results=[
            MatchResultItem(
                rank=1,
                character_id="mario",
                name="Mario",
                score=0.87,
                feature_scores={"global": 0.87},
            )
        ],
    )
    data = resp.model_dump()
    assert data["query_id"] == "q_001"
    assert len(data["results"]) == 1
    assert data["results"][0]["character_id"] == "mario"
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/test_schemas.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add app/schemas/ tests/test_schemas.py
git commit -m "feat: add API request/response schemas"
```

---

### Task 9: Match API 接口 + FastAPI 应用

**Files:**
- Create: `app/config.py`
- Create: `app/api/match.py`
- Create: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write app/config.py**

```python
"""应用级全局配置。"""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AppConfig:
    project_root: Path = Path(__file__).parent.parent
    pipeline_config_path: str = "configs/pipeline.yaml"
    debug: bool = False


_app_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
    return _app_config


def set_config(config: AppConfig) -> None:
    global _app_config
    _app_config = config
```

- [ ] **Step 2: Write app/api/match.py**

```python
"""匹配接口。"""
import base64
import io
import time
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image

from app.schemas.requests import MatchRequest
from app.schemas.responses import MatchResponse, MatchResultItem

router = APIRouter(prefix="/api/match", tags=["match"])

# 全局管线实例，由 main.py 在启动时注入
_pipeline = None


def set_pipeline(pipeline):
    global _pipeline
    _pipeline = pipeline


def _decode_image(image_base64: str) -> np.ndarray:
    """将 base64 字符串解码为 RGB numpy array。"""
    try:
        # 去掉可能的 data:image/...;base64, 前缀
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


@router.post("/image", response_model=MatchResponse)
async def match_image(req: MatchRequest):
    """单图匹配：上传一张图片，返回 Top-K 相似角色。"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    image = _decode_image(req.image_base64)

    try:
        results = _pipeline.run(image, preset_name=req.preset, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {e}")

    query_id = f"q_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    return MatchResponse(
        query_id=query_id,
        results=[
            MatchResultItem(
                rank=r.rank,
                character_id=r.character_id,
                name=r.name,
                score=r.score,
                feature_scores=r.feature_scores,
                match_reasons=r.match_reasons,
                safe=r.safe,
            )
            for r in results
        ],
    )
```

- [ ] **Step 3: Write app/main.py**

```python
"""FastAPI 应用入口。"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api.match import router as match_router, set_pipeline
from engine.pipeline import LookalikePipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化管线，关闭时释放资源。"""
    pipeline = LookalikePipeline()
    set_pipeline(pipeline)
    yield
    # 关闭时清理 GPU 显存
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Look-Alike Cam Engine",
    version="0.1.0",
    description="多特征相似度匹配系统 - 娱乐向 Look-Alike Cam",
    lifespan=lifespan,
)

app.include_router(match_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Write tests/test_api.py**

```python
import base64
import json
import numpy as np
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def test_app(tmp_path, monkeypatch):
    """构造一个不加载真实模型的测试应用。"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    monkeypatch.chdir(tmp_path)

    # 创建最小配置文件
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "pipeline.yaml").write_text("""
pipeline:
  detectors: []
  embedders: []
  retrievers: []
  ranker:
    class_path: engine.rankers.simple_ranker.SimpleRanker
""")
    (configs_dir / "presets.yaml").write_text("""
presets:
  default:
    weights:
      global: 1.0
""")
    (configs_dir / "safety.yaml").write_text("""
min_final_score: 0.0
blocked_character_tags: []
save_audience_images: false
save_audience_embeddings: false
embedding_ttl_minutes: 60
require_human_approval: false
""")

    # 创建角色数据
    chars_dir = tmp_path / "data" / "characters" / "mario"
    chars_dir.mkdir(parents=True)
    (chars_dir / "card.json").write_text(json.dumps({
        "id": "mario",
        "name": "Mario",
        "type": "game_character",
        "safety": {"allowed": True, "risk_level": "low"},
    }))

    from app.main import app
    # 替换 lifespan，避免启动时加载真实模型
    from app.api.match import set_pipeline
    from engine.pipeline import LookalikePipeline

    pipeline = LookalikePipeline(config_path=str(configs_dir / "pipeline.yaml"))
    set_pipeline(pipeline)

    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_match_image_invalid_data(client):
    resp = client.post("/api/match/image", json={
        "image_base64": "not-valid-base64!!!",
    })
    assert resp.status_code == 400


def _make_test_image_base64():
    """生成一张测试图片的 base64。"""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_match_image_returns_empty_when_no_index(client):
    """没有建索引时，返回空结果。"""
    resp = client.post("/api/match/image", json={
        "image_base64": _make_test_image_base64(),
        "preset": "default",
        "top_k": 5,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []
    assert data["query_id"].startswith("q_")
```

- [ ] **Step 5: Run tests**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/test_api.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add app/ tests/test_api.py
git commit -m "feat: add POST /api/match/image endpoint and FastAPI app"
```

---

### Task 10: 角色导入脚本

**Files:**
- Create: `scripts/import_characters.py`

- [ ] **Step 1: Write scripts/import_characters.py**

```python
#!/usr/bin/env python3
"""批量导入角色：将含 card.json 的目录复制到 data/characters/ 下。"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Import character cards into the library")
    parser.add_argument(
        "source_dir",
        type=str,
        help="Source directory containing character subdirectories with card.json",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/characters",
        help="Target character library directory (default: data/characters)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing characters",
    )
    args = parser.parse_args()

    source = Path(args.source_dir)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0

    for card_path in sorted(source.glob("*/card.json")):
        char_dir = card_path.parent
        char_id = card_path.parent.name
        dest_dir = target / char_id

        if dest_dir.exists() and not args.overwrite:
            print(f"SKIP {char_id}: already exists (use --overwrite)")
            skipped += 1
            continue

        # 验证 card.json 格式
        try:
            card = json.loads(card_path.read_text())
            assert "id" in card, "Missing 'id' field"
            assert "name" in card, "Missing 'name' field"
            assert card["id"] == char_id, f"id '{card['id']}' != directory name '{char_id}'"
        except (json.JSONDecodeError, AssertionError) as e:
            print(f"ERROR {char_id}: invalid card.json - {e}")
            continue

        # 复制整个目录
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(char_dir, dest_dir)
        print(f"IMPORTED {char_id}: {card.get('name', char_id)}")
        imported += 1

    print(f"\nDone. Imported {imported}, skipped {skipped}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试导入脚本**

Run:
```bash
cd /home/ljz/vibe_coding/lookalike-cam-engine
mkdir -p /tmp/test_chars/mario
cat > /tmp/test_chars/mario/card.json << 'JSONEOF'
{"id": "mario", "name": "Mario", "type": "game_character", "safety": {"allowed": true}}
JSONEOF
python3 scripts/import_characters.py /tmp/test_chars --target /tmp/test_library
cat /tmp/test_library/mario/card.json
```

Expected: 输出 `IMPORTED mario: Mario`，且 card.json 内容正确。

- [ ] **Step 3: Commit**

```bash
git add scripts/import_characters.py
git commit -m "feat: add character import script"
```

---

### Task 11: 索引构建脚本

**Files:**
- Create: `scripts/build_index.py`

- [ ] **Step 1: Write scripts/build_index.py**

```python
#!/usr/bin/env python3
"""构建 FAISS 索引：遍历角色库，为每张参考图生成 embedding，建立索引。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def build_global_index(
    characters_dir: str = "data/characters",
    output_path: str = "indexes/global.index",
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cuda",
):
    """为角色库中所有图片提取 OpenCLIP global embedding，构建 FAISS 索引。"""
    from engine.embedders.clip_embedder import ClipEmbedder
    from engine.retrievers.faiss_retriever import FaissRetriever

    chars_path = Path(characters_dir)
    if not chars_path.exists():
        print(f"ERROR: Characters directory not found: {characters_dir}")
        sys.exit(1)

    print(f"Loading OpenCLIP model ({model_name}, {pretrained}) on {device}...")
    embedder = ClipEmbedder(model_name=model_name, pretrained=pretrained, device=device)

    all_vectors = []
    all_ids = []
    stats = {"characters": 0, "images": 0, "skipped": 0}

    for card_path in sorted(chars_path.glob("*/card.json")):
        card = json.loads(card_path.read_text())
        char_id = card["id"]

        # 跳过不安全角色
        if not card.get("safety", {}).get("allowed", True):
            print(f"SKIP {char_id}: blocked by safety policy")
            stats["skipped"] += 1
            continue

        images_dir = card_path.parent / "images"
        if not images_dir.exists():
            print(f"WARN {char_id}: no images/ directory")
            continue

        image_files = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        if not image_files:
            print(f"WARN {char_id}: no images found in images/")
            continue

        char_vectors = []
        for img_path in image_files:
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                fv = embedder.embed(img)
                char_vectors.append(fv.vector)
            except Exception as e:
                print(f"WARN {char_id}/{img_path.name}: {e}")
                continue

        if not char_vectors:
            continue

        # 对同一角色的多张图取平均作为该角色的代表向量
        mean_vector = np.mean(char_vectors, axis=0)
        mean_vector = mean_vector / np.linalg.norm(mean_vector)

        all_vectors.append(mean_vector)
        all_ids.append(char_id)
        stats["characters"] += 1
        stats["images"] += len(char_vectors)
        print(f"  {char_id}: {len(char_vectors)} images -> embedding ({embedder.dimension}d)")

    if not all_vectors:
        print("ERROR: No valid characters found. Nothing to index.")
        sys.exit(1)

    vec_matrix = np.stack(all_vectors, axis=0)
    print(f"\nBuilding FAISS index: {vec_matrix.shape[0]} vectors x {vec_matrix.shape[1]} dims")

    retriever = FaissRetriever(dimension=embedder.dimension)
    retriever.build_index(vec_matrix, all_ids)
    retriever.save(output_path)

    print(f"Index saved to {output_path}")
    print(f"Stats: {stats['characters']} characters, {stats['images']} images, {stats['skipped']} skipped")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from character library")
    parser.add_argument("--characters-dir", default="data/characters")
    parser.add_argument("--output", default="indexes/global.index")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    build_global_index(
        characters_dir=args.characters_dir,
        output_path=args.output,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/build_index.py
git commit -m "feat: add index building script (OpenCLIP + FAISS)"
```

---

### Task 12: 示例数据与端到端验证

**Files:**
- Create: `data/characters/mario/card.json`
- Create: `data/characters/luigi/card.json`
- Create: `data/characters/pikachu/card.json`
- Create: `data/characters/sailor_moon/card.json`
- Create: `tests/test_e2e.py`

- [ ] **Step 1: 创建示例角色卡**

Write `data/characters/mario/card.json`:
```json
{
  "id": "mario",
  "name": "Mario",
  "type": "game_character",
  "source": "Super Mario",
  "tags": ["red_hat", "mustache", "blue_overalls", "cartoon"],
  "slots": {
    "face": ["mustache", "round_nose"],
    "hat": ["red_cap"],
    "upper_body": ["red_shirt"],
    "lower_body": ["blue_overalls"],
    "dominant_colors": ["red", "blue"]
  },
  "safety": {
    "allowed": true,
    "risk_level": "low"
  }
}
```

Write `data/characters/luigi/card.json`:
```json
{
  "id": "luigi",
  "name": "Luigi",
  "type": "game_character",
  "source": "Super Mario",
  "tags": ["green_hat", "mustache", "green_overalls", "cartoon"],
  "slots": {
    "face": ["mustache"],
    "hat": ["green_cap"],
    "upper_body": ["green_shirt"],
    "lower_body": ["blue_overalls"],
    "dominant_colors": ["green", "blue"]
  },
  "safety": {
    "allowed": true,
    "risk_level": "low"
  }
}
```

Write `data/characters/pikachu/card.json`:
```json
{
  "id": "pikachu",
  "name": "Pikachu",
  "type": "anime_character",
  "source": "Pokemon",
  "tags": ["yellow", "electric", "cute", "anime"],
  "slots": {
    "dominant_colors": ["yellow", "red", "black"]
  },
  "safety": {
    "allowed": true,
    "risk_level": "low"
  }
}
```

Write `data/characters/sailor_moon/card.json`:
```json
{
  "id": "sailor_moon",
  "name": "Sailor Moon",
  "type": "anime_character",
  "source": "Sailor Moon",
  "tags": ["blonde_hair", "sailor_outfit", "anime", "magical_girl"],
  "slots": {
    "hair": ["blonde", "twin_tails"],
    "upper_body": ["sailor_top"],
    "dominant_colors": ["white", "pink", "blue", "yellow"]
  },
  "safety": {
    "allowed": true,
    "risk_level": "low"
  }
}
```

For each character, create `images/` directory. Users place their own character images there.

- [ ] **Step 2: Write tests/test_e2e.py**

```python
"""端到端测试：完整链路 — 建索引 → 加载 → 匹配。"""
import json
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engine.embedders.clip_embedder import ClipEmbedder
from engine.retrievers.faiss_retriever import FaissRetriever


def _create_sample_characters(chars_dir: Path):
    """创建 4 个测试角色，每个角色有 1 张纯色测试图。"""
    characters = [
        {"id": "mario", "name": "Mario", "color": (255, 0, 0)},     # 红
        {"id": "luigi", "name": "Luigi", "color": (0, 255, 0)},     # 绿
        {"id": "pikachu", "name": "Pikachu", "color": (255, 255, 0)},  # 黄
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
```

- [ ] **Step 3: Run E2E test**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/test_e2e.py -v -m slow`
Expected: 1 PASS（需要下载 OpenCLIP 模型，约 2-5 分钟）

- [ ] **Step 4: Commit**

```bash
git add data/characters/ tests/test_e2e.py scripts/
git commit -m "feat: add sample character cards and end-to-end test"
```

---

### Task 13: 最终验证

- [ ] **Step 1: 启动服务**

Run:
```bash
cd /home/ljz/vibe_coding/lookalike-cam-engine
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 5
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 2: 测试匹配接口（无索引时返回空）**

Run:
```bash
# 生成一张测试图并 base64 编码
python3 -c "
import base64, io
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, 'JPEG')
print(base64.b64encode(buf.getvalue()).decode()[:50] + '...')
"
```

Run: `curl -s -X POST http://localhost:8000/api/match/image -H "Content-Type: application/json" -d '{"image_base64":"'"$(python3 -c "import base64,io;from PIL import Image;import numpy as np;img=Image.fromarray(np.random.randint(0,255,(224,224,3),dtype=np.uint8));b=io.BytesIO();img.save(b,'JPEG');print(base64.b64encode(b.getvalue()).decode())")"'"}' | python3 -m json.tool`

Expected: `{"query_id": "...", "results": []}`（空结果，因为还没建索引）

- [ ] **Step 3: 查看 API 文档**

访问: `http://localhost:8000/docs`

- [ ] **Step 4: 停止服务**

Run: `kill %1`

- [ ] **Step 5: 运行全部测试**

Run: `cd /home/ljz/vibe_coding/lookalike-cam-engine && python3 -m pytest tests/ -v --ignore=tests/test_e2e.py`
Expected: 所有非 slow 测试 PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: M1 complete — pluggable pipeline, single-image matching, E2E verified"
```
