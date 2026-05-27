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
            cls = _import_class(det_cfg.class_path)
            self._detectors.append(cls(**det_cfg.kwargs))

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
