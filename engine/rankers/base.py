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
