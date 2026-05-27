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
