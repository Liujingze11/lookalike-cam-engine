"""API 响应模型。"""
from pydantic import BaseModel, Field


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
