"""API 响应模型。"""
from pydantic import BaseModel, Field


class MatchResultItem(BaseModel):
    rank: int = Field(..., description="排名")
    character_id: str = Field(..., description="角色英文ID")
    name: str = Field(..., description="角色显示名称")
    score: float = Field(..., description="相似度总分 (0~1)")
    feature_scores: dict[str, float] = Field(default_factory=dict, description="各维度分数")
    match_reasons: list[str] = Field(default_factory=list, description="匹配理由（如：红帽子、胡子）")
    safe: bool = Field(default=True, description="是否安全可上屏")


class MatchResponse(BaseModel):
    query_id: str = Field(..., description="查询唯一标识")
    results: list[MatchResultItem] = Field(default_factory=list, description="匹配结果列表")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误详情")
