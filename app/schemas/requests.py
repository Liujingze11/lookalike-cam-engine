"""API 请求模型。"""
from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 编码的图片（JPEG/PNG）")
    preset: str = Field(default="default", description="匹配预设：default=通用")
    top_k: int = Field(default=5, ge=1, le=20, description="返回 Top-K 结果，1~20")
