"""API 请求模型。"""
from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 编码的图片数据")
    preset: str = Field(default="default", description="预设名称")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
