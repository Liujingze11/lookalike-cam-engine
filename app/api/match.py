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

router = APIRouter(prefix="/api/match", tags=["图片匹配"])

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
        raise HTTPException(status_code=400, detail=f"图片数据无效: {e}")


@router.post("/image", response_model=MatchResponse, summary="单图匹配")
async def match_image(req: MatchRequest):
    """上传一张观众图片，返回最相似的 Top-K 角色。

    支持 base64 编码的 JPEG/PNG 图片，可以通过 preset 切换不同匹配模式。
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="匹配引擎尚未初始化")

    image = _decode_image(req.image_base64)

    try:
        results = _pipeline.run(image, preset_name=req.preset, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"匹配失败: {e}")

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
