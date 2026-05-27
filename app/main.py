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
