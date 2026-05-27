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
