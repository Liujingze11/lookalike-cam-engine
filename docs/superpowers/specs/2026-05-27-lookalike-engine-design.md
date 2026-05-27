# Look-Alike Cam Engine 设计文档

## 项目定位

娱乐向多模态相似度匹配系统。输入一张观众图片，输出与其最相似的明星、动漫角色、游戏角色或 meme 形象。

**不是身份识别系统。** 不判断观众真实身份，不做认证。

## 核心架构：插件化流水线

整个系统是一条五阶段流水线，每个阶段通过统一接口接入可替换的插件：

```
输入图片/视频帧
    │
    ▼
┌──────────────────────────────────────────────┐
│  检测层 (Detectors)                            │
│  检测图片里有什么：人脸在哪、人体在哪、帽子在哪      │
│  接口: detect(image) -> list[Region]          │
│  M1: NullDetector (全图当作一个区域)             │
│  M2: +FaceDetector (InsightFace 检测人脸)       │
│  M3: +BodyDetector (YOLO 检测人体各区域)         │
└──────────────────┬───────────────────────────┘
                   │ 裁剪后的区域列表
                   ▼
┌──────────────────────────────────────────────┐
│  特征提取层 (Embedders)                         │
│  把每个区域变成特征向量                            │
│  接口: embed(image | region) -> Vector         │
│  M1: ClipEmbedder (全图语义)                    │
│  M2: +FaceEmbedder (InsightFace 人脸向量)       │
│  M3: +OutfitEmbedder + ColorEmbedder          │
└──────────────────┬───────────────────────────┘
                   │ 多组特征向量 (embedding_type -> vector)
                   ▼
┌──────────────────────────────────────────────┐
│  检索层 (Retrievers)                           │
│  拿着特征去角色库向量索引里搜最相似的                │
│  接口: search(vector, index_name) -> list[Candidate] │
│  M1: 一个 FAISS global_index                  │
│  M2: +face_index                              │
│  M3: +outfit_index, +color_index               │
└──────────────────┬───────────────────────────┘
                   │ 各维度召回的候选角色
                   ▼
┌──────────────────────────────────────────────┐
│  排序层 (Ranker)                               │
│  融合各维度分数、排序、安全过滤、生成解释            │
│  接口: rank(candidates, weights, safety) -> list[Result] │
│  M1: SimpleRanker (单维度分数即总分)              │
│  M3+: WeightedRanker (多维度加权融合)             │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
              Top-K 结果
```

## 插件接口定义

```python
# 检测器：输入图片，输出检测到的区域
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectedRegion]:
        ...

# 特征提取器：输入图片或区域，输出特征向量和元数据
class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def embedding_type(self) -> str:  # "global" | "face" | "outfit" | "color"
        ...

    @abstractmethod
    def embed(self, image: np.ndarray) -> FeatureVector:
        ...

# 检索器：输入向量和索引名，返回候选列表
class BaseRetriever(ABC):
    @abstractmethod
    def search(self, vector: np.ndarray, index_name: str, top_k: int) -> list[Candidate]:
        ...

# 排序器：融合 + 过滤 + 解释
class BaseRanker(ABC):
    @abstractmethod
    def rank(self, candidates: list[Candidate], preset: Preset) -> list[MatchResult]:
        ...
```

## 管线调度器

```python
class LookalikePipeline:
    """不关心插件细节，只按配置调用各层插件"""

    def __init__(self, config: PipelineConfig):
        self.detectors: list[BaseDetector] = config.build_detectors()
        self.embedders: dict[str, BaseEmbedder] = config.build_embedders()
        self.retrievers: dict[str, BaseRetriever] = config.build_retrievers()
        self.ranker: BaseRanker = config.build_ranker()

    def run(self, image: np.ndarray, preset_name: str) -> list[MatchResult]:
        # 1. 检测
        regions = []
        for detector in self.detectors:
            regions.extend(detector.detect(image))

        # 2. 特征提取
        vectors: dict[str, FeatureVector] = {}
        for region in regions:
            etype = region.embedder_type
            if etype in self.embedders:
                vectors[etype] = self.embedders[etype].embed(region.image)

        # 3. 多路检索
        all_candidates: list[Candidate] = []
        for etype, vec in vectors.items():
            if etype in self.retrievers:
                all_candidates.extend(
                    self.retrievers[etype].search(vec.vector, f"{etype}_index", top_k=50)
                )

        # 4. 融合排序
        preset = load_preset(preset_name)
        return self.ranker.rank(all_candidates, preset)
```

## 项目目录结构

```
lookalike-engine/
├── app/                        # Web 层
│   ├── main.py                 # FastAPI 启动
│   ├── api/
│   │   ├── match.py            # POST /api/match/image
│   │   ├── characters.py       # CRUD 角色（M5）
│   │   └── feedback.py         # 反馈收集（M7）
│   └── schemas/                # Pydantic 模型
│       ├── requests.py
│       └── responses.py
│
├── engine/                     # 算法引擎
│   ├── pipeline.py             # 管线调度器
│   ├── detectors/
│   │   ├── base.py
│   │   ├── null_detector.py
│   │   ├── face_detector.py    # M2
│   │   └── body_detector.py    # M3
│   ├── embedders/
│   │   ├── base.py
│   │   ├── clip_embedder.py
│   │   ├── face_embedder.py    # M2
│   │   ├── outfit_embedder.py  # M3
│   │   └── color_embedder.py   # M3
│   ├── retrievers/
│   │   ├── base.py
│   │   └── faiss_retriever.py
│   └── rankers/
│       ├── base.py
│       ├── simple_ranker.py    # M1
│       └── weighted_ranker.py  # M3+
│
├── data/
│   └── characters/             # 角色库
│       └── {character_id}/
│           ├── card.json       # 角色元数据
│           └── images/         # 参考图
│
├── indexes/                    # FAISS 索引文件（自动生成）
│
├── configs/
│   ├── pipeline.yaml           # 管线插件配置
│   ├── presets.yaml            # 权重配置
│   └── safety.yaml             # 安全规则
│
├── scripts/
│   ├── build_index.py          # 构建/重建索引
│   ├── import_characters.py    # 批量导入角色
│   └── test_match.py           # 命令行测试
│
└── requirements.txt
```

## 数据格式

### Character Card (角色卡)

```json
{
  "id": "mario",
  "name": "Mario",
  "type": "game_character",
  "source": "Super Mario",
  "images": ["front.png", "jump.png", "movie.png"],
  "tags": ["red_hat", "mustache", "blue_overalls", "cartoon"],
  "slots": {
    "face": ["mustache", "round_nose"],
    "hat": ["red_cap"],
    "upper_body": ["red_shirt"],
    "lower_body": ["blue_overalls"],
    "dominant_colors": ["red", "blue"]
  },
  "safety": {
    "allowed": true,
    "risk_level": "low",
    "notes": ""
  }
}
```

M1 只用 `id`、`name`、`type`、`images`、`tags`。`slots` 字段 M3 启用。

### 配置文件

```yaml
# pipeline.yaml — 管线的插件组成
pipeline:
  detectors: []                          # M1 空、M2 加 face、M3 加 body

  embedders:
    - name: clip_global
      class: engine.embedders.clip_embedder.ClipEmbedder
      model_name: ViT-B-32
      pretrained: laion2b_s34b_b79k
      device: cuda
      source: full_image                 # 输入来源：全图

  retrievers:
    - name: global_retriever
      class: engine.retrievers.faiss_retriever.FaissRetriever
      index_path: indexes/global.index
      embedding_type: global

  ranker:
    class: engine.rankers.simple_ranker.SimpleRanker
```

```yaml
# presets.yaml — 不同玩法的权重
presets:
  celebrity_face:
    weights:
      face: 0.55
      global: 0.20
      hair: 0.10
      outfit: 0.08
      accessory: 0.05
      color: 0.02
  anime_vibe:
    weights:
      face: 0.10
      global: 0.30
      outfit: 0.25
      color: 0.15
      hair: 0.10
      accessory: 0.08
      pose: 0.02
  outfit_match:
    weights:
      face: 0.05
      global: 0.15
      outfit: 0.40
      color: 0.20
      accessory: 0.15
      pose: 0.05
  meme_funny:
    weights:
      face: 0.10
      global: 0.25
      outfit: 0.20
      color: 0.10
      accessory: 0.15
      pose: 0.10
      surprise: 0.10
```

M1 中 presets.yaml 存在但只有一个 `default` preset。

## 完整路线图

### M1：单模型 MVP（当前实现）

**目标：** 上传图片 → 返回 Top-K 最相似角色。验证整套架构能跑。

**实现内容：**
- 项目骨架 + 目录结构
- NullDetector（全图即区域）
- ClipEmbedder（OpenCLIP ViT-B/32）
- FaissRetriever（单 index）
- SimpleRanker
- `/api/match/image` 接口
- `build_index.py` 脚本
- `card.json` 格式 + 示例角色
- 配置文件驱动

**不实现：**
- 人脸检测/特征
- 颜色/服饰
- 多索引
- 多权重融合
- 视频/实时
- 前端

**验收标准：**
- 导入 50+ 角色，每个角色至少 2 张参考图
- 单图匹配返回 Top 5，响应 < 3 秒
- Pipeline 完全由配置文件驱动

### M2：人脸模块

**新增插件：** FaceDetector、FaceEmbedder
**改进检索：** 增加 face_index，并行检索
**改进排序：** 升级为 WeightedRanker，双维度加权
**不动的代码：** pipeline.py、所有 base 接口

### M3：服饰 + 颜色 + 可解释性

**新增插件：** BodyDetector、OutfitEmbedder、ColorEmbedder
**改进检索：** 增加 outfit_index
**改进排序：** 多维度加权 + 颜色标签解释
**不动的代码：** pipeline 主流程、所有已有插件

### M4：多 Preset 配置

**改进：** 完善 presets.yaml，前端可切换
**新增：** preset 管理 API
**不动的代码：** 引擎层

### M5：角色库管理

**新增：** 前端管理界面（Next.js）
**改进：** 角色 CRUD API
**不动的代码：** 引擎层

### M6：视频 + 实时

**新增：** 视频抽帧、人物跟踪、实时流处理
**改进：** 帧质量评估、结果缓存
**不动的代码：** 单帧匹配逻辑复用 M1-M3 的管线

### M7：训练 Reranker

**新增：** 反馈收集、LightGBM 精排模型
**改进：** 排序层可插拔
**不动的代码：** 检测、特征提取、检索层

## API 设计

### POST /api/match/image

```json
// Request
{
  "image_base64": "...",          // 必填
  "preset": "default",            // 可选，默认 default
  "top_k": 5,                     // 可选，默认 5
}

// Response
{
  "query_id": "q_20260527_001",
  "results": [
    {
      "rank": 1,
      "character_id": "mario",
      "name": "Mario",
      "score": 0.87,
      "feature_scores": {         // M1 只有 global
        "global": 0.87
      },
      "match_reasons": [],        // M3 启用
      "safe": true
    }
  ]
}
```

### POST /api/characters/import（M5）

导入角色。Body 含角色目录路径和 card.json 路径。

### POST /api/feedback（M7）

提交用户反馈。Body 含 query_id、character_id、label (0-3)。

## 安全设计

```yaml
# safety.yaml
require_human_approval: false       # M1 关闭，M6 开启
save_audience_images: false
save_audience_embeddings: false
embedding_ttl_minutes: 60
blocked_character_tags:
  - political
  - criminal
  - disease
  - body_shaming
  - racial_stereotype
min_broadcast_score: 0.70
min_final_score: 0.62
```

所有角色在导入时检查 `safety.allowed` 字段。结果为 `false` 的角色不参与检索。

## 技术选型

| 组件 | 选型 | 原因 |
|------|------|------|
| 图片特征 | OpenCLIP ViT-B/32 | 6GB 显存可跑，多语言 |
| 向量检索 | FAISS IndexFlatIP | <10000 向量暴力搜索够快 |
| 后端 | FastAPI | 异步、自动文档、生态好 |
| 配置 | YAML (PyYAML) | 可读性好 |
| 图像处理 | PIL + OpenCV | 标准工具 |
| 深度学习 | PyTorch + transformers | OpenCLIP 依赖 |

## 模型内存规划

6GB 显存约束下的加载策略：

- OpenCLIP ViT-B/32: ~400MB
- 启动时仅加载 OpenCLIP
- InsightFace (M2): ~200MB，首次人脸请求时 lazy load
- 不用的模型调用 `model.to("cpu")` 腾空间
- 批处理 inference 时 batch_size=4

## 关键设计决策

1. **插件化从第一天开始** — 每个插件实现统一接口，Pipeline 不耦合具体实现
2. **配置文件驱动** — 换模型、调权重、加插件都只需改 YAML
3. **Character Card 一步到位** — 数据结构兼容到 M7，按需启用字段
4. **不用数据库** — 500 字符以内 JSON 文件 + 目录结构足够
5. **暴力搜索** — IndexFlatIP 在 1000 向量内比 IVF 索引更快更准
6. **每阶段独立可运行** — 不阻塞等待后续功能
