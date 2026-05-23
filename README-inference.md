# Screen Detector V3 - Inference

两阶段 CNN + FFT Branch 推理系统。

## 功能

- 两阶段 CNN 推理 (EfficientNet-B0 + FFT Branch)
- OOD 检测 (unknown 类别)
- TTA (Test Time Augmentation)
- FFT 预处理缓存
- FastAPI REST API 服务
- 批量检测支持

## 快速开始

```bash
# 安装依赖 (在项目根目录)
uv sync

# 启动 API 服务
uv run python main.py
```

## 目录结构

```
inference/
├── README.md
├── models/
│   ├── stage1_natural_vs_screenlike.onnx
│   └── stage2_screenlike_vs_screenphoto.onnx
├── api/                # FastAPI 服务
├── config.py           # 配置 (模型路径/阈值/标签)
├── predictor.py        # 两阶段推理器
├── preprocess.py       # RGB 预处理
├── fft_transform.py    # FFT 频谱变换
├── batch_detect.py     # 批量检测
├── image_index.py      # 图片索引
└── scheduler.py        # 后台清理
```

## API 接口

### POST /api/detect/upload

文件上传检测。

```bash
curl -X POST http://localhost:8325/api/detect/upload \
  -F "file=@test.jpg"
```

**响应:**
```json
{
  "image_id": "uuid",
  "filename": "test.jpg",
  "class_name": "screen_photo",
  "confidence": 0.9759,
  "probabilities": {"natural": 0.01, "screen_like": 0.01, "screen_photo": 0.98},
  "stage": 2,
  "confidence_tier": "high",
  "action": "accept"
}
```

### POST /api/detect

URL 检测。

```bash
curl -X POST http://localhost:8325/api/detect \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/test.jpg"}'
```

### GET /api/health

健康检查。

## Python 调用

```python
from inference.src.predictor import ScreenDetectorPredictor

predictor = ScreenDetectorPredictor()
result = predictor.predict("path/to/image.jpg")

print(result["class"])        # natural/screen_like/screen_photo/unknown
print(result["confidence"])   # 0.0 - 1.0
print(result["stage"])        # 1 or 2
print(result["confidence_tier"])  # high/medium/low/ood
```

## 推理流程

```
Image → Stage 1 CNN → natural/screen_like
                         ↓ screen_like
                    OOD 检测 (max_prob < 0.65 → unknown)
                         ↓
                    Stage 2 CNN → screenshot/screen_photo
                         ↓
                    置信度分级 (accept/review/ignore)
```

## 配置

`src/config.py` 中的关键配置:

- `STAGE1_MODEL_PATH` - Stage 1 模型路径
- `STAGE2_MODEL_PATH` - Stage 2 模型路径
- `OOD_THRESHOLD = 0.65` - OOD 检测阈值
- `CONFIDENCE_HIGH = 0.92` - 高置信度阈值
- `CONFIDENCE_MEDIUM = 0.75` - 中置信度阈值
- `API_PORT = 8325` - API 端口
