# Screen Detector V3

基于 Python + OpenCV + CNN 的三类图像来源识别系统。

## 系统架构

采用**两阶段 CNN + FFT Branch**架构：

```
Image
   ↓
Stage 1 CNN (EfficientNet-B0 + FFT Branch)
   ↓
natural / screen_like?
   ↓ natural → 返回 "natural"
   ↓ screen_like
   ↓
Stage 2 CNN (EfficientNet-B0 + FFT Branch)
   ↓
screenshot / screen_photo?
   ↓ → 返回 "screen_like" 或 "screen_photo"
```

### 标签体系

| 标签 | 含义 | 包含内容 |
|------|------|----------|
| `natural` | 真实自然图像 | 风景、人像、室内、动物、食物、街景、天空、树木 |
| `screen_like` | 屏幕内容 | 截图、PPT、IDE、UI、terminal、聊天记录、软件界面 |
| `screen_photo` | 相机拍摄屏幕 | 手机拍摄的屏幕照片 |

### 置信度分级

| 置信度 | 处理方式 |
|--------|----------|
| >= 0.92 | 直接输出 (accept) |
| 0.75 - 0.92 | 人工审核 (review) |
| < 0.75 | 忽略 (ignore) |
| < 0.65 | OOD 检测，返回 unknown |

## 快速开始

### 安装依赖

```bash
uv sync
```

### 启动 API 服务

```bash
uv run python main.py
```

API 服务运行在 `http://localhost:8325`

### 测试接口

```bash
# 健康检查
curl http://localhost:8325/api/health

# 文件上传检测
curl -X POST http://localhost:8325/api/detect/upload \
  -F "file=@test.jpg"

# URL 检测
curl -X POST http://localhost:8325/api/detect \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/test.jpg"}'
```

## 项目结构

```
opencv-screen-detector/
├── main.py                         # API 入口
├── pyproject.toml                  # 推理端依赖
├── config/
│   └── thresholds.json             # 阈值配置
├── inference/                      # 推理系统
│   ├── models/
│   │   ├── stage1_natural_vs_screenlike.onnx
│   │   └── stage2_screenlike_vs_screenphoto.onnx
│   └── src/
│       ├── config.py               # 推理配置
│       ├── predictor.py            # 两阶段推理器 (TTA/OOD/缓存)
│       ├── preprocess.py           # RGB 预处理
│       ├── fft_transform.py        # FFT 频谱变换
│       ├── unified_api.py          # FastAPI 服务
│       ├── batch_detect.py         # 批量检测
│       ├── image_index.py          # 图片索引
│       └── scheduler.py            # 后台清理
├── trainer/                        # 训练系统
│   ├── pyproject.toml              # 训练端依赖
│   └── src/
│       ├── config.py               # 训练配置
│       ├── model.py                # 融合模型 (EfficientNet + FFT Branch)
│       ├── fft_branch.py           # Frequency Branch (ResBlock)
│       ├── dataset.py              # 双输入数据集
│       ├── train.py                # 两阶段训练 (AMP)
│       ├── validate.py             # 验证指标
│       ├── augment.py              # 数据增强
│       └── export_onnx.py          # ONNX 导出
├── tests/                          # 测试
│   ├── conftest.py
│   ├── test_accuracy.py
│   ├── test_predictor.py
│   ├── test_fft_transform.py
│   ├── test_dataset.py
│   └── test_api.py
├── data/
│   ├── input/
│   │   ├── natural_photo/          # 自然照片 (含子目录)
│   │   ├── screen_like/            # 屏幕内容
│   │   ├── screenshot/             # 截图
│   │   ├── screen_photo/           # 拍屏照片
│   │   └── hard_negative/          # 难例负样本
│   ├── upload/                     # API 上传缓存
│   └── image_index.json            # 图片索引
└── scripts/
    └── fetch_natural_photos.py     # 数据爬取脚本
```

## API 文档

### POST /api/detect/upload

文件上传检测。

**请求**: `multipart/form-data`，字段 `file`

**响应**:
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

**请求**: `application/json`
```json
{"url": "https://example.com/test.jpg"}
```

### GET /api/health

健康检查。

## 训练指南

```bash
cd trainer
uv sync

# 训练两个阶段
uv run python -m src.train

# 导出 ONNX 模型
uv run python -m src.export_onnx
```

## 测试

```bash
uv run pytest tests/ -v
```

## 依赖

### 推理端
- opencv-python-headless
- numpy
- pillow
- fastapi + uvicorn
- httpx
- onnxruntime

### 训练端
- torch + torchvision
- timm (EfficientNet)
- albumentations
- scikit-learn
- matplotlib

## License

MIT
