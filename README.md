# Screen Detector V3

基于 Python + OpenCV + CNN 的三类图像来源识别系统。

## 系统架构

采用**两阶段 CNN + FFT Branch**架构：

```
Image
   ↓
Stage 1 CNN (EfficientNet-B0 + FFT Branch)
   ↓
natural / screenshot?
   ↓ natural → 返回 "natural"
   ↓ screenshot
   ↓
Stage 2 CNN (EfficientNet-B0 + FFT Branch)
   ↓
screenshot / screen_photo?
   ↓ → 返回 "screenshot" 或 "screen_photo"
```

### 标签体系

| 标签 | 含义 | 包含内容 |
|------|------|----------|
| `natural` | 真实自然图像 | 风景、人像、室内、动物、食物、街景、天空、树木 |
| `screenshot` | 屏幕内容 | 截图、PPT、IDE、UI、terminal、聊天记录、软件界面 |
| `screen_photo` | 相机拍摄屏幕 | 手机拍摄的屏幕照片 |

### 置信度分级

| 置信度 | 处理方式 |
|--------|----------|
| >= 0.92 | 直接输出 (accept) |
| 0.75 - 0.92 | 人工审核 (review) |
| < 0.75 | 忽略 (ignore) |
| < 0.50 | OOD 检测，返回 unknown |

### 训练准确率

| 阶段 | 任务 | 验证准确率 |
|------|------|-----------|
| Stage 1 | natural vs screenshot | **96.12%** |
| Stage 2 | screenshot vs screen_photo | **93.99%** |

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
├── shared/                         # 共享模块
│   └── fft_transform.py            # FFT 频谱变换 (训练/推理共享)
├── inference/                      # 推理系统
│   ├── models/
│   │   ├── stage1_natural_vs_screenshot.onnx
│   │   └── stage2_screenshot_vs_screenphoto.onnx
│   ├── config.py                   # 推理配置 (Settings dataclass)
│   ├── predictor.py                # 两阶段推理器 (TTA/OOD)
│   ├── model_loader.py             # ONNX 模型加载
│   ├── fft_service.py              # FFT 缓存服务 (LRU)
│   ├── preprocess.py               # RGB 预处理 (normalize_rgb)
│   ├── api/
│   │   ├── app.py                  # FastAPI 应用
│   │   ├── router.py               # API 路由
│   │   ├── predictor.py            # 预测器生命周期管理
│   │   ├── schema.py               # Pydantic 模型
│   │   └── utils.py                # 工具函数
│   ├── batch_detect.py             # 批量检测
│   ├── image_index.py              # 图片索引 (异步 I/O)
│   └── scheduler.py                # 后台清理
├── trainer/                        # 训练系统
│   ├── config.py                   # 训练配置
│   ├── model.py                    # 融合模型 (EfficientNet + FFT Branch)
│   ├── fft_branch.py               # Frequency Branch (ResBlock)
│   ├── dataset.py                  # 双输入数据集
│   ├── train.py                    # 两阶段训练 (AMP)
│   ├── validate.py                 # 验证指标
│   ├── augment.py                  # 数据增强
│   └── export_onnx.py              # ONNX 导出
├── tests/                          # 测试
│   ├── conftest.py
│   ├── test_fft_transform.py
│   ├── test_dataset.py
│   ├── test_package.py
│   └── test_classify_extracted.py
├── data/
│   ├── input/
│   │   ├── natural_photo/          # 自然照片
│   │   ├── screenshot/             # 截图 + 屏幕内容
│   │   ├── screen_photo/           # 拍屏照片
│   │   └── hard_negative/          # 难例负样本
│   └── upload/                     # API 上传缓存
└── scripts/
    └── fetch_natural_photos.py     # Unsplash 数据爬取脚本
```

## API 文档

### POST /api/detect/upload

文件上传检测。

**请求**: `multipart/form-data`，字段 `file`

**响应**:
```json
{
  "image_id": "hash",
  "is_screen": true
}
```

### POST /api/detect

URL 检测。

**请求**: `application/json`
```json
{"url": "https://example.com/test.jpg"}
```

### GET /api/health

健康检查。返回模型加载状态和错误信息。

### POST /api/package

打包指定时间戳之后的图片为 ZIP 文件。

### POST /api/classify

更新图片分类。

## 训练指南

```bash
# 安装训练依赖
uv sync --group train

# 训练两个阶段
uv run python -m trainer train

# 导出 ONNX 模型
uv run python -m trainer export
```

### 数据爬取

```bash
export UNSPLASH_ACCESS_KEY="your_key"
uv run scripts/fetch_natural_photos.py
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
