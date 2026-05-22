# Screen Detector V3 - Trainer

两阶段 CNN + FFT Branch 训练系统。

## 功能

- 两阶段独立训练 (Stage 1: natural vs screen_like, Stage 2: screenshot vs screen_photo)
- EfficientNet-B0 + FFT Frequency Branch 融合模型
- Mixed Precision Training (AMP)
- 数据增强 (albumentations)
- ONNX 双模型导出 (dynamic_axes)

## 快速开始

```bash
uv sync --group train

# 训练两个阶段
uv run python -m trainer.train

# 导出 ONNX 模型
uv run python -m trainer.export_onnx
```

## 目录结构

```
trainer/
├── README.md
├── pyproject.toml
├── config.py           # 训练配置 (数据映射/超参数)
├── model.py            # 融合模型 (EfficientNet + FFT Branch)
├── fft_branch.py       # Frequency Branch (ResBlock)
├── dataset.py          # 双输入数据集 (RGB + FFT)
├── train.py            # 两阶段训练 (AMP)
├── validate.py         # 验证指标 (accuracy/precision/recall/f1/fpr)
├── augment.py          # 数据增强
└── export_onnx.py      # ONNX 导出 (双输入 dynamic_axes)
```

## 数据要求

```
data/input/
├── natural_photo/      # 自然照片 (含子目录递归扫描)
├── screen_like/        # 屏幕内容 (截图/PPT/IDE)
├── screenshot/         # 截图
├── screen_photo/       # 拍屏照片
└── hard_negative/      # 难例负样本 (UI 海报/深色模式/动漫 UI)
```

**最低数据量要求**:
- Stage 1: natural 3000+, screen_like 3000+
- Stage 2: screenshot 2000+, screen_photo 2000+

## 两阶段训练

### Stage 1: natural vs screen_like

```
natural_photo/  →  "natural"
screen_like/    →  "screen_like"
screenshot/     →  "screen_like"
hard_negative/  →  "screen_like"
```

### Stage 2: screenshot vs screen_photo

```
screenshot/     →  "screenshot"
screen_photo/   →  "screen_photo"
```

## 模型架构

```
Input Image
    ├─→ EfficientNet-B0 (spatial) → LayerNorm(1280)
    └─→ FFT → FrequencyBranch (ResBlock×2) → LayerNorm(256)
                    ↓
            Concat (1536)
                    ↓
        Dropout→Linear(1536,512)→ReLU→Dropout→Linear(512,2)
```

## 配置

`src/config.py` 中的关键配置:

- `IMAGE_SIZE = 224` - 输入尺寸
- `BATCH_SIZE = 32` - 批次大小
- `LEARNING_RATE = 1e-4` - 学习率
- `NUM_EPOCHS = 30` - 训练轮数
- `TRAIN_VAL_SPLIT = 0.8` - 训练/验证比例
- `STAGE1_DATA_MAP` / `STAGE2_DATA_MAP` - 数据映射

## 输出

训练完成后在 `trainer/` 目录生成:
- `stage1_best.pth` / `stage2_best.pth` - PyTorch 权重
- `stage1_natural_vs_screenlike.onnx` - Stage 1 ONNX 模型
- `stage2_screenlike_vs_screenphoto.onnx` - Stage 2 ONNX 模型

将 ONNX 模型复制到 `inference/models/` 即可用于推理。

## 环境要求

- Python >= 3.11, < 3.13
- CUDA 支持的 GPU（推荐）
- 至少 4GB GPU 显存

## 依赖

- torch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- albumentations >= 1.3.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- onnx >= 1.14.0
- onnxruntime >= 1.16.0
