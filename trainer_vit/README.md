# ViT Training Pipeline

Vision Transformer (ViT-B/16) 训练管道，用于与现有 CNN+FFT Branch 模型进行对比实验。

## 架构

```
Dataset (三分类: natural, screenshot, screen_photo)
    │
    ▼
ViT-B/16 (Transfer Learning)
    │
    ▼
Accuracy/Precision/Recall/F1 对比实验
```

## 使用方法

### 安装依赖

```bash
cd trainer_vit
uv sync
```

### 训练模型

```bash
uv run python -m src.train
```

### 验证模型

```bash
uv run python -m src.validate
```

### 导出ONNX

```bash
uv run python -m src.export_onnx
```

## 输出

- `outputs/vit_checkpoint.pth` - 模型checkpoint
- `outputs/metrics.json` - 评估指标
- `outputs/vit_model.onnx` - ONNX模型
