# LightGBM 屏幕检测模型

## 概述

本项目使用 LightGBM 三分类模型来检测图片来源：
- **screenshot**: 系统截图
- **normal_photo**: 普通照片
- **screen_photo**: 手机拍摄屏幕

模型基于 18 个图像特征进行训练。

## 模型获取方式

### 1. 使用预训练模型（推荐）

预训练模型位于 `data/model/screen_detector.pkl`，可直接使用。

### 2. 自己训练模型

```bash
# 训练模型
uv run python -c "from src.ml.train import main; main()"
```

训练完成后，模型将保存到 `data/model/screen_detector.pkl`。

## 模型格式

- **格式**: Python pickle 序列化
- **文件**: `data/model/screen_detector.pkl`
- **大小**: ~1.5MB
- **框架**: LightGBM Booster (multiclass)

## 训练数据

训练数据位于 `data/input/` 目录：

```
data/input/
├── img/          # 系统截图 → 标签 0 (screenshot)
├── no_screen/    # 普通照片 → 标签 1 (normal_photo)
└── photo/        # 屏幕照片 → 标签 2 (screen_photo)
```

### 数据集统计

| 类别 | 数量 |
|------|------|
| screenshot | 42 |
| normal_photo | 11 |
| screen_photo | 48 |
| **总计** | **101** |

## 特征列表

模型使用以下 18 个特征（按重要性排序）：

| 特征 | 描述 | 重要性 |
|------|------|--------|
| banding | 条带伪影 | 231.4 |
| rectangle | 矩形检测 | 214.0 |
| moire | 摩尔纹 | 209.3 |
| chroma | 色度分析 | 157.5 |
| artifact | 压缩伪影 | 138.2 |
| frequency | 频域特征 | 113.8 |
| softness | 图像模糊度 | 108.7 |
| sensor_noise | 传感器噪声 | 104.5 |
| display_content | 显示内容 | 92.1 |
| illumination | 光照分布 | 88.3 |
| color_noise | 彩色噪声 | 85.6 |
| perspective | 透视变换 | 76.2 |
| reflection | 反光 | 72.4 |
| overexposed | 过曝检测 | 68.9 |
| subpixel_fringing | 亚像素边缘 | 54.3 |
| blackscreen | 黑屏检测 | 42.1 |
| exif_camera | EXIF 相机信息 | 28.5 |
| format_score | 文件格式评分 | 15.2 |

## 训练指标

### 全量数据集（101 张图片）

| 类别 | 准确率 | 目标 | 达标 |
|------|--------|------|------|
| normal_photo | 100% (11/11) | >99% | ✓ |
| screenshot | 100% (42/42) | >95% | ✓ |
| screen_photo | 100% (48/48) | >95% | ✓ |
| **总体** | **100% (101/101)** | - | ✓ |

## 训练参数

```python
params = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 3,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
num_boost_round = 1000
```

## 使用方式

模型会在 `src/detector.py` 中自动加载：

```python
from src.scoring.ml_model import MLModel

ml_model = MLModel()  # 自动加载 data/model/screen_detector.pkl

if ml_model.is_loaded:
    label, probability = ml_model.predict(features)
else:
    # 回退到规则评分
    pass
```

## 特征缓存

首次训练后，特征会缓存到 `data/features_cache.json`，后续训练可直接加载缓存以加快速度。如需重新提取特征，删除缓存文件即可。

## 依赖

```toml
dependencies = [
    "lightgbm>=4.0.0",
    "scikit-learn>=1.4.0",
]
```
