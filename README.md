# opencv-screen-detector

基于 Python + OpenCV 的屏幕拍照检测系统。通过图像特征分析，区分摄像头拍摄的屏幕照片（`screen_photo`）和普通图片（`screen_photo` vs `normal`）。

## 功能

- 支持单张图片和文件夹批量检测
- 17 个图像特征提取（显示内容、文件格式、色彩噪声、传感器噪声等）
- 加权评分系统，自动分类为 `screen_photo` 或 `normal`
- 输出 JSON 格式检测结果，包含各特征分数

## 快速开始

```bash
# 安装依赖
uv sync

# 运行检测（读取 data/input/，输出 data/output/result.json）
uv run main.py

# 运行测试
uv run python -m unittest discover test/
```

## 项目结构

```
├── main.py                    # 入口
├── src/
│   ├── main.py                # 流程编排
│   ├── detector.py            # ScreenDetector 检测器
│   ├── preprocess.py          # 图像预处理（灰度 + 高斯模糊）
│   ├── feature/               # 特征提取模块
│   │   ├── display_content.py # 边缘/线条密度（权重最高）
│   │   ├── format_score.py    # PNG/JPEG 实际格式检测
│   │   ├── color_noise.py     # HSV 色彩噪声
│   │   ├── sensor_noise.py    # CMOS 传感器噪声
│   │   ├── perspective.py     # 边缘锐度
│   │   ├── moire.py           # JPEG 块效应
│   │   └── ...                # 其他特征
│   ├── scoring/
│   │   ├── rules.py           # 加权评分 + 阈值分类
│   │   └── ml_model.py        # ML 模型（占位符）
│   └── utils/                 # 图像 I/O、EXIF、JSON 导出
├── test/                      # 单元测试和集成测试
└── data/
    ├── input/                 # 测试图片
    │   ├── img/               # 截图 + 普通图片（期望：normal）
    │   ├── photo/             # 屏幕拍照（期望：screen_photo）
    │   └── no_screen/         # 无关图片（期望：normal）
    └── output/                # 检测结果
```

## 评分机制

检测器提取图像特征后，通过加权求和计算分数，与阈值（0.25）比较进行分类。权重通过 LDA（线性判别分析）从数据中自动学习得到：

| 特征 | 权重 | 说明 |
|------|------|------|
| sensor_noise | +0.215 | 高频残差噪声（屏幕照片传感器噪声更高）— 最强正向特征 |
| softness | +0.181 | 图像模糊度（屏幕拍照更模糊） |
| illumination | +0.131 | 光照分布 |
| rectangle | +0.111 | 矩形检测 |
| format_score | +0.088 | 实际图片格式（PNG=0, JPEG=0.5） |
| exif_camera | +0.054 | EXIF 相机信息 |
| banding | +0.048 | 条纹检测 |
| color_noise | +0.020 | HSV 饱和度噪声 |
| frequency | -0.251 | 频域特征（截屏频率特征更强）— 最强负向特征 |
| display_content | -0.116 | 边缘密度（截屏边缘密度更高） |
| artifact | -0.081 | JPEG 块效应（截屏压缩伪影更明显） |
| moire | -0.066 | 摩尔纹 |
| chroma | -0.055 | 色度特征 |
| perspective | -0.013 | 边缘锐度 |
| subpixel_fringing | -0.009 | 亚像素边缘 |
| reflection | -0.008 | 反射/高光区域 |
| overexposed | -0.002 | 过曝区域 |

核心逻辑：屏幕拍照的传感器噪声（sensor_noise）和模糊度（softness）显著高于截屏，而截屏的频域特征（frequency）和边缘密度（display_content）更高。LDA 通过这些特征组合实现分类。

## 检测效果

在测试数据集上（49 张图片）的检测准确率为 **49/49 (100%)**：

- `img/`（截图 + 普通图片）：16/16 正确
- `photo/`（屏幕拍照）：29/29 正确
- `no_screen/`（无关图片）：4/4 正确

## 依赖

- Python 3.14
- opencv-python >= 4.10.0
- numpy >= 2.0.0
- pillow >= 11.0.0
- 构建系统：hatchling
- 包管理：uv
