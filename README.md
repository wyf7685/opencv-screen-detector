# opencv-screen-detector

基于 Python + OpenCV 的屏幕拍照检测系统。通过图像特征分析，区分摄像头拍摄的屏幕照片（`screen_photo`）和普通图片（`normal`）。

## 功能

- 支持单张图片和文件夹批量检测
- 18 个图像特征提取（显示内容、文件格式、色彩噪声、传感器噪声等）
- 加权评分系统 + 后处理规则，自动分类为 `screen_photo` 或 `normal`
- 输出 JSON 格式检测结果，包含各特征分数

## 快速开始

```bash
# 安装依赖
uv sync

# 运行检测（读取 data/input/，输出 data/output/result.json）
uv run main.py

# 检测准确率
uv run python test/test_accuracy.py

# 运行测试
uv run python -m pytest test/

# 代码检查和格式化
ruff check src/ test/
ruff format src/ test/

# 参数优化（网格搜索最优权重）
uv run python test/test_parameter_optimization.py
```

## 项目结构

```
├── main.py                    # 入口
├── src/
│   ├── main.py                # 流程编排
│   ├── detector.py            # ScreenDetector 检测器
│   ├── preprocess.py          # 图像预处理（灰度 + 高斯模糊）
│   ├── feature/               # 特征提取模块
│   │   ├── sensor_noise.py    # CMOS 传感器噪声
│   │   ├── softness.py        # 图像模糊度
│   │   ├── display_content.py # 边缘/线条密度
│   │   ├── blackscreen.py     # 黑屏检测
│   │   ├── moire.py           # 摩尔纹
│   │   ├── artifact.py        # JPEG 块效应
│   │   └── ...                # 其他特征
│   ├── scoring/
│   │   ├── rules.py           # 加权评分 + 阈值分类 + 后处理规则
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

检测器提取图像特征后，通过加权求和计算分数，与阈值（0.23）比较进行分类。针对边界情况设有后处理规则：

| 规则 | 条件 | 调整 | 说明 |
|------|------|------|------|
| Rule 1 | sensor>0.95, softness<0.74, artifact<0.10, moire>0.80 | score -= 0.12 | UI 截图的异常高传感器噪声 |
| Rule 2 | moire>0.95, softness>0.95, blackscreen>0.50 | score += 0.06 | 黑屏照片 |
| Rule 3 | softness>0.90, moire>0.95, artifact<0.08, rectangle>0.10 | score -= 0.06 | 干净截图模拟屏幕拍照 |
| Rule 4 | softness>0.80, moire>0.90, artifact<0.10, rectangle>0.15 | score -= 0.08 | 中等软度+高摩尔纹+低伪影 |

**特征权重：**

| 特征 | 权重 | 说明 |
|------|------|------|
| softness | +0.181 | 图像模糊度（屏幕拍照更模糊） |
| sensor_noise | +0.150 | CMOS 传感器噪声（屏幕照片噪声更高） |
| illumination | +0.131 | 光照分布 |
| rectangle | +0.111 | 矩形检测 |
| format_score | +0.088 | 实际图片格式（PNG=0, JPEG=0.5） |
| exif_camera | +0.054 | EXIF 相机信息 |
| banding | +0.048 | 条纹检测 |
| blackscreen | +0.030 | 黑屏检测 |
| color_noise | +0.020 | HSV 饱和度噪声 |
| frequency | -0.150 | 频域特征（截屏频率特征更强） |
| display_content | -0.116 | 边缘密度（截屏边缘密度更高） |
| artifact | -0.081 | JPEG 块效应（截屏压缩伪影更明显） |
| moire | -0.060 | 摩尔纹 |
| chroma | -0.055 | 色度特征 |
| perspective | -0.013 | 边缘锐度 |
| subpixel_fringing | -0.009 | 亚像素边缘 |
| reflection | -0.008 | 反射/高光区域 |
| overexposed | -0.002 | 过曝区域 |

核心逻辑：屏幕拍照的模糊度（softness）和传感器噪声（sensor_noise）显著高于截屏，而截屏的频域特征（frequency）和边缘密度（display_content）更高。

## 检测效果

在测试数据集上（74 张图片）的检测准确率为 **74/74 (100%)**：

- `img/`（截图 + 普通图片）：正确识别
- `photo/`（屏幕拍照）：正确识别
- `no_screen/`（无关图片）：正确识别

### 已成功识别的拍摄场景

以下场景的手机拍屏图片均可被正确检测：

- 站在电脑屏幕上俯瞰的手机拍摄
- 侧着电脑屏幕旁边的手机拍摄
- 趴在床上仰视电脑屏幕的手机拍摄
- **关闭电脑后打开手机闪光灯的手机拍摄**
- 用苹果手机拍摄的电脑屏幕图片
- 只拍一点点电脑屏幕的手机拍摄
- 各种游戏截图（如 Slay the Spire2,endfeild,Escape the Tarkov）

## 依赖

- Python 3.14
- opencv-python >= 4.10.0
- numpy >= 2.0.0
- pillow >= 11.0.0
- 构建系统：hatchling
- 包管理：uv

## 注意

本项目还在初始阶段，欢迎大家提供一些手机拍屏图片来优化和改进项目。
