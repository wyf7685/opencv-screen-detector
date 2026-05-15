# opencv-screen-detector

基于 Python + OpenCV + Image Forensics 的图像来源识别系统。通过成像链特征分析，区分系统截图（`screenshot`）、手机拍摄屏幕（`screen_photo`）和普通照片（`normal_photo`）。

## 功能

- FastAPI REST API，支持通过图片 URL 远程检测
- 支持单张图片和文件夹批量检测
- 三分类：`screenshot`（系统截图）、`screen_photo`（手机拍屏）、`normal_photo`（普通照片）
- 18维图像特征提取（透视畸变、CMOS噪声、摩尔纹、反光等）
- LightGBM 三分类模型，准确率达 100%
- JSON 规则引擎（17条规则）作为备用方案
- 输出 JSON 格式检测结果，包含各特征分数

## 快速开始

```bash
# 安装依赖
uv sync

# 启动 API 服务
uv run main.py

# 测试检测接口
curl -X POST http://localhost:8325/api/detect \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/test.jpg"}'
# 返回: {"is_screen": true} 或 {"is_screen": false}

# 批量检测（读取 data/input/，输出 data/output/result.json）
uv run python test/test_batch_detect.py

# 训练模型
uv run python -c "from src.ml.train import main; main()"

# 运行测试
uv run python -m pytest test/

# 检查准确率
uv run python test/test_accuracy.py

# 代码检查和格式化
ruff check src/ main.py test/
ruff format src/ main.py test/
```

## 项目结构

```
├── main.py                    # FastAPI 入口
├── src/
│   ├── main.py                # 批量检测流程编排
│   ├── detector.py            # ScreenDetector 核心检测器
│   ├── api/                   # FastAPI API 模块
│   │   └── detect.py          # POST /api/detect 端点
│   ├── preprocess.py          # 图像预处理
│   ├── feature/               # 18个特征提取模块
│   │   ├── frequency.py       # 频域特征
│   │   ├── banding.py         # 条带伪影
│   │   ├── blackscreen.py     # 黑屏检测
│   │   ├── chroma.py          # 色度分析
│   │   ├── softness.py        # 图像模糊度
│   │   ├── illumination.py    # 光照分布
│   │   ├── artifact.py        # 压缩伪影
│   │   ├── rectangle.py       # 矩形检测
│   │   ├── display_content.py # 显示内容
│   │   ├── overexposed.py     # 过曝检测
│   │   ├── perspective.py     # 透视畸变
│   │   ├── moire.py           # 摩尔纹
│   │   ├── reflection.py      # 反光
│   │   ├── sensor_noise.py    # 传感器噪声
│   │   ├── subpixel_fringing.py # 亚像素边缘
│   │   └── color_noise.py     # 彩色噪声
│   ├── ml/                    # 机器学习模块
│   │   ├── train.py           # LightGBM模型训练
│   │   └── predict.py         # 模型预测
│   ├── scoring/               # 评分系统
│   │   ├── rules.py           # JSON规则引擎
│   │   └── ml_model.py        # ML模型封装
│   └── utils/                 # 图像 I/O、EXIF、JSON 导出
├── config/
│   └── rules.json             # 规则配置文件
├── test/                      # 单元测试和集成测试
├── data/
│   ├── input/                 # 测试图片
│   │   ├── img/               # 系统截图 (42张)
│   │   ├── no_screen/         # 普通照片 (11张)
│   │   └── photo/             # 屏幕拍照 (48张)
│   ├── output/                # 检测结果
│   └── model/                 # 训练好的模型
│       └── screen_detector.pkl
└── LIGHTGBM_MODEL.md          # 模型详细文档
```

## 检测准确率

| 类别 | 准确率 | 目标 | 达标 |
|------|--------|------|------|
| normal_photo | 100% (11/11) | >99% | ✓ |
| screenshot | 100% (42/42) | >95% | ✓ |
| screen_photo | 100% (48/48) | >95% | ✓ |
| **总体** | **100% (101/101)** | - | ✓ |

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

## 输出格式

```json
{
  "filename": "test.jpg",
  "score": 0.9999,
  "result": "screenshot",
  "model_probability": 0.9999,
  "rule_score": 0.2165,
  "features": {
    "banding": 0.895,
    "frequency": 0.111,
    "softness": 0.896,
    "sensor_noise": 0.573,
    "moire": 1.0,
    "perspective": 0.580
  }
}
```

## 依赖

- Python 3.14
- opencv-python >= 4.10.0
- numpy >= 2.0.0
- pillow >= 11.0.0
- lightgbm >= 4.0.0
- scikit-learn >= 1.4.0
- fastapi >= 0.115.0
- uvicorn >= 0.34.0
- httpx >= 0.28.0
- 构建系统：hatchling
- 包管理：uv

## 技术路线

本项目采用图像取证（Image Forensics）技术路线，从内容检测转向成像链检测：

1. **V1**: OpenCV + LightGBM（当前版本）
2. **V2**: CNN（EfficientNet）端到端检测
3. **V3**: 法医级检测（PRNU、CFA、JPEG fingerprint）

## 许可证

MIT License
