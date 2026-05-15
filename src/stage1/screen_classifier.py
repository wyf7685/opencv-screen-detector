"""屏幕内容分类器 - 综合判断图像是否包含屏幕内容"""

import numpy as np


def classify_screen_content(features: dict[str, float]) -> float:
    """
    根据UI特征综合判断屏幕内容概率

    Args:
        features: 包含UI特征的字典
            - text_density: 文本密度
            - ui_line_density: UI线条密度
            - rectangle_score: 矩形检测分数

    Returns:
        屏幕内容概率 (0.0-1.0)
    """
    text_density = features.get("text_density", 0.0)
    ui_line_density = features.get("ui_line_density", 0.0)
    rectangle_score = features.get("rectangle_score", 0.0)

    # 加权组合
    # 文本密度权重最高，因为UI通常包含大量文本
    # ui_line_density权重极低，因为几乎所有图像都有大量水平/垂直线
    weights = {
        "text_density": 0.60,
        "ui_line_density": 0.05,
        "rectangle_score": 0.35,
    }

    screen_probability = (
        text_density * weights["text_density"]
        + ui_line_density * weights["ui_line_density"]
        + rectangle_score * weights["rectangle_score"]
    )

    # 应用非线性变换，增强区分度
    # 使用sigmoid-like函数
    screen_probability = _sigmoid(screen_probability, center=0.3, scale=5.0)

    return min(max(screen_probability, 0.0), 1.0)


def _sigmoid(x: float, center: float = 0.5, scale: float = 1.0) -> float:
    """
    Sigmoid变换函数

    Args:
        x: 输入值
        center: 中心点
        scale: 缩放因子

    Returns:
        变换后的值
    """
    return 1.0 / (1.0 + np.exp(-scale * (x - center)))
