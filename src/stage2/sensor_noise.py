"""CMOS传感器噪声检测模块 - 检测摄像头传感器噪声"""

import cv2
import numpy as np


def analyze_sensor_noise(image: np.ndarray) -> float:
    """
    检测图像中的CMOS传感器噪声

    摄像头拍摄存在传感器随机噪声，截图不存在。

    Args:
        image: BGR格式的输入图像

    Returns:
        传感器噪声分数 (0.0-1.0)，值越高表示噪声越明显
    """
    if image is None or image.size == 0:
        return 0.0

    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 使用高斯模糊检测噪声（与旧算法一致）
    gray_float = gray.astype(float)
    low_pass = cv2.GaussianBlur(gray_float, (9, 9), 0)
    residual = gray_float - low_pass

    # 创建边缘掩码（排除边缘区域）
    edge_mask = cv2.Canny(gray, 50, 150) == 0
    if not np.any(edge_mask):
        edge_mask = np.ones_like(gray, dtype=bool)

    # 计算噪声
    noise = residual[edge_mask]
    if noise.size == 0:
        return 0.0

    # 计算噪声标准差并归一化
    score = float(np.std(noise)) / 18.0
    return max(0.0, min(1.0, score))
