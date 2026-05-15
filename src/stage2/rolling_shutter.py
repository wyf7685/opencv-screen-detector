"""滚动快门检测模块 - 检测CMOS滚动快门效应"""

import cv2
import numpy as np


def analyze_rolling_shutter(image: np.ndarray) -> float:
    """
    检测图像中的滚动快门效应

    CMOS滚动快门在拍摄运动物体或快速变化的屏幕时会产生倾斜。

    Args:
        image: BGR格式的输入图像

    Returns:
        滚动快门分数 (0.0-1.0)，值越高表示滚动快门效应越明显
    """
    if image is None or image.size == 0:
        return 0.0

    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. 检测水平线倾斜
    tilt_score = _detect_horizontal_line_tilt(gray)

    # 2. 检测运动模糊方向性
    motion_score = _detect_directional_motion_blur(gray)

    # 3. 检测扫描线伪影
    scanline_score = _detect_scanline_artifacts(gray)

    # 综合评分
    rolling_shutter_score = tilt_score * 0.4 + motion_score * 0.3 + scanline_score * 0.3

    return min(max(rolling_shutter_score, 0.0), 1.0)


def _detect_horizontal_line_tilt(gray: np.ndarray) -> float:
    """
    检测水平线倾斜 - 滚动快门会导致水平线倾斜

    Args:
        gray: 灰度图像

    Returns:
        水平线倾斜分数 (0.0-1.0)
    """
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # HoughLinesP检测直线
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10,
    )

    if lines is None:
        return 0.0

    # 统计接近水平的线
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # 接近水平的线（角度接近0或180）
        if angle < 20 or angle > 160:
            horizontal_lines.append(angle)

    if not horizontal_lines:
        return 0.0

    # 计算水平线的角度方差
    # 滚动快门会导致水平线有轻微倾斜
    angle_variance = np.var(horizontal_lines)

    # 归一化
    max_variance = 100.0
    return min(angle_variance / max_variance, 1.0)


def _detect_directional_motion_blur(gray: np.ndarray) -> float:
    """
    检测运动模糊方向性

    Args:
        gray: 灰度图像

    Returns:
        运动模糊方向性分数 (0.0-1.0)
    """
    # 计算水平和垂直方向的梯度
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度方向
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    # 计算梯度强度
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 分析梯度方向的一致性
    # 滚动快门会导致特定方向的模糊
    h, w = gradient_direction.shape

    # 将图像分成小块
    block_size = 32
    direction_consistencies = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block_direction = gradient_direction[y : y + block_size, x : x + block_size]
            block_magnitude = gradient_magnitude[y : y + block_size, x : x + block_size]

            # 只考虑有显著梯度的区域
            if np.mean(block_magnitude) > 10:
                # 计算方向的一致性
                direction_std = np.std(block_direction)
                direction_consistencies.append(direction_std)

    if not direction_consistencies:
        return 0.0

    # 低标准差表示方向一致（可能是滚动快门）
    avg_consistency = np.mean(direction_consistencies)
    return 1.0 - min(avg_consistency / np.pi, 1.0)


def _detect_scanline_artifacts(gray: np.ndarray) -> float:
    """
    检测扫描线伪影

    Args:
        gray: 灰度图像

    Returns:
        扫描线伪影分数 (0.0-1.0)
    """
    # 计算水平方向的梯度
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算每行的平均梯度
    row_gradients = np.mean(np.abs(sobel_y), axis=1)

    # 计算梯度的周期性
    # 使用自相关检测周期性
    if len(row_gradients) < 10:
        return 0.0

    # 计算自相关
    correlation = np.correlate(row_gradients, row_gradients, mode="full")
    correlation = correlation[len(correlation) // 2 :]

    # 检测周期性峰值
    peaks = [
        i
        for i in range(2, len(correlation) - 2)
        if (
            correlation[i] > correlation[i - 1]
            and correlation[i] > correlation[i + 1]
            and correlation[i] > np.mean(correlation) + np.std(correlation)
        )
    ]

    # 滚动快门会导致特定周期的扫描线
    if not peaks:
        return 0.0

    # 计算周期的一致性
    if len(peaks) > 1:
        peak_diffs = np.diff(peaks)
        mean_diff = np.mean(peak_diffs)
        period_consistency = (
            1.0 - np.std(peak_diffs) / mean_diff if mean_diff > 0 else 0.0
        )
    else:
        period_consistency = 0.5

    return period_consistency
