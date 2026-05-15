"""反光检测模块 - 检测摄像头拍摄产生的反光"""

import cv2
import numpy as np


def analyze_reflection(image: np.ndarray) -> float:
    """
    检测图像中的反光

    拍摄屏幕时会有光斑，截图不会。

    Args:
        image: BGR格式的输入图像

    Returns:
        反光分数 (0.0-1.0)，值越高表示反光越明显
    """
    if image is None or image.size == 0:
        return 0.0

    # 转换为HSV格式
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        # 灰度图无法检测反光
        return 0.0

    # 1. 检测高亮低纹理区域
    highlight_score = _detect_highlight_regions(hsv)

    # 2. 检测光斑
    flare_score = _detect_flare_regions(hsv)

    # 3. 检测镜面反射
    specular_score = _detect_specular_reflection(hsv)

    # 综合评分
    reflection_score = highlight_score * 0.3 + flare_score * 0.4 + specular_score * 0.3

    return min(max(reflection_score, 0.0), 1.0)


def _detect_highlight_regions(hsv: np.ndarray) -> float:
    """
    检测高亮低纹理区域

    Args:
        hsv: HSV格式图像

    Returns:
        高亮区域分数 (0.0-1.0)
    """
    # 提取亮度通道
    v = hsv[:, :, 2].astype(np.float64)

    # 检测高亮区域（V > 220，提高阈值）
    highlight_mask = v > 220

    if np.sum(highlight_mask) == 0:
        return 0.0

    # 计算高亮区域的纹理
    # 使用局部方差作为纹理度量
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    kernel = kernel / (kernel_size * kernel_size)

    # 计算局部均值
    mean = cv2.filter2D(v, -1, kernel)

    # 计算局部方差
    sqr_mean = cv2.filter2D(v**2, -1, kernel)
    variance = sqr_mean - mean**2

    # 高亮区域的低纹理特征
    highlight_variance = variance[highlight_mask]
    avg_variance = np.mean(highlight_variance) if highlight_variance.size > 0 else 0.0

    # 低方差表示可能是反光
    texture_score = 1.0 - min(avg_variance / 500.0, 1.0)

    # 结合高亮区域比例
    highlight_ratio = np.sum(highlight_mask) / highlight_mask.size

    # 降低高亮区域比例的影响
    return texture_score * 0.7 + highlight_ratio * 0.3


def _detect_flare_regions(hsv: np.ndarray) -> float:
    """
    检测光斑区域

    Args:
        hsv: HSV格式图像

    Returns:
        光斑分数 (0.0-1.0)
    """
    # 提取各通道
    s = hsv[:, :, 1].astype(np.float64)
    v = hsv[:, :, 2].astype(np.float64)

    # 光斑特征：高亮度、低饱和度
    # 提高阈值，减少误检测
    flare_mask = (v > 200) & (s < 30)

    if np.sum(flare_mask) == 0:
        return 0.0

    # 计算光斑区域的连通性
    flare_mask_uint8 = flare_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        flare_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 统计光斑区域
    flare_count = 0
    flare_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:  # 提高面积阈值，过滤噪声
            flare_count += 1
            flare_area += area

    # 计算光斑分数
    total_area = hsv.shape[0] * hsv.shape[1]
    flare_ratio = flare_area / total_area

    # 光斑通常较少但较大
    return min(flare_count / 3.0, 1.0) * 0.5 + min(flare_ratio * 5, 1.0) * 0.5


def _detect_specular_reflection(hsv: np.ndarray) -> float:
    """
    检测镜面反射

    Args:
        hsv: HSV格式图像

    Returns:
        镜面反射分数 (0.0-1.0)
    """
    # 提取各通道
    s = hsv[:, :, 1].astype(np.float64)
    v = hsv[:, :, 2].astype(np.float64)

    # 镜面反射特征：极高亮度、极低饱和度
    # 提高阈值，减少误检测
    specular_mask = (v > 250) & (s < 20)

    if np.sum(specular_mask) == 0:
        return 0.0

    # 计算镜面反射区域的连续性
    specular_mask_uint8 = specular_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        specular_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 统计镜面反射区域
    specular_count = 0
    specular_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # 提高面积阈值，过滤噪声
            specular_count += 1
            specular_area += area

    # 计算镜面反射分数
    total_area = hsv.shape[0] * hsv.shape[1]
    specular_ratio = specular_area / total_area

    # 镜面反射通常很小但很亮
    return min(specular_count / 2.0, 1.0) * 0.5 + min(specular_ratio * 10, 1.0) * 0.5
