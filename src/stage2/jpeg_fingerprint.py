"""JPEG指纹检测模块 - 分析JPEG量化表指纹"""

import cv2
import numpy as np


def analyze_jpeg_fingerprint(image: np.ndarray) -> float | np.floating:
    """
    分析JPEG量化表指纹

    相机有固定JPEG量化表，截图可能是PNG/WebP或软件压缩，分布不同。

    Args:
        image: BGR格式的输入图像

    Returns:
        JPEG指纹分数 (0.0-1.0)，值越高表示越可能是相机JPEG
    """
    if image is None or image.size == 0:
        return 0.0

    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. 分析DCT块直方图
    dct_score = _analyze_dct_histogram(gray)

    # 2. 分析JPEG压缩伪影
    artifact_score = _analyze_jpeg_artifacts(gray)

    # 3. 分析量化表特征
    quantization_score = _analyze_quantization_pattern(gray)

    # 综合评分
    jpeg_score = dct_score * 0.4 + artifact_score * 0.3 + quantization_score * 0.3

    return min(max(jpeg_score, 0.0), 1.0)


def _analyze_dct_histogram(gray: np.ndarray) -> float | np.floating:
    """
    分析DCT块直方图 - JPEG压缩会在DCT域留下特征

    Args:
        gray: 灰度图像

    Returns:
        DCT直方图分数 (0.0-1.0)
    """
    # 将图像分成8x8块（JPEG标准块大小）
    h, w = gray.shape
    block_size = 8

    dct_coefficients = []

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray[y : y + block_size, x : x + block_size].astype(np.float64)

            # 应用DCT
            dct_block = cv2.dct(block)

            # 收集DCT系数
            dct_coefficients.append(dct_block.flatten())

    if not dct_coefficients:
        return 0.0

    # 计算DCT系数的统计特征
    dct_array = np.array(dct_coefficients)

    # 计算各频率分量的方差
    dct_variance = np.var(dct_array, axis=0)

    # JPEG压缩会导致高频分量方差较小
    low_freq_var = np.mean(dct_variance[:10])  # 低频
    high_freq_var = np.mean(dct_variance[30:])  # 高频

    # 计算高频/低频方差比
    freq_ratio = high_freq_var / low_freq_var if low_freq_var > 0 else 0.0

    # JPEG压缩会使这个比值较小
    return 1.0 - min(freq_ratio, 1.0)


def _analyze_jpeg_artifacts(gray: np.ndarray) -> float | np.floating:
    """
    分析JPEG压缩伪影

    Args:
        gray: 灰度图像

    Returns:
        JPEG伪影分数 (0.0-1.0)
    """
    # 检测8x8块边界
    h, w = gray.shape
    block_size = 8

    # 计算块边界处的梯度变化
    boundary_diffs = []

    for y in range(block_size, h - block_size, block_size):
        # 水平边界
        upper = gray[y - 1, :].astype(np.float64)
        lower = gray[y, :].astype(np.float64)
        boundary_diff = np.mean(np.abs(upper - lower))
        boundary_diffs.append(boundary_diff)

    for x in range(block_size, w - block_size, block_size):
        # 垂直边界
        left = gray[:, x - 1].astype(np.float64)
        right = gray[:, x].astype(np.float64)
        boundary_diff = np.mean(np.abs(left - right))
        boundary_diffs.append(boundary_diff)

    if not boundary_diffs:
        return 0.0

    # 计算边界处的梯度均值
    mean_boundary_diff = np.mean(boundary_diffs)

    # JPEG压缩会在块边界处产生特定的梯度模式
    return min(mean_boundary_diff / 10.0, 1.0)


def _analyze_quantization_pattern(gray: np.ndarray) -> float | np.floating:
    """
    分析量化表特征

    Args:
        gray: 灰度图像

    Returns:
        量化表特征分数 (0.0-1.0)
    """
    # 将图像分成8x8块
    h, w = gray.shape
    block_size = 8

    # 收集各块的统计特征
    block_means = []
    block_vars = []

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray[y : y + block_size, x : x + block_size].astype(np.float64)
            block_means.append(np.mean(block))
            block_vars.append(np.var(block))

    if not block_means:
        return 0.0

    # 计算块间一致性
    mean_variance = np.var(block_means)

    # JPEG压缩会导致块间统计特征更一致
    return 1.0 - min(mean_variance / 1000.0, 1.0)
