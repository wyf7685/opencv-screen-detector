"""亚像素彩边检测模块 - 检测LCD屏幕的RGB亚像素彩边"""

import cv2
import numpy as np


def analyze_subpixel_fringing(image: np.ndarray) -> float:
    """
    检测图像中的亚像素彩边

    手机拍LCD时会出现RGB亚像素彩边。

    Args:
        image: BGR格式的输入图像

    Returns:
        亚像素彩边分数 (0.0-1.0)，值越高表示彩边越明显
    """
    if image is None or image.size == 0:
        return 0.0

    # 转换为BGR格式（如果不是）
    if len(image.shape) == 2:
        # 灰度图无法检测彩边
        return 0.0

    # 1. 检测文字边缘的RGB偏移
    text_edge_score = _detect_text_edge_chromatic_aberration(image)

    # 2. 检测边缘颜色分离
    edge_color_score = _detect_edge_color_separation(image)

    # 3. 检测亚像素结构
    subpixel_structure_score = _detect_subpixel_structure(image)

    # 综合评分
    subpixel_score = (
        text_edge_score * 0.4 + edge_color_score * 0.3 + subpixel_structure_score * 0.3
    )

    return min(max(subpixel_score, 0.0), 1.0)


def _detect_text_edge_chromatic_aberration(image: np.ndarray) -> float:
    """
    检测文字边缘的色差

    Args:
        image: BGR格式图像

    Returns:
        文字边缘色差分数 (0.0-1.0)
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测边缘
    edges = cv2.Canny(gray, 50, 150)

    # 提取各颜色通道
    b, g, r = cv2.split(image.astype(np.float64))

    # 在边缘位置检测颜色差异
    edge_mask = edges > 0

    if np.sum(edge_mask) == 0:
        return 0.0

    # 计算边缘处的颜色差异
    r_edge = r[edge_mask]
    g_edge = g[edge_mask]
    b_edge = b[edge_mask]

    # 计算通道间差异
    r_g_diff = np.abs(r_edge - g_edge)
    r_b_diff = np.abs(r_edge - b_edge)
    g_b_diff = np.abs(g_edge - b_edge)

    avg_diff = np.mean([np.mean(r_g_diff), np.mean(r_b_diff), np.mean(g_b_diff)])

    # 归一化
    max_diff = 50.0
    return min(avg_diff / max_diff, 1.0)


def _detect_edge_color_separation(image: np.ndarray) -> float:
    """
    检测边缘颜色分离

    Args:
        image: BGR格式图像

    Returns:
        边缘颜色分离分数 (0.0-1.0)
    """
    # 提取各颜色通道
    b, g, r = cv2.split(image.astype(np.float64))

    # 计算各通道的边缘
    r_edges = cv2.Canny(r.astype(np.uint8), 50, 150)
    g_edges = cv2.Canny(g.astype(np.uint8), 50, 150)
    b_edges = cv2.Canny(b.astype(np.uint8), 50, 150)

    # 计算边缘位置的差异
    r_edge_mask = r_edges > 0
    g_edge_mask = g_edges > 0
    b_edge_mask = b_edges > 0

    # 计算边缘重叠度
    overlap = np.sum(r_edge_mask & g_edge_mask & b_edge_mask)
    total_edges = np.sum(r_edge_mask | g_edge_mask | b_edge_mask)

    if total_edges == 0:
        return 0.0

    # 低重叠度表示颜色分离
    overlap_ratio = overlap / total_edges
    return 1.0 - overlap_ratio


def _detect_subpixel_structure(image: np.ndarray) -> float:
    """
    检测亚像素结构

    Args:
        image: BGR格式图像

    Returns:
        亚像素结构分数 (0.0-1.0)
    """
    # 提取各颜色通道
    b, g, r = cv2.split(image.astype(np.float64))

    # 计算各通道的局部方差
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    kernel = kernel / (kernel_size * kernel_size)

    # 计算局部均值
    r_mean = cv2.filter2D(r, -1, kernel)
    g_mean = cv2.filter2D(g, -1, kernel)
    b_mean = cv2.filter2D(b, -1, kernel)

    # 计算局部方差
    r_var = cv2.filter2D(r**2, -1, kernel) - r_mean**2
    g_var = cv2.filter2D(g**2, -1, kernel) - g_mean**2
    b_var = cv2.filter2D(b**2, -1, kernel) - b_mean**2

    # 计算通道间方差差异
    var_diff = np.abs(r_var - g_var) + np.abs(r_var - b_var) + np.abs(g_var - b_var)
    avg_var_diff = np.mean(var_diff)

    # 亚像素结构会导致通道间方差差异
    return min(avg_var_diff / 100.0, 1.0)
