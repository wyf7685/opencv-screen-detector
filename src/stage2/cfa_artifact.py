"""CFA插值伪影检测模块 - 检测Bayer CFA插值留下的RGGB pattern"""

import cv2
import numpy as np


def analyze_cfa_artifact(image: np.ndarray) -> float:
    """
    检测图像中的CFA插值伪影

    手机摄像头使用Bayer CFA插值，会留下RGGB pattern，截图没有。

    Args:
        image: BGR格式的输入图像

    Returns:
        CFA伪影分数 (0.0-1.0)，值越高表示CFA伪影越明显
    """
    if image is None or image.size == 0:
        return 0.0

    # 转换为BGR格式（如果不是）
    if len(image.shape) == 2:
        # 灰度图无法检测CFA
        return 0.0

    # 1. 分析2x2像素相关性
    correlation_score = _analyze_2x2_correlation(image)

    # 2. 分析颜色通道差异
    channel_diff_score = _analyze_channel_differences(image)

    # 3. 分析频域特征
    frequency_score = _analyze_frequency_pattern(image)

    # 综合评分
    cfa_score = (
        correlation_score * 0.4 + channel_diff_score * 0.3 + frequency_score * 0.3
    )

    return min(max(cfa_score, 0.0), 1.0)


def _analyze_2x2_correlation(image: np.ndarray) -> float:
    """
    分析2x2像素相关性 - CFA插值会导致特定的像素相关模式

    Args:
        image: BGR格式图像

    Returns:
        2x2相关性分数 (0.0-1.0)
    """
    # 转换为浮点数
    img_float = image.astype(np.float64)

    # 提取各颜色通道
    b, g, r = cv2.split(img_float)

    # 向量化计算2x2块内方差
    h, w = r.shape
    h2, w2 = (h // 2) * 2, (w // 2) * 2

    # 提取2x2块并reshape为(n_blocks, 4)
    r_blocks = np.stack(
        [r[0:h2:2, 0:w2:2], r[0:h2:2, 1:w2:2], r[1:h2:2, 0:w2:2], r[1:h2:2, 1:w2:2]],
        axis=-1,
    ).reshape(-1, 4)
    g_blocks = np.stack(
        [g[0:h2:2, 0:w2:2], g[0:h2:2, 1:w2:2], g[1:h2:2, 0:w2:2], g[1:h2:2, 1:w2:2]],
        axis=-1,
    ).reshape(-1, 4)
    b_blocks = np.stack(
        [b[0:h2:2, 0:w2:2], b[0:h2:2, 1:w2:2], b[1:h2:2, 0:w2:2], b[1:h2:2, 1:w2:2]],
        axis=-1,
    ).reshape(-1, 4)

    if r_blocks.size == 0:
        return 0.0

    # 计算块内方差
    r_vars = np.var(r_blocks, axis=1)
    g_vars = np.var(g_blocks, axis=1)
    b_vars = np.var(b_blocks, axis=1)

    # CFA插值会导致块内方差较小（像素值更相似）
    avg_var = np.mean([np.mean(r_vars), np.mean(g_vars), np.mean(b_vars)])

    # 归一化
    max_var = 1000.0
    return 1.0 - min(avg_var / max_var, 1.0)


def _analyze_channel_differences(image: np.ndarray) -> float:
    """
    分析颜色通道差异 - CFA插值会导致特定的通道间关系

    Args:
        image: BGR格式图像

    Returns:
        通道差异分数 (0.0-1.0)
    """
    # 转换为浮点数
    img_float = image.astype(np.float64)

    # 提取各颜色通道
    b, g, r = cv2.split(img_float)

    # 计算相邻像素的通道差异
    r_g_diff = np.abs(r - g)
    r_b_diff = np.abs(r - b)
    g_b_diff = np.abs(g - b)

    # 向量化计算8x8块的方差
    block_size = 8
    h, w = r.shape
    h8, w8 = (h // block_size) * block_size, (w // block_size) * block_size

    # reshape为(block_count, block_size, block_size)然后计算每块方差
    def block_variance(diff: np.ndarray) -> np.ndarray:
        cropped = diff[:h8, :w8]
        bh, bw = h8 // block_size, w8 // block_size
        blocks = cropped.reshape(bh, block_size, bw, block_size)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size * block_size)
        return np.var(blocks, axis=1)

    r_g_vars = block_variance(r_g_diff)
    r_b_vars = block_variance(r_b_diff)
    g_b_vars = block_variance(g_b_diff)

    if r_g_vars.size == 0:
        return 0.0

    # CFA插值会导致差异方差较小
    avg_diff_var = np.mean([(r_g_vars + r_b_vars + g_b_vars) / 3.0])

    # 归一化
    max_var = 500.0
    return 1.0 - min(avg_diff_var / max_var, 1.0)


def _analyze_frequency_pattern(image: np.ndarray) -> float:
    """
    分析频域特征 - CFA插值会在频域留下特定模式

    Args:
        image: BGR格式图像

    Returns:
        频域特征分数 (0.0-1.0)
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 转换为浮点数
    gray_float = gray.astype(np.float64)

    # 应用FFT
    f_transform = np.fft.fft2(gray_float)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # 计算频谱
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    # 检测1/4和1/2频率处的峰值（CFA特征）
    quarter_y, quarter_x = h // 4, w // 4
    half_y, half_x = h // 2, w // 2

    # 计算各频率区域的能量
    total_energy = np.sum(magnitude**2)

    # 1/4频率区域能量
    quarter_region = magnitude[
        center_y - quarter_y : center_y + quarter_y,
        center_x - quarter_x : center_x + quarter_x,
    ]
    quarter_energy = np.sum(quarter_region**2)

    # 1/2频率区域能量
    half_region = magnitude[
        center_y - half_y : center_y + half_y,
        center_x - half_x : center_x + half_x,
    ]
    half_energy = np.sum(half_region**2)

    # 计算能量比
    quarter_ratio = quarter_energy / total_energy if total_energy > 0 else 0.0
    half_ratio = half_energy / total_energy if total_energy > 0 else 0.0

    # CFA插值会导致特定频率区域能量较高
    frequency_score = (quarter_ratio + half_ratio) / 2.0

    return min(frequency_score * 2.0, 1.0)
