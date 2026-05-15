"""摩尔纹检测模块 - 检测LCD/OLED屏幕的摩尔纹"""

import cv2
import numpy as np


def analyze_moire(image: np.ndarray) -> float:
    """
    检测图像中的摩尔纹

    拍摄LCD/OLED屏幕时，屏幕像素与CMOS像素会形成摩尔纹。

    Args:
        image: BGR格式的输入图像

    Returns:
        摩尔纹分数 (0.0-1.0)，值越高表示摩尔纹越明显
    """
    if image is None or image.size == 0:
        return 0.0

    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. FFT频域分析
    fft_score = _analyze_fft_pattern(gray)

    # 2. 检测离散周期峰值
    peak_score = _detect_periodic_peaks(gray)

    # 3. 分析局部纹理周期性
    texture_score = _analyze_texture_periodicity(gray)

    # 综合评分
    moire_score = fft_score * 0.4 + peak_score * 0.3 + texture_score * 0.3

    return min(max(moire_score, 0.0), 1.0)


def _analyze_fft_pattern(gray: np.ndarray) -> float:
    """
    分析FFT频域模式 - 摩尔纹会在频域产生特定峰值

    Args:
        gray: 灰度图像

    Returns:
        FFT模式分数 (0.0-1.0)
    """
    # 转换为浮点数
    gray_float = gray.astype(np.float64)

    # 应用FFT
    f_transform = np.fft.fft2(gray_float)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # 计算频谱
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    # 创建径向频率掩码
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # 计算各频率环的能量
    max_radius = min(h, w) // 2
    ring_energies = []

    for r in range(10, max_radius, 10):
        mask = (radius >= r - 5) & (radius < r + 5)
        if np.sum(mask) > 0:
            ring_energy = np.mean(magnitude[mask] ** 2)
            ring_energies.append(ring_energy)

    if not ring_energies:
        return 0.0

    # 计算能量分布的峰值
    ring_energies = np.array(ring_energies)
    mean_energy = np.mean(ring_energies)
    std_energy = np.std(ring_energies)

    # 摩尔纹会导致特定频率处有尖锐峰值
    # 计算峰值检测
    peaks = 0
    for i in range(1, len(ring_energies) - 1):
        if (
            ring_energies[i] > ring_energies[i - 1]
            and ring_energies[i] > ring_energies[i + 1]
            and ring_energies[i] > mean_energy + 2 * std_energy
        ):
            peaks += 1

    # 归一化
    return min(peaks / 5.0, 1.0)


def _detect_periodic_peaks(gray: np.ndarray) -> float:
    """
    检测离散周期峰值 - 摩尔纹的特征

    Args:
        gray: 灰度图像

    Returns:
        周期峰值分数 (0.0-1.0)
    """
    # 转换为浮点数
    gray_float = gray.astype(np.float64)

    # 应用FFT
    f_transform = np.fft.fft2(gray_float)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # 计算频谱
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    # 排除中心区域（DC分量）
    mask = np.ones((h, w), dtype=bool)
    mask[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10] = False

    # 计算频谱的统计特征
    magnitude_masked = magnitude[mask]
    mean_mag = np.mean(magnitude_masked)
    std_mag = np.std(magnitude_masked)

    # 检测显著峰值
    threshold = mean_mag + 3 * std_mag
    peaks = np.sum(magnitude_masked > threshold)

    # 归一化
    total_pixels = np.sum(mask)
    peak_ratio = peaks / total_pixels

    # 摩尔纹会导致特定比例的峰值
    return min(peak_ratio * 100, 1.0)


def _analyze_texture_periodicity(gray: np.ndarray) -> float:
    """
    分析局部纹理周期性

    Args:
        gray: 灰度图像

    Returns:
        纹理周期性分数 (0.0-1.0)
    """
    # 将图像分成小块
    block_size = 64
    h, w = gray.shape

    periodic_blocks = 0
    total_blocks = 0

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y : y + block_size, x : x + block_size]

            # 转换为浮点数并确保是2D
            block_float = block.astype(np.float32)

            # 计算自相关 - 使用归一化互相关
            # 计算块的FFT
            f_transform = np.fft.fft2(block_float)
            power_spectrum = np.abs(f_transform) ** 2

            # 计算自相关（通过IFFT）
            autocorr = np.fft.ifft2(power_spectrum).real

            # 归一化
            autocorr = autocorr / autocorr[0, 0] if autocorr[0, 0] != 0 else autocorr

            # 检测周期性峰值（排除中心）
            center_y, center_x = block_size // 2, block_size // 2
            autocorr[center_y - 2 : center_y + 2, center_x - 2 : center_x + 2] = 0

            # 计算峰值数量
            peaks = np.sum(autocorr > 0.5)

            if peaks > 0:  # 有周期性
                periodic_blocks += 1

            total_blocks += 1

    if total_blocks == 0:
        return 0.0

    # 计算周期性块的比例
    periodic_ratio = periodic_blocks / total_blocks

    return min(periodic_ratio * 2.0, 1.0)
