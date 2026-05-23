"""FFT 频谱变换模块

将图像转换为 FFT 频谱图，用于频域特征分析。
训练和推理两端共享此模块。
"""

import cv2
import numpy as np


def compute_fft_spectrum(image: np.ndarray, size: int = 224) -> np.ndarray:
    """将图像转换为 FFT 频谱图

    Args:
        image: 输入图像 (BGR 或灰度)
        size: 输出尺寸

    Returns:
        FFT 频谱图，形状 (1, 1, H, W)
    """
    # Step 1: 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 先 resize 到目标尺寸，避免大图片导致内存不足
    gray = cv2.resize(gray, (size, size))

    # Step 2: FFT
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Step 3: 频谱图 (magnitude)
    magnitude = np.log(np.abs(fshift) + 1)

    # Step 4: 归一化到 [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # Step 6: 归一化到 [0, 1] 然后 ImageNet 灰度归一化
    magnitude = magnitude.astype(np.float32) / 255.0
    mean, std = 0.449, 0.226
    magnitude = (magnitude - mean) / std

    # 转换为 (1, 1, H, W) 形状
    return magnitude.reshape(1, 1, size, size)


def compute_fft_spectrum_from_bytes(image_bytes: bytes, size: int = 224) -> np.ndarray:
    """从字节数据计算 FFT 频谱图

    Args:
        image_bytes: 图片字节数据
        size: 输出尺寸

    Returns:
        FFT 频谱图，形状 (1, 1, H, W)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert image is not None
    return compute_fft_spectrum(image, size)
