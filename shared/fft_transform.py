"""FFT 频谱变换模块

将图像转换为 FFT 频谱图，用于频域特征分析。
训练和推理两端共享此模块。
"""

import cv2
import numpy as np

# ImageNet 灰度归一化常量
FFT_NORM_MEAN = 0.449
FFT_NORM_STD = 0.226


def compute_fft_spectrum(
    image: np.ndarray,
    size: int = 224,
    color_space: str = "bgr",
) -> np.ndarray:
    """将图像转换为 FFT 频谱图

    Args:
        image: 输入图像 (H, W) 灰度 或 (H, W, 3) 彩色
        size: 输出尺寸
        color_space: 彩色图像的色彩空间，"bgr" 或 "rgb"

    Returns:
        FFT 频谱图，形状 (1, 1, H, W)
    """
    # 灰度化
    if len(image.shape) == 3:
        code = cv2.COLOR_BGR2GRAY if color_space == "bgr" else cv2.COLOR_RGB2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        gray = image

    # resize 到目标尺寸
    gray = cv2.resize(gray, (size, size))

    # FFT
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # 频谱图 (magnitude)
    magnitude = np.log(np.abs(fshift) + 1)

    # 归一化到 [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # 归一化到 [0, 1] 然后 ImageNet 灰度归一化
    magnitude = magnitude.astype(np.float32) / 255.0
    magnitude = (magnitude - FFT_NORM_MEAN) / FFT_NORM_STD

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
    if image is None:
        raise ValueError("Failed to decode image from bytes")
    return compute_fft_spectrum(image, size)
