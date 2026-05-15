"""透视畸变检测模块 - 检测摄像头拍摄产生的透视变形"""

import cv2
import numpy as np


def analyze_perspective(image: np.ndarray) -> float:
    """
    检测图像中的透视畸变

    手机拍摄屏幕几乎一定存在透视变形，而截图永远是严格平行的。

    Args:
        image: BGR格式的输入图像

    Returns:
        透视畸变分数 (0.0-1.0)，值越高表示透视畸变越明显
    """
    if image is None or image.size == 0:
        return 0.0

    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 使用拉普拉斯算子和Sobel算子检测边缘锐度（与旧算法一致）
    gray_f = gray.astype(float)
    laplacian = np.abs(cv2.Laplacian(gray_f, cv2.CV_64F))
    sobelx = np.abs(cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3))
    sobely = np.abs(cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3))
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)

    threshold = np.percentile(gradient_mag, 90)
    edge_mask = gradient_mag > threshold
    if not np.any(edge_mask):
        return 0.0

    lap_at_edges = float(np.mean(laplacian[edge_mask]))
    grad_at_edges = float(np.mean(gradient_mag[edge_mask]))

    if grad_at_edges <= 0.0:
        return 0.0

    sharpness_ratio = lap_at_edges / grad_at_edges
    score = min(1.0, sharpness_ratio / 0.4)
    return max(0.0, score)
