"""UI内容检测模块 - 检测图像中是否包含UI元素"""

import cv2
import numpy as np


def detect_ui_content(image: np.ndarray) -> dict[str, float]:
    """
    检测图像中是否包含UI内容（文本、菜单、代码、浏览器等）

    Args:
        image: BGR格式的输入图像

    Returns:
        包含UI检测结果的字典：
        - text_density: 文本密度 (0.0-1.0)
        - ui_line_density: UI线条密度 (0.0-1.0)
        - rectangle_score: 矩形检测分数 (0.0-1.0)
    """
    if image is None or image.size == 0:
        return {"text_density": 0.0, "ui_line_density": 0.0, "rectangle_score": 0.0}

    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. 文本密度检测
    text_density = _detect_text_density(gray)

    # 2. UI线条密度检测
    ui_line_density = _detect_ui_lines(gray)

    # 3. 矩形检测
    rectangle_score = _detect_rectangles(gray)

    return {
        "text_density": text_density,
        "ui_line_density": ui_line_density,
        "rectangle_score": rectangle_score,
    }


def _detect_text_density(gray: np.ndarray) -> float:
    """
    检测文本密度 - 使用边缘检测和形态学操作

    Args:
        gray: 灰度图像

    Returns:
        文本密度分数 (0.0-1.0)
    """
    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 形态学操作，连接相邻的文本区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 统计小矩形区域（文本特征）
    text_regions = 0
    total_area = gray.shape[0] * gray.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < total_area * 0.01:  # 小区域可能是文本
            _, _, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if 0.5 < aspect_ratio < 5.0:  # 文本区域的宽高比
                text_regions += 1

    # 归一化
    max_regions = 500
    return min(text_regions / max_regions, 1.0)


def _detect_ui_lines(gray: np.ndarray) -> float:
    """
    检测UI线条 - 使用HoughLinesP检测直线

    Args:
        gray: 灰度图像

    Returns:
        UI线条密度分数 (0.0-1.0)
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

    # 统计水平和垂直线（UI特征）
    horizontal_lines = 0
    vertical_lines = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        if angle < 10 or angle > 170:  # 水平线
            horizontal_lines += 1
        elif 80 < angle < 100:  # 垂直线
            vertical_lines += 1

    # UI通常有较多的水平和垂直线
    ui_lines = horizontal_lines + vertical_lines
    max_lines = 200
    return min(ui_lines / max_lines, 1.0)


def _detect_rectangles(gray: np.ndarray) -> float:
    """
    检测矩形 - 检测UI窗口和边框

    Args:
        gray: 灰度图像

    Returns:
        矩形检测分数 (0.0-1.0)
    """
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    total_area = gray.shape[0] * gray.shape[1]

    for contour in contours:
        # 近似轮廓为多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 如果是四边形
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > total_area * 0.001:  # 过滤太小的矩形
                # 检查是否接近矩形
                _, _, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:  # 合理的宽高比
                    rectangles.append(area)

    # 计算矩形分数
    if not rectangles:
        return 0.0

    # 考虑矩形数量和大小
    rect_count = len(rectangles)
    rect_area_ratio = sum(rectangles) / total_area

    # 归一化
    count_score = min(rect_count / 20, 1.0)
    area_score = min(rect_area_ratio * 5, 1.0)

    return count_score * 0.4 + area_score * 0.6
