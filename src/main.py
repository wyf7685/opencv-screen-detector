"""主程序入口 - 使用ScreenDetector进行图像来源识别"""

import logging
from pathlib import Path

from src.detector import ScreenDetector
from src.utils.json_export import save_json

logger = logging.getLogger(__name__)


def main() -> None:
    """主函数"""
    # 初始化检测器
    detector = ScreenDetector()

    # 获取所有图像路径
    input_dir = Path("./data/input")
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_paths.extend(input_dir.rglob(ext))
    image_paths.sort()

    # 检测图像
    results = []
    for image_path in image_paths:
        result = detector.detect(image_path)
        results.append(result)

    # 保存结果
    output_path = "./data/output/result.json"
    save_json(results, output_path)

    # 打印统计信息
    print_statistics(results)


def print_statistics(results: list[dict]) -> None:
    """
    打印统计信息

    Args:
        results: 检测结果列表
    """
    total = len(results)
    if total == 0:
        logger.info("No images processed")
        return

    # 统计各类别数量
    normal_count = 0
    screen_photo_count = 0

    for result in results:
        if result.get("result") == "screen_photo":
            screen_photo_count += 1
        else:
            normal_count += 1

    # 打印统计信息
    logger.info("=" * 50)
    logger.info("Detection Statistics")
    logger.info("=" * 50)
    logger.info("Total images: %d", total)
    logger.info("Normal: %d", normal_count)
    logger.info("Screen photo: %d", screen_photo_count)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
