"""批量检测脚本 - 从旧 main.py 迁移"""

from pathlib import Path

from src.main import main


def test_batch_detect() -> None:
    """运行批量检测并验证结果文件生成"""
    main()
    result_path = Path("./data/output/result.json")
    assert result_path.exists(), "result.json should be generated"


if __name__ == "__main__":
    main()
