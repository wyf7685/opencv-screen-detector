import runpy
from pathlib import Path


def main() -> None:
    src_main = Path(__file__).resolve().parent / "src" / "main.py"
    runpy.run_path(str(src_main), run_name="__main__")


if __name__ == "__main__":
    main()
