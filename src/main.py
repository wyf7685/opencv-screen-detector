from src.detector import ScreenDetector
from src.utils.image_io import load_images
from src.utils.json_export import save_json


def main() -> None:
    detector = ScreenDetector()
    images = load_images("./data/input")

    results = []
    for path in images:
        results.append(detector.detect(path))

    save_json(results, "./data/output/result.json")


if __name__ == "__main__":
    main()