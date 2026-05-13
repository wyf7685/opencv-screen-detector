"""Check detection accuracy against expected labels.

Usage:
    uv run python test/test_accuracy.py

Reads data/output/result.json and compares against expected labels
based on the directory structure (photo/ -> screen_photo, img/ & no_screen/ -> normal).
"""

import json
import sys
from pathlib import Path

RESULT_PATH = Path(__file__).parent.parent / "data" / "output" / "result.json"
INPUT_DIR = Path(__file__).parent.parent / "data" / "input"


def main():
    with RESULT_PATH.open(encoding="utf-8") as f:
        results = json.load(f)

    img_files = {p.name for p in (INPUT_DIR / "img").iterdir()}
    photo_files = {p.name for p in (INPUT_DIR / "photo").iterdir()}
    no_screen_dir = INPUT_DIR / "no_screen"
    no_screen_files = (
        {p.name for p in no_screen_dir.iterdir()} if no_screen_dir.is_dir() else set()
    )

    total = len(results)
    correct = 0
    errors = []

    for r in results:
        fname = r["filename"]
        result = r["result"]
        score = r["score"]

        if fname in photo_files:
            expected = "screen_photo"
        elif fname in img_files or fname in no_screen_files:
            expected = "normal"
        else:
            continue

        if result == expected:
            correct += 1
        else:
            errors.append((fname, expected, result, score))

    print(
        f"Total: {total}, Correct: {correct}, Errors: {len(errors)}, "
        f"Accuracy: {correct / total * 100:.2f}%"
    )

    if errors:
        print("\nMisclassified images:")
        for fname, expected, actual, score in errors:
            print(f"  {fname}: expected={expected}, actual={actual}, score={score:.3f}")
        sys.exit(1)
    else:
        print("All images correctly classified!")


if __name__ == "__main__":
    main()
