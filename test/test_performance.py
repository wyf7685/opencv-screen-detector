"""Performance test for the screen detector.

Usage:
    uv run python test/test_performance.py

Measures execution time of the full pipeline and individual feature extractors.
"""

import statistics
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import ScreenDetector
from src.utils.image_io import load_images


def main():
    print("Starting performance test...")

    # Load images
    input_dir = Path(__file__).parent.parent / "data" / "input"
    images = load_images(input_dir)
    print(f"Loaded {len(images)} images")

    # Initialize detector
    detector = ScreenDetector()

    # Measure total execution time (multiple runs for accuracy)
    num_runs = 3
    run_times = []

    for _run in range(num_runs):
        start_time = time.time()

        [detector.detect(path) for path in images]

        end_time = time.time()
        run_times.append(end_time - start_time)

    avg_time = statistics.mean(run_times)
    std_time = statistics.stdev(run_times) if len(run_times) > 1 else 0

    print(
        f"\nTotal execution time (average of {num_runs} runs): {avg_time:.2f} seconds"
    )
    print(f"Standard deviation: {std_time:.2f} seconds")
    print(f"Average time per image: {avg_time / len(images):.4f} seconds")
    print(f"Processed {len(images)} images")

    # Measure individual feature extraction times
    print("\nFeature extraction times:")

    # Test with a sample image
    if images:
        sample_path = images[0]
        print(f"\nTesting with sample image: {Path(sample_path).name}")

        from src.preprocess import preprocess_image
        from src.utils.image_io import load_image

        # Load and preprocess image
        image = load_image(sample_path)
        processed = preprocess_image(image)

        # Import feature functions
        from src.feature.artifact import analyze_artifact
        from src.feature.banding import analyze_banding
        from src.feature.blackscreen import analyze_blackscreen
        from src.feature.chroma import analyze_chroma
        from src.feature.color_noise import analyze_color_noise
        from src.feature.display_content import analyze_display_content
        from src.feature.format_score import analyze_format_score
        from src.feature.frequency import analyze_frequency
        from src.feature.illumination import analyze_illumination
        from src.feature.moire import analyze_moire
        from src.feature.overexposed import analyze_overexposed
        from src.feature.perspective import analyze_perspective
        from src.feature.rectangle import analyze_rectangle
        from src.feature.reflection import analyze_reflection
        from src.feature.sensor_noise import analyze_sensor_noise
        from src.feature.softness import analyze_softness
        from src.feature.subpixel_fringing import analyze_subpixel_fringing
        from src.utils.image_metadata import camera_exif_score

        features = {
            "frequency": (analyze_frequency, processed),
            "banding": (analyze_banding, processed),
            "blackscreen": (analyze_blackscreen, image),
            "chroma": (analyze_chroma, image),
            "softness": (analyze_softness, image),
            "illumination": (analyze_illumination, processed),
            "artifact": (analyze_artifact, processed),
            "rectangle": (analyze_rectangle, processed),
            "display_content": (analyze_display_content, image),
            "overexposed": (analyze_overexposed, processed),
            "perspective": (analyze_perspective, image),
            "moire": (analyze_moire, image),
            "reflection": (analyze_reflection, image),
            "sensor_noise": (analyze_sensor_noise, image),
            "subpixel_fringing": (analyze_subpixel_fringing, image),
            "exif_camera": (camera_exif_score, sample_path),
            "format_score": (analyze_format_score, sample_path),
            "color_noise": (analyze_color_noise, image),
        }

        feature_times = {}
        for name, (func, arg) in features.items():
            start = time.time()
            func(arg)
            end = time.time()
            feature_times[name] = end - start

        # Sort by time (descending)
        sorted_features = sorted(
            feature_times.items(), key=lambda x: x[1], reverse=True
        )

        print("\nFeature extraction times (sorted by duration):")
        for name, duration in sorted_features:
            print(f"  {name}: {duration:.4f} seconds")

        # Calculate total feature extraction time
        total_feature_time = sum(feature_times.values())
        print(f"\nTotal feature extraction time: {total_feature_time:.4f} seconds")
        print(
            f"Average per feature: "
            f"{total_feature_time / len(feature_times):.4f} seconds"
        )

    # Performance improvement calculation
    baseline_time = 112.28  # Original execution time from first run
    improvement = ((baseline_time - avg_time) / baseline_time) * 100
    print(f"\nPerformance improvement: {improvement:.1f}%")
    print("Target: 20% improvement")
    if improvement >= 20:
        print("Status: ACHIEVED")
    else:
        print("Status: NOT ACHIEVED")


if __name__ == "__main__":
    main()
