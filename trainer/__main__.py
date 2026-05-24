import sys


def main() -> None:
    usage = "Usage: python -m trainer <train|export>"
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    match sys.argv[1].lower():
        case "train":
            from .train import main as train_main

            train_main()
        case "export":
            from .export_onnx import main as export_main

            export_main()
        case x:
            print(f"Unknown command: {x}")
            print(usage)
            sys.exit(1)


if __name__ == "__main__":
    main()
