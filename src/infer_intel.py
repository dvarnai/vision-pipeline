import argparse
import json
import sys

from src.inference.prediction import load_inference_bundle, predict_image
from src.inference.validation import (
    DEFAULT_MAX_IMAGE_BYTES,
    InvalidInputError,
    validate_image_path,
)


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-image Intel scene classification inference.")
    parser.add_argument("checkpoint", type=str, help="path to the project checkpoint")
    parser.add_argument("image", type=str, help="path to the image to score")
    parser.add_argument("--config", type=str, default=None, help="override config module from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="inference device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--top-k", type=positive_int, default=1, help="include the top K labels")
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=DEFAULT_MAX_IMAGE_BYTES / (1024 * 1024),
        help="maximum accepted image size in MiB",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    args = parser.parse_args()

    try:
        max_bytes = int(args.max_file_size_mb * 1024 * 1024)
        image = validate_image_path(args.image, max_bytes=max_bytes)
        bundle = load_inference_bundle(args.checkpoint, config_override=args.config, device=args.device)
        result = predict_image(bundle, image, top_k=args.top_k)
    except InvalidInputError as exc:
        print(
            json.dumps({"error": "invalid_input", "code": exc.code, "message": str(exc)}),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            json.dumps({"error": "inference_failed", "message": str(exc)}),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
