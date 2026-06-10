import argparse
import json
import statistics
import time

import torch

from src.inference.prediction import load_inference_bundle, predict_image, resolve_device
from src.inference.validation import DEFAULT_MAX_IMAGE_BYTES, validate_image_path


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        return spec, spec
    name, checkpoint_path = spec.split("=", 1)
    if not name or not checkpoint_path:
        raise argparse.ArgumentTypeError("--model must be NAME=CHECKPOINT or CHECKPOINT")
    return name, checkpoint_path


def summarize(latencies: list[float]) -> dict[str, float | int]:
    return {
        "runs": len(latencies),
        "mean_ms": round(statistics.fmean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
    }


def empty_cuda_cache(device):
    selected_device = resolve_device(device)
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()


def measure_model(
    *,
    name: str,
    checkpoint_path: str,
    image,
    device: str,
    top_k: int,
    cold_runs: int,
    warmup_runs: int,
    warm_runs: int,
) -> dict:
    cold_latencies = []
    cold_example = None
    for _ in range(cold_runs):
        empty_cuda_cache(device)
        started_at = time.perf_counter()
        bundle = load_inference_bundle(checkpoint_path, device=device)
        cold_example = predict_image(bundle, image, top_k=top_k)
        cold_latencies.append((time.perf_counter() - started_at) * 1000)

    bundle = load_inference_bundle(checkpoint_path, device=device)
    for _ in range(warmup_runs):
        predict_image(bundle, image, top_k=top_k)

    warm_latencies = []
    warm_example = None
    for _ in range(warm_runs):
        warm_example = predict_image(bundle, image, top_k=top_k)
        warm_latencies.append(warm_example["latency_ms"])

    example = warm_example or cold_example
    return {
        "name": name,
        "checkpoint": checkpoint_path,
        "model_version": example["model_version"] if example else None,
        "preprocessing_version": example["preprocessing_version"] if example else None,
        "label_contract_version": example["label_contract_version"] if example else None,
        "cold": summarize(cold_latencies),
        "warm": summarize(warm_latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure cold and warm single-image checkpoint latency.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=parse_model_spec,
        help="model to benchmark, as NAME=CHECKPOINT or CHECKPOINT; repeat for selected and rejected models",
    )
    parser.add_argument("--image", required=True, help="image used for latency measurement")
    parser.add_argument("--device", default="auto", help="benchmark device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--top-k", type=positive_int, default=1)
    parser.add_argument("--cold-runs", type=positive_int, default=3)
    parser.add_argument("--warmup-runs", type=positive_int, default=5)
    parser.add_argument("--warm-runs", type=positive_int, default=50)
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=DEFAULT_MAX_IMAGE_BYTES / (1024 * 1024),
    )
    args = parser.parse_args()

    image = validate_image_path(args.image, max_bytes=int(args.max_file_size_mb * 1024 * 1024))
    results = [
        measure_model(
            name=name,
            checkpoint_path=checkpoint_path,
            image=image,
            device=args.device,
            top_k=args.top_k,
            cold_runs=args.cold_runs,
            warmup_runs=args.warmup_runs,
            warm_runs=args.warm_runs,
        )
        for name, checkpoint_path in args.model
    ]
    print(json.dumps({"device": str(resolve_device(args.device)), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
