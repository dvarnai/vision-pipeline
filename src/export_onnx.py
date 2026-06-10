import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import torch

from src.inference.prediction import load_inference_bundle


def positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed_value


def _to_plain(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value


def _shape_from_onnx_value(value):
    return [
        dim.dim_param if dim.dim_param else dim.dim_value
        for dim in value.type.tensor_type.shape.dim
    ]


def read_onnx_io_info(onnx_path):
    import onnx

    model = onnx.load(onnx_path)
    return {
        "input_shape": _shape_from_onnx_value(model.graph.input[0]),
        "output_shape": _shape_from_onnx_value(model.graph.output[0]),
        "opsets": {op.domain or "": op.version for op in model.opset_import},
    }


def build_metadata(bundle, dummy_input_shape, onnx_io_info, requested_dynamic_batch):
    input_shape = onnx_io_info["input_shape"]
    exported_dynamic_batch = bool(input_shape and isinstance(input_shape[0], str))
    return {
        "checkpoint_path": bundle.checkpoint_path,
        "checkpoint_epoch": bundle.checkpoint_epoch,
        "config_module": bundle.config_module_name,
        "model_version": bundle.model_version,
        "preprocessing_version": bundle.preprocessing_version,
        "label_contract_version": bundle.label_contract_version,
        "class_names": list(bundle.class_names),
        "input_name": "images",
        "output_name": "logits",
        "input_shape": input_shape,
        "output_shape": onnx_io_info["output_shape"],
        "dummy_input_shape": list(dummy_input_shape),
        "input_dtype": "float32",
        "output_dtype": "float32",
        "output_semantics": "raw logits; apply softmax over axis 1 for class probabilities",
        "opsets": onnx_io_info["opsets"],
        "requested_dynamic_batch": requested_dynamic_batch,
        "dynamic_batch": exported_dynamic_batch,
        "preprocessing": {
            "note": "ONNX graph expects already-preprocessed NCHW tensors.",
            "transform": str(bundle.transform),
        },
        "runtime_config": _to_plain(bundle.runtime_config),
    }


def write_metadata_json(metadata, metadata_path):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def attach_onnx_metadata(onnx_path, metadata):
    import onnx

    model = onnx.load(onnx_path)
    keep_keys = {
        "model_version",
        "preprocessing_version",
        "label_contract_version",
        "class_names",
        "input_shape",
        "output_semantics",
    }
    existing = {prop.key: prop for prop in model.metadata_props}
    for key in keep_keys:
        prop = existing.get(key)
        if prop is None:
            prop = model.metadata_props.add()
            prop.key = key
        prop.value = json.dumps(metadata[key]) if isinstance(metadata[key], (list, dict)) else str(metadata[key])
    onnx.save(model, onnx_path)


def export_onnx(
    *,
    checkpoint_path,
    output_path,
    config_override,
    batch_size,
    opset,
    dynamic_batch,
    metadata_path,
):
    bundle = load_inference_bundle(checkpoint_path, config_override=config_override, device="cpu")
    model = bundle.model.eval()

    dummy_input_shape = (
        batch_size,
        int(bundle.config.in_channels),
        int(bundle.config.in_height),
        int(bundle.config.in_width),
    )
    dummy_input = torch.randn(dummy_input_shape, dtype=torch.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = ({0: torch.export.Dim("batch")},)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["logits"],
            dynamic_shapes=dynamic_shapes,
        )

    onnx_io_info = read_onnx_io_info(output_path)
    metadata = build_metadata(bundle, dummy_input_shape, onnx_io_info, dynamic_batch)
    if dynamic_batch and not metadata["dynamic_batch"]:
        warnings.warn(
            "Dynamic batch was requested, but the exported ONNX graph has a fixed batch dimension.",
            stacklevel=2,
        )
    attach_onnx_metadata(output_path, metadata)

    resolved_metadata_path = Path(metadata_path) if metadata_path is not None else output_path.with_suffix(".json")
    write_metadata_json(metadata, resolved_metadata_path)

    return metadata, resolved_metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an Intel scene checkpoint to ONNX.")
    parser.add_argument("checkpoint", help="path to the project checkpoint")
    parser.add_argument("output", help="path to write the ONNX model")
    parser.add_argument("--config", default=None, help="override config module from checkpoint")
    parser.add_argument("--batch-size", type=positive_int, default=1, help="dummy export batch size")
    parser.add_argument("--opset", type=positive_int, default=18, help="ONNX opset version")
    parser.add_argument(
        "--dynamic-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export a dynamic batch dimension",
    )
    parser.add_argument("--metadata", default=None, help="optional path for metadata JSON sidecar")
    args = parser.parse_args()

    metadata, metadata_path = export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_override=args.config,
        batch_size=args.batch_size,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        metadata_path=args.metadata,
    )

    print(json.dumps({
        "onnx_path": str(Path(args.output)),
        "metadata_path": str(metadata_path),
        "input_shape": metadata["input_shape"],
        "dynamic_batch": metadata["dynamic_batch"],
        "model_version": metadata["model_version"],
        "preprocessing_version": metadata["preprocessing_version"],
        "label_contract_version": metadata["label_contract_version"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
