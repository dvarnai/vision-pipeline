from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from src.core.checkpoint import load_checkpoint
from src.data.labels import INTEL_CLASS_NAMES, INTEL_LABEL_CONTRACT_VERSION


@dataclass(frozen=True)
class InferenceBundle:
    checkpoint_path: str
    config_module_name: str
    checkpoint_epoch: int | None
    config: Any
    runtime_config: dict[str, Any]
    class_weights: torch.Tensor | None
    model: torch.nn.Module
    transform: Callable[[Image.Image], torch.Tensor]
    device: torch.device
    class_names: tuple[str, ...]
    label_contract_version: str
    model_version: str
    preprocessing_version: str


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _checkpoint_source_digest(checkpoint: dict[str, Any]) -> str | None:
    source = checkpoint.get("config_source")
    if source is None:
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def _build_model(config: Any, num_classes: int) -> torch.nn.Module:
    build_model = config.build_model
    try:
        signature = inspect.signature(build_model)
    except (TypeError, ValueError):
        return build_model(num_classes=num_classes)

    if "pretrained" in signature.parameters:
        return build_model(num_classes=num_classes, pretrained=False)
    if "weights" in signature.parameters:
        return build_model(num_classes=num_classes, weights=None)
    return build_model(num_classes=num_classes)


def _class_contract_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    label_contract = checkpoint.get("label_contract") or {}
    class_names = label_contract.get("class_names") or INTEL_CLASS_NAMES
    version = label_contract.get("version") or INTEL_LABEL_CONTRACT_VERSION
    return tuple(str(class_name) for class_name in class_names), str(version)


def _build_model_version(checkpoint_path: str | Path, checkpoint: dict[str, Any], config_module_name: str) -> str:
    checkpoint_name = Path(checkpoint_path).name
    epoch = checkpoint.get("epoch", "unknown")
    source_digest = _checkpoint_source_digest(checkpoint)
    source_suffix = f":config-sha-{source_digest}" if source_digest else ""
    return f"{checkpoint_name}:epoch-{epoch}:config-{config_module_name}{source_suffix}"


def _build_preprocessing_version(
    *,
    config_module_name: str,
    checkpoint: dict[str, Any],
    training_stats: dict[str, Any],
    transform: Callable[[Image.Image], torch.Tensor],
) -> str:
    payload = {
        "config_module": config_module_name,
        "config_source_sha": _checkpoint_source_digest(checkpoint),
        "mean": _to_plain(training_stats["mean"]),
        "std": _to_plain(training_stats["std"]),
        "transform": str(transform),
    }
    return f"preprocess-{_digest(payload)}"


def load_inference_bundle(
    checkpoint_path: str | Path,
    *,
    config_override: str | None = None,
    device: str | torch.device | None = None,
) -> InferenceBundle:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    config_module_name = config_override or checkpoint.get("config_module")
    if config_module_name is None:
        raise ValueError("checkpoint does not contain config_module; pass a config override")

    training_stats = checkpoint.get("training_stats", {})
    if "mean" not in training_stats or "std" not in training_stats:
        raise ValueError("checkpoint is missing training_stats mean/std required for preprocessing")

    class_names, label_contract_version = _class_contract_from_checkpoint(checkpoint)
    config_module = importlib.import_module(config_module_name)
    config = config_module.build_config()
    transform = config.build_val_transform(training_stats["mean"], training_stats["std"])

    selected_device = resolve_device(device)
    if (
        selected_device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(selected_device) >= (8, 0)
    ):
        torch.set_float32_matmul_precision("high")

    model = _build_model(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()

    return InferenceBundle(
        checkpoint_path=str(checkpoint_path),
        config_module_name=config_module_name,
        checkpoint_epoch=checkpoint.get("epoch"),
        config=config,
        runtime_config=dict(checkpoint.get("runtime_config", {})),
        class_weights=checkpoint.get("class_weights"),
        model=model,
        transform=transform,
        device=selected_device,
        class_names=class_names,
        label_contract_version=label_contract_version,
        model_version=_build_model_version(checkpoint_path, checkpoint, config_module_name),
        preprocessing_version=_build_preprocessing_version(
            config_module_name=config_module_name,
            checkpoint=checkpoint,
            training_stats=training_stats,
            transform=transform,
        ),
    )


def predict_batch_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    selected_device = resolve_device(device)
    with torch.inference_mode():
        return model(images.to(selected_device, non_blocking=True))


def predict_image(
    bundle: InferenceBundle,
    image: Image.Image,
    *,
    top_k: int = 1,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    k = min(top_k, len(bundle.class_names))
    _sync_device(bundle.device)
    started_at = time.perf_counter()

    with torch.inference_mode():
        image_tensor = bundle.transform(image).unsqueeze(0).to(bundle.device)
        logits = bundle.model(image_tensor)
        _sync_device(bundle.device)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

    latency_ms = (time.perf_counter() - started_at) * 1000
    confidences, indices = torch.topk(probabilities, k=k)
    top_predictions = [
        {
            "label": bundle.class_names[int(index)],
            "confidence": round(float(confidence), 6),
        }
        for confidence, index in zip(confidences, indices)
    ]

    result = {
        "predicted_label": top_predictions[0]["label"],
        "confidence": top_predictions[0]["confidence"],
        "model_version": bundle.model_version,
        "preprocessing_version": bundle.preprocessing_version,
        "label_contract_version": bundle.label_contract_version,
        "latency_ms": round(latency_ms, 3),
    }
    if top_k > 1:
        result["top_k"] = top_predictions
    return result
