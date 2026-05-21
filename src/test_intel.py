import argparse
import importlib
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from src.core.checkpoint import load_checkpoint
from src.data.dataset import IntelImageClassificationDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda" and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="path to checkpoint to evaluate")
    parser.add_argument("--config", type=str, default=None, help="override config module from checkpoint")
    parser.add_argument("--images-path", type=str, default=None, help="path to Intel dataset root")
    parser.add_argument("--batch-size", type=int, default=None, help="override evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="override number of dataloader workers")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    runtime_config = checkpoint.get("runtime_config", {})

    config_module_name = args.config or checkpoint.get("config_module")
    if config_module_name is None:
        parser.error("checkpoint does not contain config_module; pass --config explicitly")

    config_module = importlib.import_module(config_module_name)
    config = config_module.build_config()

    images_path = args.images_path if args.images_path is not None else runtime_config.get("images_path", "data/intel")
    batch_size = args.batch_size if args.batch_size is not None else runtime_config.get("batch_size", config.batch_size)
    num_workers = args.num_workers if args.num_workers is not None else runtime_config.get("num_workers", config.num_workers)

    training_stats = checkpoint.get("training_stats", {})
    if "mean" not in training_stats or "std" not in training_stats:
        raise ValueError("checkpoint is missing training_stats mean/std required for test transforms")

    test_dataset = IntelImageClassificationDataset(
        images_path=os.path.join(images_path, "seg_test/seg_test"),
        transform=config.build_val_transform(training_stats["mean"], training_stats["std"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    model = config.build_model(num_classes=len(test_dataset.classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    class_weights = checkpoint.get("class_weights")
    if class_weights is not None:
        class_weights = class_weights.to(DEVICE, dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    running_loss = 0.0
    running_count = 0
    all_labels = []
    all_logits = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            running_loss += loss.item()
            running_count += 1
            all_labels.append(labels.cpu().int())
            all_logits.append(logits.cpu())

    test_loss = running_loss / running_count
    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()
    all_preds = np.argmax(all_logits, axis=1)

    accuracy = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=test_dataset.class_names,
        zero_division=0,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {config_module_name}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print()
    print(report)


if __name__ == "__main__":
    main()
