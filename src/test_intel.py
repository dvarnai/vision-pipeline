import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import IntelImageClassificationDataset
from src.inference.prediction import load_inference_bundle, predict_batch_logits


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

    try:
        bundle = load_inference_bundle(args.checkpoint, config_override=args.config, device=DEVICE)
    except ValueError as exc:
        parser.error(str(exc))

    runtime_config = bundle.runtime_config
    images_path = args.images_path if args.images_path is not None else runtime_config.get("images_path", "data/intel")
    batch_size = args.batch_size if args.batch_size is not None else runtime_config.get("batch_size", bundle.config.batch_size)
    num_workers = args.num_workers if args.num_workers is not None else runtime_config.get("num_workers", bundle.config.num_workers)

    test_dataset = IntelImageClassificationDataset(
        images_path=os.path.join(images_path, "seg_test/seg_test"),
        transform=bundle.transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    model = bundle.model
    class_weights = bundle.class_weights
    if class_weights is not None:
        class_weights = class_weights.to(DEVICE, dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    running_loss = 0.0
    running_count = 0
    all_labels = []
    all_logits = []

    with torch.inference_mode():
        for images, labels in test_loader:
            labels = labels.to(DEVICE, non_blocking=True)

            logits = predict_batch_logits(model, images, bundle.device)
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
        target_names=bundle.class_names,
        zero_division=0,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {bundle.config_module_name}")
    print(f"Checkpoint epoch: {bundle.checkpoint_epoch}")
    print(f"Model version: {bundle.model_version}")
    print(f"Preprocessing version: {bundle.preprocessing_version}")
    print(f"Label contract: {bundle.label_contract_version}")
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
