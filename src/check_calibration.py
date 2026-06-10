import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import IntelImageClassificationDataset
from src.inference.prediction import load_inference_bundle, predict_batch_logits


def format_optional(value):
    return "" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="path to checkpoint to evaluate")
    parser.add_argument("--config", type=str, default=None, help="override config module from checkpoint")
    parser.add_argument("--images-path", type=str, default=None, help="path to Intel dataset root")
    parser.add_argument("--batch-size", type=int, default=64, help="evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="number of dataloader workers")
    parser.add_argument("--bins", type=int, default=10, help="number of confidence bins for ECE")
    parser.add_argument("--device", type=str, default="auto", help="device to use: auto, cpu, cuda, ...")
    args = parser.parse_args()

    if args.bins < 1:
        parser.error("--bins must be at least 1")

    bundle = load_inference_bundle(args.checkpoint, config_override=args.config, device=args.device)
    images_path = args.images_path or bundle.runtime_config.get("images_path", "data/intel")

    dataset = IntelImageClassificationDataset(
        images_path=os.path.join(images_path, "seg_test/seg_test"),
        transform=bundle.transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    confidences = []
    correct = []
    all_probs = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = predict_batch_logits(bundle.model, images, bundle.device)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)

            confidences.append(confidence.cpu())
            correct.append(predicted.cpu().eq(labels))
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    confidences = torch.cat(confidences)
    correct = torch.cat(correct).to(dtype=torch.float32)
    probabilities = torch.cat(all_probs)
    labels = torch.cat(all_labels).long()

    accuracy = correct.mean().item()
    average_confidence = confidences.mean().item()
    nll = torch.nn.functional.nll_loss(torch.log(probabilities.clamp_min(1e-12)), labels).item()
    one_hot = torch.nn.functional.one_hot(labels, num_classes=probabilities.shape[1]).float()
    brier = ((probabilities - one_hot) ** 2).sum(dim=1).mean().item()

    ece = 0.0
    rows = []
    for bin_index in range(args.bins):
        low = bin_index / args.bins
        high = (bin_index + 1) / args.bins
        if bin_index == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)

        count = int(mask.sum().item())
        if count == 0:
            rows.append((low, high, count, None, None, None))
            continue

        bin_accuracy = correct[mask].mean().item()
        bin_confidence = confidences[mask].mean().item()
        gap = abs(bin_accuracy - bin_confidence)
        ece += (count / len(confidences)) * gap
        rows.append((low, high, count, bin_accuracy, bin_confidence, gap))

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {bundle.config_module_name}")
    print(f"Samples: {len(confidences)}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Average confidence: {average_confidence:.6f}")
    print(f"ECE@{args.bins}: {ece:.6f}")
    print(f"NLL: {nll:.6f}")
    print(f"Brier: {brier:.6f}")
    print()
    print("| Confidence bin | Count | Accuracy | Avg confidence | Gap |")
    print("|---|---:|---:|---:|---:|")
    for low, high, count, bin_accuracy, bin_confidence, gap in rows:
        print(
            f"| ({low:.1f}, {high:.1f}] | {count} | "
            f"{format_optional(bin_accuracy)} | {format_optional(bin_confidence)} | {format_optional(gap)} |"
        )


if __name__ == "__main__":
    main()
