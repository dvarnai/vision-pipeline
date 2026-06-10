import argparse
import csv
import importlib
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.core.checkpoint import load_checkpoint
from src.data.dataset import IntelImageClassificationDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda" and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class ModelRun:
    name: str
    checkpoint_path: str
    config_module_name: str
    epoch: int | None
    logits: np.ndarray
    probs: np.ndarray
    preds: np.ndarray
    confidence: np.ndarray
    true_prob: np.ndarray


def resolve_runtime_value(cli_value, runtime_config, config, key):
    if cli_value is not None:
        return cli_value
    if key in runtime_config:
        return runtime_config[key]
    return getattr(config, key)


def build_dataset(config, checkpoint, images_path):
    training_stats = checkpoint.get("training_stats", {})
    if "mean" not in training_stats or "std" not in training_stats:
        raise ValueError("checkpoint is missing training_stats mean/std required for transforms")

    return IntelImageClassificationDataset(
        images_path=os.path.join(images_path, "seg_test/seg_test"),
        transform=config.build_val_transform(training_stats["mean"], training_stats["std"]),
    )


def load_run(
    *,
    name,
    checkpoint_path,
    config_override,
    images_path_override,
    batch_size_override,
    num_workers_override,
):
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    runtime_config = checkpoint.get("runtime_config", {})

    config_module_name = config_override or checkpoint.get("config_module")
    if config_module_name is None:
        raise ValueError(f"{checkpoint_path} does not contain config_module; pass an explicit config override")

    config_module = importlib.import_module(config_module_name)
    config = config_module.build_config()

    images_path = images_path_override if images_path_override is not None else runtime_config.get("images_path", "data/intel")
    batch_size = resolve_runtime_value(batch_size_override, runtime_config, config, "batch_size")
    num_workers = resolve_runtime_value(num_workers_override, runtime_config, config, "num_workers")

    dataset = build_dataset(config, checkpoint, images_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    model = config.build_model(num_classes=len(dataset.classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_logits = []
    with torch.inference_mode():
        for images, _labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            all_logits.append(model(images).cpu())

    logits = torch.cat(all_logits).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    labels = np.array(dataset.labels, dtype=np.int64)
    true_prob = probs[np.arange(len(labels)), labels]

    return (
        ModelRun(
            name=name,
            checkpoint_path=checkpoint_path,
            config_module_name=config_module_name,
            epoch=checkpoint.get("epoch"),
            logits=logits,
            probs=probs,
            preds=preds,
            confidence=confidence,
            true_prob=true_prob,
        ),
        dataset,
    )


def validate_same_dataset(left_dataset, right_dataset):
    if left_dataset.class_names.tolist() != right_dataset.class_names.tolist():
        raise ValueError("checkpoint configs produced different class name order")
    if left_dataset.labels != right_dataset.labels:
        raise ValueError("checkpoint configs produced different label order")
    if [Path(path).name for path in left_dataset.image_paths] != [Path(path).name for path in right_dataset.image_paths]:
        raise ValueError("checkpoint configs produced different image order")


def accuracy(labels, preds, mask):
    count = int(np.sum(mask))
    if count == 0:
        return None
    return float(np.mean(preds[mask] == labels[mask]))


def outcome_for(cnn_correct, vit_correct):
    if cnn_correct and vit_correct:
        return "both_correct"
    if cnn_correct and not vit_correct:
        return "cnn_wins"
    if vit_correct and not cnn_correct:
        return "vit_wins"
    return "both_wrong"


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_class(labels, class_names, cnn_run, vit_run):
    rows = []
    cnn_correct = cnn_run.preds == labels
    vit_correct = vit_run.preds == labels

    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        count = int(np.sum(mask))
        cnn_wins = int(np.sum(mask & cnn_correct & ~vit_correct))
        vit_wins = int(np.sum(mask & vit_correct & ~cnn_correct))
        both_correct = int(np.sum(mask & cnn_correct & vit_correct))
        both_wrong = int(np.sum(mask & ~cnn_correct & ~vit_correct))
        rows.append(
            {
                "slice": class_name,
                "count": count,
                "cnn_accuracy": accuracy(labels, cnn_run.preds, mask),
                "vit_accuracy": accuracy(labels, vit_run.preds, mask),
                "cnn_wins": cnn_wins,
                "vit_wins": vit_wins,
                "net_cnn_wins": cnn_wins - vit_wins,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
            }
        )

    return sorted(rows, key=lambda row: row["net_cnn_wins"], reverse=True)


def summarize_confusion_pairs(labels, class_names, cnn_run, vit_run):
    rows = defaultdict(lambda: Counter())
    cnn_correct = cnn_run.preds == labels
    vit_correct = vit_run.preds == labels

    for index, label in enumerate(labels):
        key = {
            "true_class": class_names[label],
            "cnn_pred": class_names[cnn_run.preds[index]],
            "vit_pred": class_names[vit_run.preds[index]],
        }
        key_tuple = tuple(key.items())
        rows[key_tuple][outcome_for(cnn_correct[index], vit_correct[index])] += 1

    flattened = []
    for key_tuple, counts in rows.items():
        row = dict(key_tuple)
        row["count"] = sum(counts.values())
        row["cnn_wins"] = counts["cnn_wins"]
        row["vit_wins"] = counts["vit_wins"]
        row["net_cnn_wins"] = counts["cnn_wins"] - counts["vit_wins"]
        row["both_correct"] = counts["both_correct"]
        row["both_wrong"] = counts["both_wrong"]
        flattened.append(row)

    return sorted(flattened, key=lambda row: (abs(row["net_cnn_wins"]), row["count"]), reverse=True)


def build_example_rows(labels, image_paths, class_names, cnn_run, vit_run):
    rows = []
    cnn_correct = cnn_run.preds == labels
    vit_correct = vit_run.preds == labels

    for index, image_path in enumerate(image_paths):
        outcome = outcome_for(cnn_correct[index], vit_correct[index])
        rows.append(
            {
                "index": index,
                "outcome": outcome,
                "true_class": class_names[labels[index]],
                "cnn_pred": class_names[cnn_run.preds[index]],
                "vit_pred": class_names[vit_run.preds[index]],
                "cnn_confidence": float(cnn_run.confidence[index]),
                "vit_confidence": float(vit_run.confidence[index]),
                "cnn_true_prob": float(cnn_run.true_prob[index]),
                "vit_true_prob": float(vit_run.true_prob[index]),
                "true_prob_margin_cnn_minus_vit": float(cnn_run.true_prob[index] - vit_run.true_prob[index]),
                "image_path": image_path,
            }
        )

    return rows


def copy_examples(rows, output_dir, limit):
    for outcome in ("cnn_wins", "vit_wins"):
        selected = sorted(
            (row for row in rows if row["outcome"] == outcome),
            key=lambda row: abs(row["true_prob_margin_cnn_minus_vit"]),
            reverse=True,
        )[:limit]

        outcome_dir = output_dir / outcome
        outcome_dir.mkdir(parents=True, exist_ok=True)
        for row in selected:
            source = Path(row["image_path"])
            destination = outcome_dir / (
                f"{row['index']:05d}_true-{row['true_class']}"
                f"_cnn-{row['cnn_pred']}_vit-{row['vit_pred']}{source.suffix}"
            )
            shutil.copy2(source, destination)


def print_top(title, rows, key, limit):
    print()
    print(title)
    for row in sorted(rows, key=lambda item: item[key], reverse=True)[:limit]:
        print(row)


def print_rows(title, rows, limit):
    print()
    print(title)
    for row in rows[:limit]:
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Intel test slices where a CNN transfer checkpoint wins vs a ViT transfer checkpoint."
    )
    parser.add_argument("--cnn-checkpoint", required=True, help="path to the CNN transfer checkpoint")
    parser.add_argument("--vit-checkpoint", required=True, help="path to the ViT transfer checkpoint")
    parser.add_argument("--cnn-config", default=None, help="override CNN config module from checkpoint")
    parser.add_argument("--vit-config", default=None, help="override ViT config module from checkpoint")
    parser.add_argument("--images-path", default=None, help="override Intel dataset root")
    parser.add_argument("--batch-size", type=int, default=None, help="override evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="override dataloader workers")
    parser.add_argument("--output-dir", default="reports/transfer_slice_inspection", help="directory for CSVs/examples")
    parser.add_argument("--example-limit", type=int, default=24, help="number of win examples to copy per model")
    parser.add_argument("--top-k", type=int, default=10, help="number of top slices to print")
    args = parser.parse_args()

    cnn_run, cnn_dataset = load_run(
        name="cnn",
        checkpoint_path=args.cnn_checkpoint,
        config_override=args.cnn_config,
        images_path_override=args.images_path,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
    )
    vit_run, vit_dataset = load_run(
        name="vit",
        checkpoint_path=args.vit_checkpoint,
        config_override=args.vit_config,
        images_path_override=args.images_path,
        batch_size_override=args.batch_size,
        num_workers_override=args.num_workers,
    )
    validate_same_dataset(cnn_dataset, vit_dataset)

    labels = np.array(cnn_dataset.labels, dtype=np.int64)
    class_names = cnn_dataset.class_names.tolist()
    image_paths = cnn_dataset.image_paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_rows = summarize_by_class(labels, class_names, cnn_run, vit_run)
    pair_rows = summarize_confusion_pairs(labels, class_names, cnn_run, vit_run)
    example_rows = build_example_rows(labels, image_paths, class_names, cnn_run, vit_run)

    write_csv(
        output_dir / "class_slices.csv",
        class_rows,
        [
            "slice",
            "count",
            "cnn_accuracy",
            "vit_accuracy",
            "cnn_wins",
            "vit_wins",
            "net_cnn_wins",
            "both_correct",
            "both_wrong",
        ],
    )
    write_csv(
        output_dir / "prediction_pair_slices.csv",
        pair_rows,
        ["true_class", "cnn_pred", "vit_pred", "count", "cnn_wins", "vit_wins", "net_cnn_wins", "both_correct", "both_wrong"],
    )
    write_csv(
        output_dir / "examples.csv",
        example_rows,
        [
            "index",
            "outcome",
            "true_class",
            "cnn_pred",
            "vit_pred",
            "cnn_confidence",
            "vit_confidence",
            "cnn_true_prob",
            "vit_true_prob",
            "true_prob_margin_cnn_minus_vit",
            "image_path",
        ],
    )

    if args.example_limit > 0:
        copy_examples(example_rows, output_dir / "examples", args.example_limit)

    cnn_correct = cnn_run.preds == labels
    vit_correct = vit_run.preds == labels
    outcome_counts = Counter(row["outcome"] for row in example_rows)

    print(f"Device: {DEVICE}")
    print(f"CNN checkpoint: {cnn_run.checkpoint_path} ({cnn_run.config_module_name}, epoch={cnn_run.epoch})")
    print(f"ViT checkpoint: {vit_run.checkpoint_path} ({vit_run.config_module_name}, epoch={vit_run.epoch})")
    print(f"Samples: {len(labels)}")
    print(f"CNN accuracy: {np.mean(cnn_correct):.4f}")
    print(f"ViT accuracy: {np.mean(vit_correct):.4f}")
    print(f"Both correct: {outcome_counts['both_correct']}")
    print(f"CNN wins: {outcome_counts['cnn_wins']}")
    print(f"ViT wins: {outcome_counts['vit_wins']}")
    print(f"Both wrong: {outcome_counts['both_wrong']}")
    print(f"Output directory: {output_dir}")

    print_top("Top class slices for CNN wins", class_rows, "cnn_wins", args.top_k)
    print_top("Top class slices for ViT wins", class_rows, "vit_wins", args.top_k)
    print_rows("Top prediction-pair slices by absolute net wins", pair_rows, args.top_k)


if __name__ == "__main__":
    main()
