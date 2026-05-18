import argparse
import time

import numpy as np
import torch
from numpy import dtype
from sklearn.metrics import f1_score, multilabel_confusion_matrix, accuracy_score, hamming_loss, jaccard_score
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.statistics import compute_mean_std, find_best_thresholds
from src.models.basic_cnn import BasicCNN
from src.data.dataset import SeverstalSteelDefectDataset
from src.data.split import stratified_train_test_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE == 'cuda' and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision('high')

def main():
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--images-path", type=str,  default="data/severstal/train_images", help="path to directory containing images")
    parser.add_argument("--label-csv", type=str, default="data/severstal/train.csv", help="path to CSV file containing labels")
    parser.add_argument("--test-size", type=float, default=0.2, help="test set size for stratified split")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for the dataloaders")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--validate-every-n-epochs", type=int, default=1, help="validate every n epochs")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Set up dataloaders
    dataset = SeverstalSteelDefectDataset(
        images_path=args.images_path,
        label_csv=args.label_csv
    )
    train, val = stratified_train_test_split(
        dataset,
        targets=dataset.targets,
        test_size=args.test_size,
        random_state=args.seed
    )

    print(f"Split dataset into {len(train)} training and {len(val)} validation samples")

    # Compute training set class weights

    class_weights = torch.tensor(compute_class_weight(
        class_weight="balanced",
        classes=dataset.classes,
        y=np.concatenate(dataset.labels[train.indices])
    ))

    targets = torch.tensor(dataset.targets[train.indices])
    positive_weights = torch.sqrt((targets==0).sum(dim=0)/targets.sum(dim=0))

    print(f"Class weights: {class_weights}")
    print(f"Positive weight: {positive_weights}")

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True
    )

    # Compute training set statistics

    dataset.transform = transforms.v2.Compose([
        transforms.v2.ToImage(),
        transforms.v2.ToDtype(torch.float32)
    ])

    train_mean, train_std = compute_mean_std(train_loader)

    print(f"Training set statistics: mean={train_mean}, std={train_std}")

    # Set up dataset transforms
    dataset.transform = transforms.v2.Compose([
        transforms.v2.ToImage(),
        transforms.v2.ToDtype(torch.float32),
        transforms.v2.Normalize(mean=[train_mean], std=[train_std])
    ])

    # set up the model
    model = torch.compile(BasicCNN(num_classes=len(dataset.classes), in_channels=1, in_width=256, in_height=1600).to(DEVICE), mode='reduce-overhead')

    # set up the optimizer
    total_steps = args.epochs * len(train) * train_loader.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=total_steps)

    # set up the loss function
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(DEVICE))

    # training loop
    start_time = time.time_ns()
    for epoch in range(args.epochs):

        # training loop
        model.train()
        running_train_loss = 0.0
        running_train_count = 0
        for i, (images, labels, _) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images.to(DEVICE))
            loss = loss_fn(outputs, labels.to(DEVICE, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_train_loss += loss.item()
            running_train_count += 1

        # validation loop
        if (epoch + 1) % args.validate_every_n_epochs == 0:
            model.eval()

            running_val_loss = 0.0
            running_val_count = 0

            all_labels = []
            all_preds = []
            all_probs = []

            with torch.inference_mode():
                for i, (images, labels, _) in enumerate(val_loader):
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, dtype=torch.float32, non_blocking=True)

                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    probs = torch.sigmoid(logits)

                    running_val_loss += loss.item()
                    running_val_count += 1

                    all_labels.append(labels.cpu().int())
                    all_probs.append(probs.cpu())

            train_loss = running_train_loss / running_train_count
            val_loss = running_val_loss / running_val_count

            all_labels = torch.cat(all_labels).numpy()
            all_probs = torch.cat(all_probs).numpy()

            thresholds = find_best_thresholds(all_labels, all_probs)

            print("Best thresholds:", thresholds)
            print("Mean probs:", all_probs.mean(axis=0))
            print("Max probs:", all_probs.max(axis=0))

            all_preds = (all_probs >= thresholds).astype(int)

            # Multi-label set-level metrics
            subset_accuracy = accuracy_score(all_labels, all_preds)
            hamming = hamming_loss(all_labels, all_preds)

            sample_jaccard = jaccard_score(
                all_labels,
                all_preds,
                average="samples",
                zero_division=0,
            )

            sample_f1 = f1_score(
                all_labels,
                all_preds,
                average="samples",
                zero_division=0,
            )

            macro_f1 = f1_score(
                all_labels,
                all_preds,
                average="macro",
                zero_division=0,
            )

            micro_f1 = f1_score(
                all_labels,
                all_preds,
                average="micro",
                zero_division=0,
            )

            # One binary confusion matrix per class
            cms = multilabel_confusion_matrix(all_labels, all_preds)

            print(
                f"Epoch {epoch + 1}/{args.epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Subset Acc: {subset_accuracy:.4f}, "
                f"Hamming Loss: {hamming:.4f}, "
                f"Sample Jaccard: {sample_jaccard:.4f}, "
                f"Sample F1: {sample_f1:.4f}, "
                f"Macro F1: {macro_f1:.4f}, "
                f"Micro F1: {micro_f1:.4f}, "
                f"Time: {(time.time_ns() - start_time) / 1e6:.2f} ms"
            )

            for class_name, cm in zip(dataset.classes, cms):
                tn, fp, fn, tp = cm.ravel()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                f1 = (
                    2 * tp / (2 * tp + fp + fn)
                    if (2 * tp + fp + fn) > 0
                    else 0.0
                )

                print(
                    f"  Class {class_name}: "
                    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}, "
                    f"Precision={precision:.4f}, "
                    f"Recall={recall:.4f}, "
                    f"Specificity={specificity:.4f}, "
                    f"F1={f1:.4f}"
                )
            start_time = time.time_ns()

if __name__ == "__main__":
    main()