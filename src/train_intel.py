import argparse
import time
import os

import numpy as np
import torch
from numpy import dtype
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import IntelImageClassificationDataset
from src.data.statistics import compute_mean_std
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
    parser.add_argument("--images-path", type=str,  default="data/intel", help="path to directory containing images")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for the dataloaders")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--validate-every-n-epochs", type=int, default=1, help="validate every n epochs")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Set up dataloaders
    train_dataset = IntelImageClassificationDataset(
        images_path=os.path.join(args.images_path, "seg_train/seg_train")
    )
    val_dataset = IntelImageClassificationDataset(
        images_path=os.path.join(args.images_path, "seg_test/seg_test")
    )

    print(f"Dataset has {len(train_dataset)} training and {len(val_dataset)} validation samples")

    # Compute training set class weights

    class_weights = torch.tensor(compute_class_weight(
        class_weight="balanced",
        classes=train_dataset.classes,
        y=np.array(train_dataset.labels)
    ), dtype=torch.float32)

    print(f"Class weights: {class_weights}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True
    )

    # Compute training set statistics

    train_dataset.transform = transforms.v2.Compose([
        transforms.v2.ToImage(),
        transforms.v2.Resize((150,150)),
        transforms.v2.ToDtype(torch.float32, scale=True),
        transforms.v2.ToPureTensor()
    ])

    train_mean, train_std = compute_mean_std(train_loader)

    print(f"Training set statistics: mean={train_mean}, std={train_std}")

    # Set up dataset transforms
    train_dataset.transform = val_dataset.transform = transforms.v2.Compose([
        transforms.v2.ToImage(),
        transforms.v2.Resize((150, 150)),
        transforms.v2.ToDtype(torch.float32, scale=True),
        transforms.v2.Normalize(mean=train_mean, std=train_std),
        transforms.v2.ToPureTensor()
    ])

    # set up the model
    model = BasicCNN(num_classes=len(train_dataset.classes), in_channels=3, in_width=150, in_height=150).to(DEVICE)

    # set up the optimizer
    total_steps = args.epochs * len(train_dataset) * train_loader.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=total_steps)

    # set up the loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # training loop
    start_time = time.time_ns()
    for epoch in range(args.epochs):

        # training loop
        model.train()
        running_train_loss = 0.0
        running_train_count = 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images.to(DEVICE))
            loss = loss_fn(outputs, labels.to(DEVICE))
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
                for i, (images, labels) in enumerate(val_loader):
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)

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
            all_preds = np.argmax(all_probs, axis=1)

            # Multi-class set-level metrics
            subset_accuracy = accuracy_score(all_labels, all_preds)

            weighted_f1 = f1_score(
                all_labels,
                all_preds,
                average="weighted",
                zero_division=0
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
            cms = confusion_matrix(all_labels, all_preds)

            print(
                f"Epoch {epoch + 1}/{args.epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Subset Acc: {subset_accuracy:.4f}, "
                f"Weighted F1: {weighted_f1:.4f}, "
                f"Macro F1: {macro_f1:.4f}, "
                f"Micro F1: {micro_f1:.4f}, "
                f"Time: {(time.time_ns() - start_time) / 1e6:.2f} ms"
            )

            print(classification_report(all_labels, all_preds, target_names=val_dataset.class_names))
            start_time = time.time_ns()

if __name__ == "__main__":
    main()