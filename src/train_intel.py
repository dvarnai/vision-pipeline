import argparse
import os
import random
import sched
import time
import wandb

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import IntelImageClassificationDataset
from src.data.statistics import compute_mean_std
from src.models.basic_cnn import BasicCNN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE == 'cuda' and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision('high')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--images-path", type=str,  default="data/intel", help="path to directory containing images")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for the dataloaders")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--num-workers", type=int, default=16, help="number of workers for dataloader")
    parser.add_argument("--validate-every-n-epochs", type=int, default=1, help="validate every n epochs")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
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

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
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
    train_dataset.transform = transforms.v2.Compose([
        transforms.v2.ToImage(),
        transforms.v2.Resize((150, 150)),
        #transforms.v2.RandomCrop(size=(100, 100), pad_if_needed=True),
        transforms.v2.RandomResizedCrop(size=(150, 150), scale=(0.8,1.0), ratio=(0.75,1.33)),
        transforms.v2.RandomHorizontalFlip(),
        transforms.v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.v2.ToDtype(torch.float32, scale=True),
        transforms.v2.Normalize(mean=train_mean, std=train_std),
        transforms.v2.ToPureTensor()
    ])

    val_dataset.transform = transforms.v2.Compose([
        transforms.v2.ToImage(),
        transforms.v2.Resize((150, 150)),
        transforms.v2.ToDtype(torch.float32, scale=True),
        transforms.v2.Normalize(mean=train_mean, std=train_std),
        transforms.v2.ToPureTensor()
    ])

    # set up the model
    model = torch.compile(BasicCNN(num_classes=len(train_dataset.classes), in_channels=3, in_width=150, in_height=150).to(DEVICE), mode='reduce-overhead')

    # set up the optimizer
    total_steps = args.epochs * len(train_dataset)
    #optimizer = torch.optim.SGD(model.parameters(), lr=5e-6, weight_decay=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-7, T_max=total_steps)

    # set up the loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # set up wandb
    run = wandb.init(
        entity="qvai",
        project="vision-pipeline-intel",
        config={
            "class_weights": class_weights,
            "total_steps": total_steps ,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "weight_decay": optimizer.param_groups[0]['weight_decay'],
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "epochs": args.epochs,
            "batch_size": train_loader.batch_size,
            "num_classes": len(train_dataset.classes),
            "transforms": [str(transform) for transform in train_dataset.transform.transforms],
            "model": str(model),
            "architecture": str(model.architecture()),
            "in_channels": 3,
            "in_width": 150,
            "in_height": 150,
        }
    )
    run.watch(model, loss_fn, log="all", log_freq=10)

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
            if scheduler:
                scheduler.step()
            running_train_loss += loss.item()
            running_train_count += 1

        # validation loop
        if (epoch + 1) % args.validate_every_n_epochs == 0:
            model.eval()

            running_val_loss = 0.0
            running_val_count = 0

            all_labels = []
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
            report = classification_report(all_labels, all_preds, target_names=val_dataset.class_names, output_dict=True)

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

            run.log({
                "Learning Rate": scheduler.get_last_lr()[0],
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Subset Acc": subset_accuracy,
                "Weighted F1": weighted_f1,
                "Macro F1": macro_f1,
                "Micro F1": micro_f1,
                "Classes": report
            }, step=epoch)
            start_time = time.time_ns()

    run.finish()

if __name__ == "__main__":
    main()