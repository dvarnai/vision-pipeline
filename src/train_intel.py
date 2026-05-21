import argparse
import importlib
import os
import random
import time
import wandb

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader

from src.core.checkpoint import load_checkpoint, read_config_source, save_checkpoint
from src.data.dataset import IntelImageClassificationDataset
from src.data.statistics import compute_mean_std

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if DEVICE == 'cuda' and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision('high')

def resolve_run_name(config_module_name, config):
    return config.run_name or config_module_name.rsplit(".", 1)[-1]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        help="config module containing build_config(); required unless --resume is used",
    )
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="override random seed for reproducibility")
    parser.add_argument("--images-path", type=str,  default=None, help="path to directory containing images")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size for the dataloaders")
    parser.add_argument("--epochs", type=int, default=None, help="override number of training epochs")
    parser.add_argument("--num-workers", type=int, default=None, help="override number of workers for dataloader")
    parser.add_argument("--validate-every-n-epochs", type=int, default=None, help="override validation interval")
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=None, help="override checkpoint interval")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="override checkpoint directory")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="compile model with torch.compile")
    args = parser.parse_args()

    # A resumed run can recover its config module from the checkpoint metadata.
    resume_checkpoint = load_checkpoint(args.resume) if args.resume is not None else None
    checkpoint_runtime_config = resume_checkpoint.get("runtime_config", {}) if resume_checkpoint is not None else {}
    checkpoint_config_module = resume_checkpoint.get("config_module") if resume_checkpoint is not None else None

    if args.config is None and checkpoint_config_module is None:
        parser.error("config is required unless --resume points to a checkpoint with config_module")

    # Prefer an explicitly provided config; otherwise use the module saved in the checkpoint.
    config_module_name = args.config or checkpoint_config_module
    config_module = importlib.import_module(config_module_name)
    config = config_module.build_config()
    run_name = resolve_run_name(config_module_name, config)
    config_file, config_source = read_config_source(config_module)

    # Keep the originally saved config source attached to resumed checkpoints for reproducibility.
    if resume_checkpoint is not None and args.config is None:
        config_file = resume_checkpoint.get("config_file", config_file)
        config_source = resume_checkpoint.get("config_source", config_source)

    # Runtime values resolve in this order: CLI override, checkpoint runtime metadata, config default.
    seed = args.seed if args.seed is not None else checkpoint_runtime_config.get("seed", config.seed)
    batch_size = args.batch_size if args.batch_size is not None else checkpoint_runtime_config.get("batch_size", config.batch_size)
    epochs = args.epochs if args.epochs is not None else checkpoint_runtime_config.get("epochs", config.epochs)
    num_workers = args.num_workers if args.num_workers is not None else checkpoint_runtime_config.get("num_workers", config.num_workers)
    validate_every_n_epochs = (
        args.validate_every_n_epochs
        if args.validate_every_n_epochs is not None
        else checkpoint_runtime_config.get("validate_every_n_epochs", config.validate_every_n_epochs)
    )
    checkpoint_every_n_epochs = (
        args.checkpoint_every_n_epochs
        if args.checkpoint_every_n_epochs is not None
        else checkpoint_runtime_config.get("checkpoint_every_n_epochs", config.checkpoint_every_n_epochs)
    )
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else checkpoint_runtime_config.get("checkpoint_dir", config.checkpoint_dir)
    )
    images_path = args.images_path if args.images_path is not None else checkpoint_runtime_config.get("images_path", "data/intel")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Set up dataloaders
    train_dataset = IntelImageClassificationDataset(
        images_path=os.path.join(images_path, "seg_train/seg_train")
    )
    val_dataset = IntelImageClassificationDataset(
        images_path=os.path.join(images_path, "seg_test/seg_test")
    )

    print(f"Dataset has {len(train_dataset)} training and {len(val_dataset)} validation samples")

    # Compute training set class weights

    # Reuse checkpoint class weights so resumed loss behavior matches the original run.
    if resume_checkpoint is not None and "class_weights" in resume_checkpoint:
        class_weights = resume_checkpoint["class_weights"].detach().cpu().to(dtype=torch.float32)
    else:
        class_weights = torch.tensor(compute_class_weight(
            class_weight="balanced",
            classes=train_dataset.classes,
            y=np.array(train_dataset.labels)
        ), dtype=torch.float32)

    print(f"Class weights: {class_weights}")

    g = torch.Generator()
    g.manual_seed(seed if seed is not None else 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Compute training set statistics

    train_dataset.transform = config.stats_transform

    # Reuse saved normalization stats instead of recomputing them on resume.
    checkpoint_training_stats = resume_checkpoint.get("training_stats", {}) if resume_checkpoint is not None else {}
    if "mean" in checkpoint_training_stats and "std" in checkpoint_training_stats:
        train_mean = checkpoint_training_stats["mean"]
        train_std = checkpoint_training_stats["std"]
    elif config.train_mean is not None and config.train_std is not None:
        train_mean = config.train_mean
        train_std = config.train_std
    else:
        train_mean, train_std = compute_mean_std(train_loader)

    print(f"Training set statistics: mean={train_mean}, std={train_std}")

    # Set up dataset transforms
    train_dataset.transform = config.build_train_transform(train_mean, train_std)
    val_dataset.transform = config.build_val_transform(train_mean, train_std)

    # set up the model
    raw_model = config.build_model(num_classes=len(train_dataset.classes))
    model_architecture = raw_model.architecture() if hasattr(raw_model, "architecture") else str(raw_model)
    # Load weights into the raw model before torch.compile wraps it.
    if resume_checkpoint is not None:
        raw_model.load_state_dict(resume_checkpoint["model_state_dict"])
    model = raw_model.to(DEVICE)
    #if args.compile:
    #    model = torch.compile(model, mode="reduce-overhead")

    # set up the optimizer
    total_steps = epochs * len(train_loader)
    optimizer = config.build_optimizer(model)
    scheduler = config.build_scheduler(optimizer, total_steps)
    # Optimizer and scheduler are built from the current config, then restored from checkpoint state.
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if scheduler is not None and resume_checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    resume_epoch = int(resume_checkpoint["epoch"]) if resume_checkpoint is not None else 0
    runtime_config = {
        "run_name": run_name,
        "config_module": config_module_name,
        "config_file": config_file,
        "resume": args.resume,
        "resume_epoch": resume_epoch,
        "images_path": images_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "validate_every_n_epochs": validate_every_n_epochs,
        "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
        "checkpoint_dir": checkpoint_dir,
        "seed": seed,
        "total_steps": total_steps,
        "in_channels": config.in_channels,
        "in_width": config.in_width,
        "in_height": config.in_height,
    }

    # set up the loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # set up wandb
    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=run_name,
        config={
            "run_name": run_name,
            "config_module": config_module_name,
            "class_weights": class_weights.tolist(),
            "total_steps": total_steps,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "weight_decay": optimizer.param_groups[0]['weight_decay'],
            "optimizer": optimizer.__class__.__name__,
            "optimizer_defaults": optimizer.defaults,
            "scheduler": scheduler.__class__.__name__ if scheduler is not None else None,
            "scheduler_state": scheduler_state,
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "num_workers": num_workers,
            "validate_every_n_epochs": validate_every_n_epochs,
            "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
            "checkpoint_dir": checkpoint_dir,
            "seed": seed,
            "num_classes": len(train_dataset.classes),
            "stats_transform": str(config.stats_transform),
            "train_transform": str(train_dataset.transform),
            "val_transform": str(val_dataset.transform),
            "model": str(model),
            "architecture": str(model_architecture),
            "in_channels": config.in_channels,
            "in_width": config.in_width,
            "in_height": config.in_height,
        }
    )
    run.watch(model, loss_fn, log="all", log_freq=10)

    # training loop
    # History starts with every completed epoch from the checkpoint, then grows one epoch at a time.
    history = list(resume_checkpoint.get("history", [])) if resume_checkpoint is not None else []
    start_time = time.time_ns()
    for epoch in range(resume_epoch, epochs):
        epoch_number = epoch + 1

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

        train_loss = running_train_loss / running_train_count
        epoch_history = {
            "epoch": epoch_number,
            "train_loss": train_loss,
            "learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else None,
        }

        # validation loop
        if epoch_number % validate_every_n_epochs == 0:
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
                f"Epoch {epoch_number}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Subset Acc: {subset_accuracy:.4f}, "
                f"Weighted F1: {weighted_f1:.4f}, "
                f"Macro F1: {macro_f1:.4f}, "
                f"Micro F1: {micro_f1:.4f}, "
                f"Time: {(time.time_ns() - start_time) / 1e6:.2f} ms"
            )

            epoch_history.update({
                "val_loss": val_loss,
                "subset_accuracy": subset_accuracy,
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "classification_report": report,
            })

            run.log({
                "Learning Rate": epoch_history["learning_rate"],
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Subset Acc": subset_accuracy,
                "Weighted F1": weighted_f1,
                "Macro F1": macro_f1,
                "Micro F1": micro_f1,
                "Classes": report
            }, step=epoch)
            start_time = time.time_ns()

        # Append before checkpointing so epoch N checkpoints contain history through epoch N.
        history.append(epoch_history)

        if checkpoint_every_n_epochs and epoch_number % checkpoint_every_n_epochs == 0:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                run_name=run_name,
                epoch=epoch_number,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                config_module_name=config_module_name,
                config_file=config_file,
                config_source=config_source,
                class_weights=class_weights,
                train_mean=train_mean,
                train_std=train_std,
                runtime_config=runtime_config,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    run.finish()

if __name__ == "__main__":
    main()
