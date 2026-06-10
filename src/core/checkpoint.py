from pathlib import Path

import torch


def get_model_for_checkpoint(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def read_config_source(config_module):
    config_path = getattr(config_module, "__file__", None)
    if config_path is None:
        return None, None

    config_path = Path(config_path)
    return str(config_path), config_path.read_text()


def load_checkpoint(checkpoint_path, map_location="cpu"):
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def save_checkpoint(
        *,
        checkpoint_dir,
        run_name,
        epoch,
        model,
        optimizer,
        scheduler,
        history,
        config_module_name,
        config_file,
        config_source,
        class_weights,
        train_mean,
        train_std,
        runtime_config,
        label_contract=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}_epoch_{epoch:04d}.pt"

    payload = {
        "epoch": epoch,
        "model_state_dict": get_model_for_checkpoint(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": list(history),
        "config_module": config_module_name,
        "config_file": config_file,
        "config_source": config_source,
        "class_weights": class_weights.detach().cpu(),
        "training_stats": {
            "mean": train_mean,
            "std": train_std,
        },
        "runtime_config": runtime_config,
    }

    if label_contract is not None:
        payload["label_contract"] = label_contract

    torch.save(payload, checkpoint_path)

    return checkpoint_path
