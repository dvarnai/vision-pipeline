from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class RunConfig:
    batch_size: int
    epochs: int
    num_workers: int
    validate_every_n_epochs: int
    checkpoint_every_n_epochs: int | None
    checkpoint_dir: str
    seed: Optional[int]
    in_channels: int
    in_width: int
    in_height: int
    stats_transform: Any
    build_model: Callable[[int], Any]
    build_train_transform: Callable[[Any, Any], Any]
    build_val_transform: Callable[[Any, Any], Any]
    build_optimizer: Callable[[Any], Any]
    build_scheduler: Callable[[Any, int], Any]
    train_mean: Any = None
    train_std: Any = None
    run_name: Optional[str] = None
    wandb_entity: str = "qvai"
    wandb_project: str = "vision-pipeline-intel"
