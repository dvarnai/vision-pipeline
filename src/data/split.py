from typing import Iterable

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset

def stratified_train_test_split(dataset: Dataset, test_size: float, labels: Iterable, random_state: int | None = None) -> tuple[Subset, Subset]:

    indices = list(range(len(dataset)))
    labels = labels if labels is not None else [dataset[i][1] for i in indices]

    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    return train_ds, val_ds
