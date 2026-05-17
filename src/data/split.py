from typing import Iterable

from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import Subset, Dataset
import numpy as np

def stratified_train_test_split(dataset: Dataset, test_size: float, targets: Iterable | None = None, random_state: int | None = None) -> tuple[Subset, Subset]:

    indices = np.array(range(len(dataset)))
    targets = targets if targets is not None else np.array([dataset[i][1] for i in indices])

    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size], random_state=random_state)
    train_idx, val_idx = next(stratifier.split(indices.reshape(-1,1), targets))

    train_ds = Subset(dataset, train_idx.reshape(-1).tolist())
    val_ds = Subset(dataset, val_idx.reshape(-1).tolist())

    return train_ds, val_ds
