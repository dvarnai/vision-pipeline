import collections
import os

import numpy as np
from PIL import Image
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import pil_to_tensor

class SeverstalSteelDefectDataset(Dataset):
    def __init__(self, images_path: str, label_csv: str, transform: collections.abc.Callable[[Image.Image], torch.Tensor] | None = None, labels: list[int] = [1,2,3,4]):
        super().__init__()
        self.images_path = images_path
        self.transform = transform

        # load label file
        try:
            csv_data: DataFrame = pd.read_csv(label_csv, sep=",", iterator=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Label CSV file not found at {label_csv}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Label CSV file is empty: {label_csv}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing label CSV file: {label_csv}. Error: {e}")

        # group by ID as it's a multi-class dataset
        self.label_data = csv_data.groupby('ImageId')['ClassId'].apply(np.array).reset_index()

        # verify there are no missing images
        image_files = os.listdir(self.images_path)
        if not image_files:
            raise ValueError(f"No images found in directory: {self.images_path}")

        for index, sample in self.label_data.iterrows():
            if os.path.isfile(os.path.join(self.images_path, sample['ImageId'])):
                continue
            raise FileNotFoundError(f"Image file not found for sample: {sample['ImageId']}")

        # validate labels
        self.classes = np.unique(csv_data['ClassId'])
        if not np.all(np.isin(self.classes, labels)):
            raise ValueError(f"Label values must be one of {labels}, found: {self.classes}")

    @property
    def image_ids(self) -> np.ndarray:
        return self.label_data['ImageId'].to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self.label_data['ClassId'].to_numpy()

    @property
    def targets(self) -> np.ndarray:
        return np.array([self.compute_target(label) for label in self.labels])

    def compute_target(self, label):
        return torch.sum(torch.nn.functional.one_hot(torch.tensor(label)-1, num_classes=len(self.classes)), dim=0)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.label_data.iloc[index]
        with Image.open(os.path.join(self.images_path, sample['ImageId'])) as image:

            image = image.convert('L')

            if self.transform:
                image = self.transform(image)

            return pil_to_tensor(image), self.compute_target(sample['ClassId']), sample['ImageId']

    def __len__(self) -> int:
        return self.label_data.shape[0]