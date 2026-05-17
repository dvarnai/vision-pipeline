import argparse

import torch
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader

from src.data.dataset import SeverstalSteelDefectDataset
from src.data.split import stratified_train_test_split

# CLI arguments

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
parser.add_argument("--images-path", type=str,  default="data/train_images", help="path to directory containing images")
parser.add_argument("--label-csv", type=str, default="data/train.csv", help="path to CSV file containing labels")
parser.add_argument("--test-size", type=float, default=0.2, help="test set size for stratified split")
parser.add_argument("--batch-size", type=int, default=32, help="batch size for the dataloaders")
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
    labels=dataset.targets,
    test_size=args.test_size,
    random_state=args.seed
)

print(f"Split dataset into {len(train)} training and {len(val)} validation samples")

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=dataset.classes,
    y=dataset.targets[train.indices]
)

print(f"Class weights: {class_weights}")

train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)