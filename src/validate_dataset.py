import argparse

import numpy as np
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
parser.add_argument("--baseline-class-weights", type=float, nargs=4, default=[1.97632312, 7.20304569, 0.34441748, 2.21372855], help="baseline class weights for comparison")
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
    test_size=args.test_size,
    random_state=args.seed
)

# Verify no overlapping samples
overlap = set(train.indices) & set(val.indices)
if len(overlap) > 0:
    print(f"Error: Training and validation sets have overlapping samples: {overlap}")
    exit(1)

# Iterate over data to verify images can be read
image_ids = np.array([id for _, _, id in dataset])

# Verify no overlapping image IDs
overlap = set.intersection(set(image_ids[train.indices]), set(image_ids[val.indices]))
if len(overlap) > 0:
    print(f"Error: Training and validation sets have {len(overlap)} overlapping image IDs: {overlap}")
    exit(1)

# Check for class distribution shifts
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=dataset.classes,
    y=np.concatenate(dataset.labels[train.indices])
)

if np.allclose(class_weights, args.baseline_class_weights, atol=1e-1):
    print("Class weights match baseline")
else:
    print(f"Class weights do not match baseline: {class_weights} vs {args.baseline_class_weights}")
    exit(1)

print("Validation successful")