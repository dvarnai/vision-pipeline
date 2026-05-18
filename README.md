# Vision Pipeline Learning Project

This repository is a hands-on project for learning how computer vision pipelines are built, trained, evaluated, and iterated on.

The main learning track now uses the [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset from Kaggle. It is a standard single-label image classification dataset, which makes it a better fit for learning the core training pipeline before moving on to harder tasks.

The earlier Severstal Steel Defect Detection dataset is still useful, but it is better treated as a stretch goal because it is a multi-label segmentation-oriented dataset.

## Goal

Build a practical vision pipeline that can classify natural scene images into one of the Intel dataset classes:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

This keeps the first version focused on the core steps of a vision workflow: dataset loading, preprocessing, training, validation, and metric reporting.

## Dataset

Dataset source:

https://www.kaggle.com/datasets/puneet6060/intel-image-classification

The dataset contains images organized by class folders. Raw dataset files should not be committed to this repository. After downloading and extracting the Kaggle dataset, keep the files in this layout:

```text
data/
  intel/
    seg_train/
      seg_train/
        buildings/
        forest/
        glacier/
        mountain/
        sea/
        street/
    seg_test/
      seg_test/
        buildings/
        forest/
        glacier/
        mountain/
        sea/
        street/
```

The project uses `seg_train` for training and `seg_test` for validation. Not all images should be assumed to be exactly `150x150`, so the preprocessing pipeline resizes images before batching.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -e .
```

Start Jupyter:

```bash
jupyter lab
```

Run the Intel baseline training script:

```bash
python -m src.train_intel
```

## Current Pipeline

The current pipeline includes:

1. Loading the Intel dataset from class folders
2. Mapping folder names to integer class labels
3. Building training and validation data loaders
4. Resizing images and converting them to tensors
5. Computing training-set normalization statistics
6. Training a small baseline CNN
7. Evaluating with accuracy and F1 scores

No data augmentation is used yet.

## Baseline Model

The first baseline is intentionally small:

```python
self.features = nn.Sequential(
    nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),

    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),

    nn.AdaptiveAvgPool2d((1, 1))
)

self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(0.2),
    nn.Linear(16, num_classes),
)
```

After 100 epochs, the baseline reached:

- Train loss: `0.9596`
- Validation loss: `2.0620`
- Accuracy: `0.4523`
- Weighted F1: `0.4258`
- Macro F1: `0.4304`
- Micro F1: `0.4523`

This is better than random chance for a six-class problem, where a uniform random classifier would be around `0.1667` accuracy.

## Initial Scope

The first version does not use data augmentation or pretrained models. The priority is to learn the major moving parts of a vision pipeline:

- data ingestion
- labeling strategy
- train/validation split
- preprocessing and augmentation
- model training
- evaluation
- experiment tracking

## Stretch Goal

A future extension may revisit the Severstal competition objective and train a model for multi-label classification or pixel-level defect segmentation.

## Status

The project has a working Intel dataset class, a training/validation loop, and a first baseline CNN result.
