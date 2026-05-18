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

Random horizontal flip has been tested as a first augmentation experiment, but it did not materially improve validation performance.

## Experiments

### Experiment 1: `1x1` Adaptive Pooling

The first baseline is intentionally small. It uses two convolution blocks followed by `1x1` adaptive average pooling:

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

After 100 epochs, Experiment 1 reached:

- Train loss: `0.9596`
- Validation loss: `2.0620`
- Accuracy: `0.4523`
- Weighted F1: `0.4258`
- Macro F1: `0.4304`
- Micro F1: `0.4523`

This is better than random chance for a six-class problem, where a uniform random classifier would be around `0.1667` accuracy.

### Experiment 2: `2x2` Adaptive Pooling

Increasing adaptive average pooling from `1x1` to `2x2` preserved more spatial information before the classifier:

```python
self.features = nn.Sequential(
    nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),

    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),

    nn.AdaptiveAvgPool2d((2, 2))
)

self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(0.2),
    nn.Linear(16 * 2 * 2, num_classes),
)
```

After 100 epochs, Experiment 2 reached:

- Train loss: `0.7920`
- Validation loss: `1.8334`
- Accuracy: `0.5860`
- Weighted F1: `0.5723`
- Macro F1: `0.5753`
- Micro F1: `0.5860`

This improved validation accuracy from `45.23%` to `58.60%`.

### Experiment 3: Random Horizontal Flip

Experiment 3 kept the `2x2` adaptive pooling model and added random horizontal flip augmentation to the training transform.

After 100 epochs, Experiment 3 reached:

- Train loss: `0.7525`
- Validation loss: `1.8210`
- Accuracy: `0.5763`
- Weighted F1: `0.5718`
- Macro F1: `0.5739`
- Micro F1: `0.5763`

This did not materially improve validation performance. Accuracy decreased slightly from `58.60%` to `57.63%`, and weighted F1 stayed almost unchanged.

## Initial Scope

The first version does not use pretrained models. The priority is to learn the major moving parts of a vision pipeline:

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
