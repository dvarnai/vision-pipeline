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

Start Jupyter if you want to inspect notebooks or experiment interactively:

```bash
jupyter lab
```

Run a training job by passing a config module:

```bash
python -m src.train_intel src.configs.intel_baseline_adam_cosine
```

Weights & Biases logging is enabled by the training script. For a local reproducibility run without syncing to W&B, set:

```bash
export WANDB_MODE=offline
```

## Reproduce the Best Run

The best saved validation result so far is the discriminative ViT fine-tuning run:

- Config: `src.configs.intel_vit_transfer_4`
- Best checkpoint: `checkpoints/intel_vit_transfer_4_epoch_0060.pt`
- Best validation epoch: `60`
- Validation loss: `0.1412`
- Accuracy: `0.9527`
- Weighted F1: `0.9525`
- Macro F1: `0.9536`
- Micro F1: `0.9527`

The run uses `ViT-B-16` with `torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1`, `224x224` inputs, ImageNet normalization, SGD, cosine annealing, and discriminative learning rates:

- ViT encoder learning rate: `1e-5`
- Classifier head learning rate: `1e-4`
- Scheduler minimum learning rate: `1e-6`
- Batch size: `64`
- Epochs: `100`
- Validation interval: every epoch
- Checkpoint interval: every `10` epochs
- Seed: not fixed in the original run

To reproduce the run from scratch, keep the Intel dataset under `data/intel` as shown above and run:

```bash
export WANDB_MODE=offline
python -m src.train_intel src.configs.intel_vit_transfer_4
```

The original run used `num_workers=16`. On smaller machines, override that without changing model behavior:

```bash
python -m src.train_intel src.configs.intel_vit_transfer_4 --num-workers 4
```

To evaluate the saved best checkpoint on the held-out Intel `seg_test` split:

```bash
python -m src.test_intel checkpoints/intel_vit_transfer_4_epoch_0060.pt
```

If the dataset is somewhere else, pass the dataset root explicitly:

```bash
python -m src.train_intel src.configs.intel_vit_transfer_4 --images-path /path/to/intel
python -m src.test_intel checkpoints/intel_vit_transfer_4_epoch_0060.pt --images-path /path/to/intel
```

Because the original run did not use a fixed seed, exact last-decimal metrics are not guaranteed when retraining from scratch. The checkpoint contains the config module name, config source, runtime settings, class weights, optimizer and scheduler state, epoch history, and normalization metadata needed to reproduce or resume the saved run.

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

The following experiments document the early custom-CNN track. They are useful for understanding the project history, but they are no longer the strongest result. The current best saved run is the ViT run documented above.

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

### Experiment 4: Early Average Pooling

Experiment 4 kept random horizontal flip augmentation enabled and added average pooling after the first convolution block:

```python
self.features = nn.Sequential(
    nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2), # [B, 8, in_height // 2, in_width // 2]

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

After 99 epochs, Experiment 4 reached:

- Train loss: `0.7336`
- Validation loss: `1.4227`
- Accuracy: `0.6597`
- Weighted F1: `0.6513`
- Macro F1: `0.6537`
- Micro F1: `0.6597`
- Epoch time: `1347.10 ms`

This is the best result so far. Accuracy improved from `57.63%` to `65.97%` compared with Experiment 3, while epoch time dropped from `1886.78 ms` to `1347.10 ms`.

### Experiment 5: Early Max Pooling

Experiment 5 kept random horizontal flip augmentation enabled and replaced the early average pooling layer with max pooling:

```python
self.features = nn.Sequential(
    nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), # [B, 8, in_height // 2, in_width // 2]

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

After 100 epochs, Experiment 5 reached:

- Train loss: `0.7215`
- Validation loss: `1.4646`
- Accuracy: `0.6280`
- Weighted F1: `0.6298`
- Macro F1: `0.6288`
- Micro F1: `0.6280`
- Epoch time: `1536.76 ms`

This was worse than the early average pooling result from Experiment 4. Accuracy decreased from `65.97%` to `62.80%`, and weighted F1 decreased from `0.6513` to `0.6298`.

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

The project has a working Intel dataset class, configurable training and evaluation scripts, checkpoint-based reproducibility metadata, early CNN baselines, ResNet transfer-learning runs, and a current best ViT transfer-learning checkpoint.
