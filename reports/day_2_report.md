# Day 2 Report

## Project Focus

Updated the project direction to use the Intel Image Classification dataset instead of the Severstal Steel Defect Detection dataset:

https://www.kaggle.com/datasets/puneet6060/intel-image-classification

The main reason for the switch is that Severstal is not a straightforward image classification dataset. It is primarily a segmentation dataset and the labels are multi-label. It also has relatively few usable samples for the simpler learning goal of building a first end-to-end image classification pipeline.

For learning purposes, the Intel Image Classification dataset is a better fit because it is a standard single-label image classification problem with more examples and clearer class structure. Severstal is still a useful dataset, but it is better treated as a stretch goal after the basic classification pipeline is working.

The technical focus for day 2 was setting up the Intel dataset class, moving preprocessing into composable PyTorch/torchvision transforms, adding dataset normalization based on the training split, and running a first baseline CNN through a full training and validation loop.

## Completed

- Updated the dataset class so image preprocessing is handled by an optional transform pipeline.
- Added composable torchvision transforms for:
  - resizing
  - tensor conversion
  - normalization
- Added a utility for computing training-set pixel statistics.
- Computed mean and standard deviation from the training split before training.
- Applied normalization using the computed training-set mean and standard deviation.
- Set up the Intel Image Classification dataset class using class folders as labels.
- Added a first baseline CNN model in `src/models/basic_cnn.py`.
- Extended the training script with:
  - configurable epoch count
  - data loaders for train and validation splits
  - class-weight calculation
  - optimizer setup
  - learning-rate scheduler
  - classification loss
  - train/validation loss logging
  - accuracy and F1 metrics
  - classification report output
- Decided to move from the Severstal dataset to the Intel Image Classification dataset for the main learning track.
- Completed a 100-epoch baseline run on the Intel dataset.

No data augmentation has been added yet. The current preprocessing is focused on resizing, tensor conversion, and normalization.

## Data Normalization

Normalization is now part of the data preprocessing pipeline instead of being hard-coded inside the dataset class.

The training script first attaches a temporary transform pipeline:

```text
Resize -> ToTensor
```

This converts images into tensors so training-set statistics can be calculated consistently.

After computing the training-set mean and standard deviation, the transform pipeline is replaced with:

```text
Resize -> ToTensor -> Normalize
```

Using the training split for normalization statistics avoids leaking validation data into preprocessing decisions. The validation set uses the same normalization values learned from the training set.

## Dataset Class

The dataset class now delegates image conversion to the configured transform callable.

Current Intel-specific behavior:

- Reads images from class folders.
- Maps class folder names to integer class labels.
- Supports the six Intel classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`.
- Loads images with Pillow.
- Applies the configured transform pipeline when one is provided.
- Returns the transformed image and integer class label.

This makes the dataset more reusable because preprocessing can be changed from the training script without editing dataset loading logic.

## Training Pipeline

The training script now performs the following high-level steps:

1. Load the dataset.
2. Use the Intel training folder for training and the Intel test folder for validation.
3. Compute class weights from the training labels.
4. Build data loaders.
5. Compute training-set mean and standard deviation.
6. Configure the final composed preprocessing transform.
7. Initialize the baseline CNN.
8. Train for the configured number of epochs.
9. Report train and validation loss after each epoch.

The baseline model is intentionally simple. Its purpose is to verify that the data path, target format, normalization, loss function, and training loop work before moving to stronger architectures.

## Baseline Model

The current baseline uses two convolution blocks followed by adaptive average pooling and a small linear classifier:

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

This architecture is intentionally small. It is a baseline for verifying the training loop, not an expected final model.

## Baseline Results

After 100 epochs, the baseline produced:

```text
Epoch 100/100, Train Loss: 0.9596, Val Loss: 2.0620, Subset Acc: 0.4523, Weighted F1: 0.4258, Macro F1: 0.4304, Micro F1: 0.4523, Time: 2669.49 ms
```

Validation classification report:

```text
              precision    recall  f1-score   support

   buildings       0.25      0.66      0.36       437
      forest       0.51      0.97      0.67       474
     glacier       0.81      0.10      0.18       553
    mountain       0.60      0.41      0.49       525
         sea       0.69      0.31      0.43       510
      street       0.62      0.35      0.45       501

    accuracy                           0.45      3000
   macro avg       0.58      0.47      0.43      3000
weighted avg       0.59      0.45      0.43      3000
```

The result is better than random chance for a six-class classification task. A uniform random baseline would be expected to produce about `16.7%` accuracy, while this baseline reached `45.23%`.

## Notes

Data normalization should be computed only from the training split. Validation and test data should be transformed with the same statistics, but should not contribute to the values.

The preprocessing flow is now easier to extend with augmentation. Future transforms such as random crops, flips, resizing, or contrast adjustments can be added to the composed training transform while keeping validation preprocessing deterministic.

The Intel dataset should not be assumed to contain only `150x150` images. During inspection, at least one file, `mountain/7400.jpg`, did not match the expected `150x150` size. The input pipeline should therefore include an explicit resize step or the model should use adaptive pooling so batch shapes remain consistent.

## Next Steps

- Add data augmentation to the training transform.
- Add shape checks for transformed batches before model training.
- Add separate transform pipelines for training and validation so augmentation can be applied only to training samples.
- Investigate why the baseline heavily favors some classes, especially high recall for `forest` and low recall for `glacier`.
- Compare the basic CNN against a stronger architecture or a pretrained model.
- Return to the Severstal dataset later as a stretch goal for multi-label learning and segmentation.
