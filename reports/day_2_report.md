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

Random horizontal flip has been tested as a first augmentation experiment. The base preprocessing remains focused on resizing, tensor conversion, and normalization.

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

## Experiment 1: Baseline Model

The first baseline uses two convolution blocks followed by `1x1` adaptive average pooling and a small linear classifier:

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

### Experiment 1 Results

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

## Experiment 2: Larger Adaptive Pooling

The second experiment kept the same convolution blocks but increased adaptive average pooling from `1x1` to `2x2`. This preserves more spatial information before the classifier while keeping the model small.

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

### Experiment 2 Results

After 100 epochs, increasing the adaptive pooling output size improved validation performance:

```text
Epoch 100/100, Train Loss: 0.7920, Val Loss: 1.8334, Subset Acc: 0.5860, Weighted F1: 0.5723, Macro F1: 0.5753, Micro F1: 0.5860, Time: 2813.71 ms
```

Validation classification report:

```text
              precision    recall  f1-score   support

   buildings       0.38      0.76      0.50       437
      forest       0.69      0.90      0.78       474
     glacier       0.85      0.26      0.40       553
    mountain       0.60      0.67      0.63       525
         sea       0.76      0.39      0.51       510
      street       0.65      0.61      0.63       501

    accuracy                           0.59      3000
   macro avg       0.65      0.60      0.58      3000
weighted avg       0.66      0.59      0.57      3000
```

Compared with Experiment 1, validation accuracy increased from `45.23%` to `58.60%`, and weighted F1 increased from `0.4258` to `0.5723`.

## Experiment 3: Random Horizontal Flip

The third experiment kept the Experiment 2 model and added random horizontal flip augmentation to the training transform. The validation transform remained deterministic.

### Experiment 3 Results

After 100 epochs, random horizontal flip did not materially improve validation performance:

```text
Epoch 100/100, Train Loss: 0.7525, Val Loss: 1.8210, Subset Acc: 0.5763, Weighted F1: 0.5718, Macro F1: 0.5739, Micro F1: 0.5763, Time: 1886.78 ms
```

Validation classification report:

```text
              precision    recall  f1-score   support

   buildings       0.32      0.78      0.45       437
      forest       0.63      0.96      0.76       474
     glacier       0.83      0.27      0.41       553
    mountain       0.79      0.41      0.54       525
         sea       0.73      0.52      0.61       510
      street       0.74      0.60      0.66       501

    accuracy                           0.58      3000
   macro avg       0.67      0.59      0.57      3000
weighted avg       0.69      0.58      0.57      3000
```

Compared with Experiment 2, validation accuracy decreased slightly from `58.60%` to `57.63%`, while weighted F1 stayed almost unchanged (`0.5723` to `0.5718`). This suggests random horizontal flip alone is not a meaningful improvement for this baseline.

## Experiment 4: Early Average Pooling

The fourth experiment kept random horizontal flip augmentation enabled and added `AvgPool2d` after the first convolution block. This reduces the spatial size of the intermediate feature maps before the second convolution, making the network cheaper to train while still keeping the `2x2` adaptive pooling output before the classifier.

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

### Experiment 4 Results

This produced the best validation result so far, despite making the network smaller and the epoch time faster:

```text
Epoch 99/100, Train Loss: 0.7336, Val Loss: 1.4227, Subset Acc: 0.6597, Weighted F1: 0.6513, Macro F1: 0.6537, Micro F1: 0.6597, Time: 1347.10 ms
```

Validation classification report:

```text
              precision    recall  f1-score   support

   buildings       0.49      0.78      0.60       437
      forest       0.70      0.96      0.81       474
     glacier       0.83      0.38      0.52       553
    mountain       0.72      0.64      0.68       525
         sea       0.62      0.67      0.64       510
      street       0.77      0.58      0.67       501

    accuracy                           0.66      3000
   macro avg       0.69      0.67      0.65      3000
weighted avg       0.70      0.66      0.65      3000
```

Compared with Experiment 3, validation accuracy increased from `57.63%` to `65.97%`, and weighted F1 increased from `0.5718` to `0.6513`. Epoch time also dropped from `1886.78 ms` to `1347.10 ms`.

## Experiment 5: Early Max Pooling

The fifth experiment kept random horizontal flip augmentation enabled and replaced the early average pooling layer from Experiment 4 with max pooling. The rest of the model stayed the same.

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

### Experiment 5 Results

Max pooling performed better than Experiment 3, which used random horizontal flip without early pooling, but worse than Experiment 4's average pooling result:

```text
Epoch 100/100, Train Loss: 0.7215, Val Loss: 1.4646, Subset Acc: 0.6280, Weighted F1: 0.6298, Macro F1: 0.6288, Micro F1: 0.6280, Time: 1536.76 ms
```

Validation classification report:

```text
              precision    recall  f1-score   support

   buildings       0.37      0.79      0.51       437
      forest       0.72      0.97      0.82       474
     glacier       0.81      0.54      0.65       553
    mountain       0.73      0.56      0.64       525
         sea       0.73      0.41      0.52       510
      street       0.74      0.55      0.63       501

    accuracy                           0.63      3000
   macro avg       0.68      0.64      0.63      3000
weighted avg       0.69      0.63      0.63      3000
```

Compared with Experiment 4, validation accuracy decreased from `65.97%` to `62.80%`, and weighted F1 decreased from `0.6513` to `0.6298`. Average pooling remains the better early pooling choice so far.

## Notes

Data normalization should be computed only from the training split. Validation and test data should be transformed with the same statistics, but should not contribute to the values.

The preprocessing flow is now easier to extend with augmentation. Future transforms such as random crops, flips, resizing, or contrast adjustments can be added to the composed training transform while keeping validation preprocessing deterministic.

The Intel dataset should not be assumed to contain only `150x150` images. During inspection, at least one file, `mountain/7400.jpg`, did not match the expected `150x150` size. The input pipeline should therefore include an explicit resize step or the model should use adaptive pooling so batch shapes remain consistent.

## Next Steps

- Try stronger or more targeted augmentation beyond random horizontal flip.
- Add shape checks for transformed batches before model training.
- Add separate transform pipelines for training and validation so augmentation can be applied only to training samples.
- Investigate why the baseline heavily favors some classes, especially high recall for `forest` and low recall for `glacier`.
- Compare the basic CNN against a stronger architecture or a pretrained model.
- Return to the Severstal dataset later as a stretch goal for multi-label learning and segmentation.
