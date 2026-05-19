# Day 3 Report

## Project Focus

Continued improving the Intel Image Classification training pipeline by adding experiment tracking with Weights & Biases, testing small architecture changes, and correcting the validation preprocessing setup.

The main finding was that increasing the model size directly has so far caused overtraining, while preserving more spatial information before the classifier improved validation performance.

## Completed

- Added Weights & Biases support for experiment tracking.
- Increased the final adaptive pooling output from `2x2` to `4x4`.
- Tested a larger network and observed that it led mainly to overtraining.
- Found and fixed an issue where data augmentation was accidentally being applied to validation data.
- Improved validation accuracy after separating training augmentation from validation preprocessing.
- Sped up training after removing augmentation from the validation path.
- Decreased the learning rate and trained for longer, which further improved validation accuracy.
- Tried replacing `AvgPool2d` downsampling with strided convolutions, but this did not improve results.
- Found that the cosine scheduler step count was configured too large, so the learning rate was barely annealing during training.

## Experiment: Larger Spatial Feature Map

The best direction so far was increasing the adaptive pooling output size from `2x2` to `4x4`. This gives the classifier more spatial information without substantially increasing the convolutional part of the network.

Current model structure:

```python
self.features = nn.Sequential(
    nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),

    nn.AdaptiveAvgPool2d((4, 4))
)

self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(0.2),
    nn.Linear(32 * 4 * 4, num_classes),
)
```

Best observed result from the 100-epoch run:

```text
Epoch 92/100, Train Loss: 0.5972, Val Loss: 1.3804, Subset Acc: 0.6727, Weighted F1: 0.6739, Macro F1: 0.6748, Micro F1: 0.6727, Time: 1494.42 ms
```

Compared with smaller adaptive pooling outputs, the `4x4` feature map improved validation accuracy. This suggests that keeping more spatial layout information before the classifier is useful for the Intel scene classes.

## Experiment: Lower Learning Rate and Longer Training

After the validation augmentation fix, decreasing the learning rate and extending training from 100 to 300 epochs further improved validation performance.

Intermediate strong result:

```text
Epoch 168/300, Train Loss: 0.5814, Val Loss: 1.1540, Subset Acc: 0.7073, Weighted F1: 0.7018, Macro F1: 0.7047, Micro F1: 0.7073, Time: 1390.99 ms
```

Best observed result so far:

```text
Epoch 231/300, Train Loss: 0.5608, Val Loss: 1.0884, Subset Acc: 0.7203, Weighted F1: 0.7150, Macro F1: 0.7182, Micro F1: 0.7203, Time: 1419.54 ms
```

Compared with the earlier 100-epoch result, validation accuracy increased from `67.27%` to `72.03%`, and weighted F1 increased from `0.6739` to `0.7150`.

## Scheduler Finding

The cosine annealing scheduler was being stepped once per mini-batch, but `T_max` was calculated from the number of samples multiplied by the batch size:

```python
total_steps = args.epochs * len(train_dataset) * train_loader.batch_size
```

This made `T_max` much larger than the actual number of optimizer steps, so the learning rate barely decreased over the run. The longer 300-epoch experiment should therefore be interpreted mostly as constant low-learning-rate training, not as a completed cosine annealing schedule.

Because `scheduler.step()` is called once per mini-batch, the correct step count is the number of batches per epoch times the number of epochs:

```python
total_steps = args.epochs * len(train_loader)
```

This may explain part of the fast overfitting behavior: the model was not receiving the intended learning-rate decay during longer runs.

## Architecture Observation

Simply increasing network size has not helped yet. Larger variants overtrained, improving training behavior without producing better validation generalization.

Replacing the `AvgPool2d` layers with strided convolutions also did not work better. For this model and dataset, explicit average-pooling downsampling appears to be the stronger simple baseline.

For now, the better tradeoff is a still-small CNN with modest downsampling and a larger final pooled feature map, rather than a substantially wider or deeper network.

## Validation Augmentation Fix

An issue was found where augmentation was accidentally being applied to the validation data. Fixing this improved accuracy and reduced training time.

The validation transform should remain deterministic and should only include preprocessing steps such as resize, tensor conversion, and normalization. Random augmentation belongs only in the training transform.

This makes the validation metrics more reliable because each validation epoch now evaluates the model on a stable, non-augmented validation distribution.

## Next Steps

- Keep training and validation transforms separate.
- Use Weights & Biases runs to compare architecture and augmentation changes more systematically.
- Continue favoring small, controlled architecture changes over simply increasing model size.
- Keep `AvgPool2d` as the default downsampling method unless a later experiment shows a clear gain from learned downsampling.
- Fix the cosine scheduler `T_max` calculation to use `args.epochs * len(train_loader)`.
- Rerun the current best small model after fixing the scheduler before making more architecture changes.
- Track whether the 300-epoch run continues improving after epoch 231 or starts overfitting.
- Consider early stopping or stronger regularization if overtraining continues.
