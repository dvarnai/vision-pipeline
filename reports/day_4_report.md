# Day 4 Report

## Project Focus

Continued iterating on the Intel Image Classification model by testing a slightly larger classifier head and moving experiment setup into run-specific Python config files.

The main finding was that adding a second linear layer with `32` hidden features made training significantly faster, but the final model quality was only comparable to the best model observed so far. The change improved training efficiency more clearly than validation performance.

## Completed

- Added a second linear layer to the classifier head.
- Used `32` hidden features between the flattened pooled features and the final class logits.
- Observed a significant training speedup from the updated classifier head.
- Confirmed that validation performance was roughly as good as the previous best model, but not clearly better.
- Started moving run-specific choices into Python config files.
- Included optimizer, scheduler, transforms, and model construction in config files so experiments can be compared more directly.
- Added configurable checkpointing support to preserve run state during longer experiments.
- Added resume support so interrupted runs can continue from saved checkpoints.
- Added a checkpoint-based Intel test evaluator for measuring held-out test performance after training.
- Started an initial transfer learning experiment using a frozen ResNet classifier backbone.

## Experiment: Two-Layer Classifier Head

The previous classifier projected the pooled CNN features directly into the output classes. The new version adds a small hidden layer:

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(0.2),
    nn.Linear(16 * 4 * 4, 32),
    nn.ReLU(),

    nn.Dropout(0.2),
    nn.Linear(32, num_classes),
)
```

This keeps the model small while giving the classifier a little more capacity after the convolutional feature extractor.

## Result

The extra linear layer sped up training significantly. This is useful because it makes experiments cheaper to run and compare.

However, the model itself did not clearly exceed the best validation result so far. Its performance was approximately on par with the previous best model, so the change should be treated as an efficiency improvement rather than a quality improvement.

Current interpretation:

- The two-layer classifier head is a reasonable default because it trains faster.
- The added classifier capacity is not enough by itself to improve generalization.
- Further gains are more likely to come from better feature extraction, regularization, augmentation choices, or scheduler/learning-rate changes.

## Transfer Learning: ResNet

Started an initial transfer learning experiment with a pretrained ResNet model. The first setup uses `ResNet-50` with ImageNet weights. All pretrained weights are frozen, and only the final fully connected layer is replaced and trained:

```python
model.fc = torch.nn.Linear(model.fc.in_features, 6)
```

The intent is to test whether pretrained visual features are already strong enough for the Intel scene classes when only a small classification head is trained. ResNet expects `224x224` inputs, so this experiment upscales the Intel images from the smaller `150x150` preprocessing size used by the custom CNN experiments. The initial transfer-learning setup also missed using the ImageNet normalization expected by pretrained ResNet models; this was corrected by using `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`. After fixing the normalization, the transfer-learning models started learning immediately.

The first optimizer choice was SGD. Before fixing normalization, SGD looked very slow and did not show much learning behavior. Adam was also tested, but it overfit almost immediately. This suspicious optimizer behavior was the clue that led to checking the preprocessing setup and finding that ImageNet normalization was missing.

After fixing ImageNet normalization, the initial `ResNet-50` transfer experiment started outperforming every previous experiment. By epoch 18 it had already reached:

```text
Epoch 18/100, Train Loss: 0.7510, Val Loss: 0.7378, Subset Acc: 0.8303, Weighted F1: 0.8296, Macro F1: 0.8321, Micro F1: 0.8303, Time: 4243.02 ms
```

By the end of the 100-epoch run, the frozen-backbone `ResNet-50` result improved further:

```text
Epoch 100/100, Train Loss: 0.4647, Val Loss: 0.4653, Subset Acc: 0.8737, Weighted F1: 0.8729, Macro F1: 0.8753, Micro F1: 0.8737, Time: 4230.29 ms
```

This is now the strongest result in the project so far. The extra ResNet test configs were removed, leaving the working `ResNet-50` transfer experiment as the baseline for this track.

The next transfer-learning experiment unfreezes `layer4` of `ResNet-50` and fine-tunes it together with the new fully connected layer. This did not materially affect the result; performance was essentially identical to the frozen-backbone baseline. Experiment 3 also unfreezes `layer3`, which produced a small improvement in accuracy and F1:

```text
Epoch 100/100, Train Loss: 0.4749, Val Loss: 0.4802, Subset Acc: 0.8843, Weighted F1: 0.8845, Macro F1: 0.8877, Micro F1: 0.8843, Time: 11214.99 ms
```

This suggests that adapting `layer3` and `layer4` helps slightly, but it is also much slower than training only the final classifier. Experiment 4 unfreezes all layers and trains the whole network with the same learning rate first. This outperformed the frozen-layer variants:

```text
Epoch 100/100, Train Loss: 0.4489, Val Loss: 0.4559, Subset Acc: 0.8870, Weighted F1: 0.8869, Macro F1: 0.8900, Micro F1: 0.8870, Time: 16103.78 ms
```

Full-network fine-tuning is now the best ResNet result, but it is also the slowest option tested so far.

## Config Refactor

Run configuration is now moving into Python files instead of being hard-coded directly in the training script.

Each run config can define:

- model construction
- optimizer setup
- scheduler setup
- training hyperparameters
- training transforms
- validation transforms

Keeping these as Python objects avoids string parsing and makes experiments easy to copy, modify, and diff.

Checkpointing was added alongside this refactor. Checkpoints are written at a configurable epoch interval and include the epoch number in the filename. Each checkpoint stores the model weights, optimizer state, scheduler state, epoch history so far, config module and source, class weights, computed training statistics, and runtime settings. This makes long runs easier to resume and makes completed experiments easier to reproduce later.

A separate Intel test script was also added. It loads a checkpoint, reconstructs the saved config and model, applies the saved training-set normalization statistics, and evaluates on the Intel test split. This gives a cleaner final evaluation path than relying only on validation metrics during training.

## Next Steps

- Keep the two-layer classifier head as a speed-focused baseline unless later runs show a regression.
- Compare future changes against the best known validation result, not only against training speed.
- Continue moving experiment-specific setup into config files.
- Use one config file per run so architecture, optimizer, scheduler, hyperparameters, and transforms are recorded together.
- Use checkpoints for longer runs so promising experiments can be resumed instead of restarted.
- Evaluate promising checkpoints on the test split before treating them as final results.
- Continue the transfer-learning track with staged fine-tuning: first `layer4`, then `layer3`, then the full network with the same learning rate first.
- Revisit custom feature extractor changes only after comparing against the stronger ResNet transfer baseline.
