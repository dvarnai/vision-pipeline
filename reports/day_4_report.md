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
- Revisit feature extractor changes, since classifier-only changes are not currently improving validation quality.
