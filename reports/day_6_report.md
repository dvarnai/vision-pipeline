# Day 6 Report

## Project Focus

Consolidated the latest ViT experiment results into the project documentation and clarified how inference confidence should be interpreted for the current single-label Intel scene classifier.

## Completed

- Reviewed the inference probability path in `src.inference.prediction`.
- Confirmed that `softmax` is the correct inference normalization for the current single-label multiclass setup.
- Clarified that top-k output means the top alternatives under one mutually exclusive probability distribution, not independent multi-label probabilities.
- Identified that the returned `confidence` field is a softmax model probability among known classes, not a calibrated real-world correctness guarantee.
- Updated `README.md` with the Day 5 ViT results.
- Updated the README shipping decision from ResNet-50 to the best recorded ViT-family run.
- Updated `reports/final_report.md` with the Day 5 ViT metrics and removed placeholder validation fields.
- Added this Day 6 report to record the documentation and inference-semantics work.

## ViT Result Incorporated

The strongest recorded Day 5 result is now the project shipping candidate:

```text
Config: src.configs.intel_vit_transfer_4
Model family: ViT-B-16
Training mode: full fine-tuning with discriminative learning rates
Best recorded epoch: 60/100
Train Loss: 0.0856
Val Loss: 0.1412
Validation Accuracy: 0.9527
Weighted F1: 0.9525
Macro F1: 0.9536
Micro F1: 0.9527
Epoch Time: 35481.59 ms
```

This replaces the previous ResNet-50 shipping recommendation. The best recorded ResNet-50 transfer run remains the fallback baseline:

```text
Validation Accuracy: 0.8870
Weighted F1: 0.8869
Macro F1: 0.8900
Micro F1: 0.8870
```

## Inference Confidence Interpretation

For the current Intel scene task, each image belongs to exactly one of six classes:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

Because the classes are mutually exclusive, `torch.softmax(logits, dim=1)` remains the correct probability transform at inference time. It converts logits into one probability distribution across the six labels.

Top-k output does not require independent class probabilities. It only returns the top-ranked classes from that same distribution. The ranking from logits and softmax is the same, but softmax makes the returned values easier to interpret as relative model probabilities among known classes.

The important caveat is calibration. The current `confidence` field should be interpreted as model softmax probability, not as a validated probability that the prediction is correct. If the product requires calibrated confidence, the next step should be temperature scaling or another validation-set calibration method.

## Current Shipping Decision

Ship the ViT-family checkpoint from `src.configs.intel_vit_transfer_4`, assuming the selected checkpoint corresponds to the epoch 60 result or a later result that matches or exceeds it.

Keep the ResNet-50 transfer run as the fallback baseline because it is simpler and faster than full ViT fine-tuning, but no longer has the best recorded validation quality.

Reject the custom CNN family for shipping because its best recorded validation result is substantially lower, around `72.03%` accuracy and `0.7150` weighted F1.

## Remaining Work

- Run `src.test_intel` on the selected ViT checkpoint to record final test-set metrics.
- Run `src.benchmark_latency` on the selected ViT checkpoint to record cold and warm latency.
- Fill in the final report latency fields once benchmark results are available.
- Consider renaming `confidence` to `probability` or adding calibration before treating it as a correctness probability.
