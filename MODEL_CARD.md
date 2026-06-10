# Model Card: Intel Scene Classifier

## Model Details

- Model family: `ViT-B-16`
- Selected config: `src.configs.intel_vit_transfer_4`
- Selected checkpoint: `checkpoints/intel_vit_transfer_4_epoch_0060.pt`
- Base weights: `torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1`
- Task: single-label image classification
- Input size: `224x224`
- Label set: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`

The selected model is a full fine-tuned Vision Transformer with discriminative learning rates:

- ViT encoder learning rate: `1e-5`
- Classifier head learning rate: `1e-4`
- Scheduler minimum learning rate: `1e-6`
- Optimizer: SGD
- Scheduler: cosine annealing
- Batch size: `64`
- Best recorded epoch: `60`

## Intended Use

This model is intended for educational and experimental classification of natural scene images into one of the six Intel Image Classification classes.

Appropriate uses:

- compare custom CNN, ResNet, and ViT transfer-learning behavior
- run local inference on Intel-style natural scene images
- study model confidence, calibration, and class-level failure modes
- use as a project baseline for future training, evaluation, and deployment exercises

The model output should be treated as a ranked prediction over the known six classes. The `confidence` value is a softmax probability under the model, not a guaranteed probability of real-world correctness.

## Out-of-Scope Use

This model should not be used for high-stakes decisions, safety-critical automation, or real-world geographic, environmental, or infrastructure assessment without additional validation.

Out-of-scope uses include:

- classifying images outside the six supported labels
- multi-label scene tagging
- object detection, segmentation, localization, or counting
- medical, legal, financial, surveillance, emergency-response, or safety-critical workflows
- inferring sensitive attributes about people or places
- treating confidence as calibrated certainty without additional calibration and thresholding policy

Images that do not belong to one of the six known scene classes will still be forced into one of those classes.

## Data

The project uses the Intel Image Classification dataset from Kaggle:

```text
data/intel/seg_train/seg_train
data/intel/seg_test/seg_test
```

The six class folders are:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

Training uses the dataset's `seg_train` split. Validation and reported evaluation use the dataset's `seg_test` split, which contains `3000` images in the local evaluation. The raw dataset is not committed to this repository.

Preprocessing for the selected ViT model:

- resize to `224x224`
- convert image to float tensor
- normalize with ImageNet mean `[0.485, 0.456, 0.406]`
- normalize with ImageNet std `[0.229, 0.224, 0.225]`

The original selected run did not use a fixed seed, so retraining from scratch may not exactly reproduce last-decimal metrics.

## Metrics

Selected checkpoint validation metrics on the Intel `seg_test` split:

| Metric | Value |
|---|---:|
| Samples | 3000 |
| Validation loss | 0.1412 |
| Accuracy | 0.9527 |
| Weighted F1 | 0.9525 |
| Macro F1 | 0.9536 |
| Micro F1 | 0.9527 |

Calibration metrics from `src.check_calibration` with 10 equal-width confidence bins:

| Metric | Value |
|---|---:|
| Accuracy | 0.952667 |
| Average confidence | 0.957520 |
| ECE@10 | 0.011785 |
| Negative log likelihood | 0.141340 |
| Brier score | 0.076020 |

Most predictions are in the `0.9-1.0` confidence bin:

| Confidence bin | Count | Accuracy | Avg confidence | Gap |
|---|---:|---:|---:|---:|
| 0.9-1.0 | 2638 | 0.980667 | 0.988402 | 0.007735 |

The low ECE suggests reasonable aggregate calibration on this split, but the model can still be confidently wrong on individual images.

## Limitations

- The model only supports six scene classes and cannot reject unknown classes by itself.
- Evaluation uses the Intel `seg_test` split as the validation/evaluation split; no separate final holdout is documented.
- The selected checkpoint was chosen based on validation performance, so final generalization claims should be conservative.
- The model is large and slower than the custom CNN and ResNet baselines.
- The original run did not use a fixed seed.
- Confidence is based on softmax output and should not be treated as a calibrated correctness guarantee for deployment decisions.
- No production latency, memory, robustness, or out-of-distribution evaluation is documented in this model card.
- The model inherits biases and coverage limits from both ImageNet pretraining and the Intel scene dataset.

## Known Failure Modes

The strongest known error pattern is confusion among visually adjacent scene categories.

Largest ViT mistake pairs from the Day 7 analysis:

```text
glacier -> mountain: 42
mountain -> glacier: 37
buildings -> street: 20
street -> buildings: 18
glacier -> sea: 7
```

High-confidence wrong ViT examples included:

```text
mountain -> sea       confidence 0.9963
mountain -> glacier   confidence 0.9951, 0.9923, 0.9897
buildings -> street   confidence 0.9941, 0.9810, 0.9801
glacier -> mountain   confidence 0.9896, 0.9871
sea -> mountain       confidence 0.9837
```

Per-class error rates from the CNN-vs-ViT slice analysis:

| Class | Count | ViT errors | ViT error rate |
|---|---:|---:|---:|
| `mountain` | 525 | 44 | 8.38% |
| `glacier` | 553 | 53 | 9.58% |
| `sea` | 510 | 7 | 1.37% |
| `street` | 501 | 18 | 3.59% |
| `buildings` | 437 | 23 | 5.26% |
| `forest` | 474 | 0 | 0.00% |

Image-quality slicing found that wrong predictions were less sharp on average than correct predictions:

| Slice | Count | Brightness mean | Contrast mean | Sharpness mean |
|---|---:|---:|---:|---:|
| ViT correct | 2855 | 114.66 | 59.27 | 2105.71 |
| ViT wrong | 145 | 119.61 | 62.02 | 1774.17 |

Practical implications:

- `glacier`/`mountain` decisions should be reviewed carefully.
- `buildings`/`street` decisions can be visually ambiguous.
- Very high confidence does not eliminate the possibility of an error.
- Blurry or lower-sharpness images are more likely to fail.

## Reproduction

Evaluate the selected checkpoint:

```bash
python -m src.test_intel checkpoints/intel_vit_transfer_4_epoch_0060.pt
```

Run the calibration check:

```bash
python -m src.check_calibration checkpoints/intel_vit_transfer_4_epoch_0060.pt
```

Use `--images-path /path/to/intel` if the Intel dataset is not under `data/intel`.
