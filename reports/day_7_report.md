# Day 7 Report

## Project Focus

Consolidated the transfer-learning experiment writeup, cleaned up unsupported comparison claims, and added a concrete calibration check for the selected ViT checkpoint.

## Completed

- Added `reports/transfer_learning_experiments.md` as the consolidated transfer-learning report.
- Removed model-family references from the transfer-learning report when there was no local experiment artifact to support them.
- Compared the ResNet/CNN transfer checkpoint against the ViT transfer checkpoint on the Intel test split at the per-image level.
- Added the CNN-vs-ViT prediction-pair slice analysis to the transfer-learning report.
- Added a calibration evaluator in `src/check_calibration.py`.
- Ran a 10-bin reliability/ECE check for the selected ViT checkpoint.
- Added `src/data/labels.py` so the dataset and inference code share one Intel label contract.
- Tightened `.gitignore` from `data/` to `/data/` so source files under `src/data/` are not accidentally ignored.

## Transfer Learning Report

Created a dedicated report:

```text
reports/transfer_learning_experiments.md
```

The report now covers:

- the best custom CNN baseline
- ResNet transfer-learning runs
- the ViT-family run table
- the selected ViT model and checkpoint
- why the ViT checkpoint won
- CNN-vs-ViT prediction-pair slices
- calibration results
- reproduction commands
- remaining limitations

The report intentionally only documents model families represented by local runs, configs, checkpoints, or explicit analysis artifacts. Unsupported model-family mentions were removed to keep the comparison evidence-based.

## Transfer Slice Analysis

Compared the ResNet/CNN transfer checkpoint against the ViT transfer checkpoint using the Intel test split:

```text
CNN checkpoint: checkpoints/intel_resnet50_transfer_4_epoch_0100.pt
ViT checkpoint: checkpoints/intel_vit_transfer_4_epoch_0080.pt
Samples: 3000
CNN accuracy: 0.8690
ViT accuracy: 0.9517
Both correct: 2556
CNN-only wins: 51
ViT-only wins: 299
Both wrong: 94
```

The generated inspection artifacts were:

```text
/tmp/transfer_slice_inspection/class_slices.csv
/tmp/transfer_slice_inspection/prediction_pair_slices.csv
/tmp/transfer_slice_inspection/examples.csv
/tmp/transfer_slice_inspection/examples_with_quality.csv
/tmp/transfer_slice_inspection/image_quality_slices.csv
```

ViT was stronger overall and stronger in every true-class aggregate. CNN still had a few local pockets where it corrected ViT mistakes:

| True class | CNN prediction | ViT prediction | Count |
| --- | --- | --- | ---: |
| glacier | glacier | mountain | 23 |
| buildings | buildings | street | 8 |
| mountain | mountain | glacier | 8 |
| street | street | buildings | 7 |

The ViT-only wins were much larger and mostly came from fixing CNN's terrain and urban-boundary mistakes:

| True class | CNN prediction | ViT prediction | Count |
| --- | --- | --- | ---: |
| mountain | glacier | mountain | 101 |
| sea | glacier | sea | 44 |
| street | buildings | street | 36 |
| glacier | mountain | glacier | 34 |
| buildings | street | buildings | 20 |
| sea | mountain | sea | 12 |
| glacier | sea | glacier | 11 |

The strongest pattern is that the CNN over-predicts `glacier`, especially for true `mountain` and `sea` images. ViT largely fixes that failure mode, although it still makes some high-confidence `glacier`/`mountain` mistakes of its own.

By true-class net wins, ViT led everywhere:

| Class | CNN wins | ViT wins | Net CNN wins |
| --- | ---: | ---: | ---: |
| forest | 0 | 14 | -14 |
| buildings | 9 | 24 | -15 |
| glacier | 24 | 47 | -23 |
| street | 7 | 43 | -36 |
| sea | 1 | 62 | -61 |
| mountain | 10 | 109 | -99 |

## Confidence Error Review

High-confidence wrong CNN predictions were mostly moderate-confidence errors:

```text
glacier -> sea        confidence around 0.73
street -> buildings   confidence around 0.73
mountain -> glacier   confidence around 0.66-0.68
mountain -> sea       confidence around 0.67
```

High-confidence wrong ViT predictions were more extreme:

```text
mountain -> sea       confidence 0.9963
mountain -> glacier   confidence 0.9951, 0.9923, 0.9897
buildings -> street   confidence 0.9941, 0.9810, 0.9801
glacier -> mountain   confidence 0.9896, 0.9871
sea -> mountain       confidence 0.9837
```

This reinforces the calibration caveat. ViT is much more accurate, but when it is wrong, it can be very confidently wrong.

Low-confidence correct predictions show the opposite side of the calibration issue. CNN has many barely-correct predictions with softmax confidence around `0.20` to `0.25`, while ViT's low-confidence correct cases are usually closer to `0.47` to `0.52`. CNN therefore appears less separated on borderline images.

## Calibration Check

Added a reusable calibration script:

```bash
python -m src.check_calibration checkpoints/intel_vit_transfer_4_epoch_0060.pt
```

The script loads a checkpoint through the inference bundle, evaluates the Intel `seg_test` split, computes a reliability table, and reports:

- accuracy
- average confidence
- expected calibration error
- negative log likelihood
- Brier score

For the selected ViT checkpoint, the 10-bin calibration result was:

```text
Samples: 3000
Accuracy: 0.952667
Average confidence: 0.957520
ECE@10: 0.011785
NLL: 0.141340
Brier: 0.076020
```

Most predictions landed in the `0.9-1.0` confidence bin:

```text
Count: 2638
Accuracy: 0.980667
Average confidence: 0.988402
Gap: 0.007735
```

The overall ECE is low, but the model is still slightly over-confident in its dominant high-confidence bin. The smaller low-confidence bins are noisy because they contain few samples, but they remain useful for identifying review or abstention candidates.

## Errors by Class

| Class | Count | CNN errors | CNN error rate | ViT errors | ViT error rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| mountain | 525 | 143 | 27.24% | 44 | 8.38% |
| glacier | 553 | 76 | 13.74% | 53 | 9.58% |
| sea | 510 | 68 | 13.33% | 7 | 1.37% |
| street | 501 | 54 | 10.78% | 18 | 3.59% |
| buildings | 437 | 38 | 8.70% | 23 | 5.26% |
| forest | 474 | 14 | 2.95% | 0 | 0.00% |

CNN's largest specific mistake pairs were:

```text
mountain -> glacier: 130
glacier -> mountain: 53
street -> buildings: 47
sea -> glacier: 46
buildings -> street: 32
```

ViT's largest specific mistake pairs were:

```text
glacier -> mountain: 42
mountain -> glacier: 37
buildings -> street: 20
street -> buildings: 18
glacier -> sea: 7
```

Both models struggle most with visually adjacent scene categories. The difference is scale: ViT makes the same kinds of mistakes, but far fewer of them.

## Errors by Image Quality

Computed simple image-quality slices directly from the source images:

- brightness: grayscale mean
- contrast: grayscale standard deviation
- sharpness: mean squared grayscale gradient energy

| Slice | Count | Brightness mean | Contrast mean | Sharpness mean |
| --- | ---: | ---: | ---: | ---: |
| all | 3000 | 114.90 | 59.41 | 2089.68 |
| CNN correct | 2607 | 114.20 | 59.10 | 2135.80 |
| CNN wrong | 393 | 119.53 | 61.42 | 1783.79 |
| ViT correct | 2855 | 114.66 | 59.27 | 2105.71 |
| ViT wrong | 145 | 119.61 | 62.02 | 1774.17 |
| both wrong | 94 | 121.19 | 62.39 | 1751.93 |

The clearest image-quality signal is sharpness. Wrong predictions are noticeably less sharp on average for both models. Error cases are also slightly brighter and slightly higher contrast, but the brightness and contrast shifts are smaller than the sharpness gap.

## Implementation Notes

The calibration and inference paths depend on a shared Intel label contract. `src/data/labels.py` now defines:

- `INTEL_CLASS_NAMES`
- `INTEL_CLASS_TO_INDEX`
- `INTEL_LABEL_CONTRACT_VERSION`

This also required tightening `.gitignore` from `data/` to `/data/`, because the broader rule ignored untracked source files under `src/data/`.

## Remaining Work

- Run `src.test_intel` on the selected ViT checkpoint to record final test-set metrics.
- Run `src.benchmark_latency` on the selected ViT checkpoint to record cold and warm latency.
- Decide whether the product should use an abstention threshold based on the calibration and reliability-curve results.
- Consider adding a calibrated-confidence field if temperature scaling or another calibration method is introduced.
