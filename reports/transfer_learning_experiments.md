# Transfer Learning Experiments

This report summarizes the Intel Image Classification transfer-learning track. The goal was to compare the custom CNN baseline against pretrained convolutional models and ViT-family models, then select the best candidate for the current project state.

Unless noted otherwise, results are validation metrics on `data/intel/seg_test/seg_test`, using the same six Intel scene classes as the rest of the project. Most completed transfer runs used:

- batch size `64`
- `num_workers=16`
- ImageNet normalization for pretrained torchvision models
- SGD optimizer
- cosine annealing scheduler
- no fixed seed in the original run

## CNN Baseline

The custom CNN track established that the data loader, transforms, class weights, training loop, validation metrics, and checkpointing path worked before introducing pretrained models.

The best early CNN result reported in the project notes was the longer low-learning-rate run from day 3:

| Model family | Run | Epoch | Val loss | Accuracy | Weighted F1 | Macro F1 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Custom CNN | best day 3 CNN | 231 / 300 | 1.0884 | 0.7203 | 0.7150 | 0.7182 | Small CNN with tuned pooling and longer training |

This was a useful baseline, but it left a large quality gap. The model still relied on a small task-specific feature extractor trained from scratch, so it struggled to match the representation quality available from ImageNet-pretrained backbones.

## ResNet Transfer

The ResNet transfer experiments showed the first large jump over the custom CNN baseline. The important fix was using the ImageNet preprocessing expected by pretrained models. Before that correction, optimizer behavior was misleading: SGD looked too slow and Adam overfit very quickly.

| Model family | Run | Input | Trainable scope | Best epoch | Val loss | Accuracy | Weighted F1 | Macro F1 | Notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| ResNet-18 | `intel_resnet18_transfer_1` | 224 | checkpoint metadata only | 74 | 0.4651 | 0.8747 | 0.8741 | 0.8765 | Smaller CNN transfer baseline |
| ResNet-50 | `intel_resnet50_transfer_2` | 224 | `layer4`, `avgpool`, `fc` | 74 | 0.5775 | 0.8750 | 0.8750 | 0.8781 | Fine-tunes only the last ResNet block |
| ResNet-50 | `intel_resnet50_transfer_3` | 224 | `layer3`, `layer4`, `avgpool`, `fc` | 74 | 0.4792 | 0.8847 | 0.8848 | 0.8879 | Slightly stronger than `layer4`-only fine-tuning |
| ResNet-50 | `intel_resnet50_transfer_4` | 150 | full network | 96 | 0.4226 | 0.8900 | 0.8900 | 0.8933 | Best saved ResNet result; smaller input reduced epoch time |

The best ResNet result was full-network `ResNet-50` fine-tuning at `150x150`. This is a practical result because ResNet uses convolutional layers plus adaptive pooling, so it can accept smaller spatial inputs without breaking the final classifier shape. In the project notes, this reduced epoch time from roughly `16` seconds at `224x224` to roughly `9` seconds at `150x150` while preserving quality.

## ViT-Family Runs

The ViT experiments used `torchvision.models.vit_b_16` with `ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1` and `224x224` inputs. The fixed input size matters more for ViT than for ResNet because the image is split into patches and matched against learned positional embeddings.

| Run | Setup | Epochs configured | Best epoch | Val loss | Accuracy | Weighted F1 | Macro F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `intel_vit_transfer_1` | frozen ViT backbone, train classifier head | 100 | 80 | 0.2164 | 0.9347 | 0.9346 | 0.9362 | Strong cheap baseline; roughly ResNet full-tune epoch time at 224 |
| `intel_vit_transfer_2` | full ViT fine-tuning, single learning rate `1e-4` | 100 | 7 | 0.1492 | 0.9507 | 0.9506 | 0.9515 | Very strong early peak, then unstable overfitting |
| `intel_vit_transfer_3` | full ViT fine-tuning, 10-epoch schedule | 10 | 8 | 0.1555 | 0.9477 | 0.9475 | 0.9489 | Shorter cosine schedule did not improve the peak |
| `intel_vit_transfer_4` | full ViT fine-tuning with discriminative learning rates | 100 | 60 | 0.1412 | 0.9527 | 0.9525 | 0.9536 | Best saved validation result |

The discriminative ViT run used a lower learning rate for the pretrained encoder and a higher learning rate for the newly initialized classifier head:

- encoder learning rate: `1e-5`
- classifier head learning rate: `1e-4`
- scheduler minimum learning rate: `1e-6`
- checkpoint selected: `checkpoints/intel_vit_transfer_4_epoch_0060.pt`

## Selected Model

The selected model is `src.configs.intel_vit_transfer_4`, using `ViT-B-16` with ImageNet SWAG linear weights and discriminative learning rates. The selected checkpoint is:

```text
checkpoints/intel_vit_transfer_4_epoch_0060.pt
```

It is selected because it has the strongest saved validation metrics in the project:

| Metric | Value |
|---|---:|
| Best epoch | 60 |
| Validation loss | 0.1412 |
| Accuracy | 0.9527 |
| Weighted F1 | 0.9525 |
| Macro F1 | 0.9536 |
| Micro F1 | 0.9527 |

## Why It Won

The selected ViT run won on validation quality and training stability.

Compared with the custom CNN baseline, it benefits from a much stronger pretrained representation. The custom CNN had to learn scene-level features from the Intel dataset alone, while the ViT starts from broad ImageNet-pretrained visual features.

Compared with ResNet transfer learning, the ViT representation transferred better to this scene classification task. The best saved ResNet run reached `0.8900` accuracy and `0.8900` weighted F1, while the selected ViT reached `0.9527` accuracy and `0.9525` weighted F1.

Compared with the earlier full ViT fine-tuning run, discriminative learning rates improved control. The single-learning-rate run peaked quickly at epoch 7, then validation quality dropped while training loss continued to fall. The discriminative run improved more gradually and reached a higher peak at epoch 60 without the same immediate collapse.

## CNN vs ViT Prediction Slices

The aggregate metrics favor ViT, and the pairwise prediction slices show the same pattern. From `prediction_pair_slices.csv`, comparing the selected ViT against the CNN baseline:

| Outcome | Count |
|---|---:|
| Total samples | 3000 |
| Both correct | 2556 |
| CNN-only wins | 51 |
| ViT-only wins | 299 |
| Both wrong | 94 |

ViT was stronger overall and in every true-class aggregate. The CNN did have real pockets where it beat ViT, but those pockets were narrow:

| True class | CNN correct | ViT predicted | Count |
|---|---|---|---:|
| `glacier` | `glacier` | `mountain` | 23 |
| `buildings` | `buildings` | `street` | 8 |
| `mountain` | `mountain` | `glacier` | 8 |
| `street` | `street` | `buildings` | 7 |

The main CNN advantage was that it sometimes handled `glacier` vs `mountain` better than ViT, but only on `23` true `glacier` images plus `8` true `mountain` images.

ViT's wins were much larger:

| True class | CNN predicted | ViT correct | Count |
|---|---|---|---:|
| `mountain` | `glacier` | `mountain` | 101 |
| `sea` | `glacier` | `sea` | 44 |
| `street` | `buildings` | `street` | 36 |
| `glacier` | `mountain` | `glacier` | 34 |
| `buildings` | `street` | `buildings` | 20 |
| `sea` | `mountain` | `sea` | 12 |
| `glacier` | `sea` | `glacier` | 11 |

The main ViT advantage was that it strongly fixed the CNN's over-prediction of `glacier`, especially for true `mountain` and `sea` images.

By true-class net wins, ViT led everywhere:

| Class | CNN wins | ViT wins | Net CNN wins |
|---|---:|---:|---:|
| `forest` | 0 | 14 | -14 |
| `buildings` | 9 | 24 | -15 |
| `glacier` | 24 | 47 | -23 |
| `street` | 7 | 43 | -36 |
| `sea` | 1 | 62 | -61 |
| `mountain` | 10 | 109 | -99 |

## Calibration Check

The selected ViT checkpoint is accurate, but calibration asks a different question: when the model says it is confident, does that confidence match empirical correctness?

Using 10 equal-width confidence bins on the Intel `seg_test` split, the selected checkpoint has low expected calibration error:

| Metric | Value |
|---|---:|
| Checkpoint | `checkpoints/intel_vit_transfer_4_epoch_0060.pt` |
| Samples | 3000 |
| Accuracy | 0.952667 |
| Average confidence | 0.957520 |
| ECE@10 | 0.011785 |
| Negative log likelihood | 0.141340 |
| Brier score | 0.076020 |

Reliability table:

| Confidence bin | Count | Accuracy | Avg confidence | Gap |
|---|---:|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a | n/a |
| 0.1-0.2 | 0 | n/a | n/a | n/a |
| 0.2-0.3 | 2 | 0.500000 | 0.282436 | 0.217564 |
| 0.3-0.4 | 3 | 0.000000 | 0.358545 | 0.358545 |
| 0.4-0.5 | 10 | 0.600000 | 0.460015 | 0.139985 |
| 0.5-0.6 | 55 | 0.600000 | 0.552783 | 0.047217 |
| 0.6-0.7 | 64 | 0.671875 | 0.649810 | 0.022065 |
| 0.7-0.8 | 80 | 0.812500 | 0.755584 | 0.056916 |
| 0.8-0.9 | 148 | 0.831081 | 0.854564 | 0.023482 |
| 0.9-1.0 | 2638 | 0.980667 | 0.988402 | 0.007735 |

Interpretation: most predictions are in the `0.9-1.0` confidence bin, where the model is slightly over-confident by about `0.0077`. The lower-confidence bins are noisier because they contain few samples, but they also identify the cases where review or abstention would be most useful.

## What It Did Not Fix

The selected model is not a complete solution.

- It did not remove the need for careful checkpoint selection. The best saved checkpoint is epoch 60, while later saved epochs did not improve the peak.
- It did not make the run cheap. Full ViT fine-tuning takes roughly `35.5` to `36` seconds per epoch in the project notes, which is much slower than the smaller CNN and `150x150` ResNet runs.
- It did not solve reproducibility exactly. The original run did not use a fixed seed, so retraining from scratch can differ in the last decimals or select a slightly different best epoch.
- It did not prove held-out generalization beyond the current validation/test folder convention. The project still uses the Intel `seg_test` split as the validation/evaluation split, so a separate final holdout or cross-validation setup would be needed for stronger claims.
- It did not address production concerns such as model size, inference latency, memory footprint, thresholding or abstention policy, and robustness to out-of-distribution images.

## Reproduction

Train the selected configuration from scratch:

```bash
export WANDB_MODE=offline
python -m src.train_intel src.configs.intel_vit_transfer_4
```

Evaluate the selected checkpoint:

```bash
python -m src.test_intel checkpoints/intel_vit_transfer_4_epoch_0060.pt
```

Run the calibration check:

```bash
python -m src.check_calibration checkpoints/intel_vit_transfer_4_epoch_0060.pt
```

Use `--images-path /path/to/intel` if the Intel dataset is not under `data/intel`.
