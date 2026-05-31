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

Start Jupyter:

```bash
jupyter lab
```

Run the Intel baseline training script:

```bash
python -m src.train_intel
```

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

### Experiment 6: ResNet-50 Transfer Learning

The transfer-learning track moved from the small custom CNN family to pretrained `ResNet-50`. The best recorded ResNet run fine-tuned the full network and reached:

- Accuracy: `0.8870`
- Weighted F1: `0.8869`
- Macro F1: `0.8900`
- Micro F1: `0.8870`
- Epoch time: `16103.78 ms`

This became the first strong shipping baseline, but it was later surpassed by the ViT experiments.

### Experiment 7: Vision Transformer Transfer Learning

The ViT track used `ViT-B-16` at `224x224`, matching the expected input size for the selected pretrained weight preset. `ViT-B-16` has `86,567,656` parameters, making it more than three times larger than `ResNet-50`, but its pretrained representation transferred very well to the Intel scene task.

With the ViT backbone frozen and only the classifier head trained, the 100-epoch run reached:

- Accuracy: `0.9347`
- Weighted F1: `0.9346`
- Macro F1: `0.9362`
- Micro F1: `0.9347`
- Epoch time: `16561.88 ms`

Full ViT fine-tuning then improved rapidly, peaking early at epoch 7:

- Accuracy: `0.9507`
- Weighted F1: `0.9506`
- Macro F1: `0.9515`
- Micro F1: `0.9507`
- Epoch time: `35864.83 ms`

The best Day 5 result came from full ViT fine-tuning with discriminative learning rates. By epoch 60, that run reached:

- Accuracy: `0.9527`
- Weighted F1: `0.9525`
- Macro F1: `0.9536`
- Micro F1: `0.9527`
- Epoch time: `35481.59 ms`

This is the strongest validation result recorded so far. The tradeoff is compute cost: the best ViT run takes roughly `35.5` seconds per epoch, while the frozen-backbone ViT run gives a cheaper high-quality baseline at roughly `16.5` seconds per epoch.

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

The project has a working Intel dataset class, a training/validation loop, custom CNN baselines, ResNet-50 transfer-learning baselines, ViT transfer-learning experiments, and a shared inference path for CLI, API, evaluation, and latency benchmarking.

## Inference

The inference path is intentionally thin. `src.inference.prediction` loads one project checkpoint, reconstructs the config-backed model, rebuilds the deterministic validation transform from the checkpoint's saved training statistics, and uses the same label contract as training and evaluation.

Run single-image CLI inference:

```bash
python -m src.infer_intel \
  checkpoints/intel_resnet50_transfer_4_epoch_0100.pt \
  data/intel/seg_test/seg_test/forest/20056.jpg \
  --top-k 3 \
  --pretty
```

Example response:

```json
{
  "predicted_label": "forest",
  "confidence": 0.982134,
  "model_version": "intel_resnet50_transfer_4_epoch_0100.pt:epoch-100:config-src.configs.intel_resnet50_transfer_4:config-sha-a1b2c3d4e5f6",
  "preprocessing_version": "preprocess-91a2b3c4d5e6",
  "label_contract_version": "intel-scene-v1",
  "latency_ms": 12.431,
  "top_k": [
    {"label": "forest", "confidence": 0.982134},
    {"label": "mountain", "confidence": 0.010442},
    {"label": "glacier", "confidence": 0.003817}
  ]
}
```

Invalid inputs return clear JSON errors on the CLI and HTTP 400 errors in the API. Validation covers unsupported file extensions, upload content types, extension/content-type mismatches, empty or oversized files, undecodable images, and unsupported Pillow image modes. Supported image modes are converted to RGB before preprocessing.

Run the FastAPI server:

```bash
CHECKPOINT_PATH=checkpoints/intel_resnet50_transfer_4_epoch_0100.pt \
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -s -X POST "http://localhost:8000/predict?top_k=3" \
  -F "file=@data/intel/seg_test/seg_test/forest/20056.jpg"
```

Build and run the container:

```bash
docker build -t vision-pipeline .
docker run --rm -p 8000:8000 \
  -e CHECKPOINT_PATH=/models/intel_resnet50_transfer_4_epoch_0100.pt \
  -v "$PWD/checkpoints:/models:ro" \
  vision-pipeline
```

Measure cold and warm single-image latency on the GPU machine:

```bash
python -m src.benchmark_latency \
  --device cuda \
  --image data/intel/seg_test/seg_test/forest/20056.jpg \
  --model ship=checkpoints/intel_resnet50_transfer_4_epoch_0100.pt \
  --model reject=checkpoints/rejected_model_epoch_0100.pt \
  --cold-runs 5 \
  --warm-runs 100
```

Cold latency includes checkpoint load plus one prediction. Warm latency loads the checkpoint once, runs warmups, then measures repeated single-image predictions.

## Training-Serving Skew

The shared contract is:

- Checkpoint: evaluation, CLI, API, and latency benchmark all call `load_inference_bundle`.
- Prediction: evaluation and serving call the same forward helper, `predict_batch_logits`; single-image serving wraps the same model and transform in `predict_image`.
- Preprocessing: inference rebuilds `config.build_val_transform(mean, std)` from the checkpoint's saved `training_stats`, never from freshly computed statistics.
- Labels: the Intel class order lives in `src.data.labels`, and new checkpoints save `label_contract.version`, `class_names`, and `class_to_index`.
- Versions: every response includes `model_version`, `preprocessing_version`, and `label_contract_version`.

Older checkpoints that do not contain `label_contract` fall back to `intel-scene-v1`, the class order used by the existing Intel dataset. Config source is stored in checkpoints and included in version hashes, but the loader imports the current config module. Treat config modules used for released checkpoints as immutable, or retrain/re-export if the config code changes.

## Shipping Decision

Ship the ViT-family checkpoint from `src.configs.intel_vit_transfer_4`, the discriminative-learning-rate full fine-tuning run. It is the strongest recorded checkpoint family in this repository, reaching validation accuracy `0.9527` and weighted F1 `0.9525` by epoch 60.

Keep the full-network `ResNet-50` transfer run from `src.configs.intel_resnet50_transfer_4` as the fallback baseline. It reached validation accuracy `0.8870` and weighted F1 `0.8869`, which is strong but materially below the ViT result.

Reject the available custom CNN family for shipping because the best recorded result is materially lower, topping out around validation accuracy `0.7203` and weighted F1 `0.7150`.

Before final release, run `src.test_intel` and `src.benchmark_latency` against the selected ViT checkpoint to record final test metrics and cold/warm latency.
