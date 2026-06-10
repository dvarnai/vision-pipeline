# Final Report

## Inference Path

The shipping path now uses one shared inference contract across evaluation, CLI inference, API serving, and latency benchmarking.

- `src.inference.prediction.load_inference_bundle` loads the checkpoint once, reconstructs the config-backed model, restores checkpoint weights, rebuilds deterministic validation preprocessing from saved training statistics, and exposes model/preprocessing/label versions.
- `src.test_intel` now evaluates through the shared inference bundle and `predict_batch_logits`.
- `src.infer_intel` scores one validated image from the CLI and returns predicted label, confidence, optional top-k labels, model version, preprocessing version, label contract version, and latency.
- `src.api` exposes the same prediction function through FastAPI.
- `src.benchmark_latency` measures cold and warm single-image latency for selected and rejected checkpoints.

## Skew Controls

- Checkpoint identity: evaluation and serving load the same `.pt` checkpoint format.
- Preprocessing identity: inference uses checkpoint `training_stats.mean/std` with `config.build_val_transform`, matching evaluation preprocessing.
- Label identity: `src.data.labels` defines the Intel class order, and new checkpoints persist the label contract.
- Output identity: responses include `model_version`, `preprocessing_version`, and `label_contract_version`.
- Input validation: CLI and API reject unsupported extensions, bad content types, mismatches, empty/oversized files, undecodable images, and unsupported image modes.

Remaining risk: older checkpoints without `label_contract` fall back to `intel-scene-v1`. Checkpoints store config source and version hashes, but the runtime imports the current config module, so released config modules should be treated as immutable.

## Shipping Decision

Ship the ViT-family checkpoint from the off-machine run.

Best recorded validation result from Day 5:

- Config module: `src.configs.intel_vit_transfer_4`
- Model family: `ViT-B-16`
- Training mode: full fine-tuning with discriminative learning rates
- Best recorded epoch: `60`
- Validation accuracy: `0.9527`
- Weighted F1: `0.9525`
- Macro F1: `0.9536`
- Micro F1: `0.9527`
- Epoch time: `35481.59 ms`
- Cold latency, CPU: mean `569.572 ms`, median `454.555 ms` over 3 runs
- Warm latency, CPU: mean `35.659 ms`, median `35.636 ms` over 50 runs
- Cold latency, CUDA: mean `636.071 ms`, median `447.855 ms` over 3 runs
- Warm latency, CUDA: mean `1.491 ms`, median `1.490 ms` over 50 runs

CPU latency benchmark command:

```bash
python -m src.benchmark_latency \
  --device auto \
  --image data/intel/seg_test/seg_test/mountain/20738.jpg \
  --model ship=checkpoints/intel_vit_transfer_4_epoch_0060.pt \
  --model fallback=checkpoints/intel_resnet50_transfer_4_epoch_0100.pt \
  --cold-runs 3 \
  --warmup-runs 5 \
  --warm-runs 50
```

Benchmark environment resolved to `cpu`. Cold latency includes checkpoint load plus one prediction. Warm latency loads the checkpoint once, runs 5 warmups, then measures repeated single-image predictions.

CUDA latency benchmark used the same command with `--device cuda`. For the selected ViT checkpoint, the CUDA run reported cold latency mean `636.071 ms`, median `447.855 ms`, min `387.555 ms`, max `1072.802 ms`; warm latency mean `1.491 ms`, median `1.490 ms`, min `1.479 ms`, max `1.525 ms`.

Why:

- The Day 5 ViT runs are materially better than the ResNet transfer runs.
- The discriminative-learning-rate ViT run reached validation accuracy `0.9527` and weighted F1 `0.9525`, compared with the best recorded ResNet validation accuracy `0.8870` and weighted F1 `0.8869`.
- The inference path is checkpoint/config driven, so the same CLI/API/evaluation code can ship the ViT checkpoint.
- The selected ViT checkpoint has measured local CPU and CUDA serving latency through the same inference bundle used by CLI, API, evaluation, and latency benchmarking.

Reject the ResNet-50 transfer run as the shipping candidate.

Why:

- The best recorded local ResNet result is validation accuracy `0.8870`, weighted F1 `0.8869`, macro F1 `0.8900`.
- That is now the fallback baseline, not the model to ship, because the ViT checkpoint family outperformed it on validation metrics.
- Local CPU latency for the fallback checkpoint was faster than ViT in this benchmark: cold mean `121.315 ms`, warm mean `10.592 ms`. CUDA warm latency was similar between the fallback (`1.449 ms`) and selected ViT (`1.491 ms`), while fallback cold latency was lower (`127.504 ms`). The fallback remains useful when latency or model size matters more than peak validation accuracy.

Reject the custom CNN family for shipping because the best recorded result is much lower: validation accuracy about `0.7203`, weighted F1 `0.7150`.
