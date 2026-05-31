# Day 5 Report

## Project Focus

Continued the transfer-learning track by checking whether the stronger ResNet baseline still works well at the original `150x150` image size, then started preparing a Vision Transformer experiment at a larger fixed input size.

The main goal was to separate two different input-size constraints:

- convolutional models such as ResNet can usually accept different spatial sizes, as long as the architecture does not contain fixed-size assumptions that are violated
- Vision Transformer models are much more tied to the image size used by their patch embedding and positional embeddings, especially when using pretrained weights

## Completed

- Added a ResNet-50 transfer-learning config using `150x150` inputs.
- Kept ImageNet normalization for the pretrained ResNet experiment.
- Confirmed the reason ResNet can be tested below `224x224`: the convolutional trunk is spatially flexible and uses pooling before classification.
- Started a ViT transfer-learning config.
- Set the initial ViT experiment direction around `224x224` inputs, since this is the common minimum size for many torchvision ImageNet ViT weight presets.
- Identified that ViT input size needs to be matched carefully to the selected pretrained weight preset.

## Experiment: ResNet at `150x150`

The previous ResNet transfer experiments used `224x224` inputs because that is the standard ImageNet preprocessing size. Day 5 tests whether ResNet-50 can still be used with the original smaller Intel image size:

```python
image_size = (150, 150)
```

This is valid for a CNN because convolutional layers slide over the image and do not require a fixed height and width by themselves. The practical constraints come from later operations:

- repeated downsampling cannot shrink the feature map too far
- fixed-size linear layers can require a specific flattened feature length
- adaptive pooling can remove much of that fixed-size requirement

ResNet handles this well because its convolutional feature extractor is followed by global adaptive pooling before the final fully connected classifier. That means the final classifier still receives a fixed feature vector even when the input image size changes.

In practice, the smaller input size did not hurt validation quality. The `150x150` ResNet-50 run performed just as well as the `224x224` run while significantly reducing training time.

Epoch time dropped from roughly `16` seconds per epoch at `224x224` to roughly `9` seconds per epoch at `150x150`. This makes the smaller input size a better default for the ResNet transfer-learning track unless later test-set evaluation shows a quality regression.

## Experiment: Vision Transformer at `224x224`

Started preparing a ViT transfer-learning experiment. Unlike a CNN, a ViT splits the image into fixed-size patches and uses positional embeddings over the resulting patch sequence. This makes the pretrained model more sensitive to input resolution.

The smallest ViT variant being considered, `ViT-B-16`, is already much larger than `ResNet-50`:

```text
ViT-B-16: 86,567,656 parameters
ResNet-50: 25,557,032 parameters
```

This makes `ViT-B-16` more than three times larger than `ResNet-50`. Because the `150x150` ResNet experiment already preserved quality while reducing epoch time, ViT should be treated as a heavier comparison point rather than a cheap next baseline.

For a `16x16` patch model, `224x224` produces:

```text
224 / 16 = 14 patches per side
14 * 14 = 196 image patches
```

The class token is then added to this sequence. Because pretrained positional embeddings are learned for a specific sequence length, the input size must match what the selected pretrained weights expect unless the implementation explicitly interpolates or adapts those embeddings.

The current direction is therefore to start ViT at `224x224`, which is the common minimum supported size for standard torchvision ImageNet ViT weights.

One implementation detail to keep in mind: not every torchvision ViT weight preset has the same expected image size. Some SWAG end-to-end ViT weights expect a larger resolution, so the selected weight preset should be checked before running the experiment.

The initial ViT result was surprisingly strong. With the ViT backbone frozen and only the classifier head unfrozen, the model already outperformed the best pretrained `ResNet-50` result within just `10` epochs.

Training cost was also more reasonable than the parameter count alone suggests. With the ViT backbone frozen and only the classifier head trained at `224x224`, one epoch took roughly `16` seconds. This is about the same epoch time as full fine-tuning `ResNet-50` at `224x224`.

By the end of the 100-epoch run, the frozen-backbone ViT result reached:

```text
Epoch 100/100, Train Loss: 0.2145, Val Loss: 0.2153, Subset Acc: 0.9347, Weighted F1: 0.9346, Macro F1: 0.9362, Micro F1: 0.9347, Time: 16561.88 ms
```

Full ViT fine-tuning was then tested by unfreezing all layers. This improved on the frozen-backbone classifier-head result within only `2` epochs:

```text
Epoch 2/100, Train Loss: 0.1731, Val Loss: 0.1739, Subset Acc: 0.9433, Weighted F1: 0.9431, Macro F1: 0.9443, Micro F1: 0.9433, Time: 35545.28 ms
```

The full fine-tuning run kept improving until epoch 7, which is the best validation result so far:

```text
Epoch 7/100, Train Loss: 0.0853, Val Loss: 0.1492, Subset Acc: 0.9507, Weighted F1: 0.9506, Macro F1: 0.9515, Micro F1: 0.9507, Time: 35864.83 ms
```

After epoch 7, the model started overfitting. Training loss continued decreasing, but validation quality dropped sharply:

```text
Epoch 8/100, Train Loss: 0.0736, Val Loss: 0.2155, Subset Acc: 0.9247, Weighted F1: 0.9244, Macro F1: 0.9268, Micro F1: 0.9247, Time: 36160.55 ms
Epoch 9/100, Train Loss: 0.0670, Val Loss: 0.2069, Subset Acc: 0.9280, Weighted F1: 0.9270, Macro F1: 0.9297, Micro F1: 0.9280, Time: 36072.37 ms
```

This is now the strongest validation result in the project so far, but it needs early stopping. The tradeoff is cost: full ViT fine-tuning takes about `35.5` to `36` seconds per epoch after the first epoch, more than twice the frozen-backbone ViT epoch time.

The next full fine-tuning experiment shortened training to `10` epochs. Because the cosine scheduler uses the configured total number of steps, reducing the run length from `100` epochs to `10` epochs made the learning rate decay much more sharply. This did not improve the best result, but the run still peaked around the same validation metrics as the longer full fine-tuning run.

The next experiment used different learning rates for different parts of the ViT. The newly initialized classifier head used a higher learning rate, while the pretrained backbone used a lower learning rate. This lets the new task-specific head adapt quickly without moving the already useful pretrained representation too aggressively.

So far, discriminative training made the run much more stable. Validation loss decreased smoothly instead of peaking early and then collapsing. By epoch 27, it was approaching the best full fine-tuning result more gradually:

```text
Epoch 27/100, Train Loss: 0.1173, Val Loss: 0.1488, Subset Acc: 0.9483, Weighted F1: 0.9482, Macro F1: 0.9494, Micro F1: 0.9483, Time: 34514.67 ms
```

Around epoch 40, the discriminative run started consistently matching or outperforming the previous full fine-tuning high from epoch 7. By epoch 43, it had matched the earlier peak:

```text
Epoch 43/100, Train Loss: 0.0974, Val Loss: 0.1425, Subset Acc: 0.9507, Weighted F1: 0.9505, Macro F1: 0.9516, Micro F1: 0.9507, Time: 35035.68 ms
```

The run then continued reaching new highs without showing the same overfitting collapse. By epoch 60, it reached the best validation result so far:

```text
Epoch 60/100, Train Loss: 0.0856, Val Loss: 0.1412, Subset Acc: 0.9527, Weighted F1: 0.9525, Macro F1: 0.9536, Micro F1: 0.9527, Time: 35481.59 ms
```

The important difference is that the discriminative run is still training without showing the same overfitting collapse. The main value of this setup is stability and better control over the pretrained backbone; it has now exceeded the previous peak while maintaining a healthier curve.

This changes the interpretation of the ViT experiment. Even though `ViT-B-16` is much larger than `ResNet-50`, its pretrained representation transfers very well to the Intel scene classification task, and both frozen-backbone training and full fine-tuning outperform the ResNet transfer-learning track. Full ViT fine-tuning currently gives the best quality, while the frozen-backbone ViT run remains the cheaper high-quality baseline.

## Current Interpretation

ResNet and ViT should not be treated the same way with respect to input size.

For ResNet:

- `224x224` is the standard ImageNet preprocessing size, not a strict architectural requirement.
- `150x150` works well because the model is fully convolutional up to adaptive pooling.
- The smaller input size preserved model quality while reducing epoch time from about `16` seconds to about `9` seconds.
- The current ResNet default should therefore move toward `150x150` unless final test metrics disagree.

For ViT:

- input size is part of the model structure through patch count and positional embeddings
- pretrained weights are usually tied to a specific training or fine-tuning resolution
- `224x224` is the right starting point only when the chosen ViT weights are also `224x224` compatible
- even `ViT-B-16` has `86,567,656` parameters, making it more than three times larger than `ResNet-50`
- despite the larger model size, the frozen-backbone ViT run outperformed the best fine-tuned `ResNet-50` result within `10` epochs
- after 100 epochs, frozen-backbone ViT reached `93.47%` validation accuracy and `0.9346` weighted F1
- the classifier-head-only ViT run at `224x224` takes about `16` seconds per epoch, roughly matching full fine-tuning `ResNet-50` at `224x224`
- full ViT fine-tuning outperformed the frozen-backbone ViT result within `2` epochs
- full ViT fine-tuning peaked at epoch 7 with `95.07%` validation accuracy and `0.9506` weighted F1
- after epoch 7, validation performance dropped while training loss kept decreasing, indicating overfitting
- full ViT fine-tuning is much slower at about `35.5` to `36` seconds per epoch after the first epoch
- shortening full ViT fine-tuning to `10` epochs made cosine annealing decay the learning rate much more aggressively, but did not improve the peak result
- discriminative learning rates made full ViT fine-tuning much more stable
- around epoch 40, the discriminative run started consistently matching or exceeding the previous full fine-tuning peak
- by epoch 60, the discriminative run reached a new high with `95.27%` validation accuracy and `0.9525` weighted F1
- unlike the earlier full fine-tuning run, the discriminative run is still training without overfitting
- ViT experiments should still be judged against both quality and compute cost, not accuracy alone