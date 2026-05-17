# Day 1 Report

## Project Focus

Set up the first version of a computer vision learning project using the Severstal Steel Defect Detection dataset.

The original Kaggle task is segmentation, but the first project milestone is image-level defect class prediction. Segmentation is intentionally deferred until the basic dataset and classification pipeline are working.

## Completed

- Created the initial project README.
- Set up Python project metadata in `pyproject.toml`.
- Installed the main project dependencies:
  - PyTorch
  - torchvision
  - pandas
  - Pillow
  - matplotlib
  - Jupyter
  - ipywidgets
  - scikit-learn
  - scikit-multilearn
- Confirmed the Kaggle dataset layout:

```text
data/
  train.csv
  train_images/
  test_images/
```

- Added `.gitignore` rules so local dataset files, virtualenv files, IDE files, and generated package metadata are not committed.
- Created the initial dataset class in `src/data/dataset.py`.
- Added stratified train/validation splitting in `src/data/split.py`.
- Started a notebook for dataset exploration in `notebooks/dataset.ipynb`.

## Dataset Class

Implemented `SeverstalSteelDefectDataset`, a PyTorch `Dataset` for loading training samples from:

- `data/train_images/`
- `data/train.csv`

Current behavior:

- Reads labels from the Kaggle `train.csv` file with pandas.
- Validates that the label CSV exists, is not empty, and can be parsed.
- Checks that referenced images exist in the configured image directory.
- Loads images with Pillow.
- Converts opened images to monochrome because the steel surface images are monochrome, reducing unnecessary color dimensions before model input.
- Supports an optional transform callable.
- Returns an image tensor and the corresponding `ClassId`.

## Train/Validation Split

Implemented `stratified_train_test_split` for splitting a PyTorch dataset into training and validation subsets.

Current behavior:

- Builds sample indices from the dataset.
- Reads each sample label from `dataset[i][1]`.
- Uses scikit-learn's `train_test_split`.
- Passes labels through `stratify=labels` so the train and validation sets preserve the class-label distribution.
- Returns PyTorch `Subset` objects for the training and validation splits.

This helps keep defect classes represented proportionally in both sets, which is important because the Severstal labels are class-imbalanced.

Stratification does not remove the underlying imbalance. Training will still need an imbalance strategy, such as class-weighted loss or weighted sampling, so minority defect classes are not overwhelmed by the dominant class.

Because some images have multiple defect rows in `train.csv`, splitting must happen at the image level rather than the row level. The same `ImageId` should always stay entirely in either the training split or the validation split. The dataset will be converted to one-hot image-level samples so each image has a single target vector representing all defect classes present in that image.

After conversion to one-hot multi-label targets, plain single-label stratification is not enough. The split should use iterative stratification through `scikit-multilearn` so combinations of defect labels are balanced across train and validation sets as well as possible.

## Notes

The dataset CSV contains `ImageId`, `ClassId`, and `EncodedPixels`.

Class distribution in `train.csv`:

| ClassId | Samples |
| --- | ---: |
| 1 | 897 |
| 2 | 247 |
| 3 | 5150 |
| 4 | 801 |

For the first pass, the dataset class uses `ClassId` as the image-level label and does not decode `EncodedPixels`. This keeps the first implementation focused on classification rather than segmentation.

The image data is treated as monochrome. Converting images to grayscale during loading avoids carrying redundant RGB channels through the pipeline.

Some images appear multiple times in `train.csv` because they contain defects of multiple types. The file has 7,095 annotation rows for 6,666 unique images. Of those images, 427 have multiple defect classes, and the maximum observed number of classes on a single image is 3. These should be represented as multi-label, one-hot image-level samples rather than independent row-level samples.

## Next Steps

- Fix the dataset length calculation so it returns the number of rows or image samples, not the total number of dataframe cells.
- Convert repeated image rows into one-hot multi-label targets.
- Update stratified splitting so duplicate `ImageId` values cannot be split across train and validation sets.
- Use `scikit-multilearn` iterative stratification for the one-hot multi-label targets.
- Add basic dataset tests or notebook checks for:
  - sample count
  - image tensor shape
  - label format
  - class distribution
- Build a small baseline training loop.
