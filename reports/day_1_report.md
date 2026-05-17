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
- Confirmed the Kaggle dataset layout:

```text
data/
  train.csv
  train_images/
  test_images/
```

- Added `.gitignore` rules so local dataset files, virtualenv files, IDE files, and generated package metadata are not committed.
- Created the initial dataset class in `src/data/dataset.py`.
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

## Notes

The dataset CSV contains `ImageId`, `ClassId`, and `EncodedPixels`.

For the first pass, the dataset class uses `ClassId` as the image-level label and does not decode `EncodedPixels`. This keeps the first implementation focused on classification rather than segmentation.

The image data is treated as monochrome. Converting images to grayscale during loading avoids carrying redundant RGB channels through the pipeline.

Some images may have more than one defect class annotation. A later iteration should decide whether to handle this as:

- a multi-label classification dataset, with one target vector per image, or
- a row-level classification dataset, where one image may appear once per class annotation.

## Next Steps

- Fix the dataset length calculation so it returns the number of rows or image samples, not the total number of dataframe cells.
- Decide the target representation for images with multiple defect classes.
- Add basic dataset tests or notebook checks for:
  - sample count
  - image tensor shape
  - label format
  - class distribution
- Build a small baseline training loop.
