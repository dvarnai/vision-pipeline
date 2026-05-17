# Vision Pipeline Learning Project

This repository is a hands-on project for learning how computer vision pipelines are built, trained, evaluated, and iterated on.

The project uses the [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection/) dataset from Kaggle. The original competition evaluated pixel-level defect segmentation, but this project starts with a simpler training objective: classify which defect classes appear in each annotated steel sample image.

## Goal

Build a practical vision pipeline that can classify steel surface images into one or more Severstal defect classes:

- `1`: defect class 1
- `2`: defect class 2
- `3`: defect class 3
- `4`: defect class 4

This keeps the first version focused on the core steps of a vision workflow without taking on pixel-level mask prediction immediately.

## Dataset

Dataset source:

https://www.kaggle.com/competitions/severstal-steel-defect-detection/

The dataset contains steel surface images and defect annotations. For this learning project, the segmentation annotations can be converted into image-level class labels using the `ClassId` column from `train.csv`. Some images may have more than one defect class, so the initial task is multi-label classification over the four defect classes, not binary defective/non-defective classification.

Raw dataset files should not be committed to this repository. After downloading and extracting the official Kaggle dataset zip, keep the files in this layout:

```text
data/
  train.csv
  train_images/
  test_images/
```

`train.csv` contains the original segmentation annotations with `ImageId`, `ClassId`, and `EncodedPixels` columns. The initial classification task uses the annotated training samples and predicts image-level defect classes from `ClassId`; it does not use `EncodedPixels` to train a segmentation model yet.

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

## Planned Pipeline

The initial project pipeline will focus on:

1. Downloading and organizing the Kaggle dataset locally
2. Creating image-level defect class labels from the original annotations
3. Exploring class balance and sample images
4. Building a baseline multi-label image classifier
5. Evaluating model performance with multi-label classification metrics
6. Iterating on preprocessing, augmentation, model choice, and validation strategy

## Initial Scope

The first version will not train a segmentation model. The priority is to learn the major moving parts of a vision pipeline:

- data ingestion
- labeling strategy
- train/validation split
- preprocessing and augmentation
- model training
- evaluation
- experiment tracking

## Stretch Goal

A future extension may revisit the original competition objective and train a segmentation model that predicts defect masks rather than only image-level defect presence.

## Status

This project is in the setup phase.
