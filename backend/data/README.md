# Dataset Folder

This folder contains the base training dataset used by `backend/train.py`.

## Expected files

- `Fake.csv`
- `True.csv`

The loader in `train.py` also accepts:

- `fake.csv`
- `False.csv`
- `false.csv`
- `true.csv`

## Source

- https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

## Current repository state

The current repository includes the CSV dataset files so the training pipeline can run without a separate download step.
