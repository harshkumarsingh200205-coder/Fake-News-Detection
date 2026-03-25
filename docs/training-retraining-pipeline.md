# Training and Retraining Pipeline

This document explains the implemented training workflow in `backend/train.py` and the verified retraining path orchestrated by `backend/main.py`.

## Base Training Dataset

The training loader supports the following filename variants:

- fake class: `Fake.csv`, `fake.csv`, `False.csv`, `false.csv`
- real class: `True.csv`, `true.csv`

The loader reads the CSV files from `backend/data/`.

## Dataset Assembly

`load_dataset()`:

1. reads fake and real CSV files with pandas
2. assigns labels:
   `0` for fake
   `1` for real
3. constructs `full_text` as `title + text`
4. concatenates both classes
5. shuffles rows with `random_state=42`

## Dataset Preprocessing

`preprocess_dataset()` applies the shared `TextPreprocessor` to every `full_text` item and stores the result in `processed_text`.

Rows that become empty after preprocessing are removed.

## Fixed Holdout Strategy

`build_training_splits()` creates a deterministic train/validation split using:

- `test_size=0.2`
- `random_state=42`
- `stratify=y`

The resulting split is saved to:

```text
backend/models/training_splits.joblib
```

That saved split is reused for retraining so evaluation stays comparable over time.

## Implemented Model Configuration

The training code constructs:

```text
TfidfVectorizer(
  max_features=10000,
  ngram_range=(1, 2),
  min_df=2,
  max_df=0.95,
  sublinear_tf=True,
  strip_accents="unicode",
  lowercase=True
)

LogisticRegression(
  C=1.0,
  solver="lbfgs",
  class_weight="balanced",
  max_iter=1000,
  random_state=42
)
```

## Evaluation Outputs

`evaluate()` computes:

- accuracy
- precision
- recall
- F1 score
- confusion matrix
- classification report

The code then augments metrics with:

- `train_samples`
- `validation_samples`
- `validation_strategy`
- `evaluated_at`

## Saved Outputs

Training writes:

- `fake_news_model.joblib`
- `model_metrics.json`
- `training_splits.joblib`

## Verified Retraining Pipeline

The verified retraining path works like this:

```text
verified rows from SQLite
-> preprocess verified texts
-> load saved base train/validation split
-> append verified samples to base training split
-> retrain model
-> evaluate on fixed validation holdout
-> save new model and metrics
-> replace in-memory model
```

## Minimum Data Requirement

Retraining requires at least `50` verified samples after preprocessing.

Why that matters:

- the raw number of verified rows is not enough
- some rows may become empty after preprocessing
- the threshold is checked on usable processed samples

## Auto-Retraining Check

The backend increments a prediction counter after each successful prediction.

By default, after every `50` successful predictions:

- the backend checks whether enough verified data exists
- if enough data exists, it attempts retraining
- otherwise it logs that retraining was skipped

This interval is controlled by:

```text
FAKE_NEWS_AUTO_RETRAIN_CHECK_INTERVAL
```
