# Training script
"""
Model Training Script for Fake News Detection
"""

import os
import sys
from datetime import datetime

import joblib # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import get_preprocessor
from model import FakeNewsModel, MODELS_DIR, MODEL_PATH

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAINING_SPLITS_PATH = os.path.join(MODELS_DIR, "training_splits.joblib")
TRAIN_TEST_SPLIT = 0.2
TRAINING_RANDOM_STATE = 42


def resolve_dataset_path(*candidate_names):
    """Return the first matching dataset filename from the data directory."""
    for candidate_name in candidate_names:
        candidate_path = os.path.join(DATA_DIR, candidate_name)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def load_dataset():
    """Load the fake/real news dataset."""
    fake_path = resolve_dataset_path("Fake.csv", "fake.csv", "False.csv", "false.csv")
    true_path = resolve_dataset_path("True.csv", "true.csv")

    if not fake_path or not true_path:
        print("\n" + "=" * 50)
        print("DATASET NOT FOUND!")
        print("=" * 50)
        print("Please download from:")
        print("https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset")
        print(f"\nPlace Fake.csv/False.csv and True.csv in: {DATA_DIR}")
        return None, None

    print(f"\nLoading dataset from {DATA_DIR}...")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    fake_df["full_text"] = fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
    true_df["full_text"] = true_df["title"].fillna("") + " " + true_df["text"].fillna("")

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=TRAINING_RANDOM_STATE).reset_index(drop=True)

    print(f"Loaded {len(df):,} articles")
    print(f"  - Fake: {len(fake_df):,}")
    print(f"  - Real: {len(true_df):,}")

    return df, {"total": len(df), "fake": len(fake_df), "real": len(true_df)}


def preprocess_dataset(df, preprocessor):
    """Preprocess all texts."""
    print("\nPreprocessing texts...")

    processed = []
    total = len(df)

    for index, text in enumerate(df["full_text"]):
        if (index + 1) % 5000 == 0:
            print(f"  {index + 1:,}/{total:,}...")
        processed.append(preprocessor.preprocess(str(text)))

    df["processed_text"] = processed
    df = df[df["processed_text"].str.len() > 0]

    print(f"Preprocessing complete: {len(df):,} articles")
    return df


def build_training_splits(df):
    """Build a deterministic train/validation split for training and retraining."""
    X = df["processed_text"].tolist()
    y = df["label"].tolist()

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT,
        random_state=TRAINING_RANDOM_STATE,
        stratify=y,
    )

    return {
        "train_texts": X_train,
        "train_labels": y_train,
        "validation_texts": X_validation,
        "validation_labels": y_validation,
        "validation_strategy": "fixed_holdout_v1",
        "random_state": TRAINING_RANDOM_STATE,
        "test_size": TRAIN_TEST_SPLIT,
        "created_at": datetime.now().isoformat(),
    }


def save_training_splits(training_splits, splits_path: str = TRAINING_SPLITS_PATH):
    """Persist the split so retraining always reuses the same holdout set."""
    joblib.dump(training_splits, splits_path, protocol=4)
    print(f"Training splits saved to {splits_path}")


def load_training_splits(splits_path: str = TRAINING_SPLITS_PATH):
    """Load the cached deterministic split if it exists."""
    if not os.path.exists(splits_path):
        return None
    return joblib.load(splits_path)


def get_or_create_training_splits(force_rebuild: bool = False):
    """Load cached splits or rebuild them from the source dataset."""
    if not force_rebuild:
        cached_splits = load_training_splits()
        if cached_splits is not None:
            return cached_splits

    df, _ = load_dataset()
    if df is None:
        raise FileNotFoundError(
            f"Could not build training splits because the dataset is missing from {DATA_DIR}"
        )

    preprocessor = get_preprocessor()
    df = preprocess_dataset(df, preprocessor)
    training_splits = build_training_splits(df)
    save_training_splits(training_splits)
    return training_splits


def preprocess_labeled_texts(texts, labels, preprocessor=None):
    """Preprocess labeled texts and drop items that become empty."""
    active_preprocessor = preprocessor or get_preprocessor()
    processed_texts = []
    processed_labels = []

    for text, label in zip(texts, labels):
        processed_text = active_preprocessor.preprocess(str(text))
        if processed_text.strip():
            processed_texts.append(processed_text)
            processed_labels.append(label)

    return processed_texts, processed_labels


def build_retraining_bundle(verified_texts, verified_labels):
    """Combine the fixed base training split with externally verified feedback."""
    training_splits = get_or_create_training_splits(force_rebuild=False)
    processed_verified_texts, processed_verified_labels = preprocess_labeled_texts(
        verified_texts,
        verified_labels,
    )

    return {
        "train_texts": list(training_splits["train_texts"]) + processed_verified_texts,
        "train_labels": list(training_splits["train_labels"]) + processed_verified_labels,
        "validation_texts": list(training_splits["validation_texts"]),
        "validation_labels": list(training_splits["validation_labels"]),
        "base_train_samples": len(training_splits["train_texts"]),
        "verified_samples_used": len(processed_verified_texts),
        "validation_samples": len(training_splits["validation_texts"]),
        "validation_strategy": training_splits.get("validation_strategy", "fixed_holdout_v1"),
    }


def train_model(training_splits):
    """Train on the fixed training split and evaluate on the fixed holdout."""
    print("\n" + "=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)

    X_train = training_splits["train_texts"]
    y_train = training_splits["train_labels"]
    X_validation = training_splits["validation_texts"]
    y_validation = training_splits["validation_labels"]

    print(f"Training:   {len(X_train):,} samples")
    print(f"Validation: {len(X_validation):,} samples")

    model = FakeNewsModel(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    print("\nTraining...")
    model.fit(X_train, y_train)

    # Evaluate on the fixed holdout before saving the candidate model.
    print("\nEvaluating...")
    metrics = model.evaluate(X_validation, y_validation)
    metrics.update(
        {
            "train_samples": len(X_train),
            "validation_samples": len(X_validation),
            "validation_strategy": training_splits.get(
                "validation_strategy", "fixed_holdout_v1"
            ),
            "evaluated_at": datetime.now().isoformat(),
        }
    )
    model.metrics = metrics

    print("\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    return model, metrics


def main():
    print("\n" + "=" * 50)
    print("FAKE NEWS MODEL TRAINING")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df, _ = load_dataset()
    if df is None:
        return

    preprocessor = get_preprocessor()
    df = preprocess_dataset(df, preprocessor)
    training_splits = build_training_splits(df)
    save_training_splits(training_splits)

    model, _ = train_model(training_splits)

    print("\nSaving model...")
    model.save()

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")

    print("\nQuick Test:")
    test_texts = [
        "BREAKING: Scientists discover miracle cure!",
        "The Federal Reserve announced interest rate changes.",
    ]

    for text in test_texts:
        processed_text = preprocessor.preprocess(text)
        prediction, confidence = model.get_confidence([processed_text])[0]
        print(f"  '{text[:40]}...' -> {prediction} ({confidence:.1%})")


if __name__ == "__main__":
    main()
