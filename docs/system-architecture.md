# System Architecture

The Fake News Detector is structured as a full-stack machine learning application with four main layers: a Next.js frontend, a FastAPI backend, a machine learning inference pipeline, and a persistence plus retraining layer. The goal is to support end-to-end fake-news analysis while keeping the system easy to demo, test, and extend.

## High-Level Flow

```text
User -> Frontend -> FastAPI -> ML Model -> Response -> Frontend
                           |
                           v
              Database (history + feedback)
                           |
                           v
                  Retraining pipeline
```

## Frontend Layer

The frontend is built with Next.js and provides the product surface of the project. It supports:

- text-based fake news analysis
- URL-based article analysis
- model metrics and health status views
- prediction history browsing
- manual verification of historical predictions
- manual retraining triggers

The frontend focuses on presentation and interaction. It does not perform local machine learning inference.

## Backend API Layer

The FastAPI backend coordinates the application:

- validates incoming requests
- scrapes article text when a URL is submitted
- preprocesses text for inference
- performs predictions
- stores prediction history
- exposes metrics, health, verification, and retraining endpoints

At startup, the backend initializes the SQLite database, loads the trained model if present, and prepares the text preprocessing pipeline.

## Preprocessing and Inference

The backend preprocessing stage:

- strips HTML, links, emails, and noisy characters
- normalizes casing and punctuation
- tokenizes text
- removes stopwords
- optionally lemmatizes when NLTK resources are available

Inference is handled by `FakeNewsPredictor`, which delegates classification to a TF-IDF plus Logistic Regression pipeline. If a fitted model is unavailable, the app still returns a fallback prediction and heuristic keywords so the UI remains functional for demos.

## Persistence Layer

Prediction history is stored in SQLite. Each record can include:

- source type
- input text or URL
- predicted label
- fake/real probabilities
- confidence
- keywords
- processing time
- verification label
- timestamp

This keeps the project easy to run locally while still supporting a realistic product flow.

## Retraining Workflow

Verified predictions create a feedback loop:

1. A user marks a past prediction as `REAL` or `FAKE`.
2. The backend stores the verified label in SQLite.
3. Verified records are preprocessed and merged with the base training split.
4. A fresh candidate model is trained.
5. The candidate is evaluated against a fixed holdout set.
6. If training succeeds, the new model replaces the in-memory model and saved artifacts.

This structure is intentionally conservative: the model can learn from verified feedback without redefining the validation set every time.

## Why This Architecture Works Well

- It separates UI, API, ML, and storage concerns clearly.
- It supports both live demos and offline training workflows.
- It makes explainability visible at the product level.
- It keeps infrastructure lightweight for interviews and academic evaluation.
- It provides a concrete story about iteration and model improvement, not just a one-off classifier.
