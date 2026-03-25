# Backend

This folder contains the API, persistence layer, ML pipeline, and training scripts for the web application.

## What belongs here

- API source files such as `main.py`, `inference.py`, `db.py`, and `train.py`
- Tests under `tests/`
- Dependency files such as `requirements.txt`
- Dataset files under `data/`
- Generated model outputs under `models/`

## What the backend does

- serves FastAPI endpoints for prediction, history, metrics, and retraining
- stores prediction history in SQLite
- preprocesses text before inference and training
- runs offline training from the CSV dataset
- retrains from verified history labels while keeping a fixed validation holdout

## Repository state

- The current repository includes the base dataset CSVs in `data/`
- Generated model files such as `fake_news_model.joblib`, `training_splits.joblib`, and `model_metrics.json` are created in `models/` during training
- Some presentation plot images may be committed, but large generated artifacts are generally treated as local outputs
- The API can still start without trained artifacts because inference falls back to demo heuristics

## Important local folders

- `data/`: base dataset files used by `train.py`
- `models/`: trained model, metrics, split bundle, and plots
- `nltk_data/`: optional local NLP resources used by preprocessing

## Key files

- `main.py`: FastAPI entrypoint and route orchestration
- `db.py`: SQLite schema and history helpers
- `inference.py`: prediction path, URL scraping, and fallback heuristics
- `preprocessing.py`: text cleanup, tokenization, stopword removal, and lemmatization
- `model.py`: TF-IDF plus Logistic Regression wrapper
- `train.py`: offline training and verified retraining bundle logic
