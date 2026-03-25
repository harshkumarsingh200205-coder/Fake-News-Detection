# Backend

This folder is kept source-first so it looks clean in the repository and stays easy to review.

## What belongs here

- API source files such as `main.py`, `inference.py`, `db.py`, and `train.py`
- Tests under `tests/`
- Dependency files such as `requirements.txt`
- Small placeholder docs inside `data/` and `models/`

## What is intentionally not committed

- Virtual environments like `venv/`
- Downloaded NLTK resources in `nltk_data/`
- Raw datasets in `data/`
- Trained models, metrics, and plots in `models/`
- SQLite databases, logs, cache folders, and `__pycache__`

## Expected local folders

- `data/`: place `Fake.csv` and `True.csv` here before training
- `models/`: created outputs such as `fake_news_model.joblib` and `model_metrics.json`
- `nltk_data/`: optional local NLP resources downloaded for preprocessing

The API still starts without trained artifacts because inference falls back to demo mode until a model is trained.
