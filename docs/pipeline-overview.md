# Pipeline Overview

This document gives a compact map of the full application pipeline and points to the more detailed markdown files for each stage.

## End-to-End Flow

```text
User input
-> Next.js web page
-> FastAPI endpoint
-> preprocessing
-> TF-IDF + Logistic Regression inference
-> keyword explanation
-> SQLite history persistence
-> optional verification
-> optional retraining
```

## Major Pipeline Parts

### 1. Frontend to Backend

The web UI in `frontend/src/app/page.tsx` sends requests to FastAPI using `fetchBackend()` from `frontend/src/lib/backend.ts`.

Detailed doc:

- [frontend-backend-flow.md](frontend-backend-flow.md)

### 2. Text Preprocessing

All training and inference text passes through `backend/preprocessing.py`.

Detailed doc:

- [preprocessing-pipeline.md](preprocessing-pipeline.md)

### 3. Prediction and URL Scraping

Prediction requests are handled by `backend/inference.py`, which supports both raw text and scraped article URLs.

Detailed doc:

- [inference-pipeline.md](inference-pipeline.md)

### 4. History and Persistence

Predictions are stored in SQLite by `backend/db.py`, which also stores verification labels used for retraining.

Detailed doc:

- [history-persistence-pipeline.md](history-persistence-pipeline.md)

### 5. Offline Training and Verified Retraining

The base model is trained by `backend/train.py`, while `backend/main.py` orchestrates retraining from verified history labels.

Detailed doc:

- [training-retraining-pipeline.md](training-retraining-pipeline.md)

## Core Algorithms Used

- Text cleanup, tokenization, stopword removal, and optional lemmatization
- TF-IDF vectorization over unigrams and bigrams
- balanced Logistic Regression classification
- coefficient-based keyword importance for explanations
- heuristic fallback predictions when no fitted model is available
- fixed-holdout evaluation for retraining

## Supporting Architecture Docs

- [system-architecture.md](system-architecture.md)
- [flowchart.md](flowchart.md)
- [complete_project_report.md](complete_project_report.md)
