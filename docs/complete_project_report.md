## ✅ Project Summary

- Project name: Fake News Detector
- Type: Full-stack web app
- Backend: Python FastAPI
- Frontend: Next.js + React + Tailwind (shadcn UI)
- Model: TF-IDF + Logistic Regression
- Dataset: Kaggle ISOT Fake and Real News (Fake.csv, True.csv)
- Output: classification (FAKE/REAL), confidence, top keywords, metrics/history

## 🗂️ File structure

```
/
  README.md
  diagram.mmd
  pipeline.mmd
  docs/flowchart.md
  documentation/complete_project_report.md
  backend/
    main.py
    inference.py
    model.py
    preprocessing.py
    train.py
    generate_plots.py
    requirements.txt
    data/Fake.csv
    data/True.csv
    models/fake_news_model.joblib
    models/model_metrics.json
    nltk_data/
  frontend/
    src/app/page.tsx
    src/lib/backend.ts
    app/api/
      health/route.ts
      metrics/route.ts
      history/route.ts
      predict/route.ts
      predict-url/route.ts
    src/components/ui/
    src/hooks/use-toast.ts
    src/hooks/use-mobile.ts
    …
```

## 🧩 Backend

### backend/main.py

- FastAPI app with lifespan initialization
- CORS open access
- Endpoints:
  - `/` (info)
  - `/health` (status, model_loaded)
  - `/predict` (text predictions)
  - `/predict-url` (URL scraping + predict)
  - `/metrics` (model metrics)
  - `/history` (prediction history)
  - `/history/stats` (fake/real counts, avg confidence)
- Pydantic request/response models

### backend/preprocessing.py

- `TextPreprocessor`:
  - HTML, URL, email, mention, hashtag cleanup
  - lowercase, punctuation/numbers cleaning
  - tokenization (NLTK `word_tokenize`)
  - stopword removal (`nltk.corpus.stopwords` + custom)
  - lemmatization (`WordNetLemmatizer`)
  - minimum word length filter
- `download_nltk_resources`: auto-download required resources
- `get_preprocessor()` factory

### backend/model.py

- `FakeNewsModel` class:
  - TF-IDF vectorizer + logistic regression classifier pipeline
  - training: `fit()`; predict: `predict` and `predict_proba`
  - `get_keyword_importance(text, top_n)` using coefficient \* tfidf
  - `get_top_fake_keywords` and `get_top_real_keywords`
  - `evaluate` metrics + confusion matrix; `cross_validate`
  - `save` and `load` for persistence
  - `get_model_info` metadata
- `get_model()` loads persisted model if exists

### backend/inference.py

- Global singletons `_model`, `_preprocessor`
- `URLScraper`: URL validation, fetch (requests), parse (BeautifulSoup)
- `FakeNewsPredictor`:
  - `predict(text)`: preprocess + model predict/proba + keyword extraction
  - `_mock_prediction()`: heuristic fallback if model missing
  - `predict_from_url(url)`: scrape then predict

### backend/train.py

- `load_dataset()`: read CSVs, label fake=0, real=1, shuffle
- `preprocess_dataset()`: preprocess each article
- `train_model()`: train/test split, fit, evaluate
- `main()`: complete training cycle + sample quick tests

## 🖥️ Frontend

### frontend/src/lib/backend.ts

- `BACKEND_URL` from env or default
- `buildBackendUrl(path)` and `fetchBackend(path, init)`

### frontend/src/app/page.tsx

- states: input modes, input values, result, history, metrics, status, errors, dark mode
- `loadBackendData()`: health + metrics + history
- `handlePredict()`: POST predict/predict-url, error handling, history update
- chart config via Recharts
- `normalizePredictionLabel`, keyword mapping, history filtering

### frontend/app/api/\*/route.ts

- proxy calls: `/api/health`, `/api/metrics`, `/api/history`, `/api/predict`, `/api/predict-url`

## 🔄 Pipeline flow (diagram/pipeline files)

### diagram.mmd

- user input → endpoint → preprocess → model → return prediction

### pipeline.mmd

- includes training pipeline + inference path

## 📊 Model & Data

- dataset: `Fake.csv` + `True.csv`
- vectorization: TF-IDF uni/bigram
- classifier: LogisticRegression (balanced weights)
- threshold: 0.5
- output: fake & real probability, determination by max value

## 📈 Metrics & reporting

- accuracy, precision, recall, F1, confusion matrix
- saved at `backend/models/model_metrics.json`

## 🧪 Quick local tests

Backend:

```bash
cd backend
python main.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

API:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"BREAKING: ..."}'
```

## 🚀 Deployment notes

- ensure backend hosted and CORS configured
- set `NEXT_PUBLIC_BACKEND_URL` for production
- keep `backend/models`, `backend/nltk_data` persisted

## 🛠️ Enhancements

- retrain endpoint
- streaming / async queue with batch inference
- per-user history + authentication
- add evaluation charts in UI
- multiple language support
