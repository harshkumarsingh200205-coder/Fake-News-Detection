# Pipeline

This document describes the full application pipeline, from user input to retraining.

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

#### Main Components

- `frontend/src/app/page.tsx`: primary web UI
- `frontend/src/lib/backend.ts`: backend URL resolver and fetch helper
- `backend/main.py`: FastAPI application and endpoint orchestration

#### Request Path

The main page mostly calls FastAPI directly through `fetchBackend()`.

Effective flow:

```text
Browser
-> fetchBackend()
-> FastAPI
-> response JSON
-> React state update
-> UI render
```

The project also contains Next.js route handlers under `frontend/app/api/`, but they are not the main request path for the current page.

#### Page Load Flow

When the page loads, the UI:

1. calls `/health`
2. checks whether the backend is reachable and healthy
3. calls `/metrics`
4. calls `/history?limit=20`
5. updates dashboard, history, and status badges

#### Text Prediction Flow

```text
User enters article text
-> page validates minimum length
-> page sends POST /predict
-> FastAPI validates the payload
-> backend runs preprocessing and inference
-> backend stores the result in SQLite history
-> frontend renders label, confidence, and keyword explanation
```

#### URL Prediction Flow

Similar to text prediction, but with URL scraping first.

### 2. Text Preprocessing

All training and inference text passes through `backend/preprocessing.py`.

#### Purpose

The preprocessing stage converts raw article text into normalized tokens that are more suitable for TF-IDF vectorization and Logistic Regression.

#### Resource Strategy

The preprocessor looks for local NLTK resources under `backend/nltk_data/`.

Resources checked:

- `punkt`
- `punkt_tab`
- `stopwords`
- `wordnet`
- `omw-1.4`

If some resources are missing, the code falls back gracefully instead of failing.

#### Step-by-Step Pipeline

1. **Raw Text Cleanup**: `clean_text()` removes or normalizes HTML tags, URLs, email addresses, @mentions, #hashtags, repeated whitespace.

2. **Lowercasing**: All text is converted to lowercase.

3. **Punctuation Handling**: Punctuation is removed.

4. **Number Handling**: Numbers are preserved.

5. **Tokenization**: Uses NLTK when available, otherwise regex fallback.

6. **Stopword Removal**: Removes common stopwords.

7. **Lemmatization**: Optional, using WordNet.

8. **Filtering**: Removes short words (min length 2).

### 3. Prediction and URL Scraping

Prediction requests are handled by `backend/inference.py`, which supports both raw text and scraped article URLs.

#### Main Classes and Functions

- `initialize()`
- `get_instances()`
- `replace_model()`
- `URLScraper`
- `FakeNewsPredictor`

#### Shared Instance Strategy

The backend keeps cached module-level references to the model and preprocessor to avoid rebuilding them on every request.

#### Text Prediction Flow

```text
Raw text
-> preprocess()
-> check minimum usable token count
-> if model is fitted: predict_proba()
-> if model is not fitted: heuristic fallback
-> build response
-> attach keywords
```

#### Validation Rule

After preprocessing, the predictor requires at least 3 tokens.

#### Fitted-Model Prediction Path

1. Preprocessed text is vectorized through TF-IDF.
2. `predict_proba()` returns probabilities for FAKE and REAL.
3. Larger probability decides the label.
4. Keywords are extracted based on coefficients.

#### URL Scraping

- Validates URL
- Fetches HTML with requests
- Extracts text with BeautifulSoup
- Removes scripts, styles, nav, etc.
- Extracts from article, content, main tags

### 4. History and Persistence

Predictions are stored in SQLite by `backend/db.py`, which also stores verification labels used for retraining.

#### Database Type

File-based SQLite database, default `backend/fake_news_history.db`.

#### Initialization

`init_db()` runs on startup, creates `query_history` table if missing.

#### Stored Fields

- source, input_text, url, prediction, confidence, fake_probability, real_probability, keywords, processing_time, error, verified_label, verified_at, timestamp

#### Insert Flow

After prediction, `insert_query_history()` writes to SQLite.

#### Query Flows

- GET /history: paginated history
- GET /history/stats: statistics
- POST /history/{id}/verify: store verified label

### 5. Offline Training and Verified Retraining

The base model is trained by `backend/train.py`, while `backend/main.py` orchestrates retraining from verified history labels.

#### Base Training Dataset

Supports variants: Fake.csv/true.csv, etc.

#### Dataset Assembly

1. Read CSVs with pandas
2. Assign labels (0 fake, 1 real)
3. Construct full_text = title + text
4. Concatenate and shuffle

#### Dataset Preprocessing

Apply TextPreprocessor to every full_text, store processed_text. Remove empty rows.

#### Fixed Holdout Strategy

Deterministic train/validation split (80/20), stratified, saved to training_splits.joblib for reuse in retraining.

#### Model Training

- TF-IDF with max_features=10000, ngram_range=(1,2), etc.
- LogisticRegression with balanced class_weight
- Fit on train, evaluate on validation
- Save model, metrics

#### Verified Retraining

- Load verified samples from DB
- Combine with base train split
- Retrain, evaluate on same holdout
- Replace model

## Core Algorithms

- Text preprocessing with NLTK fallbacks
- TF-IDF vectorization
- Logistic Regression
- Keyword importance via coefficients
- Heuristic fallbacks
- Fixed-holdout evaluation
