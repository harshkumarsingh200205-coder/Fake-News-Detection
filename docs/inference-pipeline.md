# Inference Pipeline

This document explains how prediction works in `backend/inference.py`.

## Main Classes and Functions

- `initialize()`
- `get_instances()`
- `replace_model()`
- `URLScraper`
- `FakeNewsPredictor`

## Shared Instance Strategy

The backend keeps cached module-level references to:

- the model
- the preprocessor

This avoids rebuilding them on every request.

`get_instances()` also supports a useful runtime behavior:

- if the process started before a trained artifact existed
- and `backend/models/fake_news_model.joblib` later appears
- the backend attempts to reload the trained model automatically

## Text Prediction Flow

```text
Raw text
-> preprocess()
-> check minimum usable token count
-> if model is fitted: predict_proba()
-> if model is not fitted: heuristic fallback
-> build response
-> attach keywords
```

## Validation Rule

After preprocessing, the predictor requires at least `3` tokens. If the cleaned text is too short, it returns an error response instead of forcing a prediction.

## Fitted-Model Prediction Path

When a trained model exists:

1. the preprocessed text is vectorized through the saved TF-IDF pipeline
2. `predict_proba()` returns probabilities for `FAKE` and `REAL`
3. the larger probability decides the label
4. confidence is the maximum probability

The returned payload includes:

- `prediction`
- `confidence`
- `fake_probability`
- `real_probability`
- `processing_time`

## Keyword Explanation

If keyword explanation is requested, the predictor calls `model.get_keyword_importance(...)`.

The importance score for each term is based on:

```text
term TF-IDF weight * Logistic Regression coefficient
```

Interpretation:

- negative signed contribution -> fake-leaning term
- positive signed contribution -> real-leaning term

## Heuristic Fallback Path

If no fitted model is available, the predictor uses `_mock_prediction(...)`.

That fallback:

- counts heuristic fake indicators such as `shocking`, `miracle`, `secret`, and `viral`
- counts heuristic real indicators such as `official`, `reported`, `study`, and `confirmed`
- converts those counts into approximate fake and real probabilities
- still returns keyword explanations so the UI remains useful

This is a demo fallback, not a trained statistical model.

## URL Prediction Flow

`predict_from_url(...)` uses `URLScraper.scrape_url(...)` first.

URL scraping steps:

1. validate URL format
2. fetch HTML with `requests.get(...)`
3. parse the page with BeautifulSoup
4. remove non-content sections such as `script`, `style`, `nav`, `header`, `footer`, and `aside`
5. try article-focused selectors such as `article`, `[class*="content"]`, and `main`
6. collect paragraph text
7. pass the extracted text into the standard prediction path

If scraping fails or extracts too little text, the backend returns an error instead of pretending the analysis succeeded.
