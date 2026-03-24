# Flowchart and Component Workflow

## Overall Flowchart (Mermaid)

Overall system view retained in `docs/diagram.mmd`.

```mermaid
flowchart TD
    User[User] --> Frontend[Next.js frontend]
    Frontend -->|Text analysis| PredictText[POST /predict]
    Frontend -->|URL analysis| PredictUrl[POST /predict-url]
    Frontend -->|Dashboard reads| StatusRoutes[GET health, metrics, history, training stats, retrain status]
    Frontend -->|Verify label| VerifyRoute[POST /history/:id/verify]
    Frontend -->|Manual retrain| RetrainRoute[POST /retrain]

    PredictText --> API[FastAPI backend]
    PredictUrl --> API
    StatusRoutes --> API
    VerifyRoute --> API
    RetrainRoute --> API

    subgraph Startup[Startup and shared state]
        AppStart[App startup] --> InitDb[init_db]
        AppStart --> InitPredictor[FakeNewsPredictor initialize]
        InitPredictor --> LoadAssets[get_model plus get_preprocessor]
        LoadAssets --> ModelArtifacts[(Model artifacts)]
        InitDb --> HistoryDb[(SQLite query_history)]
    end

    subgraph Prediction[Prediction request flow]
        API --> TextPredict[FakeNewsPredictor.predict]
        API --> Scrape[URLScraper.scrape_url]
        Scrape --> Fetch[Validate URL, fetch HTML, extract article text]
        Fetch --> TextPredict
        TextPredict --> Preprocess[TextPreprocessor.preprocess]
        Preprocess --> ModelReady{Model fitted}
        ModelReady -->|Yes| PredictProba[FakeNewsModel.predict_proba]
        ModelReady -->|No| Mock[_mock_prediction fallback]
        PredictProba --> BuildResult[Prediction, confidence, probabilities]
        PredictProba --> Keywords[get_keyword_importance]
        BuildResult --> Response[Response JSON]
        Keywords --> Response
        Mock --> Response
    end

    Response --> SaveHistory[insert_query_history]
    SaveHistory --> HistoryDb
    Response --> Frontend

    API --> ReadHistory[History, verification, and retrain status queries]
    ReadHistory --> HistoryDb
    API --> ReadMetrics[Metrics endpoint reads saved model metrics]
    ReadMetrics --> ModelArtifacts

    subgraph Feedback[Verification and retraining loop]
        API --> VerifyEntry[get_history_entry plus verify_history_entry]
        VerifyEntry --> HistoryDb
        API --> AutoCheck[maybe_auto_retrain every 50 successful predictions]
        AutoCheck --> RetrainFlow[run_verified_retraining]
        API --> RetrainFlow
        RetrainFlow --> VerifiedData[get_verified_training_data]
        VerifiedData --> HistoryDb
        RetrainFlow --> Bundle[build_retraining_bundle]
        Bundle --> TrainCandidate[train_model on base split plus verified labels]
        TrainCandidate --> SaveCandidate[save model, save metrics, replace_model]
        SaveCandidate --> ModelArtifacts
    end

    subgraph Training[Offline training pipeline]
        TrainPy[train.py main] --> LoadData[load_dataset Fake.csv plus True.csv]
        LoadData --> PrepData[preprocess_dataset]
        PrepData --> BuildSplit[build_training_splits]
        BuildSplit --> FitModel[FakeNewsModel.fit]
        FitModel --> Evaluate[evaluate fixed holdout]
        Evaluate --> Persist[save model, metrics, and training_splits]
        Persist --> ModelArtifacts
    end
```

## Preprocessing Flow

Focused preprocessing view retained in `docs/preprocessing_diagram.mmd`.

```mermaid
flowchart TD
    User[User input] --> Mode{Input type}
    Mode -->|Text| TextRequest[POST /predict]
    Mode -->|URL| UrlRequest[POST /predict-url]

    TextRequest --> Predictor[FakeNewsPredictor.predict]
    UrlRequest --> Scraper[URLScraper.scrape_url]
    Scraper --> Validate[Validate URL]
    Validate --> Fetch[Fetch HTML with requests]
    Fetch --> Extract[Extract article text with BeautifulSoup]
    Extract --> Predictor

    Predictor --> Clean[TextPreprocessor.preprocess]
    Clean --> Tokens[Cleaned and normalized text]
```

## Prediction Flow

Focused prediction view retained in `docs/prediction_diagram.mmd`.

```mermaid
flowchart TD
    Input[Processed text] --> Check{Model fitted}
    Check -->|Yes| Predict[FakeNewsModel.predict_proba]
    Check -->|No| Fallback[_mock_prediction]

    Predict --> Label[Prediction plus confidence]
    Predict --> KeywordScores[get_keyword_importance]
    KeywordScores --> Keywords[Top real or fake keywords]

    Label --> Response[Response JSON]
    Keywords --> Response
    Fallback --> DemoKeywords[_build_demo_keywords]
    DemoKeywords --> Response
```

## History and Retraining Flow

Focused storage and feedback loop retained in `docs/history_retraining_diagram.mmd`.

```mermaid
flowchart TD
    Response[Prediction response] --> SaveHistory[insert_query_history]
    SaveHistory --> HistoryDb[(SQLite query_history)]

    HistoryDb --> HistoryApi[GET /history and /history/stats]
    HistoryDb --> VerifyApi[POST /history/:id/verify]
    VerifyApi --> Verified[verified_label plus verified_at stored]

    Response --> AutoCheck[maybe_auto_retrain]
    AutoCheck --> RetrainFlow[run_verified_retraining]
    Verified --> RetrainFlow
    ManualRetrain[POST /retrain] --> RetrainFlow
    StatusApi[GET /training/stats and /retrain/status] --> HistoryDb

    RetrainFlow --> VerifiedData[get_verified_training_data]
    VerifiedData --> Bundle[build_retraining_bundle]
    Bundle --> Train[train_model]
    Train --> SaveModel[save model, metrics, replace_model]
```

## Offline Training Flow

Focused training pipeline retained in `docs/training_pipeline_diagram.mmd`.

```mermaid
flowchart TD
    Start[train.py main] --> LoadData[load_dataset]
    LoadData --> Merge[Load Fake.csv and True.csv]
    Merge --> Label[Assign labels and build full_text]
    Label --> Prep[preprocess_dataset]
    Prep --> Split[build_training_splits]
    Split --> Fit[FakeNewsModel.fit]
    Fit --> Evaluate[evaluate fixed holdout]
    Evaluate --> SaveModel[save fake_news_model.joblib]
    Evaluate --> SaveMetrics[save model_metrics.json]
    Split --> SaveSplits[save training_splits.joblib]
```

## Backend Algorithmic Workflow

1. `main.py` starts by initializing SQLite history storage and the shared predictor components.
2. the frontend calls FastAPI directly for health, metrics, history, prediction, verification, and retraining endpoints.
3. text requests go to `FakeNewsPredictor.predict()`, while URL requests first pass through `URLScraper.scrape_url()`.
4. extracted text is normalized by `TextPreprocessor.preprocess()`.
5. if a trained model exists, `FakeNewsModel.predict_proba()` and `get_keyword_importance()` build the final result.
6. if the model is unavailable, `_mock_prediction()` returns a fallback prediction and demo keywords.
7. each prediction is written to SQLite with probabilities, keywords, timing, and any error details.
8. successful predictions also trigger the periodic auto-retraining readiness check.

## Verification and Retraining Workflow

1. the frontend can verify a stored prediction with `POST /history/:id/verify`.
2. verified labels are stored in SQLite and become eligible retraining samples.
3. `/training/stats` and `/retrain/status` summarize current verified data and retraining readiness.
4. `/retrain` and automatic checks both call `run_verified_retraining()`.
5. retraining combines the fixed base training split with verified feedback, evaluates on the fixed holdout, then saves the new model and metrics.

## Data Science Flow

- Train with `train.py` using `Fake.csv` and `True.csv`
- Preprocess the full dataset before building a deterministic train and validation split
- Evaluate metrics on a fixed holdout: accuracy, precision, recall, F1, confusion matrix
- Save model pipeline to `backend/models/fake_news_model.joblib`
- Save metrics to `backend/models/model_metrics.json`
- Save reusable split metadata to `backend/models/training_splits.joblib`
