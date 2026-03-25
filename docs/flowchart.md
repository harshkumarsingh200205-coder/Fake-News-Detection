# Flowchart and Workflow Notes

This file summarizes the current runtime flows in the repository and keeps the Mermaid diagrams aligned with the actual code.

## Overall Flowchart

Overall system view retained in `docs/diagram.mmd`.

```mermaid
flowchart TD
    User[User] --> Frontend[Next.js frontend]
    Frontend -->|Text analysis| PredictText[POST /predict]
    Frontend -->|URL analysis| PredictUrl[POST /predict-url]
    Frontend -->|Dashboard reads| StatusRoutes[GET health, metrics, history, training stats, retrain status]
    Frontend -. API available; not exposed in current page UI .-> VerifyRoute[POST /history/{entry_id}/verify]
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
        API --> AutoCheck[maybe_auto_retrain every 50 successful predictions by default]
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
        TrainPy[train.py main] --> LoadData[load_dataset Fake.csv or False.csv plus True.csv]
        LoadData --> PrepData[preprocess_dataset]
        PrepData --> BuildSplit[build_training_splits]
        BuildSplit --> FitModel[FakeNewsModel.fit]
        FitModel --> Evaluate[evaluate fixed holdout]
        Evaluate --> Persist[save model, metrics, and training_splits]
        Persist --> ModelArtifacts
    end
```

## Focused Pipeline Docs

The repository now includes separate markdown explanations for each major pipeline section:

- [pipeline-overview.md](pipeline-overview.md)
- [frontend-backend-flow.md](frontend-backend-flow.md)
- [preprocessing-pipeline.md](preprocessing-pipeline.md)
- [inference-pipeline.md](inference-pipeline.md)
- [history-persistence-pipeline.md](history-persistence-pipeline.md)
- [training-retraining-pipeline.md](training-retraining-pipeline.md)

## Diagram Files

- `docs/diagram.mmd`: overall system flow
- `docs/preprocessing_diagram.mmd`: focused preprocessing flow
- `docs/prediction_diagram.mmd`: focused prediction flow
- `docs/history_retraining_diagram.mmd`: history and retraining loop
- `docs/training_pipeline_diagram.mmd`: offline training flow

## Workflow Notes

1. `backend/main.py` initializes SQLite and the predictor during app startup.
2. The current main page mostly calls FastAPI directly through `fetchBackend()` instead of going through Next.js `/api/*` routes.
3. Text requests go to `FakeNewsPredictor.predict()`.
4. URL requests first pass through `URLScraper.scrape_url()`, then reuse the same prediction pipeline.
5. Predictions are written to SQLite with probabilities, keywords, timings, and error details.
6. Successful predictions increment the auto-retraining counter.
7. Verification is implemented at the API layer through `POST /history/{entry_id}/verify`, but the current main page does not surface it yet.
8. Manual retraining and retrain-status checks are already surfaced in the current web UI.
