# Frontend and Backend Flow

This document explains how the web application talks to the backend in the current codebase.

## Main Components

- `frontend/src/app/page.tsx`: primary web UI
- `frontend/src/lib/backend.ts`: backend URL resolver and fetch helper
- `backend/main.py`: FastAPI application and endpoint orchestration

## Request Path Used by the Current UI

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

## Page Load Flow

When the page loads, the UI:

1. calls `/health`
2. checks whether the backend is reachable and healthy
3. calls `/metrics`
4. calls `/history?limit=20`
5. updates dashboard, history, and status badges

## Text Prediction Flow

```text
User enters article text
-> page validates minimum length
-> page sends POST /predict
-> FastAPI validates the payload
-> backend runs preprocessing and inference
-> backend stores the result in SQLite history
-> frontend renders label, confidence, and keyword explanation
```

## URL Prediction Flow

```text
User enters article URL
-> page sends POST /predict-url
-> FastAPI validates URL format
-> backend fetches and parses article text
-> backend reuses the normal text prediction path
-> backend stores the result in SQLite history
-> frontend renders the analysis result
```

## Dashboard and History Flow

The dashboard and history tabs read backend state through:

- `GET /metrics`
- `GET /history`
- `GET /training/stats`
- `GET /retrain/status`

These endpoints allow the web UI to show:

- model metrics
- prediction counts
- confidence trends
- recent history
- retraining readiness

## Retraining Flow

```text
User opens Retrain tab
-> page loads /training/stats and /retrain/status
-> user clicks Retrain Model
-> page sends POST /retrain
-> backend retrains using verified labels plus base split
-> backend saves updated model and metrics
-> page refreshes training status and metrics
```

## Important Limitation

The backend supports `POST /history/{entry_id}/verify`, but the current page does not expose a verification control yet. So retraining-ready samples must currently be created through the API layer rather than the visible UI.
