# Development

## Local Development Setup

### Prerequisites

- Python 3.8+
- Node.js 18+
- PowerShell

### Setup

Run the setup script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

This creates:

- `backend/.env` from `backend/.env.example`
- `frontend/.env.local` from `frontend/.env.local.example`

### Running the Application

#### Terminal 1 (Backend)

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Runs on http://localhost:8000

#### Terminal 2 (Frontend)

```powershell
cd frontend
npm run dev
```

Runs on http://localhost:3000

#### Browser

Open http://localhost:3000 to use the Fake News Detector.

### Running Checks

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1
```

## Environment Variables

### Frontend

```env
BACKEND_URL=http://127.0.0.1:8000
```

### Backend

```env
FAKE_NEWS_DB_FILENAME=fake_news_history.db
FAKE_NEWS_AUTO_RETRAIN_CHECK_INTERVAL=50
FAKE_NEWS_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

## Project Structure

```
backend/
  main.py           # FastAPI app
  db.py             # SQLite database
  inference.py      # Prediction logic
  preprocessing.py  # Text preprocessing
  model.py          # ML model
  train.py          # Training script
  data/             # Datasets
  models/           # Model artifacts
  tests/            # Unit tests

frontend/
  src/app/page.tsx  # Main UI
  src/lib/backend.ts # Backend client
  src/components/   # UI components
  app/api/          # Next.js API routes
```
