# FastAPI app
"""
FastAPI Application for Fake News Detector
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    API_NAME,
    APP_NAME,
    AUTO_RETRAIN_CHECK_INTERVAL,
    CORS_ALLOWED_ORIGINS,
)
from db import (
    get_history as db_get_history,
    get_history_entry,
    get_history_stats as db_get_history_stats,
    get_history_total as db_get_history_total,
    get_training_data_stats,
    get_verified_training_data,
    init_db,
    insert_query_history,
    verify_history_entry,
    MIN_VERIFIED_SAMPLES_FOR_RETRAINING,
)
from inference import FakeNewsPredictor, replace_model
from model import METRICS_PATH
from train import build_retraining_bundle, train_model

prediction_counter = 0


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="News text to analyze")
    return_keywords: bool = Field(True, description="Return influential keywords")
    top_keywords: int = Field(10, ge=1, le=50, description="Number of keywords")


class URLPredictRequest(BaseModel):
    url: str = Field(..., description="URL to scrape and analyze")
    return_keywords: bool = Field(True)
    top_keywords: int = Field(10, ge=1, le=50)

    @validator("url")
    def url_must_be_valid(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value


class VerifyHistoryRequest(BaseModel):
    verified_label: str = Field(..., description="Ground-truth label: REAL or FAKE")

    @validator("verified_label")
    def verified_label_must_be_valid(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"REAL", "FAKE"}:
            raise ValueError("verified_label must be REAL or FAKE")
        return normalized


class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    fake_probability: Optional[float] = None
    real_probability: Optional[float] = None
    keywords: Optional[List[Dict[str, Any]]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MetricsResponse(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    model_loaded: bool
    train_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    verified_samples_used: Optional[int] = None
    validation_strategy: Optional[str] = None
    retrained_at: Optional[str] = None
    last_updated: Optional[str] = None


def run_verified_retraining(limit: int = 1000) -> Dict[str, Any]:
    """Retrain using only externally verified labels and a fixed validation set."""
    verified_texts, verified_labels = get_verified_training_data(limit=limit)

    if len(verified_texts) < MIN_VERIFIED_SAMPLES_FOR_RETRAINING:
        return {
            "success": False,
            "error": (
                "Insufficient verified training data. "
                f"Need at least {MIN_VERIFIED_SAMPLES_FOR_RETRAINING} verified samples, "
                f"got {len(verified_texts)}"
            ),
            "available_samples": len(verified_texts),
            "min_samples_required": MIN_VERIFIED_SAMPLES_FOR_RETRAINING,
        }

    retraining_bundle = build_retraining_bundle(verified_texts, verified_labels)
    verified_samples_used = retraining_bundle["verified_samples_used"]

    if verified_samples_used < MIN_VERIFIED_SAMPLES_FOR_RETRAINING:
        return {
            "success": False,
            "error": (
                "Verified samples became too short after preprocessing. "
                f"Need at least {MIN_VERIFIED_SAMPLES_FOR_RETRAINING} usable samples, "
                f"got {verified_samples_used}"
            ),
            "available_samples": verified_samples_used,
            "min_samples_required": MIN_VERIFIED_SAMPLES_FOR_RETRAINING,
        }

    retrained_model, metrics = train_model(retraining_bundle)
    retrained_at = datetime.now().isoformat()
    metrics.update(
        {
            "base_train_samples": retraining_bundle["base_train_samples"],
            "verified_samples_used": verified_samples_used,
            "total_training_samples": len(retraining_bundle["train_texts"]),
            "validation_samples": retraining_bundle["validation_samples"],
            "validation_strategy": retraining_bundle["validation_strategy"],
            "retrained_at": retrained_at,
        }
    )
    retrained_model.metrics = metrics
    retrained_model.save()
    replace_model(retrained_model)

    return {
        "success": True,
        "message": (
            f"Model retrained with {verified_samples_used} verified samples "
            f"plus {retraining_bundle['base_train_samples']} base samples"
        ),
        "verified_samples_used": verified_samples_used,
        "base_train_samples": retraining_bundle["base_train_samples"],
        "training_samples": len(retraining_bundle["train_texts"]),
        "validation_samples": retraining_bundle["validation_samples"],
        "metrics": metrics,
        "retrained_at": retrained_at,
    }


def maybe_auto_retrain() -> None:
    """Periodically check whether enough verified labels exist for retraining."""
    global prediction_counter

    prediction_counter += 1
    if prediction_counter < AUTO_RETRAIN_CHECK_INTERVAL:
        return

    try:
        result = run_verified_retraining(limit=1000)
        if result.get("success"):
            print(
                "Auto-retraining completed with "
                f"{result.get('verified_samples_used', 0)} verified samples."
            )
        else:
            print(f"Auto-retraining skipped: {result.get('error')}")
    except Exception as exc:
        print(f"Auto-retraining failed: {exc}")
    finally:
        prediction_counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting {API_NAME}...")
    init_db()

    try:
        predictor = FakeNewsPredictor()
        print(f"Model initialized. Is fitted: {predictor.model.is_fitted}")
    except Exception as exc:
        print(f"Model initialization failed: {exc}")
        raise

    yield

    print(f"Shutting down {API_NAME}...")


app = FastAPI(
    title=API_NAME,
    description="Detect fake news using TF-IDF + Logistic Regression",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": API_NAME,
        "version": "1.1.0",
        "endpoints": {
            "/predict": "POST - Predict if news text is fake or real",
            "/predict-url": "POST - Predict from URL content",
            "/metrics": "GET - Get model performance metrics",
            "/history": "GET - Get prediction history",
            "/history/stats": "GET - Get history statistics",
            "/history/{id}/verify": "POST - Store a verified ground-truth label",
            "/training/stats": "GET - Get verified retraining data statistics",
            "/retrain": "POST - Retrain using verified labels and a fixed holdout set",
            "/retrain/status": "GET - Check if enough verified data exists for retraining",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health_check():
    predictor = FakeNewsPredictor()
    return {
        "status": "healthy",
        "model_loaded": predictor.model.is_fitted,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_text(request: PredictRequest):
    try:
        predictor = FakeNewsPredictor()
        result = predictor.predict(
            request.text,
            return_keywords=request.return_keywords,
            top_keywords=request.top_keywords,
        )

        if result.get("success"):
            insert_query_history(
                source="text",
                input_text=request.text.strip(),
                url=None,
                prediction=result.get("prediction"),
                confidence=result.get("confidence"),
                fake_probability=result.get("fake_probability"),
                real_probability=result.get("real_probability"),
                keywords=result.get("keywords", []),
                processing_time=result.get("processing_time"),
                error=None,
            )
            maybe_auto_retrain()
        else:
            insert_query_history(
                source="text",
                input_text=request.text.strip(),
                url=None,
                prediction=None,
                confidence=None,
                fake_probability=None,
                real_probability=None,
                keywords=[],
                processing_time=result.get("processing_time"),
                error=result.get("error"),
            )

        return PredictionResponse(**result, timestamp=datetime.now().isoformat())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict-url", response_model=PredictionResponse)
async def predict_from_url(request: URLPredictRequest):
    try:
        predictor = FakeNewsPredictor()
        result = predictor.predict_from_url(request.url)

        insert_query_history(
            source="url",
            input_text=result.get("source_text"),
            url=request.url,
            prediction=result.get("prediction"),
            confidence=result.get("confidence"),
            fake_probability=result.get("fake_probability"),
            real_probability=result.get("real_probability"),
            keywords=result.get("keywords", []),
            processing_time=result.get("processing_time"),
            error=result.get("error"),
        )

        if result.get("success"):
            maybe_auto_retrain()

        return PredictionResponse(**result, timestamp=datetime.now().isoformat())
    except Exception as exc:
        insert_query_history(
            source="url",
            input_text=None,
            url=request.url,
            prediction=None,
            confidence=None,
            fake_probability=None,
            real_probability=None,
            keywords=[],
            processing_time=None,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    predictor = FakeNewsPredictor()
    model_loaded = predictor.model.is_fitted
    metrics = predictor.model.metrics if predictor.model.metrics else {}

    if model_loaded and not metrics and os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
        except Exception:
            metrics = {}

    return MetricsResponse(
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        confusion_matrix=metrics.get("confusion_matrix"),
        model_loaded=model_loaded,
        train_samples=metrics.get("train_samples"),
        validation_samples=metrics.get("validation_samples"),
        verified_samples_used=metrics.get("verified_samples_used"),
        validation_strategy=metrics.get("validation_strategy"),
        retrained_at=metrics.get("retrained_at"),
        last_updated=datetime.now().isoformat(),
    )


@app.get("/history")
async def get_history(limit: int = Query(20, ge=1, le=100)):
    items = db_get_history(limit)
    total = db_get_history_total()
    return {"total": total, "items": items}


@app.get("/history/stats")
async def get_history_stats():
    return db_get_history_stats()


@app.post("/history/{entry_id}/verify")
async def verify_history(entry_id: int, request: VerifyHistoryRequest):
    existing_entry = get_history_entry(entry_id)
    if existing_entry is None:
        raise HTTPException(status_code=404, detail="History entry not found")

    if not existing_entry.get("input_text"):
        raise HTTPException(
            status_code=400,
            detail="Only entries with stored source text can be verified for retraining",
        )

    try:
        updated_entry = verify_history_entry(entry_id, request.verified_label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if updated_entry is None:
        raise HTTPException(status_code=404, detail="History entry not found")

    return {
        "success": True,
        "message": f"Stored verified label {request.verified_label} for entry {entry_id}",
        "item": updated_entry,
    }


@app.get("/training/stats")
async def get_training_stats():
    """Get statistics about verified data available for retraining."""
    return get_training_data_stats()


@app.post("/retrain")
async def retrain_model():
    """Retrain the model using verified history data and a fixed validation set."""
    try:
        result = run_verified_retraining(limit=1000)
        return result
    except FileNotFoundError as exc:
        return {
            "success": False,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


@app.get("/retrain/status")
async def get_retrain_status():
    """Check if retraining is recommended based on verified labels only."""
    stats = get_training_data_stats()
    min_samples = stats["min_samples_for_retraining"]
    current_samples = stats["total_samples"]

    return {
        "should_retrain": current_samples >= min_samples,
        "current_samples": current_samples,
        "min_samples_required": min_samples,
        "verified_samples": stats["verified_samples"],
        "unverified_predictions": stats["unverified_predictions"],
        "data_balance": {
            "real_ratio": stats["real_samples"] / max(current_samples, 1),
            "fake_ratio": stats["fake_samples"] / max(current_samples, 1),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
