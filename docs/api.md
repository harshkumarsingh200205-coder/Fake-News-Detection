# API Reference

The backend provides a REST API via FastAPI.

## Base URL

Default: `http://127.0.0.1:8000`

## Endpoints

### GET /

Root endpoint with API overview.

**Response:**

```json
{
  "name": "Fake News Detector API",
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
    "/health": "GET - Health check"
  }
}
```

### GET /health

Health check.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-04-06T..."
}
```

### POST /predict

Predict from text.

**Request Body:**

```json
{
  "text": "News article text...",
  "return_keywords": true,
  "top_keywords": 10
}
```

**Response:**

```json
{
  "success": true,
  "prediction": "REAL",
  "confidence": 85.2,
  "fake_probability": 14.8,
  "real_probability": 85.2,
  "keywords": [{ "word": "official", "importance": 0.45, "type": "real" }],
  "processing_time": 0.123
}
```

### POST /predict-url

Predict from URL.

**Request Body:**

```json
{
  "url": "https://example.com/article",
  "return_keywords": true,
  "top_keywords": 10
}
```

**Response:** Similar to /predict, plus `source_text`, `scraped_text_length`.

### GET /metrics

Get model metrics.

**Response:**

```json
{
  "accuracy": 0.92,
  "precision": 0.89,
  "recall": 0.91,
  "f1_score": 0.9,
  "confusion_matrix": [
    [100, 10],
    [5, 85]
  ],
  "model_loaded": true,
  "train_samples": 1000,
  "validation_samples": 200,
  "verified_samples_used": 50,
  "validation_strategy": "fixed_holdout_v1",
  "retrained_at": "2026-04-06T...",
  "last_updated": "2026-04-06T..."
}
```

### GET /history

Get prediction history.

**Query Params:**

- `limit`: int, default 20, max 100

**Response:**

```json
{
  "total": 150,
  "items": [
    {
      "id": 1,
      "input_text": "Article text...",
      "prediction": "FAKE",
      "confidence": 78.5,
      "timestamp": "2026-04-06T..."
    }
  ]
}
```

### GET /history/stats

Get history statistics.

**Response:**

```json
{
  "total_predictions": 150,
  "real_predictions": 80,
  "fake_predictions": 70,
  "verified_predictions": 20,
  "unverified_predictions": 130
}
```

### POST /history/{entry_id}/verify

Verify a prediction.

**Request Body:**

```json
{
  "verified_label": "REAL"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Stored verified label REAL for entry 1",
  "item": {...}
}
```

### GET /training/stats

Get verified training data stats.

**Response:**

```json
{
  "total_samples": 20,
  "verified_samples": 20,
  "unverified_predictions": 130,
  "real_samples": 12,
  "fake_samples": 8,
  "min_samples_for_retraining": 50
}
```

### POST /retrain

Manual retraining.

**Response:**

```json
{
  "success": true,
  "message": "Model retrained with 50 verified samples plus 1000 base samples",
  "verified_samples_used": 50,
  "base_train_samples": 1000,
  "training_samples": 1050,
  "validation_samples": 200,
  "metrics": {...},
  "retrained_at": "2026-04-06T..."
}
```

### GET /retrain/status

Check retraining status.

**Response:**

```json
{
  "should_retrain": false,
  "current_samples": 20,
  "min_samples_required": 50,
  "verified_samples": 20,
  "unverified_predictions": 130,
  "data_balance": {
    "real_ratio": 0.6,
    "fake_ratio": 0.4
  }
}
```
