# History and Persistence Pipeline

This document explains how prediction history and retraining labels are stored in `backend/db.py`.

## Database Type

The backend uses a file-based SQLite database.

Default path:

```text
backend/fake_news_history.db
```

The filename can be changed with `FAKE_NEWS_DB_FILENAME`.

## Initialization

`init_db()` runs on backend startup.

It creates the `query_history` table with `CREATE TABLE IF NOT EXISTS`, which means:

- the table is created if missing
- existing rows are preserved
- startup does not reset the history database

## Stored Fields

The `query_history` table stores:

- `source`
- `input_text`
- `url`
- `prediction`
- `confidence`
- `fake_probability`
- `real_probability`
- `keywords`
- `processing_time`
- `error`
- `verified_label`
- `verified_at`
- `timestamp`

## Insert Flow

After prediction requests, `insert_query_history(...)` writes a row to SQLite.

This happens for:

- successful text predictions
- failed text predictions
- successful URL predictions
- failed URL predictions

That design lets the application track both model behavior and failure cases.

## Read Flow

The history layer supports:

- `get_history(limit)`
- `get_history_entry(entry_id)`
- `get_history_total()`
- `get_history_stats()`

The UI uses these functions indirectly through FastAPI to show recent history and dashboard summaries.

## Preview Behavior

`get_history()` returns a shortened preview of long text inputs using `_build_history_preview(...)` so the history tab stays readable.

## Verification Flow

The backend supports `verify_history_entry(entry_id, verified_label)`.

Rules:

- only `REAL` and `FAKE` are accepted
- the target entry must exist
- the route layer requires stored source text before verification

Once verification is stored:

- `verified_label` is set
- `verified_at` is set
- the row becomes eligible for retraining

## Training-Data Queries

The persistence layer also exposes:

- `get_verified_training_data(limit)`
- `get_training_data_stats()`

These functions convert verified labels into the binary format expected by the training code:

- `REAL -> 1`
- `FAKE -> 0`

## Why This Layer Matters

The database is more than an audit log. It is the bridge between:

- live inference
- user feedback
- retraining readiness
- verified retraining data generation

Without this persistence layer, the application could still predict, but it would not have a feedback loop for model improvement.
