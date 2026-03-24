import os
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from config import DB_FILENAME

VALIDATED_LABELS = {"REAL", "FAKE"}
HISTORY_PREVIEW_LENGTH = 220
MIN_VERIFIED_SAMPLES_FOR_RETRAINING = 50


def get_db_path() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, DB_FILENAME)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_query_history_columns(cursor: sqlite3.Cursor) -> None:
    existing_columns = {
        row["name"]
        for row in cursor.execute("PRAGMA table_info(query_history)").fetchall()
    }
    required_columns = {
        "verified_label": "TEXT",
        "verified_at": "TEXT",
    }

    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE query_history ADD COLUMN {column_name} {column_type}"
            )


def _build_history_preview(
    input_text: Optional[str],
    url: Optional[str],
    max_length: int = HISTORY_PREVIEW_LENGTH,
) -> Optional[str]:
    source_text = (input_text or url or "").strip()
    if not source_text:
        return None

    if len(source_text) <= max_length:
        return source_text

    return f"{source_text[:max_length].rstrip()}..."


def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            input_text TEXT,
            url TEXT,
            prediction TEXT,
            confidence REAL,
            fake_probability REAL,
            real_probability REAL,
            keywords TEXT,
            processing_time REAL,
            error TEXT,
            verified_label TEXT,
            verified_at TEXT,
            timestamp TEXT NOT NULL
        )
        """
    )
    _ensure_query_history_columns(cursor)
    conn.commit()
    conn.close()


def insert_query_history(
    source: str,
    input_text: Optional[str],
    url: Optional[str],
    prediction: Optional[str],
    confidence: Optional[float],
    fake_probability: Optional[float],
    real_probability: Optional[float],
    keywords: Optional[List[Dict[str, Any]]],
    processing_time: Optional[float],
    error: Optional[str],
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    keywords_json = json.dumps(keywords or [])
    timestamp = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO query_history
        (source, input_text, url, prediction, confidence, fake_probability, real_probability, keywords, processing_time, error, verified_label, verified_at, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source,
            input_text,
            url,
            prediction,
            confidence,
            fake_probability,
            real_probability,
            keywords_json,
            processing_time,
            error,
            None,
            None,
            timestamp,
        ),
    )
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id # pyright: ignore[reportReturnType]


def get_history(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM query_history ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        result.append({
            'id': row['id'],
            'source': row['source'],
            'input_text': _build_history_preview(row['input_text'], row['url']),
            'url': row['url'],
            'prediction': row['prediction'],
            'confidence': row['confidence'],
            'fake_probability': row['fake_probability'],
            'real_probability': row['real_probability'],
            'keywords': json.loads(row['keywords'] or '[]'),
            'processing_time': row['processing_time'],
            'error': row['error'],
            'verified_label': row['verified_label'],
            'verified_at': row['verified_at'],
            'timestamp': row['timestamp'],
        })
    return result


def get_history_entry(entry_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM query_history WHERE id = ?",
        (entry_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        'id': row['id'],
        'source': row['source'],
        'input_text': row['input_text'],
        'url': row['url'],
        'prediction': row['prediction'],
        'confidence': row['confidence'],
        'fake_probability': row['fake_probability'],
        'real_probability': row['real_probability'],
        'keywords': json.loads(row['keywords'] or '[]'),
        'processing_time': row['processing_time'],
        'error': row['error'],
        'verified_label': row['verified_label'],
        'verified_at': row['verified_at'],
        'timestamp': row['timestamp'],
    }


def verify_history_entry(entry_id: int, verified_label: str) -> Optional[Dict[str, Any]]:
    normalized_label = verified_label.strip().upper()
    if normalized_label not in VALIDATED_LABELS:
        raise ValueError("verified_label must be either REAL or FAKE")

    conn = get_connection()
    cursor = conn.cursor()
    verified_at = datetime.now().isoformat()
    cursor.execute(
        """
        UPDATE query_history
        SET verified_label = ?, verified_at = ?
        WHERE id = ?
        """,
        (normalized_label, verified_at, entry_id),
    )
    updated_rows = cursor.rowcount
    conn.commit()
    conn.close()

    if updated_rows == 0:
        return None

    return get_history_entry(entry_id)


def get_verified_training_data(limit: Optional[int] = 1000) -> tuple[list[str], list[int]]:
    """Get verified examples for retraining the model."""
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT input_text, verified_label
        FROM query_history
        WHERE verified_label IS NOT NULL
          AND input_text IS NOT NULL
          AND TRIM(input_text) != ''
        ORDER BY COALESCE(verified_at, timestamp) DESC
    """
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    texts: list[str] = []
    labels: list[int] = []

    for row in rows:
        text = (row['input_text'] or "").strip()
        if text.strip():
            texts.append(text.strip())
            labels.append(1 if row['verified_label'].upper() == 'REAL' else 0)

    return texts, labels


def get_training_data_stats() -> Dict[str, Any]:
    """Get statistics about verified data available for retraining."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) AS total FROM query_history WHERE verified_label IS NOT NULL"
    )
    total = cursor.fetchone()['total'] or 0

    cursor.execute(
        "SELECT COUNT(*) AS real_count FROM query_history WHERE verified_label = 'REAL'"
    )
    real_count = cursor.fetchone()['real_count'] or 0

    cursor.execute(
        "SELECT COUNT(*) AS fake_count FROM query_history WHERE verified_label = 'FAKE'"
    )
    fake_count = cursor.fetchone()['fake_count'] or 0

    cursor.execute(
        "SELECT COUNT(*) AS total_predictions FROM query_history WHERE prediction IS NOT NULL"
    )
    total_predictions = cursor.fetchone()['total_predictions'] or 0

    conn.close()

    return {
        'total_samples': total,
        'real_samples': real_count,
        'fake_samples': fake_count,
        'verified_samples': total,
        'unverified_predictions': max(total_predictions - total, 0),
        'min_samples_for_retraining': MIN_VERIFIED_SAMPLES_FOR_RETRAINING,
    }


def get_history_total() -> int:
    """Get total number of history records"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS total FROM query_history")
    total = cursor.fetchone()['total'] or 0
    conn.close()
    return total


def get_history_stats() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS total FROM query_history")
    total = cursor.fetchone()['total'] or 0

    cursor.execute("SELECT COUNT(*) AS fake_count FROM query_history WHERE prediction = 'FAKE'")
    fake_count = cursor.fetchone()['fake_count'] or 0

    cursor.execute("SELECT COUNT(*) AS real_count FROM query_history WHERE prediction = 'REAL'")
    real_count = cursor.fetchone()['real_count'] or 0

    cursor.execute("SELECT AVG(confidence) AS avg_confidence FROM query_history WHERE confidence IS NOT NULL")
    avg_confidence = cursor.fetchone()['avg_confidence']
    avg_confidence = float(avg_confidence) if avg_confidence is not None else 0.0

    conn.close()

    return {
        'total': total,
        'fake_count': fake_count,
        'real_count': real_count,
        'avg_confidence': round(avg_confidence, 2),
    }
