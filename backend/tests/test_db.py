from pathlib import Path
import uuid

import db


def test_build_history_preview_shortens_long_text() -> None:
    long_text = "a" * 300

    preview = db._build_history_preview(long_text, None, max_length=50)

    assert preview is not None
    assert preview.endswith("...")
    assert len(preview) == 53


def test_verified_training_data_uses_temp_database(monkeypatch) -> None:
    temp_db = Path(__file__).resolve().parent / f"history-{uuid.uuid4().hex}.db"

    monkeypatch.setattr(db, "get_db_path", lambda: str(temp_db))

    try:
        db.init_db()
        entry_id = db.insert_query_history(
            source="text",
            input_text="Official sources confirmed the policy update.",
            url=None,
            prediction="REAL",
            confidence=91.2,
            fake_probability=8.8,
            real_probability=91.2,
            keywords=[],
            processing_time=0.2,
            error=None,
        )
        db.verify_history_entry(entry_id, "REAL")

        texts, labels = db.get_verified_training_data(limit=10)

        assert texts == ["Official sources confirmed the policy update."]
        assert labels == [1]
    finally:
        if temp_db.exists():
            temp_db.unlink()
