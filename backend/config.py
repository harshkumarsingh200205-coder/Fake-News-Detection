import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - safe fallback if dotenv is unavailable
    load_dotenv = None


BACKEND_DIR = Path(__file__).resolve().parent
ENV_PATH = BACKEND_DIR / ".env"

if load_dotenv is not None:
    load_dotenv(ENV_PATH)


def get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def get_list_env(name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    return [item.strip() for item in raw_value.split(",") if item.strip()]


APP_NAME = "Fake News Detector"
API_NAME = f"{APP_NAME} API"
DB_FILENAME = os.getenv("FAKE_NEWS_DB_FILENAME", "fake_news_history.db")
AUTO_RETRAIN_CHECK_INTERVAL = get_int_env(
    "FAKE_NEWS_AUTO_RETRAIN_CHECK_INTERVAL",
    50,
)
CORS_ALLOWED_ORIGINS = get_list_env(
    "FAKE_NEWS_CORS_ORIGINS",
    ["http://localhost:3000", "http://127.0.0.1:3000"],
)
