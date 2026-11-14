
from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")

RAW_CSV = ROOT / "data" / "raw" / "bank-full.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "bank_clean.csv"

MODEL_FILE = ROOT / "artifacts" / "model.joblib"
METADATA_FILE = ROOT / "artifacts" / "model_metadata.json"

REPORTS_DIR = ROOT / "reports"


def database_engine():
    url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER", "bankuser"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        database=os.getenv("DB_NAME", "bankpredict"),
    )

    return create_engine(url, pool_pre_ping=True)
